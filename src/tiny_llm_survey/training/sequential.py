from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from tiny_llm_survey.config import CONFIGS, load_models, merge_configs
from tiny_llm_survey.eval.runner import run_tracked_task_eval
from tiny_llm_survey.training.collator import TaskEmbeddingCollator
from tiny_llm_survey.training.data import load_training_dataset, task_id_map
from tiny_llm_survey.training.simple_dataset import CausalLMDataset
from tiny_llm_survey.training.task_embedding import OneHotTaskEmbedding, inject_task_embedding
from tiny_llm_survey.training.trainer import TaskEmbeddingDataset, TaskEmbeddingTrainer, _resolve_dtype


class SequentialEvalCallback(TrainerCallback):
    """Run tracked benchmark evals at fixed step intervals during a training phase."""

    def __init__(
        self,
        *,
        model_key: str,
        eval_tasks: list[str],
        eval_interval: int,
        global_step_offset: int,
        phase_index: int,
        phase_task: str,
        trajectory: list[dict[str, Any]],
        adapter_dir: Path,
        eval_limit: int | None,
        task_module: OneHotTaskEmbedding | None = None,
        base_model=None,
    ) -> None:
        self.model_key = model_key
        self.eval_tasks = eval_tasks
        self.eval_interval = max(1, eval_interval)
        self.global_step_offset = global_step_offset
        self.phase_index = phase_index
        self.phase_task = phase_task
        self.trajectory = trajectory
        self.adapter_dir = adapter_dir
        self.eval_limit = eval_limit
        self.task_module = task_module
        self.base_model = base_model
        self._last_eval_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.eval_interval != 0:
            return
        if state.global_step == self._last_eval_step:
            return
        self._last_eval_step = state.global_step
        global_step = self.global_step_offset + state.global_step
        self._record_eval(global_step, kwargs.get("model"))

    def on_train_end(self, args, state, control, **kwargs):
        global_step = self.global_step_offset + state.global_step
        if global_step == self._last_eval_step:
            return
        self._record_eval(global_step, kwargs.get("model"))

    def _record_eval(self, global_step: int, model) -> None:
        adapter_path = self._save_checkpoint(model)
        print(f"[sequential] Eval at step {global_step} (phase {self.phase_index}: {self.phase_task})")
        scores = run_tracked_task_eval(
            self.model_key,
            self.eval_tasks,
            adapter_path=None if self.task_module is not None else adapter_path,
            checkpoint_path=adapter_path if self.task_module is not None else None,
            limit=self.eval_limit,
        )
        self.trajectory.append(
            {
                "global_step": global_step,
                "phase_index": self.phase_index,
                "phase_task": self.phase_task,
                "scores": scores,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._persist_trajectory()

    def _save_checkpoint(self, model) -> Path:
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        if self.task_module is not None:
            torch.save(self.task_module.state_dict(), self.adapter_dir / "task_embedding.pt")
            self.base_model.save_pretrained(self.adapter_dir / "model")
            return self.adapter_dir / "model"
        model.save_pretrained(self.adapter_dir)
        return self.adapter_dir

    def _persist_trajectory(self) -> None:
        path = self.adapter_dir.parent / "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trajectory, f, indent=2)


def _normalize_scores(
    trajectory: list[dict[str, Any]],
    eval_tasks: list[str],
    baseline_scores: dict[str, float],
) -> list[dict[str, float]]:
    """Normalize each task relative to its pre-training baseline (0 = baseline, 1 = perfect)."""
    peaks: dict[str, float] = dict(baseline_scores)
    normalized: list[dict[str, float]] = []

    for point in trajectory:
        raw = point["scores"]
        norm: dict[str, float] = {}
        for task in eval_tasks:
            score = raw.get(task)
            base = baseline_scores.get(task)
            if score is None or base is None:
                continue
            peaks[task] = max(peaks.get(task, base), score)
            denom = max(1.0 - base, 1e-6)
            norm[task] = (score - base) / denom
        normalized.append(norm)
    return normalized


def run_sequential_forgetting(
    model_key: str,
    method_config_path: Path | str,
    *,
    eval_limit: int | None = 50,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Train on tasks sequentially (phase 1 → phase 2 → …) while periodically
    evaluating all tracked benchmarks.  Produces a trajectory JSON suitable
    for forgetting-curve plots like SDFT vs SFT figures.
    """
    models = load_models(include_optional=True)
    if model_key not in models:
        raise KeyError(f"Unknown model key '{model_key}'")

    model_cfg = models[model_key]
    train_cfg = merge_configs(
        CONFIGS / "training" / "base.yaml",
        method_config_path,
    )
    method = train_cfg["method"]
    task_order: list[str] = train_cfg["task_order"]
    eval_tasks: list[str] = train_cfg.get("eval_tasks", task_order)
    steps_per_phase = int(train_cfg["steps_per_phase"])
    eval_interval = int(train_cfg.get("eval_interval_steps", 50))
    max_samples = int(train_cfg.get("max_samples_per_task", 500))

    hf_id = model_cfg["hf_id"]
    dtype = _resolve_dtype(train_cfg["dtype"])
    device = train_cfg.get("device", "cuda")

    run_name = f"{model_key}__sequential_{Path(method_config_path).stem}"
    out_dir = Path(output_dir or train_cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print(f"[sequential] {model_key} | method={method} | phases={task_order}")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=model_cfg.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    print("[sequential] Baseline eval (step 0) ...")
    baseline_scores = run_tracked_task_eval(model_key, eval_tasks, limit=eval_limit)
    trajectory: list[dict[str, Any]] = [
        {
            "global_step": 0,
            "phase_index": -1,
            "phase_task": "baseline",
            "scores": baseline_scores,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ]

    task_module: OneHotTaskEmbedding | None = None
    peft_model = None

    if method == "lora":
        lora_config = LoraConfig(
            r=int(train_cfg["lora_r"]),
            lora_alpha=int(train_cfg["lora_alpha"]),
            lora_dropout=float(train_cfg["lora_dropout"]),
            target_modules=list(train_cfg["target_modules"]),
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_config)
    elif method == "one_hot_task_emb":
        task_module = OneHotTaskEmbedding(
            num_tasks=len(task_order),
            hidden_size=base_model.config.hidden_size,
            embed_dim=int(train_cfg.get("task_embedding_dim", 64)),
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            task_module = task_module.to(device)
    else:
        raise ValueError(f"Sequential training supports 'lora' or 'one_hot_task_emb', got '{method}'")

    global_step_offset = 0
    phase_boundaries: list[int] = [0]

    for phase_idx, phase_task in enumerate(task_order):
        print(f"[sequential] === Phase {phase_idx + 1}/{len(task_order)}: {phase_task} ===")
        phase_ds = load_training_dataset([phase_task], max_samples_per_task=max_samples)
        phase_ckpt = ckpt_root / f"phase_{phase_idx:02d}_{phase_task}"

        training_args = TrainingArguments(
            output_dir=str(phase_ckpt / "trainer_state"),
            per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
            learning_rate=float(train_cfg["learning_rate"]),
            max_steps=steps_per_phase,
            warmup_ratio=float(train_cfg["warmup_ratio"]),
            logging_steps=int(train_cfg["logging_steps"]),
            bf16=train_cfg["dtype"] == "bfloat16",
            fp16=train_cfg["dtype"] == "float16",
            report_to=[],
            seed=int(train_cfg["seed"]) + phase_idx,
            remove_unused_columns=False,
            save_strategy="no",
        )

        callback = SequentialEvalCallback(
            model_key=model_key,
            eval_tasks=eval_tasks,
            eval_interval=eval_interval,
            global_step_offset=global_step_offset,
            phase_index=phase_idx,
            phase_task=phase_task,
            trajectory=trajectory,
            adapter_dir=phase_ckpt,
            eval_limit=eval_limit,
            task_module=task_module,
            base_model=base_model,
        )

        if method == "lora":
            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=CausalLMDataset(phase_ds, tokenizer, max_length=int(train_cfg["max_seq_length"])),
                data_collator=collator,
                callbacks=[callback],
            )
        else:
            id_map = {name: idx for idx, name in enumerate(task_order)}
            dataset = TaskEmbeddingDataset(
                phase_ds,
                tokenizer,
                max_length=int(train_cfg["max_seq_length"]),
                task_to_id=id_map,
            )
            trainer = TaskEmbeddingTrainer(
                model=base_model,
                base_model=base_model,
                task_module=task_module,
                args=training_args,
                train_dataset=dataset,
                data_collator=TaskEmbeddingCollator(),
                callbacks=[callback],
            )

        trainer.train()
        global_step_offset += steps_per_phase
        phase_boundaries.append(global_step_offset)

        if method == "lora":
            peft_model.save_pretrained(out_dir / "adapter")
        else:
            torch.save(task_module.state_dict(), out_dir / "task_embedding.pt")
            base_model.save_pretrained(out_dir / "model")

    normalized = _normalize_scores(trajectory, eval_tasks, baseline_scores)

    payload: dict[str, Any] = {
        "model_key": model_key,
        "method": method,
        "method_config": str(method_config_path),
        "task_order": task_order,
        "eval_tasks": eval_tasks,
        "steps_per_phase": steps_per_phase,
        "eval_interval_steps": eval_interval,
        "phase_boundaries": phase_boundaries,
        "baseline_scores": baseline_scores,
        "trajectory": trajectory,
        "normalized_trajectory": [
            {"global_step": trajectory[i]["global_step"], "scores": normalized[i]}
            for i in range(len(trajectory))
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
    }

    if method == "lora":
        payload["lora_r"] = int(train_cfg["lora_r"])

    traj_path = out_dir / "trajectory.json"
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    meta_path = out_dir / "sequential_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if k != "trajectory"}, f, indent=2)

    print(f"[sequential] Done. Trajectory saved to {traj_path}")
    return payload
