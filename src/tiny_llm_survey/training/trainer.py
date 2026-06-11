from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from tiny_llm_survey.config import CONFIGS, load_models, load_tasks, load_yaml, merge_configs
from tiny_llm_survey.training.data import load_training_dataset, task_id_map
from tiny_llm_survey.training.collator import TaskEmbeddingCollator
from tiny_llm_survey.training.simple_dataset import CausalLMDataset
from tiny_llm_survey.training.task_embedding import (
    OneHotTaskEmbedding,
    TaskDescriptionEmbedding,
    inject_task_embedding,
)


class TaskEmbeddingDataset(TorchDataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length: int = 2048,
        task_to_id: dict[str, int] | None = None,
    ):
        self.rows = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_to_id = task_to_id or task_id_map()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        item["task_id"] = torch.tensor(self.task_to_id[row["task"]], dtype=torch.long)
        return item


class TaskEmbeddingTrainer(Trainer):
    """Trainer that injects task embeddings into input_embeds."""

    def __init__(self, *args, task_module=None, base_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_module = task_module
        self.base_model = base_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        task_ids = inputs.pop("task_id")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        labels = inputs.pop("labels")

        input_embeds = inject_task_embedding(self.base_model, input_ids, task_ids, self.task_module)
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def _resolve_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def run_multitask_training(
    model_key: str,
    method_config_path: Path | str,
    max_samples_per_task: int = 2000,
) -> dict[str, Any]:
    """
    Train on new tasks with one of:
      - one_hot_task_emb
      - task_desc_emb
      - lora (ranks S/M/L via config)
    """
    models = load_models(include_optional=True)
    if model_key not in models:
        raise KeyError(f"Unknown model key '{model_key}'")

    model_cfg = models[model_key]
    train_cfg = merge_configs(CONFIGS / "training" / "base.yaml", method_config_path)
    tasks_cfg = load_tasks()
    method = train_cfg["method"]

    hf_id = model_cfg["hf_id"]
    dtype = _resolve_dtype(train_cfg["dtype"])
    device = train_cfg.get("device", "cuda")

    print(f"[train] {model_key} | method={method} | hf_id={hf_id}")

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

    task_names = tasks_cfg["task_splits"]["new_tasks"]
    raw_ds = load_training_dataset(task_names, max_samples_per_task=max_samples_per_task)
    dataset = TaskEmbeddingDataset(raw_ds, tokenizer, max_length=int(train_cfg["max_seq_length"]))

    run_name = f"{model_key}__{Path(method_config_path).stem}"
    out_dir = Path(train_cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        bf16=train_cfg["dtype"] == "bfloat16",
        fp16=train_cfg["dtype"] == "float16",
        report_to=[],
        seed=int(train_cfg["seed"]),
        remove_unused_columns=False,
    )

    meta: dict[str, Any] = {
        "model_key": model_key,
        "hf_id": hf_id,
        "method": method,
        "method_config": str(method_config_path),
        "train_tasks": task_names,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
    }

    if method == "lora":
        lora_config = LoraConfig(
            r=int(train_cfg["lora_r"]),
            lora_alpha=int(train_cfg["lora_alpha"]),
            lora_dropout=float(train_cfg["lora_dropout"]),
            target_modules=list(train_cfg["target_modules"]),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=CausalLMDataset(raw_ds, tokenizer, max_length=int(train_cfg["max_seq_length"])),
            data_collator=collator,
        )
        trainer.train()
        model.save_pretrained(out_dir / "adapter")
        meta["lora_r"] = train_cfg["lora_r"]

    elif method == "one_hot_task_emb":
        num_tasks = len(task_names)
        task_module = OneHotTaskEmbedding(
            num_tasks=num_tasks,
            hidden_size=base_model.config.hidden_size,
            embed_dim=int(train_cfg.get("task_embedding_dim", 64)),
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            task_module = task_module.to(device)

        trainer = TaskEmbeddingTrainer(
            model=base_model,
            base_model=base_model,
            task_module=task_module,
            args=training_args,
            train_dataset=dataset,
            data_collator=TaskEmbeddingCollator(),
        )
        trainer.train()
        torch.save(task_module.state_dict(), out_dir / "task_embedding.pt")
        base_model.save_pretrained(out_dir / "model")
        meta["task_embedding_dim"] = train_cfg.get("task_embedding_dim", 64)

    elif method == "task_desc_emb":
        descriptions = [tasks_cfg["task_descriptions"][t] for t in task_names]
        task_module = TaskDescriptionEmbedding(
            descriptions=descriptions,
            tokenizer=tokenizer,
            base_model=base_model,
            embed_dim=int(train_cfg.get("task_embedding_dim", 64)),
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            task_module = task_module.to(device)

        trainer = TaskEmbeddingTrainer(
            model=base_model,
            base_model=base_model,
            task_module=task_module,
            args=training_args,
            train_dataset=dataset,
            data_collator=TaskEmbeddingCollator(),
        )
        trainer.train()
        torch.save(task_module.state_dict(), out_dir / "task_desc_embedding.pt")
        base_model.save_pretrained(out_dir / "model")
        meta["task_embedding_dim"] = train_cfg.get("task_embedding_dim", 64)

    else:
        raise ValueError(f"Unknown training method: {method}")

    meta_path = out_dir / "training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[train] Done. Artifacts in {out_dir}")
    return meta
