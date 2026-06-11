from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiny_llm_survey.config import CONFIGS, load_models, load_yaml, merge_configs


def _resolve_dtype(dtype_name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _extract_scores(
    results: dict[str, Any],
    benchmarks: list[dict[str, Any]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for bench in benchmarks:
        bench_name = bench["name"]
        task_key = bench["lm_eval_task"]
        metric_key = bench["metric"]
        if task_key not in results:
            continue
        task_res = results[task_key]
        if metric_key in task_res:
            scores[bench_name] = float(task_res[metric_key])
            continue
        for k, v in task_res.items():
            if k.endswith(",none") and isinstance(v, (int, float)):
                scores[bench_name] = float(v)
                break
    return scores


def _extract_mmlu_subjects(raw_results: dict[str, Any]) -> dict[str, float]:
    """Pull per-subject MMLU scores when lm-eval returns them."""
    subjects: dict[str, float] = {}
    mmlu_res = raw_results.get("mmlu", {})
    for key, value in mmlu_res.items():
        if not key.endswith(",none"):
            continue
        if key == "acc,none":
            continue
        if isinstance(value, (int, float)):
            subject = key.replace(",none", "").replace("acc,", "")
            subjects[subject] = float(value)
    return subjects


def _build_hf_lm(
    model_key: str,
    suite_cfg: dict[str, Any],
    *,
    adapter_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
):
    """Load an lm-eval HFLM wrapper for base, merged, or LoRA checkpoints."""
    import torch
    from lm_eval.models.huggingface import HFLM

    models = load_models(include_optional=True)
    model_cfg = models[model_key]
    dtype = _resolve_dtype(suite_cfg.get("dtype", "bfloat16"))
    hf_id = model_cfg["hf_id"]
    trust = model_cfg.get("trust_remote_code", False)
    device = suite_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None:
        pretrained = str(checkpoint_path)
        peft = None
    elif adapter_path is not None:
        pretrained = hf_id
        peft = str(adapter_path)
    else:
        pretrained = hf_id
        peft = None

    return HFLM(
        pretrained=pretrained,
        peft=peft,
        dtype=dtype,
        trust_remote_code=trust,
        batch_size=suite_cfg.get("batch_size", "auto"),
        device=device,
    )


def run_eval_suite(
    model_key: str,
    suite_config_path: Path | str | None = None,
    suite_cfg: dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
    limit: int | None = None,
    output_filename: str = "results.json",
    adapter_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """Run lm-eval for a benchmark suite defined in a YAML config or dict."""
    from lm_eval import evaluator

    models = load_models(include_optional=True)
    if model_key not in models:
        raise KeyError(f"Unknown model key '{model_key}'. Available: {sorted(models)}")

    model_cfg = models[model_key]
    if suite_cfg is None:
        if suite_config_path is None:
            raise ValueError("Provide suite_config_path or suite_cfg")
        suite_cfg = load_yaml(suite_config_path)
    benchmarks = suite_cfg["benchmarks"]
    task_names = [b["lm_eval_task"] for b in benchmarks]
    suite_name = suite_cfg.get("suite") or (
        Path(suite_config_path).stem if suite_config_path else "eval"
    )

    out = Path(output_dir or suite_cfg["output_dir"]) / model_key
    if save_results:
        out.mkdir(parents=True, exist_ok=True)

    hf_id = model_cfg["hf_id"]

    load_label = checkpoint_path or adapter_path or hf_id
    print(f"[{suite_name}] Loading {load_label} ({model_key}) ...")
    lm = _build_hf_lm(
        model_key,
        suite_cfg,
        adapter_path=adapter_path,
        checkpoint_path=checkpoint_path,
    )

    lm_eval_cfg = suite_cfg.get("lm_eval", {})
    num_fewshot = lm_eval_cfg.get("num_fewshot")
    eval_limit = limit if limit is not None else lm_eval_cfg.get("limit")

    # Per-task fewshot from benchmark config when global num_fewshot is null
    if num_fewshot is None and len(benchmarks) == 1 and benchmarks[0].get("shots") is not None:
        num_fewshot = benchmarks[0]["shots"]

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=num_fewshot,
        limit=eval_limit,
        bootstrap_iters=lm_eval_cfg.get("bootstrap_iters", 0),
    )

    scores = _extract_scores(results["results"], benchmarks)

    if len(scores) > 1:
        avg = sum(scores.values()) / len(scores)
        if suite_name == "baseline_9task":
            scores["avg_9_tasks"] = avg
        else:
            scores[f"avg_{len(scores)}_tasks"] = avg

    payload: dict[str, Any] = {
        "suite": suite_name,
        "model_key": model_key,
        "hf_id": hf_id,
        "params_m": model_cfg["params_m"],
        "tier": model_cfg["tier"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_fewshot": num_fewshot,
        "scores": scores,
        "raw_results": results["results"],
    }

    if suite_name == "mmlu":
        payload["mmlu_subjects"] = _extract_mmlu_subjects(results["results"])

    if save_results:
        out_file = out / output_filename
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        headline = scores.get("mmlu") or scores.get("avg_9_tasks") or next(iter(scores.values()), None)
        print(f"[{suite_name}] Saved {out_file}")
        if headline is not None:
            print(f"[{suite_name}] headline score = {headline:.4f}")
    return payload


def run_tracked_task_eval(
    model_key: str,
    task_names: list[str],
    *,
    adapter_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    limit: int | None = 50,
    eval_config_path: Path | str | None = None,
) -> dict[str, float]:
    """Evaluate a subset of benchmark tasks (used during sequential training)."""
    from tiny_llm_survey.config import benchmark_by_name

    suite_cfg = merge_configs(CONFIGS / "eval.yaml")
    if eval_config_path:
        suite_cfg.update(load_yaml(eval_config_path))

    benches = benchmark_by_name()
    scores: dict[str, float] = {}
    for name in task_names:
        if name == "mmlu":
            payload = run_eval_suite(
                model_key,
                suite_config_path=CONFIGS / "mmlu.yaml",
                limit=limit,
                adapter_path=adapter_path,
                checkpoint_path=checkpoint_path,
                save_results=False,
            )
            scores.update(payload.get("scores", {}))
            continue
        if name not in benches:
            print(f"[tracked_eval] Unknown task '{name}', skipping.")
            continue
        bench = benches[name]
        mini_cfg = {
            **suite_cfg,
            "suite": f"tracked_{name}",
            "benchmarks": [bench],
        }
        payload = run_eval_suite(
            model_key,
            suite_cfg=mini_cfg,
            limit=limit,
            adapter_path=adapter_path,
            checkpoint_path=checkpoint_path,
            save_results=False,
        )
        scores.update(payload.get("scores", {}))
    return scores


def run_all_eval_suite(
    suite_config_path: Path | str,
    model_keys: list[str] | None = None,
    limit: int | None = None,
    output_filename: str = "results.json",
) -> list[dict[str, Any]]:
    keys = model_keys or list(load_models().keys())
    return [
        run_eval_suite(k, suite_config_path, limit=limit, output_filename=output_filename)
        for k in keys
    ]
