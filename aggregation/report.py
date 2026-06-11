from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tiny_llm_survey.config import load_models, load_tasks, project_root


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_mmlu_table(results_dir: Path | None = None) -> pd.DataFrame:
    root = results_dir or (project_root() / "results" / "mmlu")
    rows = []
    for model_dir in sorted(root.iterdir()) if root.exists() else []:
        data = _load_json(model_dir / "mmlu.json")
        if not data:
            continue
        row = {
            "model_key": data["model_key"],
            "params_m": data["params_m"],
            "tier": data["tier"],
            "mmlu": data.get("scores", {}).get("mmlu"),
            "num_fewshot": data.get("num_fewshot"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def collect_baseline_table(results_dir: Path | None = None) -> pd.DataFrame:
    root = results_dir or (project_root() / "results" / "baseline")
    rows = []
    for model_dir in sorted(root.iterdir()) if root.exists() else []:
        data = _load_json(model_dir / "baseline.json")
        if not data:
            continue
        row = {
            "model_key": data["model_key"],
            "params_m": data["params_m"],
            "tier": data["tier"],
            **data.get("scores", {}),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def collect_latency_table(results_dir: Path | None = None) -> pd.DataFrame:
    root = results_dir or (project_root() / "results" / "latency")
    rows = []
    for model_dir in sorted(root.iterdir()) if root.exists() else []:
        data = _load_json(model_dir / "latency.json")
        if not data:
            continue
        lat = data.get("latency", {})
        rows.append({
            "model_key": data["model_key"],
            "params_m": data["params_m"],
            "tier": data["tier"],
            "decode_tps": lat.get("decode_tokens_per_sec_mean"),
            "ttft_ms": lat.get("ttft_ms_mean"),
            "peak_vram_mb": lat.get("peak_vram_mb"),
        })
    return pd.DataFrame(rows)


def collect_forgetting_table(results_dir: Path | None = None) -> pd.DataFrame:
    root = results_dir or (project_root() / "results" / "forgetting")
    rows = []
    for run_dir in sorted(root.iterdir()) if root.exists() else []:
        data = _load_json(run_dir / "forgetting.json")
        if not data:
            continue
        f = data.get("forgetting", {})
        rows.append({
            "model_key": data["model_key"],
            "method": data["method"],
            "training_run": data["training_run"],
            "retention_mean": f.get("retention_mean"),
            "forgetting_score": f.get("forgetting_score"),
            "avg_9_after": data.get("after_scores", {}).get("avg_9_tasks"),
            "avg_9_before": data.get("baseline_scores", {}).get("avg_9_tasks"),
        })
    return pd.DataFrame(rows)


def build_table1_forgetting_detail(results_dir: Path | None = None) -> pd.DataFrame:
    """Reproduce reference Table 1 style: per-task scores by method."""
    root = results_dir or (project_root() / "results" / "forgetting")
    tasks_cfg = load_tasks()
    bench_names = ["mmlu"] + [b["name"] for b in tasks_cfg["benchmarks"]]

    rows = []
    for run_dir in sorted(root.iterdir()) if root.exists() else []:
        data = _load_json(run_dir / "forgetting.json")
        if not data:
            continue
        for phase, scores in [("base", data["baseline_scores"]), ("after", data["after_scores"])]:
            row = {
                "model_key": data["model_key"],
                "method": data["method"] if phase == "after" else "base",
                "phase": phase,
            }
            for name in bench_names:
                row[name] = scores.get(name)
            row["avg_9_tasks"] = scores.get("avg_9_tasks")
            rows.append(row)
    return pd.DataFrame(rows)


def plot_latency_vs_performance(
    latency_df: pd.DataFrame,
    output_path: Path,
    mmlu_df: pd.DataFrame | None = None,
    baseline_df: pd.DataFrame | None = None,
) -> None:
    if latency_df.empty:
        print("[aggregate] Skipping Pareto plot — missing latency data.")
        return

    perf_df = mmlu_df if mmlu_df is not None and not mmlu_df.empty else baseline_df
    perf_col = "mmlu" if perf_df is not None and "mmlu" in perf_df.columns else "avg_9_tasks"
    if perf_df is None or perf_df.empty or perf_col not in perf_df.columns:
        print("[aggregate] Skipping Pareto plot — missing MMLU/baseline scores.")
        return

    merged = perf_df.merge(latency_df, on="model_key", suffixes=("_p", "_l"))
    tier_col = "tier_p" if "tier_p" in merged.columns else "tier"
    params_col = "params_m_p" if "params_m_p" in merged.columns else "params_m"

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for tier in merged[tier_col].unique():
        sub = merged[merged[tier_col] == tier]
        ax.scatter(
            sub["decode_tps"],
            sub[perf_col],
            s=sub[params_col] * 0.3,
            alpha=0.75,
            label=tier,
        )
        for _, r in sub.iterrows():
            ax.annotate(r["model_key"], (r["decode_tps"], r[perf_col]), fontsize=8)

    ylabel = "MMLU (5-shot)" if perf_col == "mmlu" else "Average 9-task score"
    ax.set_xlabel("Decode throughput (tokens/sec)")
    ax.set_ylabel(ylabel)
    ax.set_title("Latency vs Performance — Tiny & Super-Tiny LLMs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] Saved {output_path}")


def plot_forgetting_heatmap(forgetting_df: pd.DataFrame, output_path: Path) -> None:
    if forgetting_df.empty:
        print("[aggregate] Skipping forgetting heatmap — no data.")
        return

    pivot = forgetting_df.pivot_table(
        index="model_key", columns="method", values="retention_mean", aggfunc="mean"
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Old-Task Retention by Model & Method")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] Saved {output_path}")


def generate_blog_assets(output_dir: Path | None = None) -> dict[str, Path]:
    out = output_dir or (project_root() / "blog_assets")
    out.mkdir(parents=True, exist_ok=True)

    mmlu_df = collect_mmlu_table()
    baseline_df = collect_baseline_table()
    latency_df = collect_latency_table()
    forgetting_df = collect_forgetting_table()
    table1_df = build_table1_forgetting_detail()

    paths: dict[str, Path] = {}
    if not mmlu_df.empty:
        p = out / "table_mmlu.csv"
        mmlu_df.to_csv(p, index=False)
        paths["mmlu"] = p

    if not baseline_df.empty:
        p = out / "table_baseline.csv"
        baseline_df.to_csv(p, index=False)
        paths["baseline"] = p

    if not latency_df.empty:
        p = out / "table_latency.csv"
        latency_df.to_csv(p, index=False)
        paths["latency"] = p

    if not forgetting_df.empty:
        p = out / "table_forgetting.csv"
        forgetting_df.to_csv(p, index=False)
        paths["forgetting"] = p

    if not table1_df.empty:
        p = out / "table1_task_scores.csv"
        table1_df.to_csv(p, index=False)
        paths["table1"] = p

    perf_for_merge = mmlu_df if not mmlu_df.empty else baseline_df
    if not perf_for_merge.empty and not latency_df.empty:
        merged = perf_for_merge.merge(latency_df, on="model_key", how="outer")
        p = out / "table_latency_vs_performance.csv"
        merged.to_csv(p, index=False)
        paths["merged"] = p
        plot_latency_vs_performance(
            latency_df,
            out / "fig_latency_vs_performance.png",
            mmlu_df=mmlu_df,
            baseline_df=baseline_df,
        )

    plot_forgetting_heatmap(forgetting_df, out / "fig_forgetting_heatmap.png")

    # Markdown summary stub for Medium
    md_lines = [
        "# Tiny LLM Survey — Results Summary\n",
        "## Body 1: Latency vs Performance\n",
        f"MMLU models: {len(mmlu_df)} | 9-task models: {len(baseline_df)}\n",
        "## Body 2: Catastrophic Forgetting\n",
        f"Training runs: {len(forgetting_df)}\n",
        "## Body 3: Suggested Improvements\n",
        "See table1_task_scores.csv and forgetting heatmap for method comparison.\n",
    ]
    md_path = out / "blog_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    paths["blog_summary"] = md_path

    print(f"[aggregate] Blog assets written to {out}")
    return paths
