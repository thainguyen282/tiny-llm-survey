"""Plot sequential catastrophic-forgetting curves (SDFT vs SFT style)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tiny_llm_survey.config import project_root

TASK_LABELS = {
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "openbookqa": "Science Q&A",
    "winogrande": "WinoGrande",
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "piqa": "PIQA",
}

METHOD_LABELS = {
    "lora": "SFT",
    "one_hot_task_emb": "Task Emb.",
    "task_desc_emb": "Task Desc. Emb.",
}


def _load_trajectory(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _trajectory_to_long_df(data: dict[str, Any], use_normalized: bool = True) -> pd.DataFrame:
    key = "normalized_trajectory" if use_normalized else "trajectory"
    points = data.get(key) or data.get("trajectory", [])
    rows = []
    for point in points:
        step = point["global_step"]
        scores = point.get("scores") or {}
        for task, value in scores.items():
            rows.append(
                {
                    "global_step": step,
                    "task": task,
                    "score": value,
                    "phase_index": point.get("phase_index"),
                    "phase_task": point.get("phase_task"),
                }
            )
    return pd.DataFrame(rows)


def plot_sequential_forgetting_curve(
    trajectory_path: Path | str,
    output_path: Path | str,
    *,
    title: str | None = None,
    plot_tasks: list[str] | None = None,
    use_normalized: bool = True,
    ylabel: str = "Normalized Performance",
) -> None:
    """Single-panel forgetting curve from one sequential run."""
    data = _load_trajectory(Path(trajectory_path))
    df = _trajectory_to_long_df(data, use_normalized=use_normalized)
    if df.empty:
        print(f"[forgetting_curve] No trajectory data in {trajectory_path}")
        return

    tasks = plot_tasks or data.get("eval_tasks") or data.get("task_order", [])
    df = df[df["task"].isin(tasks)]
    boundaries = data.get("phase_boundaries", [])
    task_order = data.get("task_order", [])
    method = data.get("method", "")
    panel_title = title or METHOD_LABELS.get(method, method)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("Blues", n_colors=max(len(tasks), 3))

    for idx, task in enumerate(tasks):
        sub = df[df["task"] == task].sort_values("global_step")
        if sub.empty:
            continue
        label = TASK_LABELS.get(task, task)
        ax.plot(
            sub["global_step"],
            sub["score"],
            label=label,
            color=palette[idx % len(palette)],
            linewidth=2,
        )

    for b in boundaries[1:-1]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ymax = max(1.0, df["score"].max() * 1.1)
    ymin = min(-0.1, df["score"].min() - 0.05)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)
    ax.legend(loc="best", fontsize=9)

    phase_labels = [f"Train on {TASK_LABELS.get(t, t)}" for t in task_order]
    if boundaries and phase_labels:
        for i, label in enumerate(phase_labels):
            if i + 1 >= len(boundaries):
                break
            x0 = boundaries[i]
            x1 = boundaries[i + 1]
            ax.text(
                (x0 + x1) / 2,
                ymax * 0.97,
                label,
                ha="center",
                va="top",
                fontsize=8,
                color="dimgray",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[forgetting_curve] Saved {output_path}")


def plot_sequential_forgetting_comparison(
    trajectory_paths: list[Path | str],
    output_path: Path | str,
    *,
    panel_titles: list[str] | None = None,
    plot_tasks: list[str] | None = None,
    use_normalized: bool = True,
) -> None:
    """Side-by-side panels comparing multiple sequential runs (e.g. SFT vs Task Emb.)."""
    paths = [Path(p) for p in trajectory_paths]
    n = len(paths)
    if n == 0:
        print("[forgetting_curve] No trajectory paths provided.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)

    for ax_idx, traj_path in enumerate(paths):
        ax = axes[0, ax_idx]
        data = _load_trajectory(traj_path)
        df = _trajectory_to_long_df(data, use_normalized=use_normalized)
        tasks = plot_tasks or data.get("eval_tasks") or data.get("task_order", [])
        df = df[df["task"].isin(tasks)]
        boundaries = data.get("phase_boundaries", [])
        task_order = data.get("task_order", [])
        method = data.get("method", "")

        if panel_titles and ax_idx < len(panel_titles):
            panel_title = panel_titles[ax_idx]
        else:
            panel_title = f"({chr(97 + ax_idx)}) {METHOD_LABELS.get(method, method)}"

        palette = sns.color_palette("Blues", n_colors=max(len(tasks), 3))
        for idx, task in enumerate(tasks):
            sub = df[df["task"] == task].sort_values("global_step")
            if sub.empty:
                continue
            ax.plot(
                sub["global_step"],
                sub["score"],
                label=TASK_LABELS.get(task, task),
                color=palette[idx % len(palette)],
                linewidth=2,
            )

        for b in boundaries[1:-1]:
            ax.axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ymax = max(1.0, df["score"].max() * 1.1) if not df.empty else 1.0
        ymin = min(-0.1, df["score"].min() - 0.05) if not df.empty else -0.1
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Gradient Steps")
        if ax_idx == 0:
            ax.set_ylabel("Normalized Performance")
        ax.set_title(panel_title)
        ax.legend(loc="best", fontsize=8)

        phase_labels = [f"Train on {TASK_LABELS.get(t, t)}" for t in task_order]
        for i, label in enumerate(phase_labels):
            if i + 1 >= len(boundaries):
                break
            x0, x1 = boundaries[i], boundaries[i + 1]
            ax.text((x0 + x1) / 2, ymax * 0.97, label, ha="center", va="top", fontsize=7, color="dimgray")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[forgetting_curve] Saved {output_path}")


def generate_forgetting_curve_assets(
    results_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Scan results/sequential/ and produce per-run and comparison figures."""
    root = results_dir or (project_root() / "results" / "sequential")
    out = output_dir or (project_root() / "blog_assets" / "forgetting_curves")
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if not root.exists():
        print(f"[forgetting_curve] No sequential results at {root}")
        return paths

    trajectories = sorted(root.glob("*/trajectory.json"))
    for traj in trajectories:
        run_name = traj.parent.name
        fig_path = out / f"fig_forgetting_{run_name}.png"
        plot_sequential_forgetting_curve(traj, fig_path)
        paths[run_name] = fig_path

    sft_traj = root / "qwen3-0.6b__sequential_sequential" / "trajectory.json"
    emb_traj = root / "qwen3-0.6b__sequential_sequential_task_emb" / "trajectory.json"
    compare_candidates = [p for p in [sft_traj, emb_traj] if p.exists()]
    if len(compare_candidates) >= 2:
        cmp_path = out / "fig_forgetting_sft_vs_task_emb.png"
        plot_sequential_forgetting_comparison(
            compare_candidates,
            cmp_path,
            panel_titles=["(a) Task Emb.", "(b) SFT"],
        )
        paths["comparison"] = cmp_path
    elif len(trajectories) >= 2:
        cmp_path = out / "fig_forgetting_comparison.png"
        plot_sequential_forgetting_comparison(
            [t for t in trajectories[:2]],
            cmp_path,
        )
        paths["comparison"] = cmp_path

    csv_rows = []
    for traj in trajectories:
        data = _load_trajectory(traj)
        for point in data.get("trajectory", []):
            for task, score in point.get("scores", {}).items():
                csv_rows.append(
                    {
                        "run": traj.parent.name,
                        "method": data.get("method"),
                        "global_step": point["global_step"],
                        "phase_task": point.get("phase_task"),
                        "task": task,
                        "score": score,
                    }
                )
    if csv_rows:
        csv_path = out / "sequential_trajectory.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        paths["csv"] = csv_path

    print(f"[forgetting_curve] Assets written to {out}")
    return paths
