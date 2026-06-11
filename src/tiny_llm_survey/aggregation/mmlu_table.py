"""Research-oriented MMLU result tables and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tiny_llm_survey.config import project_root

MMLU_CATEGORY_KEYS = {
    "stem": "mmlu_stem",
    "humanities": "mmlu_humanities",
    "social_sciences": "mmlu_social_sciences",
    "other": "mmlu_other",
}

SUMMARY_COLUMNS = [
    "model_key",
    "params_m",
    "tier",
    "mmlu",
    "stem",
    "humanities",
    "social_sciences",
    "other",
]

DISPLAY_NAMES = {
    "model_key": "Model",
    "params_m": "Params (M)",
    "tier": "Tier",
    "mmlu": "MMLU",
    "stem": "STEM",
    "humanities": "Humanities",
    "social_sciences": "Social Sci.",
    "other": "Other",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _acc(raw: dict[str, Any]) -> float | None:
    val = raw.get("acc,none")
    return float(val) if val is not None else None


def _is_subject_key(key: str) -> bool:
    if not key.startswith("mmlu_"):
        return False
    return key not in MMLU_CATEGORY_KEYS.values() and key != "mmlu"


def load_mmlu_results(results_dir: Path | None = None) -> list[dict[str, Any]]:
    root = results_dir or (project_root() / "results" / "mmlu")
    results: list[dict[str, Any]] = []
    if not root.exists():
        return results
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        data = _load_json(model_dir / "mmlu.json")
        if data:
            results.append(data)
    return results


def build_mmlu_summary_table(results_dir: Path | None = None) -> pd.DataFrame:
    """Main research table: overall MMLU plus four subject categories."""
    rows: list[dict[str, Any]] = []
    for data in load_mmlu_results(results_dir):
        raw = data.get("raw_results", {})
        row = {
            "model_key": data["model_key"],
            "params_m": data["params_m"],
            "tier": data["tier"],
            "mmlu": data.get("scores", {}).get("mmlu") or _acc(raw.get("mmlu", {})),
        }
        for col, key in MMLU_CATEGORY_KEYS.items():
            row[col] = _acc(raw.get(key, {}))
        rows.append(row)

    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if df.empty:
        return df
    return df.sort_values(["params_m", "model_key"], ignore_index=True)


def build_mmlu_subject_matrix(results_dir: Path | None = None) -> pd.DataFrame:
    """Per-subject scores: subjects as rows, models as columns (appendix table)."""
    results = load_mmlu_results(results_dir)
    if not results:
        return pd.DataFrame()

    subject_names: set[str] = set()
    model_scores: dict[str, dict[str, float | None]] = {}

    for data in results:
        model_key = data["model_key"]
        raw = data.get("raw_results", {})
        scores: dict[str, float | None] = {}
        for key, entry in raw.items():
            if not _is_subject_key(key):
                continue
            alias = entry.get("alias") or key.removeprefix("mmlu_")
            subject_names.add(alias)
            scores[alias] = _acc(entry)
        model_scores[model_key] = scores

    subjects = sorted(subject_names)
    matrix = {model: [model_scores[model].get(s) for s in subjects] for model in model_scores}
    return pd.DataFrame(matrix, index=subjects)


def format_pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value * 100:.{digits}f}"


def summary_table_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with accuracy columns formatted as percentages."""
    if df.empty:
        return df
    out = df.copy()
    for col in ["mmlu", *MMLU_CATEGORY_KEYS]:
        if col in out.columns:
            out[col] = out[col].map(lambda v: format_pct(v))
    return out.rename(columns=DISPLAY_NAMES)


def to_markdown_table(df: pd.DataFrame, caption: str = "") -> str:
    display = summary_table_display(df)
    lines: list[str] = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    headers = list(display.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def to_latex_table(
    df: pd.DataFrame,
    caption: str = "MMLU 5-shot accuracy (\\%) on tiny and super-tiny LLMs.",
    label: str = "tab:mmlu",
) -> str:
    display = summary_table_display(df)
    latex = display.to_latex(
        index=False,
        escape=True,
        column_format="lrrl" + "r" * 5,
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        "\\end{table}\n"
    )


def _best_indices(series: pd.Series) -> set[int]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return set()
    best = numeric.max()
    return set(numeric[numeric == best].index.tolist())


def plot_mmlu_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    """Render the summary table as a publication-style figure."""
    if df.empty:
        print("[visualize] Skipping summary table figure — no MMLU data.")
        return

    display = summary_table_display(df)
    score_cols = ["mmlu", *MMLU_CATEGORY_KEYS.keys()]
    best_cells: set[tuple[int, int]] = set()
    for col in score_cols:
        table_col = list(display.columns).index(DISPLAY_NAMES[col])
        for row_idx in _best_indices(df[col]):
            best_cells.add((row_idx + 1, table_col))

    n_rows, n_cols = display.shape
    fig_w = max(10, n_cols * 1.1)
    fig_h = max(2.5, 0.45 * (n_rows + 1) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
            cell.set_facecolor("#ecf0f1" if row % 2 == 0 else "#f8f9fa")
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 0 else "#f8f9fa")
            if (row, col) in best_cells:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#d5f5e3")

    ax.set_title("MMLU 5-shot Accuracy (%) — Tiny LLM Survey", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved {output_path}")


def plot_mmlu_category_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of model × MMLU category for quick comparison."""
    if df.empty:
        print("[visualize] Skipping category heatmap — no MMLU data.")
        return

    cols = ["mmlu", *MMLU_CATEGORY_KEYS.keys()]
    heat = df.set_index("model_key")[cols].multiply(100)
    heat.columns = [DISPLAY_NAMES[c] for c in cols]

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.2), max(3, len(heat) * 0.45 + 1)))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".1f",
        cmap="YlGn",
        vmin=0,
        vmax=max(60, heat.max().max() * 1.05),
        linewidths=0.5,
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("MMLU Scores by Model and Category")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved {output_path}")


def generate_mmlu_research_tables(output_dir: Path | None = None) -> dict[str, Path]:
    """Write CSV, Markdown, LaTeX, and figure assets for MMLU results."""
    out = output_dir or (project_root() / "results" / "tables")
    out.mkdir(parents=True, exist_ok=True)

    summary = build_mmlu_summary_table()
    subjects = build_mmlu_subject_matrix()
    paths: dict[str, Path] = {}

    if summary.empty:
        print("[visualize] No MMLU results found under results/mmlu/")
        return paths

    csv_path = out / "mmlu_summary.csv"
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    paths["summary_csv"] = csv_path

    md_path = out / "mmlu_summary.md"
    md_path.write_text(
        to_markdown_table(
            summary,
            caption="Table 1. MMLU 5-shot accuracy (%) on surveyed models.",
        ),
        encoding="utf-8",
    )
    paths["summary_md"] = md_path

    tex_path = out / "mmlu_summary.tex"
    tex_path.write_text(to_latex_table(summary), encoding="utf-8")
    paths["summary_tex"] = tex_path

    if not subjects.empty:
        subj_path = out / "mmlu_subjects.csv"
        subjects.multiply(100).to_csv(subj_path, float_format="%.2f")
        paths["subjects_csv"] = subj_path

    plot_mmlu_summary_table(summary, out / "mmlu_summary_table.png")
    paths["summary_fig"] = out / "mmlu_summary_table.png"

    plot_mmlu_category_heatmap(summary, out / "mmlu_category_heatmap.png")
    paths["heatmap_fig"] = out / "mmlu_category_heatmap.png"

    print(f"[visualize] MMLU research tables written to {out}")
    return paths
