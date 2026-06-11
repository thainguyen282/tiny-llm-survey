from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tiny_llm_survey.config import project_root


def _parse_models(value: str | None) -> list[str] | None:
    if not value or value.lower() == "all":
        return None
    return [m.strip() for m in value.split(",") if m.strip()]


def run_mmlu() -> None:
    from tiny_llm_survey.eval.mmlu import run_all_mmlu, run_mmlu_eval

    parser = argparse.ArgumentParser(description="Run 5-shot MMLU evaluation (run this first)")
    parser.add_argument("--model", type=str, default="all", help="Model key or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (smoke test)")
    args = parser.parse_args()

    keys = _parse_models(args.model)
    if keys:
        for k in keys:
            run_mmlu_eval(k, limit=args.limit)
    else:
        run_all_mmlu(limit=args.limit)


def run_baseline() -> None:
    from tiny_llm_survey.eval.baseline import run_all_baselines, run_baseline_eval

    parser = argparse.ArgumentParser(description="Run 9-task baseline evaluation")
    parser.add_argument("--model", type=str, default="all", help="Model key or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (smoke test)")
    args = parser.parse_args()

    keys = _parse_models(args.model)
    if keys:
        for k in keys:
            run_baseline_eval(k, limit=args.limit)
    else:
        run_all_baselines(limit=args.limit)


def run_latency() -> None:
    from tiny_llm_survey.latency.benchmark import measure_latency, run_all_latency

    parser = argparse.ArgumentParser(description="Measure inference latency")
    parser.add_argument("--model", type=str, default="all")
    args = parser.parse_args()

    keys = _parse_models(args.model)
    if keys:
        for k in keys:
            measure_latency(k)
    else:
        run_all_latency()


def run_train() -> None:
    from tiny_llm_survey.training.trainer import run_multitask_training

    parser = argparse.ArgumentParser(description="Multi-task training for forgetting study")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--method-config",
        type=str,
        required=True,
        help="e.g. configs/training/lora_s.yaml",
    )
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    cfg_path = Path(args.method_config)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    run_multitask_training(args.model, cfg_path, max_samples_per_task=args.max_samples)


def run_forgetting() -> None:
    from tiny_llm_survey.eval.forgetting import run_forgetting_eval

    parser = argparse.ArgumentParser(description="Evaluate catastrophic forgetting")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--training-run", type=str, required=True, help="Path to training output dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.training_run)
    if not run_dir.is_absolute():
        run_dir = project_root() / run_dir

    run_forgetting_eval(args.model, run_dir, limit=args.limit)


def run_aggregate() -> None:
    from tiny_llm_survey.aggregation.report import generate_blog_assets

    parser = argparse.ArgumentParser(description="Aggregate results into blog assets")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    generate_blog_assets(out)


def run_visualize() -> None:
    from tiny_llm_survey.aggregation.mmlu_table import generate_mmlu_research_tables

    parser = argparse.ArgumentParser(description="Generate research tables and figures from results/")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/tables/)",
    )
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    generate_mmlu_research_tables(out)


def run_sequential() -> None:
    from tiny_llm_survey.training.sequential import run_sequential_forgetting

    parser = argparse.ArgumentParser(
        description="Sequential task training with periodic eval (forgetting curves)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--method-config",
        type=str,
        default="configs/training/sequential.yaml",
        help="Sequential config (sequential.yaml=SFT/LoRA, sequential_task_emb.yaml=task embeddings)",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=30,
        help="lm-eval sample limit per task during training (lower = faster)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.method_config)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    run_sequential_forgetting(args.model, cfg_path, eval_limit=args.eval_limit)


def run_forgetting_curves() -> None:
    from tiny_llm_survey.aggregation.forgetting_curve import (
        generate_forgetting_curve_assets,
        plot_sequential_forgetting_comparison,
        plot_sequential_forgetting_curve,
    )

    parser = argparse.ArgumentParser(description="Plot sequential forgetting curves from trajectory JSON")
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Path to trajectory.json (omit to scan results/sequential/)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="*",
        default=None,
        help="Two or more trajectory paths for side-by-side comparison",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else None

    if args.compare:
        out_path = out or (project_root() / "blog_assets" / "forgetting_curves" / "fig_comparison.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_sequential_forgetting_comparison(args.compare, out_path)
    elif args.trajectory:
        traj = Path(args.trajectory)
        if not traj.is_absolute():
            traj = project_root() / traj
        out_path = out or (traj.parent / "forgetting_curve.png")
        plot_sequential_forgetting_curve(traj, out_path)
    else:
        generate_forgetting_curve_assets(output_dir=out)


def run_list_models() -> None:
    from tiny_llm_survey.config import load_models

    models = load_models(include_optional=True)
    for key, cfg in models.items():
        print(f"{key:25s} {cfg['params_m']:>6}M  {cfg['tier']:12s}  {cfg['hf_id']}")


def main() -> None:
    cmds = {
        "mmlu": run_mmlu,
        "baseline": run_baseline,
        "latency": run_latency,
        "train": run_train,
        "sequential": run_sequential,
        "forgetting": run_forgetting,
        "forgetting-curves": run_forgetting_curves,
        "aggregate": run_aggregate,
        "visualize": run_visualize,
        "list-models": run_list_models,
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: tls-<command> [args]  OR  python -m tiny_llm_survey.cli <command> [args]")
        print(f"Commands: {', '.join(cmds)}")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    command = sys.argv.pop(1)
    if command not in cmds:
        print(f"Unknown command: {command}")
        print(f"Commands: {', '.join(cmds)}")
        sys.exit(1)

    cmds[command]()


if __name__ == "__main__":
    main()
