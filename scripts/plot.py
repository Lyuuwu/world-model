import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = ("gru", "mamba")
DEFAULT_PHASES = ("eval", "train")
DEFAULT_COLORS = {
    "gru": "#2563eb",
    "mamba": "#dc2626",
}
NON_METRIC_KEYS = {"phase", "env_step", "step"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot DreamerV3 train/eval jsonl logs across seeds. Each task gets "
            "one output folder containing GRU vs Mamba mean +/- std figures."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help=(
            "Task names under runs/dreamerv3. Use 'all' for every task. "
            "Aliases like atari_bank_heist resolve to atari100k_bank_heist."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds to plot starting from 0. Example: 3 means seed_0..seed_2.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model folders to compare. Default: gru mamba.",
    )
    parser.add_argument(
        "--profile",
        default="small",
        help="Profile suffix used in model profile folders, e.g. gru_small and mamba_small.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("train", "eval"),
        default=list(DEFAULT_PHASES),
        help="Which phase jsonl files to plot.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric names to plot. By default, plots every numeric metric found.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "dreamerv3",
        help="Root directory containing task run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "imgs",
        help="Directory where imgs/{task}/ figures are written.",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Also plot system/* metrics such as CUDA memory.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Optional rolling mean window applied per seed before aggregation.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved image DPI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be greater than 0.")
    if args.smooth <= 0:
        raise SystemExit("--smooth must be greater than 0.")

    run_root = args.run_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    tasks = resolve_tasks(run_root, args.tasks)

    if not tasks:
        print(f"No tasks found in {run_root}", file=sys.stderr)
        return 1

    for task_dir in tasks:
        plot_task(
            task_dir=task_dir,
            out_dir=out_dir / task_dir.name,
            models=args.models,
            profile=args.profile,
            seed_count=args.seeds,
            phases=args.phases,
            requested_metrics=args.metrics,
            include_system=args.include_system,
            smooth=args.smooth,
            dpi=args.dpi,
        )
    return 0


def resolve_tasks(run_root: Path, task_names: list[str]) -> list[Path]:
    if not run_root.exists():
        raise SystemExit(f"Run root does not exist: {run_root}")

    available = {path.name: path for path in sorted(run_root.iterdir()) if path.is_dir()}
    if any(name.lower() == "all" for name in task_names):
        return list(available.values())

    resolved: list[Path] = []
    for task_name in task_names:
        candidates = task_aliases(task_name)
        match = next((available[name] for name in candidates if name in available), None)
        if match is None:
            known = ", ".join(available)
            print(f"Warning: task '{task_name}' not found. Available: {known}", file=sys.stderr)
            continue
        resolved.append(match)
    return dedupe_paths(resolved)


def task_aliases(task_name: str) -> list[str]:
    clean = task_name.strip().strip("/")
    aliases = [clean]
    if clean.startswith("atari_"):
        aliases.append("atari100k_" + clean.removeprefix("atari_"))
    if clean.startswith("atari100k_"):
        aliases.append("atari_" + clean.removeprefix("atari100k_"))
    return aliases


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def plot_task(
    task_dir: Path,
    out_dir: Path,
    models: list[str],
    profile: str,
    seed_count: int,
    phases: list[str],
    requested_metrics: list[str] | None,
    include_system: bool,
    smooth: int,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTask: {task_dir.name}")

    phase_data: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for phase in phases:
        phase_data[phase] = {}
        for model in models:
            frames = load_model_phase_frames(task_dir, model, profile, seed_count, phase, smooth)
            if frames:
                phase_data[phase][model] = frames
                seed_labels = ", ".join(f"seed_{int(frame.attrs['seed'])}" for frame in frames)
                print(f"  {phase:<5} {model:<8}: {len(frames)} run(s) [{seed_labels}]")
            else:
                print(f"  {phase:<5} {model:<8}: no data")

    metric_manifest: dict[str, list[str]] = {}
    for phase, model_frames in phase_data.items():
        metrics = choose_metrics(model_frames, requested_metrics, include_system)
        metric_manifest[phase] = metrics
        for metric in metrics:
            save_path = out_dir / f"{phase}_{slugify(metric)}.png"
            plot_metric(
                task_name=task_dir.name,
                phase=phase,
                metric=metric,
                model_frames=model_frames,
                save_path=save_path,
                dpi=dpi,
            )
    write_manifest(out_dir, task_dir.name, metric_manifest)


def load_model_phase_frames(
    task_dir: Path,
    model: str,
    profile: str,
    seed_count: int,
    phase: str,
    smooth: int,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for seed in range(seed_count):
        jsonl_path = latest_jsonl_for_seed(task_dir, model, profile, seed, phase)
        if jsonl_path is None:
            continue
        frame = read_jsonl_frame(jsonl_path)
        if frame.empty:
            continue
        frame = normalize_frame(frame, seed, jsonl_path, smooth)
        if not frame.empty:
            frames.append(frame)
    return frames


def latest_jsonl_for_seed(
    task_dir: Path,
    model: str,
    profile: str,
    seed: int,
    phase: str,
) -> Path | None:
    profile_dir = task_dir / model / f"{model}_{profile}" / f"seed_{seed}"
    if not profile_dir.exists():
        matches = sorted((task_dir / model).glob(f"{model}_*/seed_{seed}"))
        if len(matches) == 1:
            profile_dir = matches[0]
        else:
            return None

    candidates = sorted(
        (path / f"{phase}.jsonl" for path in profile_dir.iterdir() if path.is_dir()),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    return next((path for path in candidates if path.exists()), None)


def read_jsonl_frame(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: skip bad JSON in {path}:{line_no}: {exc}", file=sys.stderr)
    return pd.DataFrame.from_records(records)


def normalize_frame(frame: pd.DataFrame, seed: int, path: Path, smooth: int) -> pd.DataFrame:
    x_col = "env_step" if "env_step" in frame.columns else "step"
    if x_col not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame = frame.dropna(subset=[x_col]).sort_values(x_col)
    frame = frame.rename(columns={x_col: "env_step"})
    frame = frame.drop_duplicates(subset=["env_step"], keep="last")

    numeric_cols = []
    for col in frame.columns:
        if col == "env_step" or col == "phase":
            continue
        converted = pd.to_numeric(frame[col], errors="coerce")
        if converted.notna().any():
            frame[col] = converted
            numeric_cols.append(col)

    keep_cols = ["env_step", *numeric_cols]
    frame = frame[keep_cols].set_index("env_step")
    if smooth > 1:
        frame[numeric_cols] = frame[numeric_cols].rolling(
            window=smooth, min_periods=1
        ).mean()

    frame.attrs["seed"] = seed
    frame.attrs["path"] = str(path)
    return frame


def choose_metrics(
    model_frames: dict[str, list[pd.DataFrame]],
    requested_metrics: list[str] | None,
    include_system: bool,
) -> list[str]:
    available: set[str] = set()
    for frames in model_frames.values():
        for frame in frames:
            available.update(
                col for col in frame.columns
                if col not in NON_METRIC_KEYS and is_metric_column(col, include_system)
            )

    if requested_metrics is not None:
        missing = [metric for metric in requested_metrics if metric not in available]
        for metric in missing:
            print(f"Warning: metric '{metric}' was not found in this phase.", file=sys.stderr)
        return [metric for metric in requested_metrics if metric in available]

    return sorted(available, key=metric_sort_key)


def is_metric_column(column: str, include_system: bool) -> bool:
    if column in NON_METRIC_KEYS:
        return False
    if not include_system and column.startswith("system/"):
        return False
    return True


def metric_sort_key(metric: str) -> tuple[int, str]:
    priority = {
        "return_mean": 0,
        "ep_return_mean": 1,
        "return_median": 2,
        "length_mean": 3,
        "loss/total": 10,
        "loss/image": 11,
        "loss/rew": 12,
        "loss/value": 13,
        "loss/policy": 14,
        "kl/dyn_raw": 20,
        "kl/rep_raw": 21,
        "fps": 30,
        "train_updates_per_sec": 31,
    }
    return (priority.get(metric, 100), metric)


def plot_metric(
    task_name: str,
    phase: str,
    metric: str,
    model_frames: dict[str, list[pd.DataFrame]],
    save_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    plotted = False

    for model, frames in model_frames.items():
        if not any(metric in frame.columns for frame in frames):
            continue
        stats = aggregate_metric(frames, metric)
        if stats.empty:
            continue

        color = DEFAULT_COLORS.get(model, None)
        x = stats.index.to_numpy(dtype=float)
        mean = stats["mean"].to_numpy(dtype=float)
        std = stats["std"].to_numpy(dtype=float)

        ax.plot(x, mean, label=f"{model} mean", color=color, linewidth=2.0)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
            label=f"{model} +/- std",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(f"{task_name} - {phase} - {metric}")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def aggregate_metric(frames: list[pd.DataFrame], metric: str) -> pd.DataFrame:
    series = []
    for frame in frames:
        if metric not in frame.columns:
            continue
        seed = frame.attrs.get("seed", len(series))
        series.append(frame[metric].rename(f"seed_{seed}"))

    if not series:
        return pd.DataFrame()

    values = pd.concat(series, axis=1).sort_index()
    values = values.replace([np.inf, -np.inf], np.nan)
    stats = pd.DataFrame(
        {
            "mean": values.mean(axis=1, skipna=True),
            "std": values.std(axis=1, skipna=True, ddof=0),
            "count": values.count(axis=1),
        }
    )
    stats = stats[stats["count"] > 0]
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def write_manifest(out_dir: Path, task_name: str, metric_manifest: dict[str, list[str]]) -> None:
    manifest = {
        "task": task_name,
        "phases": metric_manifest,
    }
    path = out_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    total = sum(len(metrics) for metrics in metric_manifest.values())
    print(f"  saved {total} plot(s) to {out_dir}")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return slug or "metric"


if __name__ == "__main__":
    raise SystemExit(main())
