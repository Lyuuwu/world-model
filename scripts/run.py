import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scripts/train.py sequentially for profiles and consecutive seeds."
    )
    parser.add_argument(
        "--task",
        required=True,
        help='Task in "domain_game" format, e.g. atari_pong.',
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="Profiles to train in order, e.g. gru_small mamba_small.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First seed to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to run.",
    )
    parser.add_argument(
        "--agent",
        default="dreamerv3",
        help="Agent passed to scripts/train.py.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device passed to scripts/train.py, e.g. "auto", "cpu", or "cuda:0".',
    )
    parser.add_argument(
        "--override",
        default=None,
        help="Optional override string passed to every scripts/train.py run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the train commands without starting training.",
    )
    return parser.parse_args()


def seed_range(start_seed: int, seed_count: int) -> range:
    if start_seed < 0:
        raise SystemExit("--seed must be greater than or equal to 0.")
    if seed_count <= 0:
        raise SystemExit("--seeds must be greater than 0.")
    return range(start_seed, start_seed + seed_count)


def build_train_command(args: argparse.Namespace, profile: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--agent",
        args.agent,
        "--task",
        args.task,
        "--profile",
        profile,
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if args.override:
        command.extend(["--override", args.override])
    return command


def main() -> int:
    args = parse_args()
    seeds = seed_range(args.seed, args.seeds)
    job_count = len(args.profiles) * len(seeds)

    job_index = 0
    for profile in args.profiles:
        for seed in seeds:
            job_index += 1
            command = build_train_command(args, profile, seed)
            print(
                f"[Run {job_index}/{job_count}] profile={profile} seed={seed}",
                flush=True,
            )
            print(f"  {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue

            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if result.returncode != 0:
                print(
                    f"[Stopped] profile={profile} seed={seed} exited with "
                    f"code {result.returncode}.",
                    file=sys.stderr,
                )
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
