"""Show training STATUS: what was learned (checkpoints) and how it's going.

Clears a common confusion: the `.zip` saved in <vault>/.checkpoints is the "brain"
(the PPO policy weights). Its SIZE is ~constant — it depends on the network
architecture, not on "how much the agent learned". What tells you whether training is
efficient is the reward/accuracy CURVE (see the metrics below, or the curve in the run
note).

Usage:
    python -m rl.status                 # checkpoints + metrics for the .env run
    python -m rl.status --run NAME       # metrics for a specific run
"""
import argparse
import glob
import os
from datetime import datetime

from config import Config
from writer.snapshot_log import SnapshotLog, log_path_for


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def show_checkpoints(cfg: Config) -> None:
    zips = sorted(
        glob.glob(os.path.join(cfg.checkpoint_dir, "*.zip")), key=os.path.getmtime
    )
    print(f"\n== Saved brains in {cfg.checkpoint_dir} ==")
    if not zips:
        print("  (none yet — run `python -m rl.train`)")
        return
    for z in zips:
        st = os.stat(z)
        when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {os.path.basename(z):<40} {_human_size(st.st_size):>10}  {when}")
    print(
        "  Note: size is ~constant (network architecture). "
        "Efficiency = the reward curve, not the file size."
    )


def show_metrics(cfg: Config) -> None:
    snaps = SnapshotLog.read_all(log_path_for(cfg.pending_dir, cfg.run_name))
    print(f"\n== Progress of run '{cfg.run_name}' ({len(snaps)} checkpoints) ==")
    if not snaps:
        print("  (no snapshots — train with documentation enabled)")
        return
    first, last = snaps[0], snaps[-1]

    def line(s, tag):
        print(
            f"  [{tag}] {int(s.get('num_timesteps', 0)):>9,} steps | "
            f"reward {s.get('mean_reward', 0):6.2f} | "
            f"accuracy {s.get('shooting_accuracy', 0):4.0%} | "
            f"kills/ep {s.get('kills_per_episode', 0):4.1f} | "
            f"success {s.get('success_rate', 0):4.0%}"
        )

    line(first, "start")
    line(last, " now ")
    d_reward = last.get("mean_reward", 0) - first.get("mean_reward", 0)
    d_acc = last.get("shooting_accuracy", 0) - first.get("shooting_accuracy", 0)
    arrow = "[up]" if d_reward >= 0 else "[down]"
    print(f"  {arrow} change: reward {d_reward:+.2f} | accuracy {d_acc:+.0%}")


def main() -> None:
    p = argparse.ArgumentParser(description="Training status: checkpoints + metrics.")
    p.add_argument("--run", default=None, help="Specific run (default: RUN_NAME from .env).")
    args = p.parse_args()
    cfg = Config()
    if args.run:
        cfg.run_name = args.run
    show_checkpoints(cfg)
    show_metrics(cfg)


if __name__ == "__main__":
    main()
