"""
Run test_errnet.py once per subdirectory (default 00,01,02,03) under --checkpoint_base,
using each folder's errnet_latest.pt, or the highest-epoch errnet_<epoch>_<iters>.pt.
Place this script at the project root (next to test_errnet.py).

Example (real20; folder names are exact names under checkpoint_base — list E:\\ERRNet\\checkpoints to confirm):
  python run_test_multi_ckpt.py --checkpoint_base ./checkpoints --subdirs ab_00_baseline,ab_01_hyper,ab_02_dark_channel,ab_03_hyper_and_dark_channel --auto_metrics --hyper_subdirs ab_01_hyper,ab_03_hyper_and_dark_channel --dcp_subdirs ab_02_dark_channel,ab_03_hyper_and_dark_channel -- \\
    --dataset real20 --data_root ./datasets/processed_data -r --gpu_ids 0

`--auto_metrics` writes ./results/00/metrics_00.json (etc.) so each run is not truncated by overwriting.
`--hyper_subdirs` / `--dcp_subdirs` append `--hyper` and/or `--dark_channel` per subdir (arch must match checkpoint).
Add `--dcp_kernel_size` in the forwarded args if training used a non-default value (default 15).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_final_checkpoint(sub: Path) -> Path:
    latest = sub / "errnet_latest.pt"
    if latest.is_file():
        return latest
    pts = [
        p
        for p in sub.iterdir()
        if p.is_file() and p.suffix == ".pt" and p.name.startswith("errnet_") and "latest" not in p.name
    ]
    if not pts:
        raise FileNotFoundError("No errnet_*.pt or errnet_latest.pt in {}".format(sub))

    def _epoch(p: Path) -> int:
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return -1

    return max(pts, key=_epoch)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run test_errnet for multiple checkpoint subfolders (e.g. 00–03).",
    )
    ap.add_argument(
        "--checkpoint_base",
        type=Path,
        required=True,
        help="Parent directory that contains 00, 01, 02, 03 (each with a .pt).",
    )
    ap.add_argument(
        "--subdirs",
        type=str,
        default="00,01,02,03",
        help="Comma-separated subfolder names (default: 00,01,02,03).",
    )
    ap.add_argument(
        "--auto_metrics",
        action="store_true",
        help="Append --metrics_out metrics_<subdir>.json for each run (avoids one file overwrite).",
    )
    ap.add_argument(
        "--hyper_subdirs",
        type=str,
        default=None,
        help="Comma-separated subdir names (e.g. 01,03) for which to add --hyper.",
    )
    ap.add_argument(
        "--dcp_subdirs",
        type=str,
        default=None,
        help="Comma-separated subdir names (e.g. 02,03) for which to add --dark_channel.",
    )
    ap.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="All arguments for test_errnet.py (e.g. --dataset real20 ... -r). "
        "If your shell supports it, use -- before these so -r is not parsed here.",
    )
    args = ap.parse_args()
    if args.rest and args.rest[0] == "--":
        args.rest = args.rest[1:]

    root = Path(__file__).resolve().parent
    test_script = root / "test_errnet.py"
    if not test_script.is_file():
        sys.stderr.write("test_errnet.py not found next to run_test_multi_ckpt.py ({})\n".format(test_script))
        sys.exit(1)

    base = args.checkpoint_base.resolve()
    if not base.is_dir():
        sys.stderr.write("checkpoint_base is not a directory: {}\n".format(base))
        sys.exit(1)

    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    hyper_set = set()
    if args.hyper_subdirs:
        hyper_set = {s.strip() for s in args.hyper_subdirs.split(",") if s.strip()}
    dcp_set = set()
    if args.dcp_subdirs:
        dcp_set = {s.strip() for s in args.dcp_subdirs.split(",") if s.strip()}
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    for name in subdirs:
        sub = base / name
        if not sub.is_dir():
            sys.stderr.write("Skipping missing subdir: {}\n".format(sub))
            continue
        try:
            ckpt = _find_final_checkpoint(sub)
        except FileNotFoundError as e:
            sys.stderr.write("Skipping {}: {}\n".format(name, e))
            continue

        cmd = [
            sys.executable,
            str(test_script),
            *args.rest,
        ]
        if name in hyper_set:
            cmd.append("--hyper")
        if name in dcp_set:
            cmd.append("--dark_channel")
        cmd.extend(
            [
                "-r",
                "--icnn_path",
                str(ckpt),
                "--save_subdir",
                name,
            ]
        )
        if args.auto_metrics:
            cmd.extend(["--metrics_out", "metrics_{}.json".format(name)])
        print("=" * 72, flush=True)
        extra = []
        if name in hyper_set:
            extra.append("--hyper")
        if name in dcp_set:
            extra.append("--dark_channel")
        extra_s = "  [+ {}]".format(" ".join(extra)) if extra else ""
        print("Run {}  <-  {}{}".format(name, ckpt, extra_s), flush=True)
        print("=" * 72, flush=True)
        r = subprocess.run(cmd, env=env, cwd=str(root))
        if r.returncode != 0:
            sys.exit(r.returncode)

    print("All runs finished.", flush=True)


if __name__ == "__main__":
    main()
