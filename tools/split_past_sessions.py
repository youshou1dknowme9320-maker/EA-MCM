#!/usr/bin/env python
"""Session-isolated train/val/test split (8:1:1) of a P.A.S.T. manifest.

A *session* is a continuous recording block captured with a fixed microphone
setup on a separate day. Entire sessions are assigned to exactly one split so
that clips from the same acoustic session never straddle train and test,
preventing microphone/room-profile leakage. The session id is read from the
``session`` column when present, otherwise from the filename stem prefix
before the first ``_`` (e.g. ``s03_cough_0007.wav`` -> ``s03``), otherwise
from the clip's parent directory::

    python tools/split_past_sessions.py --manifest data/past/manifest.csv \
        --train-out data/past/train.csv --val-out data/past/val.csv \
        --test-out data/past/test.csv --seed 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eamcm.datasets.audio_dataset import read_manifest, write_manifest
from eamcm.utils.audio import split_session_manifest


def infer_session(row: dict) -> str:
    """Session id: ``session`` column > filename stem prefix > parent directory."""
    if row.get("session"):
        return row["session"]
    stem = Path(row["filepath"]).stem
    if "_" in stem:
        return stem.split("_")[0]
    parts = Path(row["filepath"]).parts
    return parts[-2] if len(parts) > 1 else stem


def main() -> None:
    p = argparse.ArgumentParser(description="Session-isolated P.A.S.T. split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--train-out", required=True)
    p.add_argument("--val-out", required=True)
    p.add_argument("--test-out", required=True)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   help="train/val/test session ratios")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    for r in rows:
        r["session"] = infer_session(r)

    splits = split_session_manifest(rows, tuple(args.ratios), seed=args.seed)

    for name, path in (("train", args.train_out), ("val", args.val_out), ("test", args.test_out)):
        out_rows = [{k: v for k, v in r.items() if k != "session"} for r in splits[name]]
        write_manifest(path, out_rows)
        per_class = {}
        for r in out_rows:
            per_class[r["label"]] = per_class.get(r["label"], 0) + 1
        sessions = sorted({r["session"] for r in splits[name]})
        print(f"[{name}] {len(out_rows)} clips | sessions={sessions} | per-class={per_class}")

    total = sum(len(s) for s in splits.values())
    print(f"[done] {total} clips split by ratios {args.ratios} (session isolation, seed={args.seed})")


if __name__ == "__main__":
    main()
