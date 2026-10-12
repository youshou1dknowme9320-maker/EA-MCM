#!/usr/bin/env python
"""Build manifest CSVs (filepath,label[,fold]) for every supported benchmark.

Subcommands
-----------
folder      class-per-directory audio tree (P.A.S.T., WMM, SPC-2, ...):
            every first-level sub-directory is one class.
csv         generic metadata CSV with column selection / renaming.
esc50       official ESC-50 ``meta/esc50.csv`` (preserves the fold column).
multi-csv   multi-label CSV: separator-delimited class names -> sorted
            integer ids; also emits a ``labels.json`` vocabulary.
unlabeled   glob a directory tree into a filepath-only SSL manifest.

Examples::

    python tools/prepare_manifests.py folder --root data/past/clips --out data/past/manifest.csv
    python tools/prepare_manifests.py esc50 --meta ESC-50-master/meta/esc50.csv \
        --audio-root ESC-50-master/audio --out data/esc50/manifest.csv
    python tools/prepare_manifests.py unlabeled --root data/past/continuous --out data/past/unlabeled.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eamcm.datasets.audio_dataset import write_manifest


def cmd_folder(a) -> None:
    root = Path(a.root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"no class sub-directories under {root}")
    rows = []
    for cname in classes:
        files = sorted((root / cname).rglob(f"*{a.ext}"))
        if not files:
            print(f"[warn] empty class: {cname}")
        for p in files:
            fp = str(p) if a.absolute else str(p.relative_to(root))
            rows.append({"filepath": fp, "label": cname})
    write_manifest(a.out, rows)
    print(f"[folder] {len(rows)} clips, {len(classes)} classes -> {a.out}")


def cmd_csv(a) -> None:
    import csv as _csv

    rows = []
    with open(a.meta, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            row = {"filepath": r[a.filepath_col], "label": r[a.label_col]}
            if a.audio_root:
                row["filepath"] = str(Path(a.audio_root) / row["filepath"])
            if a.fold_col and a.fold_col in r:
                row["fold"] = r[a.fold_col]
            rows.append(row)
    write_manifest(a.out, rows)
    print(f"[csv] {len(rows)} rows -> {a.out}")


def cmd_esc50(a) -> None:
    import csv as _csv

    rows = []
    with open(a.meta, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append({
                "filepath": str(Path(a.audio_root) / r["filename"]),
                "label": r["target"],
                "fold": r["fold"],
            })
    write_manifest(a.out, rows)
    folds = sorted({r["fold"] for r in rows})
    print(f"[esc50] {len(rows)} clips, 50 classes, folds {folds} -> {a.out}")


def cmd_multi_csv(a) -> None:
    import csv as _csv

    vocab = set()
    raw = []
    with open(a.meta, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            names = [x.strip() for x in r[a.label_col].split(a.sep) if x.strip()]
            vocab.update(names)
            raw.append((r[a.filepath_col], names))
    id_of = {n: i for i, n in enumerate(sorted(vocab))}
    rows = [
        {"filepath": (str(Path(a.audio_root) / fp) if a.audio_root else fp),
         "label": "|".join(str(id_of[n]) for n in names)}
        for fp, names in raw
    ]
    write_manifest(a.out, rows)
    with open(a.labels_out, "w", encoding="utf-8") as f:
        json.dump(id_of, f, indent=2)
    print(f"[multi-csv] {len(rows)} clips, {len(vocab)} classes -> {a.out} + {a.labels_out}")


def cmd_unlabeled(a) -> None:
    root = Path(a.root)
    files = sorted(root.rglob(f"*{a.ext}"))
    rows = [{"filepath": str(p) if a.absolute else str(p.relative_to(root))} for p in files]
    write_manifest(a.out, rows)
    print(f"[unlabeled] {len(rows)} files -> {a.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Manifest preparation")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("folder", help="class-per-directory tree")
    sp.add_argument("--root", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--ext", default=".wav")
    sp.add_argument("--absolute", action="store_true")
    sp.set_defaults(fn=cmd_folder)

    sp = sub.add_parser("csv", help="generic metadata CSV")
    sp.add_argument("--meta", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--filepath-col", default="filename")
    sp.add_argument("--label-col", default="target")
    sp.add_argument("--fold-col", default="")
    sp.add_argument("--audio-root", default="")
    sp.set_defaults(fn=cmd_csv)

    sp = sub.add_parser("esc50", help="official ESC-50 metadata")
    sp.add_argument("--meta", required=True, help="path to meta/esc50.csv")
    sp.add_argument("--audio-root", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_esc50)

    sp = sub.add_parser("multi-csv", help="multi-label CSV -> integer ids")
    sp.add_argument("--meta", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--labels-out", required=True)
    sp.add_argument("--filepath-col", default="filename")
    sp.add_argument("--label-col", default="labels")
    sp.add_argument("--sep", default=",")
    sp.add_argument("--audio-root", default="")
    sp.set_defaults(fn=cmd_multi_csv)

    sp = sub.add_parser("unlabeled", help="filepath-only SSL manifest")
    sp.add_argument("--root", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--ext", default=".wav")
    sp.add_argument("--absolute", action="store_true")
    sp.set_defaults(fn=cmd_unlabeled)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
