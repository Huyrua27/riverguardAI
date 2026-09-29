"""Run the ablation matrix (see docs/EVALUATION.md).

Runs the pipeline at each ablation level over a source and prints a comparison
table of system-level metrics (objects, tracks, alert cells, FPS).

Accuracy metrics that require ground-truth labels — Detection mAP, Tracking
IDF1/MOTA/HOTA, Prediction MAE/RMSE, hotspot precision/recall — are listed but
reported as N/A until a labelled test set exists (see docs/DATASET.md). This
keeps the command runnable today while making the missing pieces explicit.

Usage:
    python scripts/evaluate.py --source data/raw --config configs/default.yaml
"""
from __future__ import annotations

import argparse

from riverguard.pipeline import STAGE_PRESETS, run

ABLATIONS = list(STAGE_PRESETS)


def main() -> None:
    from riverguard.utils.console import utf8_console

    utf8_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default="data/raw", help="image / folder / video")
    parser.add_argument("--ablation", choices=ABLATIONS + ["all"], default="all")
    parser.add_argument("--output-dir", default="outputs/eval")
    args = parser.parse_args()

    levels = ABLATIONS if args.ablation == "all" else [args.ablation]
    rows = []
    for level in levels:
        res = run(args.config, args.source, stages=level,
                  output_dir=f"{args.output_dir}/{level}",
                  save_vis=False, make_report=False)
        s = res["summary"]
        n_tracks = sum(len(fr["tracks"]) for fr in res["frames"])
        rows.append({
            "ablation": level,
            "objects": s["total_detections"],
            "tracks": n_tracks,
            "HIGH": s["alert_cells"]["HIGH"],
            "fps": res["fps"],
        })

    # Print comparison table.
    print("\n=== Ablation matrix (system-level) ===")
    header = f"{'ablation':<24}{'objects':>9}{'tracks':>8}{'HIGH':>6}{'FPS':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['ablation']:<24}{r['objects']:>9}{r['tracks']:>8}"
              f"{r['HIGH']:>6}{str(r['fps']):>8}")

    print("\n=== Accuracy metrics (need labelled test set) ===")
    for group, metrics in {
        "Detection": ["Precision", "Recall", "mAP@50", "mAP@50:95"],
        "Tracking": ["IDF1", "MOTA", "HOTA"],
        "Prediction": ["MAE", "RMSE", "hotspot P/R"],
        "Early-warning": ["Top-1/Top-3 recall", "lead time"],
    }.items():
        print(f"  {group:<14} {', '.join(metrics)} — N/A (chưa có nhãn)")


if __name__ == "__main__":
    main()
