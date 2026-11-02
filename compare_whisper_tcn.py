"""Paired singer-level descriptive comparison for exploratory A/B models."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    return {(r["singer"], r["style"], str(r["number"])): r
            for fold in result["folds"] for r in fold["test_segments"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    old, new = rows(args.old), rows(args.new)
    assert old.keys() == new.keys() and len(old) == 197
    out = {"kind": "exploratory_paired_model_comparison_not_confirmatory",
           "n_segments": len(old), "n_singers": len({k[0] for k in old}), "channels": {}}
    rng = np.random.default_rng(20260930)
    for channel in ("A", "B"):
        channel_result = {}
        for metric in ("r", "nrmse", "peak_f1", "trough_f1"):
            delta = {key: new[key][channel][metric] - old[key][channel][metric]
                     for key in old}
            by_singer = defaultdict(list)
            for key, value in delta.items():
                by_singer[key[0]].append(value)
            singer_delta = np.array([np.median(by_singer[name])
                                     for name in sorted(by_singer)])
            draws = rng.choice(singer_delta, size=(10000, len(singer_delta)), replace=True)
            boots = np.median(draws, axis=1)
            channel_result[metric] = {
                "old_segment_median": float(np.median([old[k][channel][metric] for k in old])),
                "new_segment_median": float(np.median([new[k][channel][metric] for k in old])),
                "paired_segment_delta_median": float(np.median(list(delta.values()))),
                "paired_singer_delta_median": float(np.median(singer_delta)),
                "singer_bootstrap_95pct_descriptive": [float(x) for x in np.quantile(boots, [.025, .975])],
                "singers_positive": int(np.sum(singer_delta > 0)),
                "singers_negative": int(np.sum(singer_delta < 0)),
            }
        out["channels"][channel] = channel_result
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
