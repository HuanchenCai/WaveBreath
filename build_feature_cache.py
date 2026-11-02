"""Cache deterministic 5-Hz acoustic features and relative belt targets."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from breath_model_pilot import read_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["suspect"] != "1" and float(r["duration_s"]) >= 10]
    xs, ys, offsets = [], [], [0]
    meta = []
    for i, row in enumerate(rows):
        x, y = read_pair(row)
        xs.append(x); ys.append(y)
        offsets.append(offsets[-1] + len(y))
        meta.append({"singer": row["singer_en"], "style": row["style"], "number": row["number"], "seconds": len(y) / 5})
        if (i + 1) % 20 == 0:
            print(f"processed {i+1}/{len(rows)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=np.concatenate(xs), y=np.concatenate(ys), offsets=np.array(offsets),
                        metadata=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"wrote {args.out}: {len(meta)} recordings, {offsets[-1]} frames", flush=True)


if __name__ == "__main__":
    main()
