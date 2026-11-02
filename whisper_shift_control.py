"""Test-time 10-second Voice shift control for frozen Whisper A/B regression."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dual_belt_cv import predict
from evaluate_breath_pilot import metrics
from sequence_model_screen import recordings
from whisper_feature_tcn_cv import FeatureTCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--cv-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shift-seconds", type=int, default=10)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    records = recordings(args.cache)
    cv = json.loads(args.cv_results.read_text(encoding="utf-8"))
    rows = []
    for fold in cv["folds"]:
        state = torch.load(args.cv_results.parent / fold["checkpoint"],
                           map_location=device, weights_only=False)
        model = FeatureTCN(input_dim=records[0][1].shape[1]).to(device)
        model.load_state_dict(state["model"])
        for meta, x, y in records:
            if meta["singer"] not in fold["test_singers"]:
                continue
            shifted = np.roll(x, args.shift_seconds * 5, axis=0)
            pred = predict(model, shifted, state["mean"], state["std"], device)
            row = {**meta}
            for col, channel in enumerate(("A", "B")):
                row[channel] = metrics(y[:, col], pred[:, col])
            rows.append(row)
        print(f"fold {fold['fold']}/5", flush=True)
    assert len(rows) == 197
    summary = {c: {k: float(np.median([r[c][k] for r in rows]))
                   for k in ("r", "nrmse")}
               for c in ("A", "B")}
    args.out.write_text(json.dumps({"kind": "test_time_voice_shift_negative_control",
                                    "shift_seconds": args.shift_seconds,
                                    "note": "Circular shift within each held-out recording; no retraining",
                                    "summary": summary, "test_segments": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
