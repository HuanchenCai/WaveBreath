"""Compare parallel direct and latent heads, excluding their shared Whisper encoder."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_belt_autoencoder_cv import BeltAE
from latent_voice_mapper_cv import AudioMapper
from whisper_feature_tcn_cv import FeatureTCN


def measure(fn, x, device, repetitions=200):
    with torch.inference_mode():
        for _ in range(30):
            fn(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repetitions):
            begin = time.perf_counter()
            fn(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - begin) * 1000)
    return {"median_ms_per_batch": float(np.median(times)),
            "p90_ms_per_batch": float(np.quantile(times, .9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-checkpoint", type=Path, required=True)
    ap.add_argument("--latent-checkpoint", type=Path, required=True)
    ap.add_argument("--ae-checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    direct_state = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    latent_state = torch.load(args.latent_checkpoint, map_location="cpu", weights_only=False)
    ae_state = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    latent_dim = latent_state["latent_dim"]
    result = {"scope": "regression heads only; shared frozen Whisper extraction excluded",
              "window_frames": 128, "input_dim": 512, "latent_shape": [16, latent_dim],
              "devices": {}}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        direct = FeatureTCN().to(device).eval()
        direct.load_state_dict(direct_state["model"])
        mapper = AudioMapper(latent_dim=latent_dim).to(device).eval()
        mapper.load_state_dict(latent_state["mapper"])
        ae = BeltAE(latent_dim).to(device).eval()
        ae.load_state_dict(ae_state["model"])
        row = {"direct_parameters": sum(p.numel() for p in direct.parameters()),
               "latent_inference_parameters": sum(p.numel() for p in mapper.parameters())
               + sum(p.numel() for p in ae.start.parameters())
               + sum(p.numel() for block in ae.up for p in block.parameters())
               + sum(p.numel() for p in ae.end.parameters()),
               "batches": {}}
        for batch_size in (1, 32):
            x = torch.randn(batch_size, 128, 512, device=device)
            row["batches"][str(batch_size)] = {
                "direct": measure(direct, x, device),
                "latent": measure(lambda v: ae.decode(mapper(v)), x, device)}
        result["devices"][device] = row
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
