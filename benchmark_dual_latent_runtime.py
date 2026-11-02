"""Benchmark independent Voice and belt latent inference against direct parallel head."""
import argparse
import json
from pathlib import Path

import torch

from benchmark_latent_runtime import measure
from dual_latent_bridge_cv import LatentBridge
from latent_belt_autoencoder_cv import BeltAE
from voice_latent_autoencoder_cv import VoiceAE
from whisper_feature_tcn_cv import FeatureTCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-checkpoint", type=Path, required=True)
    ap.add_argument("--bridge-checkpoint", type=Path, required=True)
    ap.add_argument("--voice-ae-checkpoint", type=Path, required=True)
    ap.add_argument("--belt-ae-checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    ds = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    bs = torch.load(args.bridge_checkpoint, map_location="cpu", weights_only=False)
    vs = torch.load(args.voice_ae_checkpoint, map_location="cpu", weights_only=False)
    ys = torch.load(args.belt_ae_checkpoint, map_location="cpu", weights_only=False)
    result = {"scope": "heads only; shared frozen Whisper encoder excluded",
              "window_frames": 128, "voice_latent": [16, bs["voice_dim"]],
              "belt_latent": [16, bs["belt_dim"]], "devices": {}}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        direct = FeatureTCN().to(device).eval()
        direct.load_state_dict(ds["model"])
        voice = VoiceAE(latent_dim=bs["voice_dim"]).to(device).eval()
        voice.load_state_dict(vs["model"])
        bridge = LatentBridge(bs["voice_dim"], bs["belt_dim"]).to(device).eval()
        bridge.load_state_dict(bs["v_to_b"])
        belt = BeltAE(bs["belt_dim"]).to(device).eval()
        belt.load_state_dict(ys["model"])
        path_params = (sum(p.numel() for p in voice.encoder.parameters())
                       + sum(p.numel() for p in bridge.parameters())
                       + sum(p.numel() for p in belt.start.parameters())
                       + sum(p.numel() for block in belt.up for p in block.parameters())
                       + sum(p.numel() for p in belt.end.parameters()))
        row = {"direct_parameters": sum(p.numel() for p in direct.parameters()),
               "dual_latent_inference_parameters": path_params, "batches": {}}
        for batch_size in (1, 32):
            x = torch.randn(batch_size, 128, 512, device=device)
            row["batches"][str(batch_size)] = {
                "direct": measure(direct, x, device),
                "dual_latent": measure(
                    lambda v: belt.decode(bridge(voice.encode(v))), x, device)}
        result["devices"][device] = row
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
