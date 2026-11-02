"""Extract frozen Whisper encoder embeddings aligned to 5-Hz A/B labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperModel


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE
FRAMES_PER_CHUNK = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--reference-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    args = ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["suspect"] != "1" and float(r["duration_s"]) >= 10]
    ref = np.load(args.reference_cache, allow_pickle=False)
    offsets = ref["offsets"]
    metadata = ref["metadata"]
    assert len(rows) == len(offsets) - 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    extractor = WhisperFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model = WhisperModel.from_pretrained(args.model, local_files_only=True).encoder.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    dim = model.config.d_model
    temp = args.out.with_suffix(".working.npy")
    progress = args.out.with_suffix(".progress.json")
    if temp.exists():
        xout = np.lib.format.open_memmap(temp, mode="r+")
        start_row = (json.loads(progress.read_text(encoding="utf-8"))["completed_recordings"]
                     if progress.exists() else 0)
        assert xout.shape == (int(offsets[-1]), dim)
    else:
        xout = np.lib.format.open_memmap(temp, mode="w+", dtype="float16",
                                         shape=(int(offsets[-1]), dim))
        start_row = 0
    print(f"device={device} model={args.model} recordings={len(rows)} resume={start_row}", flush=True)
    for i in range(start_row, len(rows)):
        row = rows[i]
        audio, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        assert sr == 44100 and audio.shape[1] >= 1
        voice = resample_poly(audio[:, 0], 160, 441)
        n = int(offsets[i + 1] - offsets[i])
        pieces = []
        for left in range(0, len(voice), CHUNK_SAMPLES):
            chunk = voice[left:left + CHUNK_SAMPLES]
            count = min(FRAMES_PER_CHUNK, max(0, n - left // 3200))
            if count == 0:
                break
            features = extractor(chunk, sampling_rate=SAMPLE_RATE,
                                 return_tensors="pt").input_features.to(
                                     device=device, dtype=next(model.parameters()).dtype)
            with torch.inference_mode():
                hidden = model(features).last_hidden_state[0, :count * 10]
                pooled = hidden.reshape(count, 10, dim).mean(dim=1)
            pieces.append(pooled.cpu().numpy().astype("float16"))
        emb = np.concatenate(pieces, axis=0)
        if len(emb) != n:
            raise RuntimeError(f"alignment failure recording={i} got={len(emb)} expected={n}")
        xout[offsets[i]:offsets[i + 1]] = emb
        xout.flush()
        progress.write_text(json.dumps({"completed_recordings": i + 1, "total": len(rows),
                                        "model": args.model, "dimension": dim}, indent=2),
                            encoding="utf-8")
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"extracted {i + 1}/{len(rows)}", flush=True)
    np.savez_compressed(args.out, x=np.asarray(xout), y=ref["y"],
                        offsets=offsets, metadata=metadata,
                        feature_source=np.array(args.model),
                        feature_method=np.array("encoder last hidden; 30-s chunks; 10-frame mean to 5 Hz"))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
