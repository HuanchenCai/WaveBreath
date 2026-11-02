"""Audit audio/belt pairs without modifying the source recordings."""

import argparse
import csv
import json
import re
from pathlib import Path

import soundfile as sf

STYLES = {"\u7f8e\u58f0": "BelCanto", "\u6c11\u6b4c": "FolkSongs", "\u901a\u4fd7": "PopularMusic"}  # Chinese style labels in the source files
# Participant-specific fixes live in a private file that is not part of the repository:
# {"name_overrides": {...}, "suspect_singers": [...]}
_PRIVATE = Path(__file__).with_name("private_overrides.json")
_P = json.loads(_PRIVATE.read_text(encoding="utf-8")) if _PRIVATE.exists() else {}
NAME_OVERRIDES = _P.get("name_overrides", {})  # demographic-file spelling -> recording-folder spelling
SUSPECT_SINGERS = set(_P.get("suspect_singers", []))


def person_names(root: Path) -> dict[str, str]:
    names = {}
    for line in (root / "demographic info.txt").read_text(encoding="utf-8").splitlines()[1:]:
        fields = re.split(r"[,\uff0c]", line.strip())  # ASCII or full-width comma
        if len(fields) >= 3 and fields[1].strip() and fields[2].strip():
            names[fields[1].strip()] = fields[2].strip()
    return names


def file_number(path: Path) -> int | None:
    match = re.search(r"\((\d+)\)", path.name)
    return int(match.group(1)) if match else None


def build(root: Path) -> list[dict]:
    names = person_names(root)
    records = []
    audio_root = root / ".audio_data_separated_by_file"
    for belt in sorted((root / "._breath_data").glob("*/*/*.wav")):
        style_cn, singer_cn = belt.parent.parent.name, belt.parent.name
        singer_en = NAME_OVERRIDES.get(singer_cn, names.get(singer_cn, ""))
        number = file_number(belt)
        audio_dir = audio_root / STYLES.get(style_cn, "") / singer_en
        candidates = [p for p in audio_dir.glob("*.wav") if "voice_egg" in p.name.lower() and file_number(p) == number] if singer_en and number is not None else []
        audio = candidates[0] if len(candidates) == 1 else None
        bi = sf.info(str(belt))
        ai = sf.info(str(audio)) if audio else None
        records.append({
            "style": STYLES.get(style_cn, style_cn), "singer_cn": singer_cn,
            "singer_en": singer_en, "number": number, "belt": str(belt),
            "audio": str(audio) if audio else "", "audio_candidates": len(candidates),
            "belt_sr": bi.samplerate, "audio_sr": ai.samplerate if ai else "",
            "belt_channels": bi.channels, "audio_channels": ai.channels if ai else "",
            "belt_frames": bi.frames, "audio_frames": ai.frames if ai else "",
            "duration_s": round(bi.duration, 3),
            "frame_delta": ai.frames - bi.frames if ai and ai.samplerate == bi.samplerate else "",
            "suspect": int(style_cn == "\u7f8e\u58f0" and singer_cn in SUSPECT_SINGERS and number == 20),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = sum(bool(r["audio"]) for r in rows)
    aligned = sum(r["frame_delta"] == 0 for r in rows)
    print(f"belts={len(rows)} paired={paired} exact_frame_match={aligned} singers={len({r['singer_cn'] for r in rows})}")
    for row in rows:
        if not row["audio"] or row["frame_delta"] != 0:
            print("CHECK", row["style"], row["singer_cn"], row["number"], row["audio_candidates"], row["frame_delta"])


if __name__ == "__main__":
    main()
