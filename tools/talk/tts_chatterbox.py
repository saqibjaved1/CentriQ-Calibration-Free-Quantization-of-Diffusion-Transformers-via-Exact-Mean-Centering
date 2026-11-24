"""Narrate the talk with Chatterbox (Resemble AI, MIT licence, built-in Perth audio watermark), one WAV per slide.

Writes build/voice/NN.wav and build/voice/timing.json (per-sentence durations, used for exact captions).
Captions keep the written text; only the spoken text uses the pronunciation map.
Run: uv run --python 3.11 --with chatterbox-tts --with soundfile --with "setuptools<80" python tts_chatterbox.py
"""
import argparse, json, os, re
import numpy as np, soundfile as sf, torch
from chatterbox.tts import ChatterboxTTS

HERE = os.path.dirname(os.path.abspath(__file__))
SPOKEN = [
    ("CentriQ", "Centri-Q"), ("SVDQuant", "S V D Quant"), ("PixArt-Sigma", "Pix-Art Sigma"), ("FLUX", "Flux"),
    ("SANA", "Sana"), ("W4A4", "W four A four"), ("W4A2", "W four A two"), ("INT4", "int four"), ("BF16", "B F sixteen"),
    ("GEMM", "gem"), ("HQQ", "H Q Q"), ("RoDS", "Rods"), ("FID", "F I D"), ("LPIPS", "L-pips"), ("MJHQ", "M J H Q"),
    ("RTX PRO 6000", "R T X Pro six thousand"), ("A100", "A one hundred"), ("EPFL", "E P F L"), ("RMS", "R M S"),
    ("HadaNorm", "Hada-Norm"), ("ImageReward", "Image Reward"), ("l-p", "L P"), ("1152", "eleven fifty-two"),
    ("2240", "twenty-two forty"), ("3072", "thirty seventy-two"), ("Nunchaku", "Noon-chaku"),
    ("4-bit", "four-bit"), ("2-bit", "two-bit"), ("4 bits", "four bits"), ("four u squared", "four u squared"),
]


def spoken(s):
    for a, b in SPOKEN:
        s = s.replace(a, b)
    return s


def trim(w, sr, thr=0.008, keep=0.06):
    """Trim leading/trailing near-silence, keeping a short margin."""
    idx = np.where(np.abs(w) > thr)[0]
    if len(idx) == 0:
        return w
    a = max(0, idx[0] - int(keep * sr)); b = min(len(w), idx[-1] + int(keep * sr))
    return w[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exaggeration", type=float, default=0.5); ap.add_argument("--cfg", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    m = ChatterboxTTS.from_pretrained(device=dev); sr = m.sr
    notes = json.load(open(os.path.join(HERE, "notes.json")))
    out = os.path.join(HERE, "build", "voice"); os.makedirs(out, exist_ok=True)
    pause = np.zeros(int(0.30 * sr), dtype=np.float32)
    timing = {}
    for i, text in enumerate(notes, 1):
        sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
        chunks, rows = [], []
        for j, s in enumerate(sents):
            torch.manual_seed(a.seed + 100 * i + j)
            w = m.generate(spoken(s), exaggeration=a.exaggeration, cfg_weight=a.cfg).squeeze(0).cpu().numpy().astype(np.float32)
            w = trim(w, sr)
            chunks += [w, pause]; rows.append([s, len(w) / sr, len(pause) / sr])
        audio = np.concatenate(chunks[:-1])
        sf.write(os.path.join(out, f"{i:02d}.wav"), audio, sr)
        timing[f"{i:02d}"] = rows
        print(f"slide {i:2d}: {len(audio)/sr:5.1f} s, {len(sents)} sentences", flush=True)
    json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
