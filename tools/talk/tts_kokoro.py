"""Narrate the talk with Kokoro (open-source neural TTS, Apache-2.0), one WAV per slide.

Writes build/voice/NN.wav and build/voice/timing.json (per-sentence durations, used for captions).
Captions keep the written text; only the spoken text uses the pronunciation map below.
Run: uv run --python 3.12 --with "kokoro>=0.9.4" --with soundfile --with "misaki[en]" python tts_kokoro.py [--voice am_michael]
"""
import argparse, json, os, re
import numpy as np, soundfile as sf
from kokoro import KPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 24000
SPOKEN = [  # (written, spoken) applied in order
    ("Nataša Jovanović", "Natasha Yovanovich"), ("CentriQ", "Centri-Q"), ("SVDQuant", "S V D Quant"),
    ("PixArt-Sigma", "Pix-Art Sigma"), ("FLUX", "Flux"), ("SANA", "Sana"), ("W4A4", "W four A four"),
    ("W4A2", "W four A two"), ("INT4", "int four"), ("BF16", "B F sixteen"), ("GEMM", "gem"),
    ("HQQ", "H Q Q"), ("RoDS", "Rods"), ("FID", "F I D"), ("LPIPS", "L-pips"), ("MJHQ", "M J H Q"),
    ("RTX PRO 6000", "R T X Pro six thousand"), ("A100", "A one hundred"), ("EPFL", "E P F L"),
    ("RMS", "R M S"), ("HadaNorm", "Hada-Norm"), ("ImageReward", "Image Reward"), ("l-p", "L P"),
    ("1152", "eleven fifty-two"), ("2240", "twenty-two forty"), ("3072", "thirty seventy-two"),
    ("Nunchaku", "Noon-chaku"),
]


def spoken(s):
    for a, b in SPOKEN:
        s = s.replace(a, b)
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--voice", default="am_michael"); ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    notes = json.load(open(os.path.join(HERE, "notes.json")))
    out = os.path.join(HERE, "build", "voice"); os.makedirs(out, exist_ok=True)
    pipe = KPipeline(lang_code="a" if a.voice[0] == "a" else "b")
    pause = np.zeros(int(0.32 * SR), dtype=np.float32)
    timing = {}
    for i, text in enumerate(notes, 1):
        sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
        chunks, rows = [], []
        for s in sents:
            wav = np.concatenate([np.asarray(x, dtype=np.float32) for _, _, x in pipe(spoken(s), voice=a.voice, speed=a.speed)])
            chunks += [wav, pause]
            rows.append([s, len(wav) / SR, len(pause) / SR])
        audio = np.concatenate(chunks[:-1])
        sf.write(os.path.join(out, f"{i:02d}.wav"), audio, SR)
        timing[f"{i:02d}"] = rows
        print(f"slide {i:2d}: {len(audio)/SR:5.1f} s, {len(sents)} sentences")
    json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
