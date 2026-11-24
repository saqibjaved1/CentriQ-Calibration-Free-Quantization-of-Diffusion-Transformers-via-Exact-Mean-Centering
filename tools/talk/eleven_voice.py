"""Generate the narration with ElevenLabs, one MP3 per slide, into build/voice/NN.mp3.

The API key is read from $ELEVENLABS_API_KEY or from talk/.elevenlabs_key (never commit it).
make_video.py picks up build/voice/NN.* automatically, so run it afterwards.

    uv run --with requests python eleven_voice.py --list            # show your voices
    uv run --with requests python eleven_voice.py --voice <voice_id>
"""
import argparse, json, os, sys
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.elevenlabs.io/v1"


def key():
    k = os.environ.get("ELEVENLABS_API_KEY")
    p = os.path.join(HERE, ".elevenlabs_key")
    if not k and os.path.exists(p):
        k = open(p).read().strip()
    if not k:
        sys.exit("No ElevenLabs key: set ELEVENLABS_API_KEY or write it to talk/.elevenlabs_key")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--voice", default="nPczCjzI2devNBz1zQrb")   # "Brian", a premade ElevenLabs voice
    ap.add_argument("--model", default="eleven_multilingual_v2")
    ap.add_argument("--slides", default=None, help="comma list, e.g. 3,5 (default: all)")
    a = ap.parse_args()
    h = {"xi-api-key": key()}
    if a.list:
        r = requests.get(f"{API}/voices", headers=h, timeout=30); r.raise_for_status()
        for v in r.json()["voices"]:
            print(f'{v["voice_id"]}  {v["name"]:<24} {v.get("labels", {})}')
        return
    notes = json.load(open(os.path.join(HERE, "notes.json")))
    out = os.path.join(HERE, "build", "voice"); os.makedirs(out, exist_ok=True)
    want = set(int(x) for x in a.slides.split(",")) if a.slides else set(range(1, len(notes) + 1))
    prev = None
    for i, text in enumerate(notes, 1):
        if i not in want:
            prev = text; continue
        body = {"text": text, "model_id": a.model,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.15, "use_speaker_boost": True}}
        if prev: body["previous_text"] = prev           # keeps intonation continuous across slides
        if i < len(notes): body["next_text"] = notes[i]
        r = requests.post(f"{API}/text-to-speech/{a.voice}?output_format=mp3_44100_128",
                          headers={**h, "Content-Type": "application/json"}, json=body, timeout=180)
        if r.status_code != 200:
            sys.exit(f"slide {i}: HTTP {r.status_code}: {r.text[:300]}")
        p = os.path.join(out, f"{i:02d}.mp3"); open(p, "wb").write(r.content)
        print(f"slide {i:2d}: {len(r.content)//1024} KB -> {p}")
        prev = text


if __name__ == "__main__":
    main()
