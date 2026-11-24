# Talk deck and videos

| File | Output |
|---|---|
| `build_deck.py` | `centriq_talk.pptx` (12 slides) plus `notes.json` / `narration_script.md` with the narration |
| `make_clips.py` | the three loops in `docs/assets/video/` (hidden mean, method flow, comparison wipes) |
| `tts_chatterbox.py` | narration audio per slide (Chatterbox, balanced delivery); `tts_kokoro.py` is a lighter alternative |
| `make_video.py` | `centriq_intro.mp4` (about 6 min) and captions from the slides, clips and narration |

Keynote rejects speaker notes written by python-pptx, so the narration is added in Keynote as presenter notes
(AppleScript) and the deck is saved as `docs/assets/slides/centriq_talk.key`; `centriq_talk.pdf` is exported from it.

Narration: `tts_chatterbox.py` voices the script with Chatterbox (Resemble AI, MIT licence, with its built-in audio
watermark), writing `build/voice/NN.wav` plus per-sentence timings for exact captions. To use your own recording instead, put
`01.m4a` … `12.m4a` in `build/voice/` and rerun `make_video.py`. The video adds a slow push-in on each slide and soft
fades between segments.
