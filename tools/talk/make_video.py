"""Assemble the narrated CentriQ intro video from the slides, clips and narration.

Inputs: build/hd/slide_NN.png (from the Keynote PDF), notes.json (narration per slide), clips/*.mp4.
Voice: macOS `say` (draft voice) unless build/voice/NN.(m4a|wav|aiff|mp3) exists, which wins (your own recording).
Outputs: centriq_intro.mp4 (soft English captions), centriq_intro.srt
Run: uv run --with imageio-ffmpeg python make_video.py [--voice Samantha]
"""
import argparse, glob, json, os, re, subprocess
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "build"); SEGS = os.path.join(B, "segs"); AUD = os.path.join(B, "audio")
os.makedirs(SEGS, exist_ok=True); os.makedirs(AUD, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
# clips shown (silent, with a soft music-free pause) after the given slide number
CLIPS_AFTER = {4: ("mean_spike.mp4", 10.0), 6: ("method_flow.mp4", 6.0), 9: ("compare_wipe.mp4", 15.0)}
PAD = 0.6   # seconds of silence before and after each slide's narration
FADE = 0.35  # seconds of fade in/out on every segment
TIMING = None


def own_recording(i):
    return any(p.endswith((".m4a", ".mp3", ".aiff")) for p in glob.glob(os.path.join(B, "voice", f"{i:02d}.*")))
VF = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0b1416,fps=30,format=yuv420p"


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path):
    out = subprocess.run([FF, "-i", path], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def ts(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--voice", default="Samantha"); ap.add_argument("--rate", default="172")
    a = ap.parse_args()
    global TIMING
    tp = os.path.join(B, "voice", "timing.json")
    TIMING = json.load(open(tp)) if os.path.exists(tp) else None
    notes = json.load(open(os.path.join(HERE, "notes.json")))
    slides = sorted(glob.glob(os.path.join(B, "hd", "slide_*.png")))
    assert len(slides) == len(notes), (len(slides), len(notes))
    segs, srt, t0, k = [], [], 0.0, 1
    for i, (img, text) in enumerate(zip(slides, notes), 1):
        own = [p for p in glob.glob(os.path.join(B, "voice", f"{i:02d}.*")) if not p.endswith(".json")]
        if own:
            voice = own[0]
        else:
            voice = os.path.join(AUD, f"{i:02d}.aiff")
            txt = os.path.join(AUD, f"{i:02d}.txt"); open(txt, "w").write(text)
            run(["say", "-v", a.voice, "-r", a.rate, "-o", voice, "-f", txt])
        d = duration(voice); total = d + 2 * PAD
        seg = os.path.join(SEGS, f"{i:02d}_slide.mp4")
        nfr = int(round(total * 30)); fo = total - FADE
        # slow push-in (1.0 -> 1.035) rendered from a 2x upscale to avoid zoompan jitter, plus soft fades
        vf = (f"[0:v]scale=3840:2160:flags=lanczos,zoompan=z='1+0.035*on/{nfr}':d={nfr}:"
              f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30,"
              f"fade=t=in:st=0:d={FADE},fade=t=out:st={fo:.3f}:d={FADE},format=yuv420p[v];"
              f"[1:a]adelay={int(PAD*1000)}|{int(PAD*1000)},apad,aresample=48000,"
              f"afade=t=in:st=0:d=0.15,afade=t=out:st={fo:.3f}:d={FADE}[a]")
        run([FF, "-y", "-i", img, "-i", voice, "-filter_complex", vf, "-map", "[v]", "-map", "[a]",
             "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-c:a", "aac", "-b:a", "160k", "-ac", "2", seg])
        segs.append(seg)
        # captions: exact per-sentence timing from the TTS run when available, else share of characters
        cur = t0 + PAD
        if TIMING and f"{i:02d}" in TIMING and not own_recording(i):
            for s_, dur, gap in TIMING[f"{i:02d}"]:
                srt.append(f"{k}\n{ts(cur)} --> {ts(cur + dur)}\n{s_}\n"); k += 1; cur += dur + gap
        else:
            sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
            tot = sum(len(s) for s in sents)
            for s_ in sents:
                dur = d * len(s_) / tot
                srt.append(f"{k}\n{ts(cur)} --> {ts(cur + dur)}\n{s_}\n"); k += 1; cur += dur
        t0 += total
        if i in CLIPS_AFTER:
            name, cd = CLIPS_AFTER[i]
            seg = os.path.join(SEGS, f"{i:02d}_clip.mp4")
            run([FF, "-y", "-i", os.path.join(HERE, "clips", name), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                 "-map", "0:v", "-map", "1:a", "-vf", VF + f",fade=t=in:st=0:d={FADE},fade=t=out:st={cd-FADE:.3f}:d={FADE}",
                 "-t", f"{cd:.3f}", "-c:v", "libx264", "-preset", "medium",
                 "-crf", "18", "-c:a", "aac", "-b:a", "160k", seg])
            segs.append(seg); t0 += cd
    lst = os.path.join(SEGS, "list.txt")
    open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
    srt_path = os.path.join(HERE, "centriq_intro.srt"); open(srt_path, "w").write("\n".join(srt))
    out = os.path.join(HERE, "centriq_intro.mp4")
    run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", srt_path, "-map", "0", "-map", "1",
         "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
         "-movflags", "+faststart", out])
    print(f"wrote {out}  ({t0/60:.1f} min)  and {srt_path}")


if __name__ == "__main__":
    main()
