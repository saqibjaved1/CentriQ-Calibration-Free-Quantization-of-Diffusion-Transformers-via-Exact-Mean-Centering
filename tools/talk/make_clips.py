"""Render looping explainer clips for the CentriQ page and talk (1280x720, 30 fps, H.264 MP4).

clips/mean_spike.mp4   illustration: rotation leaves a DC spike that sets the 4-bit grid; centering removes it
clips/method_flow.mp4  one layer: center -> rotate -> INT4 -> GEMM, mean bypasses both quantizers
clips/compare_wipe.mp4 wipe from another 4-bit method to CentriQ (same prompt and seed)
Run: uv run --with pillow --with imageio-ffmpeg python make_clips.py
"""
import math, os, random, subprocess
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "clips"); os.makedirs(OUT, exist_ok=True)
PAGE = os.path.join(HERE, "..", "project_page", "assets", "img", "cmp2")
W, H, FPS = 1280, 720, 30
BG0, BG1 = (16, 46, 51), (7, 21, 24)
MINT, MINT2, ORANGE, BLUE, GRID = (234, 244, 243), (159, 189, 185), (242, 131, 74), (79, 163, 224), (60, 96, 102)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def font(size, bold=False, serif=False):
    cands = (["/System/Library/Fonts/Supplemental/Georgia.ttf"] if serif else []) + \
            (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"]) + \
            ["/System/Library/Fonts/Helvetica.ttc"]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


_BG = None
def background():
    global _BG
    if _BG is None:
        im = Image.new("RGB", (W, H)); px = im.load()
        for y in range(H):
            for x in range(W):
                d = min(1, (((x - W / 2) / 900) ** 2 + ((y - H * 0.3) / 520) ** 2) ** 0.5)
                px[x, y] = tuple(int(BG0[i] + (BG1[i] - BG0[i]) * d) for i in range(3))
        _BG = im
    return _BG.copy()


def ease(t):
    t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)


def encode(frames_fn, n, name):
    path = os.path.join(OUT, name)
    p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                          "-preset", "slow", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(frames_fn(i).tobytes())
    p.stdin.close(); p.wait()
    print("wrote", path)


# ---------------------------------------------------------------- clip 1: the spike
random.seed(7)
N = 48
bulk = sorted([abs(random.gauss(0, 1)) * 0.36 for _ in range(N)], reverse=True)
SPIKE = bulk[0] * 2.4


def spike_frame(i):
    t = i / FPS                     # 10 s loop: hold rot 0-3, morph 3-4.5, hold centered 4.5-8, morph back 8-10
    a = ease((t - 3) / 1.5) if t < 8 else 1 - ease((t - 8) / 2)
    im = background(); d = ImageDraw.Draw(im)
    x0, x1, yb, yt = 140, 1140, 600, 170
    top = SPIKE * 1.1
    v0 = SPIKE * (1 - a) + bulk[0] * a
    vals = [v0] + bulk[1:]
    mx = max(vals); step = mx / 7
    # grid levels set by the largest coordinate
    for k in range(1, 8):
        y = yb - (step * k) / top * (yb - yt)
        for xx in range(x0, x1, 14):
            d.line([(xx, y), (xx + 7, y)], fill=(BLUE if a < 0.5 else ORANGE) + (), width=1)
    bw = (x1 - x0) / N
    for j, v in enumerate(vals):
        h = v / top * (yb - yt)
        col = ORANGE if (j == 0 and a < 0.5) else (MINT2 if a < 0.5 else ORANGE)
        if j == 0 and a < 0.5:
            col = ORANGE
        elif a < 0.5:
            col = BLUE
        d.rounded_rectangle([x0 + j * bw + 2, yb - h, x0 + (j + 1) * bw - 2, yb], radius=3, fill=col)
    d.line([(x0, yb), (x1, yb)], fill=GRID, width=2)
    f1, f2, f3 = font(40, serif=True), font(24), font(18)
    if a < 0.5:
        d.text((W / 2, 70), "Rotation only", font=f1, fill=MINT, anchor="mm")
        d.text((W / 2, 118), "The token's mean lands on one coordinate and sets the 4-bit step", font=f2, fill=MINT2, anchor="mm")
        hx = x0 + bw / 2; hy = yb - v0 / top * (yb - yt)
        d.text((hx + 30, hy + 6), "DC spike", font=f2, fill=ORANGE, anchor="lm")
    else:
        d.text((W / 2, 70), "Centering + rotation", font=f1, fill=MINT, anchor="mm")
        d.text((W / 2, 118), "Spike removed: the same 16 levels cover the real values", font=f2, fill=MINT2, anchor="mm")
        g = SPIKE / bulk[0] * ease((t - 4.5) / 0.8) if t < 8 else SPIKE / bulk[0]
        d.text((x1, 210), f"grid {g:.1f}× finer", font=font(34, bold=True), fill=ORANGE, anchor="rm")
    d.text((x0, 640), "Sorted coordinate magnitudes of one token after rotation · dashed: INT4 levels · illustration",
           font=f3, fill=MINT2)
    return im


# ---------------------------------------------------------------- clip 2: method flow
BOXES = [("center", "subtract mean", ORANGE), ("pad · rotate", "Hadamard H", BLUE), ("quantize", "INT4, per token", BLUE),
         ("INT4 GEMM", "4-bit weights", BLUE), ("epilogue", "+ mean · W1 + b", MINT2)]


def method_frame(i):
    t = (i / FPS) % 6.0
    im = background(); d = ImageDraw.Draw(im)
    d.text((W / 2, 70), "One CentriQ layer", font=font(40, serif=True), fill=MINT, anchor="mm")
    d.text((W / 2, 118), "The mean takes the full-precision path around both quantizers", font=font(24), fill=MINT2, anchor="mm")
    y = 400; bw, bh, gap = 180, 92, 38
    xs = [110 + k * (bw + gap) for k in range(5)]
    # main wire
    d.line([(40, y), (W - 40, y)], fill=(BLUE[0], BLUE[1], BLUE[2]), width=3)
    # orange bypass: from center box top to epilogue top
    bx0, bx1, by = xs[0] + bw / 2, xs[4] + bw / 2, y - bh / 2 - 90
    dash = int(t * 40) % 24
    for seg in [((bx0, y - bh / 2), (bx0, by)), ((bx0, by), (bx1, by)), ((bx1, by), (bx1, y - bh / 2))]:
        (ax, ay), (cx, cy) = seg
        L = math.hypot(cx - ax, cy - ay); n = int(L // 12)
        for k in range(n):
            if (k * 12 + dash) % 24 < 12:
                p0 = (ax + (cx - ax) * k / n, ay + (cy - ay) * k / n); p1 = (ax + (cx - ax) * (k + .6) / n, ay + (cy - ay) * (k + .6) / n)
                d.line([p0, p1], fill=ORANGE, width=3)
    d.text(((bx0 + bx1) / 2, by - 24), "token mean · full precision", font=font(22, bold=True), fill=ORANGE, anchor="mm")
    # pulses
    for off in (0.0, 0.33, 0.66):
        u = ((t / 6.0) + off) % 1.0
        px = 40 + u * (W - 80)
        d.ellipse([px - 8, y - 8, px + 8, y + 8], fill=BLUE)
        # orange pulse on the bypass
        L1, L2, L3 = by and (y - bh / 2 - by), bx1 - bx0, (y - bh / 2 - by)
        s = u * (L1 + L2 + L3)
        if s < L1: q = (bx0, y - bh / 2 - s)
        elif s < L1 + L2: q = (bx0 + (s - L1), by)
        else: q = (bx1, by + (s - L1 - L2))
        d.ellipse([q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7], fill=ORANGE)
    for k, (a, b, c) in enumerate(BOXES):
        x = xs[k]
        d.rounded_rectangle([x, y - bh / 2, x + bw, y + bh / 2], radius=14, fill=(14, 38, 43), outline=c, width=3)
        d.text((x + bw / 2, y - 14), a, font=font(24, bold=True), fill=MINT, anchor="mm")
        d.text((x + bw / 2, y + 20), b, font=font(20), fill=MINT2, anchor="mm")
    d.text((22, y), "x", font=font(30, serif=True), fill=MINT, anchor="lm")
    d.text((W - 22, y), "y", font=font(30, serif=True), fill=MINT, anchor="rm")
    d.text((W / 2, 600), "Blue: 4-bit path      Orange: exact rank-1 term, never quantized", font=font(20), fill=MINT2, anchor="mm")
    return im


# ---------------------------------------------------------------- clip 3: comparison wipes
def load(p, s=520):
    return Image.open(p).convert("RGB").resize((s, s), Image.LANCZOS)

PAIRS = [
    ("PixArt-Σ · W4A4", load(os.path.join(HERE, "build", "pixart_377_hadanorm_pad.jpg")), "HadaNorm-style",
     load(os.path.join(PAGE, "pixart_377_centriq.jpg")), load(os.path.join(PAGE, "pixart_377_ref.jpg")), "FP16"),
    ("FLUX.1-schnell · W4A2", load(os.path.join(PAGE, "fluxa2_0_convrot.jpg")), "ConvRot-style",
     load(os.path.join(PAGE, "fluxa2_0_centriq.jpg")), load(os.path.join(PAGE, "fluxa2_0_ref.jpg")), "BF16"),
    ("FLUX.1-schnell · W4A2", load(os.path.join(PAGE, "fluxa2_201_rot.jpg")), "Rotation only",
     load(os.path.join(PAGE, "fluxa2_201_centriq.jpg")), load(os.path.join(PAGE, "fluxa2_201_ref.jpg")), "BF16"),
]
SEG = 5.0


def compare_frame(i):
    t = i / FPS; k = int(t // SEG) % len(PAIRS); u = (t % SEG) / SEG
    title_, other, oname, ours, ref, rname = PAIRS[k]
    im = background(); d = ImageDraw.Draw(im)
    d.text((W / 2, 52), title_, font=font(34, serif=True), fill=MINT, anchor="mm")
    S = 520; xr, xm, y = 80, 680, 120
    im.paste(ref, (xr, y)); d.text((xr + S / 2, y + S + 30), rname + " (reference)", font=font(22), fill=MINT2, anchor="mm")
    p = ease((u - 0.15) / 0.5)          # wipe from other method to CentriQ
    cut = int(S * p)
    comp = other.copy(); comp.paste(ours.crop((0, 0, cut, S)), (0, 0))
    im.paste(comp, (xm, y))
    if 0 < cut < S:
        d.line([(xm + cut, y), (xm + cut, y + S)], fill=MINT, width=3)
    if p > 0.02:
        d.rounded_rectangle([xm + 12, y + 12, xm + 132, y + 48], radius=8, fill=ORANGE)
        d.text((xm + 72, y + 30), "CentriQ", font=font(20, bold=True), fill=(27, 13, 5), anchor="mm")
    if p < 0.98:
        d.rounded_rectangle([xm + S - 12 - 200, y + 12, xm + S - 12, y + 48], radius=8, fill=(0, 0, 0))
        d.text((xm + S - 112, y + 30), oname, font=font(20, bold=True), fill=MINT, anchor="mm")
    d.text((xm + S / 2, y + S + 30), "same prompt and seed, no calibration data", font=font(22), fill=MINT2, anchor="mm")
    return im


if __name__ == "__main__":
    encode(spike_frame, int(10 * FPS), "mean_spike.mp4")
    encode(method_frame, int(6 * FPS), "method_flow.mp4")
    encode(compare_frame, int(SEG * len(PAIRS) * FPS), "compare_wipe.mp4")
    for n, fn, t in [("mean_spike", spike_frame, 2.0), ("mean_spike_b", spike_frame, 6.0), ("method_flow", method_frame, 1.0),
                     ("compare_wipe", compare_frame, 3.5)]:
        fn(int(t * FPS)).save(os.path.join(OUT, f"{n}_poster.jpg"), quality=88)
