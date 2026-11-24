"""Build the CentriQ talk deck (centriq_talk.pptx) with python-pptx.

Every number comes from the paper (arXiv:2610.06260). Speaker notes hold the narration
used for the intro video. Run: uv run --with python-pptx --with pillow python build_deck.py
"""
import os
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "build")
PAGE = os.path.join(HERE, "..", "project_page", "assets", "img")

INK = RGBColor(0x0D, 0x1B, 0x1E); INK2 = RGBColor(0x3A, 0x4A, 0x4F); MUTED = RGBColor(0x6B, 0x7B, 0x80)
LINE = RGBColor(0xE3, 0xE8, 0xEA); BG2 = RGBColor(0xF5, 0xF8, 0xF8); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = RGBColor(0xE2, 0x60, 0x1B); BLUE = RGBColor(0x16, 0x73, 0xB8); GRAY = RGBColor(0x9A, 0xA9, 0xAD)
DARK = RGBColor(0x0B, 0x1D, 0x20); MINT = RGBColor(0xEA, 0xF4, 0xF3); MINT2 = RGBColor(0x9F, 0xBD, 0xB9)
HEAD, BODY = "Cambria", "Calibri"

W, H = Inches(13.333), Inches(7.5)


def dark_bg(spike=True):
    p = os.path.join(B, "bg_dark.png" if spike else "bg_dark_plain.png")
    if not os.path.exists(p):
        w, h = 1920, 1080
        im = Image.new("RGB", (w, h))
        px = im.load()
        cx, cy = w * 0.5, h * 0.32
        for y in range(h):
            for x in range(0, w):
                d = min(1.0, (((x - cx) / 1250) ** 2 + ((y - cy) / 760) ** 2) ** 0.5)
                c0, c1 = (20, 58, 64), (7, 21, 24)
                px[x, y] = tuple(int(c0[i] + (c1[i] - c0[i]) * d) for i in range(3))
        # faint bar motif (token coordinates) along the bottom
        dr = ImageDraw.Draw(im, "RGBA")
        import random
        random.seed(3)
        n = 96; bw = w / n
        for i in range(n):
            bh = int(60 + abs(random.gauss(0, 1)) * 90)
            col = (242, 131, 74, 120) if (spike and i == n // 2) else (120, 200, 192, 45)
            if spike and i == n // 2:
                bh = 330
            if not spike:
                bh = int(bh * 0.55)
            dr.rectangle([i * bw + 2, h - bh, (i + 1) * bw - 2, h], fill=col)
        im.save(p)
    return p


prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, spacing_after=0, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing_after)
        segs = para if isinstance(para, list) else [para]
        for seg in segs:
            if isinstance(seg, str):
                seg = (seg, {})
            r = p.add_run()
            r.text = seg[0]
            o = seg[1]
            r.font.size = Pt(o.get("size", size))
            r.font.name = o.get("font", font)
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", italic)
            r.font.color.rgb = o.get("color", color)
    return tb


def title(slide, t, sub=None, dark=False):
    text(slide, 0.6, 0.45, 12.1, 0.9, t, size=36, font=HEAD, color=MINT if dark else INK, name="Title")
    if sub:
        text(slide, 0.6, 1.3, 12.1, 0.5, sub, size=16, color=MINT2 if dark else MUTED, name="Subtitle")


def card(slide, x, y, w, h, fill=BG2, line=LINE):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line; s.line.width = Pt(1)
    s.shadow.inherit = False
    return s


def dot(slide, x, y, d, color):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    return s


def picture(slide, path, x, y, w=None, h=None, frame=True):
    if frame:
        im = Image.open(path)
        ar = im.width / im.height
        if w and not h:
            h = w / ar
        if h and not w:
            w = h * ar
        card(slide, x - 0.12, y - 0.12, w + 0.24, h + 0.24, fill=WHITE)
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


NOTES = []


def notes(slide, s):
    # Keynote rejects python-pptx notes slides, so narration is stored separately and added in Keynote.
    NOTES.append(s)


def new(dark=False, spike=True):
    s = prs.slides.add_slide(BLANK)
    if dark:
        s.shapes.add_picture(dark_bg(spike), 0, 0, W, H)
    return s


def stat(slide, x, y, big, small, dark=False, accent=None, w=2.9):
    text(slide, x, y, w, 0.9, big, size=44, font=HEAD, color=accent or (MINT if dark else INK))
    text(slide, x, y + 0.95, w, 0.8, small, size=14, color=MINT2 if dark else MUTED)


# 1 ---------------------------------------------------------------- title
s = new(dark=True)
text(s, 0.6, 0.7, 12.1, 0.4, "CALIBRATION-FREE · W4A4 · DIFFUSION TRANSFORMERS", size=12, color=MINT2,
     align=PP_ALIGN.CENTER, bold=True)
text(s, 0.6, 1.35, 12.1, 1.6, [[("Centri", {}), ("Q", {"color": ORANGE, "italic": True})]], size=96, font=HEAD,
     color=MINT, align=PP_ALIGN.CENTER, name="Title")
text(s, 1.2, 3.05, 10.9, 0.9, "Calibration-Free Quantization of Diffusion Transformers via Exact Mean Centering",
     size=24, color=MINT, align=PP_ALIGN.CENTER)
text(s, 0.6, 4.15, 12.1, 0.5, "Nataša Jovanović¹ ²     Mathieu Salzmann²     Saqib Javed²", size=18, color=MINT,
     align=PP_ALIGN.CENTER)
text(s, 0.6, 4.65, 12.1, 0.4, "¹ Tenstorrent   ·   ² EPFL        arXiv:2610.06260", size=14, color=MINT2,
     align=PP_ALIGN.CENTER)
notes(s, "Hi, I'm going to present CentriQ, a way to quantize diffusion transformers to 4-bit weights and "
         "4-bit activations without any calibration data.")

# 2 ---------------------------------------------------------------- problem
s = new()
title(s, "4-bit DiTs today depend on calibration", "Every W4A4 method that works on diffusion models fits something to data first")
items = [("Smoothing factors", "fitted to activation statistics of one checkpoint"),
         ("Clipping ranges and rotations", "tuned on a prompt set and timesteps"),
         ("Low-rank corrections", "e.g. SVDQuant's 16-bit branch, fitted to calibration prompts")]
for i, (h_, d_) in enumerate(items):
    y = 2.2 + i * 1.25
    dot(s, 0.75, y + 0.08, 0.32, BLUE)
    text(s, 1.3, y, 6.0, 0.45, h_, size=20, bold=True)
    text(s, 1.3, y + 0.45, 6.0, 0.6, d_, size=15, color=MUTED)
card(s, 7.7, 2.1, 5.0, 3.9, fill=BG2)
stat(s, 8.1, 2.45, "13.0 h", "SVDQuant calibration log for FLUX.1-schnell (128 prompts × 4 steps)", accent=BLUE, w=4.3)
stat(s, 8.1, 4.25, "Per model", "repeated for every checkpoint, fine-tune and bit-width", w=4.3)
notes(s, "Quantizing both weights and activations to 4 bits is what actually speeds up a diffusion transformer, "
         "because its forward pass is compute-bound. But every method that reaches W4A4 on diffusion models today "
         "relies on calibration: smoothing factors, clipping ranges, rotations or low-rank corrections, fitted to "
         "prompts run through one specific checkpoint. For FLUX, SVDQuant's calibration took us thirteen hours, "
         "and it has to be redone for every model, fine-tune and bit-width.")

# 3 ---------------------------------------------------------------- collapse
s = new()
title(s, "Without calibration, 4-bit DiTs collapse", "W4A4 with calibration-free quantizers on three DiTs")
picture(s, os.path.join(B, "fig_collapse.png"), 0.9, 1.95, h=5.0)
text(s, 8.0, 2.3, 4.7, 4.0, [
    [("RTN and HQQ ", {"bold": True}), "turn PixArt-Σ into noise and lose the prompt on FLUX.1-schnell and SANA-1.6B."],
    [("FID 481 ", {"bold": True, "color": ORANGE}), ("on PixArt-Σ for RTN, against 16.7 at 16 bits.", {})],
    [("CentriQ ", {"bold": True}), "stays close to the 16-bit image on all three models, with no data."]],
    size=17, spacing_after=16)
notes(s, "Without calibration, things fall apart. Plain round-to-nearest and HQQ produce pure noise on PixArt-Sigma, "
         "with an FID of 481, and lose the content of the prompt on FLUX and SANA. On the right you can see CentriQ, "
         "which stays close to the 16-bit output on all three models without seeing any data. So what breaks?")

# 4 ---------------------------------------------------------------- insight
s = new()
title(s, "A hidden mean survives the rotation", "adaLN adds a per-token offset; Hadamard rotation piles it onto one coordinate")
picture(s, os.path.join(B, "sorted_magnitudes.png"), 0.9, 2.0, h=4.6)
text(s, 7.6, 2.1, 5.1, 4.4, [
    [("1  ", {"bold": True, "color": ORANGE}), "Adaptive LayerNorm shifts every token by a conditioning-dependent offset, so each activation carries a per-token mean."],
    [("2  ", {"bold": True, "color": ORANGE}), "A Hadamard rotation keeps that energy and maps it to one or a few coordinates: the DC spike."],
    [("3  ", {"bold": True, "color": ORANGE}), "The spike sets the 4-bit step for the whole token. Removing it makes the grid "],
    [("2.4× finer", {"bold": True, "size": 26, "font": HEAD})]], size=16, spacing_after=12)
notes(s, "Diffusion transformers condition every block through adaptive layer norm, which shifts each token by an "
         "offset. So every activation vector carries a per-token mean. The standard data-free fix for outliers is a "
         "Hadamard rotation, but a rotation preserves energy, and it maps this mean onto one or a few coordinates. "
         "That spike, shown here in blue, then sets the quantization step for the entire token. Remove it, and the "
         "4-bit grid for this token becomes 2.4 times finer.")

# 5 ---------------------------------------------------------------- why rotation fails
s = new()
title(s, "Why rotation alone cannot fix it", "Only regular Hadamard matrices spread the mean evenly, and none fit DiT widths")
for i, (w_, f_) in enumerate([("1152", "PixArt-Σ"), ("2240", "SANA-1.6B"), ("3072", "FLUX.1")]):
    x = 0.9 + i * 2.25
    card(s, x, 2.3, 2.0, 1.9)
    text(s, x, 2.55, 2.0, 0.8, w_, size=40, font=HEAD, align=PP_ALIGN.CENTER)
    text(s, x, 3.45, 2.0, 0.5, f_, size=14, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 0.9, 4.6, 6.6, 1.8, [
    "Regular Hadamard matrices (equal row sums) exist only at orders n = 4u².",
    "None of these hidden widths has that form, so practical rotations are block-structured or Kronecker products, which inherit the spike."],
    size=16, color=INK2, spacing_after=10)
picture(s, os.path.join(B, "dc_energy_fraction.png"), 8.0, 2.2, w=4.6)
text(s, 8.0, 5.45, 4.6, 0.8, "Top: share of token energy in the mean. Bottom: per-token max/RMS under identity, rotation, centering, and both.",
     size=11, color=MUTED)
notes(s, "Could a better rotation spread the mean evenly? Only a regular Hadamard matrix, one with equal row sums, does "
         "that, and those exist only at orders four u squared. None of the evaluated widths, 1152, 2240 or 3072, has "
         "that form. On the right: rotation alone leaves the per-token max-to-RMS ratio where it was, while centering "
         "before rotation brings it down to what incoherence predicts.")

# 6 ---------------------------------------------------------------- method
s = new()
title(s, "CentriQ: center, rotate, restore exactly", "Because the layer is linear, the mean can leave before quantization and return with no error")
steps = [(ORANGE, "Center", "Subtract the per-token mean x̄ₜ, at O(n) cost, in the same kernel launch that computes the scale."),
         (BLUE, "Rotate & quantize", "Fixed factorized Hadamard H = Hₘ ⊗ H₂ᵏ, zero-padded only when needed. One closed-form scale per token."),
         (ORANGE, "Restore", "The mean re-enters as an exact rank-1 term x̄ₜ·W1 in full precision, fused into the GEMM epilogue.")]
for i, (c, h_, d_) in enumerate(steps):
    x = 0.6 + i * 4.15
    card(s, x, 2.0, 3.9, 2.55)
    dot(s, x + 0.3, 2.3, 0.45, c)
    text(s, x + 0.3, 2.31, 0.45, 0.45, str(i + 1), size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.95, 2.33, 2.8, 0.45, h_, size=19, bold=True)
    text(s, x + 0.3, 3.0, 3.35, 1.5, d_, size=14, color=INK2)
card(s, 0.6, 4.95, 12.1, 1.35, fill=WHITE)
text(s, 0.6, 5.2, 12.1, 0.9, [[("yₜ  =  ", {}), ("Ŵ_R · Q( H P (xₜ − x̄ₜ1) )", {"color": BLUE}), ("  +  ", {}),
                               ("x̄ₜ · W1", {"color": ORANGE}), ("  +  b", {})]], size=28, font=HEAD,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 6.5, 12.1, 0.4, "Blue: 4-bit path.  Orange: full-precision rank-1 branch that bypasses both quantizers.",
     size=12, color=MUTED, align=PP_ALIGN.CENTER)
notes(s, "CentriQ does three things. It subtracts each token's mean, which costs order n and is computed in the same "
         "kernel launch as the scale. It rotates the centered token with a fixed factorized Hadamard matrix and "
         "quantizes it with one closed-form scale per token. And because the layer is linear, it adds the mean back "
         "exactly, as a rank-1 term, the mean times W times one, in full precision inside the GEMM epilogue. The mean "
         "never touches a quantizer.")

# 7 ---------------------------------------------------------------- dataflow + weights
s = new()
title(s, "One layer, and nothing sees data", "Online per-token path on top, offline weight path below")
picture(s, os.path.join(B, "layer_dataflow.png"), 0.8, 1.95, w=7.6)
card(s, 9.0, 1.85, 3.75, 4.5)
text(s, 9.3, 2.1, 3.2, 0.5, "Weights: RoDS", size=20, bold=True)
text(s, 9.3, 2.65, 3.2, 3.6, [
    "Grouped INT4 (64 columns), fitted offline under a robust ℓp objective with p = 0.7.",
    "Large residuals cost little, so the dense mode of each group sets the scale instead of its heavy tail.",
    "Solved by IRLS and half-quadratic splitting. No calibration data."], size=14, color=INK2, spacing_after=10)
notes(s, "Here is one quantized layer. The online path centers, pads and rotates each token, quantizes it to INT4 and "
         "runs the INT4 GEMM. The orange branch carries the mean around both quantizers. Offline, the weights are "
         "fitted under a robust l-p objective with p equal to 0.7, which we call RoDS. Because the marginal cost of a "
         "large residual shrinks, the dense mode of each weight group sets the scale, not its heavy tail. Neither path "
         "uses data.")

# 8 ---------------------------------------------------------------- results table
s = new()
title(s, "Matches calibrated SVDQuant at W4A4, without data", "MJHQ, 1024²; PSNR and LPIPS against the 16-bit image from the same seed")
rows = [("Model", "Method", "Calibration", "FID ↓", "ImageReward ↑", "PSNR ↑", "LPIPS ↓"),
        ("PixArt-Σ", "SVDQuant†", "yes", "19.2", "0.878", "17.6", "0.323"),
        ("", "CentriQ", "none", "19.1", "0.891", "16.15", "0.278"),
        ("FLUX.1-schnell", "SVDQuant†", "yes", "18.3", "0.951", "18.3", "0.258"),
        ("", "CentriQ", "none", "18.8", "0.939", "18.40", "0.173"),
        ("SANA-1.6B", "SVDQuant†", "yes", "19.3", "0.935", "17.8", "0.220"),
        ("", "CentriQ", "none", "16.2", "1.072", "17.64", "0.172")]
tbl = s.shapes.add_table(len(rows), 7, Inches(0.6), Inches(2.0), Inches(12.1), Inches(4.0)).table
widths = [2.3, 1.9, 1.6, 1.4, 1.9, 1.4, 1.6]
for j, wd in enumerate(widths):
    tbl.columns[j].width = Inches(wd)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = tbl.cell(i, j)
        c.text = v
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j < 3 else PP_ALIGN.RIGHT
        run = p.runs[0] if p.runs else p.add_run()
        run.font.size = Pt(15 if i else 12); run.font.name = BODY
        run.font.bold = (i == 0) or r[1] == "CentriQ"
        run.font.color.rgb = MUTED if i == 0 else INK
        c.fill.solid()
        c.fill.fore_color.rgb = RGBColor(0xFD, 0xEE, 0xE5) if r[1] == "CentriQ" else (BG2 if i == 0 else WHITE)
        c.margin_left = c.margin_right = Inches(0.12)
text(s, 0.6, 6.25, 12.1, 0.6, "† Reported by the SVDQuant authors. Regenerated under our exact protocol, the two match on FLUX.1-schnell "
     "(FID 18.73 vs 18.80) and CentriQ leads on PixArt-Σ (19.10 vs 19.27).", size=12, color=MUTED)
notes(s, "Here are the main results at W4A4 on MJHQ. Across PixArt-Sigma, FLUX and SANA, CentriQ matches calibrated "
         "SVDQuant on FID and ImageReward and has the lowest LPIPS against the 16-bit images, without using any data. "
         "Because SVDQuant's numbers come from its paper, we also regenerated it under our exact protocol: the two "
         "match on FLUX, and CentriQ is ahead on PixArt-Sigma.")

# 9 ---------------------------------------------------------------- images
s = new()
title(s, "Same prompt, same seed", "W4A4. Left to right: 16-bit, CentriQ, then two other 4-bit methods")
CMP = os.path.join(PAGE, "cmp2")
rows_img = [("PixArt-Σ", "pixart_377", ["ref", "centriq", "hadanorm_pad", "rot_pad"], ["FP16", "CentriQ", "HadaNorm-style", "Rotation only"]),
            ("FLUX.1-schnell", "flux_169", ["ref", "centriq", "svd", "hadanorm"], ["BF16", "CentriQ", "SVDQuant", "HadaNorm-style"])]
for r_, (mname, stem, keys, labels) in enumerate(rows_img):
    y = 1.95 + r_ * 2.55
    text(s, 0.6, y + 0.85, 1.6, 0.6, mname, size=15, bold=True)
    for j, (k, lab) in enumerate(zip(keys, labels)):
        x = 2.4 + j * 2.6
        src = os.path.join(B, f"{stem}_{k}.jpg") if k.endswith("_pad") else os.path.join(CMP, f"{stem}_{k}.jpg")
        s.shapes.add_picture(src, Inches(x), Inches(y), Inches(2.2), Inches(2.2))
        text(s, x, y + 2.22, 2.2, 0.3, lab, size=11, color=ORANGE if k == "centriq" else MUTED,
             bold=k == "centriq", align=PP_ALIGN.CENTER)
notes(s, "Here is what that looks like. Every image in a row shares the prompt and the initial noise. On PixArt-Sigma, "
         "CentriQ stays close to the 16-bit wolf, while HadaNorm-style centering and rotation alone, run in the same "
         "setting, break the image. On FLUX, CentriQ and calibrated SVDQuant are both close to the 16-bit image.")

# 10 --------------------------------------------------------------- speed & cost
s = new()
title(s, "Fast in the same engine, cheap to deploy", "FLUX.1-schnell transformer, seconds per image (1024², 4 steps, batch 1)")
cd = CategoryChartData()
cd.categories = ["BF16", "BF16 + compile", "SVDQuant", "CentriQ"]
cd.add_series("RTX PRO 6000", (1.166, 1.006, 0.646, 0.664))
cd.add_series("A100", (1.762, 1.407, 1.245, 1.289))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.95), Inches(7.2), Inches(4.6), cd)
ch = gf.chart
ch.has_legend = True; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(12); ch.legend.font.name = BODY
ch.plots[0].gap_width = 60; ch.plots[0].overlap = -10
ch.plots[0].has_data_labels = True
dl = ch.plots[0].data_labels; dl.font.size = Pt(11); dl.font.name = BODY; dl.number_format = "0.000"; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
for ser, col in zip(ch.series, (ORANGE, BLUE)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
ch.category_axis.tick_labels.font.size = Pt(12); ch.category_axis.tick_labels.font.name = BODY
ch.category_axis.reverse_order = True
ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False
card(s, 8.3, 1.95, 4.4, 4.6)
stat(s, 8.65, 2.2, "≤ 3.5%", "latency gap to SVDQuant, same fused Nunchaku engine", w=3.8)
stat(s, 8.65, 3.65, "9.3 min", "to convert FLUX.1-schnell, vs 13.0 h of SVDQuant calibration", accent=ORANGE, w=3.8)
stat(s, 8.65, 5.1, "26.8 GB", "peak memory (SVDQuant 26.6 GB, BF16 36.3 GB)", w=3.8)
notes(s, "Speed. Both 4-bit methods run in the same fused Nunchaku engine, so they differ only in the quantizer. "
         "CentriQ is within 3.5 percent of SVDQuant's latency on an A100, and within 2.8 percent on an RTX PRO 6000, "
         "where its transformer is 1.76 times faster than eager BF16. Converting a new checkpoint takes 9.3 minutes "
         "for FLUX, against thirteen hours of calibration, with about the same peak memory.")

# 11 --------------------------------------------------------------- low bit
s = new()
title(s, "Usable images at 2-bit activations", "GenEval prompts on FLUX.1-schnell at W4A4, W4A2 and W2A4 (simulated quantization)")
picture(s, os.path.join(B, "fig_geneval.png"), 0.9, 1.95, h=4.9)
card(s, 6.9, 1.95, 5.8, 4.9)
text(s, 7.2, 2.2, 5.3, 0.5, "Component ablation, FLUX.1-schnell, FID ↓", size=15, bold=True)
ab = [("", "W4A4", "W4A2", "W2A4"), ("No transform", "40.62", "437.41", "363.14"), ("+ rotation", "25.65", "455.30", "35.84"),
      ("+ centering", "18.46", "444.21", "21.81"), ("+ clip = CentriQ", "18.80", "35.59", "21.56"), ("BF16", "19.45", "19.45", "19.45")]
t2 = s.shapes.add_table(len(ab), 4, Inches(7.2), Inches(2.8), Inches(5.2), Inches(2.8)).table
for j, wd in enumerate([2.0, 1.05, 1.1, 1.05]):
    t2.columns[j].width = Inches(wd)
for i, r in enumerate(ab):
    for j, v in enumerate(r):
        c = t2.cell(i, j); c.text = v
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
        run = p.runs[0] if p.runs else p.add_run()
        run.font.size = Pt(13); run.font.name = BODY; run.font.bold = (i == 0 or "CentriQ" in r[0])
        run.font.color.rgb = MUTED if (i == 0 or r[0] == "BF16") else INK
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFD, 0xEE, 0xE5) if "CentriQ" in r[0] else WHITE
        c.margin_left = c.margin_right = Inches(0.1)
text(s, 7.2, 5.85, 5.3, 0.9, "Centering closes the gap at 4 bits; one shared clip percentile carries it to 2-bit activations.",
     size=13, color=INK2)
notes(s, "Below four bits. On FLUX, CentriQ is the first calibration-free method to keep usable image quality at "
         "2-bit activations, and it outperforms the strongest calibration-free method reported so far at 2-bit weights. "
         "The ablation on the right shows why: rotation helps, centering closes the gap at four bits, and a single "
         "shared clip percentile brings W4A2 from an FID of 444 down to 36.")

# 12 --------------------------------------------------------------- conclusion
s = new(dark=True, spike=False)
text(s, 0.6, 0.9, 12.1, 0.9, "Takeaways", size=40, font=HEAD, color=MINT, name="Title")
pts = [("The per-token mean from adaLN", " is what breaks data-free rotation on DiTs."),
       ("Center, rotate, restore exactly:", " the mean bypasses both quantizers as a rank-1 term."),
       ("No calibration data:", " matches calibrated SVDQuant at W4A4 on three DiTs, within 3.5% of its latency."),
       ("Below 4 bits:", " first calibration-free method with usable quality at 2-bit activations.")]
for i, (a, b_) in enumerate(pts):
    y = 2.05 + i * 0.95
    dot(s, 0.8, y + 0.1, 0.28, ORANGE if i % 2 == 0 else BLUE)
    text(s, 1.3, y, 11.0, 0.8, [[(a, {"bold": True}), (b_, {})]], size=20, color=MINT)
text(s, 0.6, 5.85, 12.1, 0.5, "Paper: arxiv.org/abs/2610.06260     ·     Code and project page: coming soon", size=16,
     color=MINT2, align=PP_ALIGN.CENTER)
notes(s, "To sum up: the per-token mean that adaptive layer norm adds is what breaks data-free rotations on diffusion "
         "transformers. CentriQ centers each token, rotates, and restores the mean exactly, so it never touches a "
         "quantizer. With no calibration data, it matches calibrated SVDQuant at W4A4 on three models, runs within "
         "3.5 percent of its latency, and is the first calibration-free method with usable quality at 2-bit "
         "activations. The paper is on arXiv, and the code and project page are coming soon. Thanks for watching.")

out = os.path.join(HERE, "centriq_talk.pptx")
prs.save(out)
import json
json.dump(NOTES, open(os.path.join(HERE, "notes.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(HERE, "narration_script.md"), "w") as f:
    f.write("# CentriQ intro video: narration script\n\n")
    for i, n in enumerate(NOTES, 1):
        f.write(f"## Slide {i}\n\n{n}\n\n")
print("saved", out, len(prs.slides), "slides", len(NOTES), "notes")
