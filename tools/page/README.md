# Tools used to build the page

| File | Purpose |
|---|---|
| `pdf2png.swift` | Render a paper figure PDF to PNG with macOS PDFKit: `swift tools/page/pdf2png.swift in.pdf out.png 3` |
| `probe.py`, `crop.py` | Detect the grid in the paper's qualitative figures and crop single tiles (used for the SANA images in `assets/img/cmp/`) |
| `rank_candidates.py` | Rank prompts by CentriQ's LPIPS advantage over the other methods |
| `make_strips.py` | Side-by-side review strips for the ranked prompts |
| `export_picks.py` | Export the chosen prompts to `assets/img/cmp2/` at 768 px |

Images come from runs of `scripts/rods_a_w4a4_eval.py` in the CentriQ code release, with
`paired_vs_bf16.json` from `scripts/paired_image_metrics.py`. Every run of a model uses the same
prompts (MJHQ-5k, sorted) and seed, so image `i` shares its prompt and initial noise across methods.
