"""Shared perceptual/pixel metric kernels (P1 gen/eval seam, c4).

One implementation of the LPIPS forward and the uint8 MSE, used by every
pre-generated-image metric driver (scripts/lpips_eval.py, scripts/psnr_eval.py,
scripts/paired_image_metrics.py, scripts/fid_eval.py). The drivers keep their own
IO contracts (reference-vs-method discovery vs index-paired vs category-nested) and,
for PSNR, their own dB formula + identical-image sentinel — this module only owns the
numerics that were duplicated verbatim.

LPIPS convention (all sites): AlexNet, 256x256 resize (antialias), ToTensor -> [0,1],
forward(..., normalize=True). lpips.LPIPS defaults to eval_mode=True (dropout off), so
the distance is deterministic; .eval() is called again here as a harmless no-op to match
the explicit lpips_eval/fid_eval construction.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image


class LpipsScorer:
    """AlexNet LPIPS with the repo-headline transform (256x256, antialias,
    ToTensor, normalize=True). Drop-in for paired_image_metrics._LPIPS: same
    `(device, net="alex", resize=256)` constructor and `__call__(a, b)`.

    `a`/`b` may be image paths (opened + converted to RGB here) or already-loaded
    PIL images. `resize=0` keeps full resolution (no Resize op).
    """

    def __init__(self, device, net: str = "alex", resize: int = 256):
        try:
            import lpips
        except ImportError as e:  # pragma: no cover - environment guard
            raise ImportError("lpips not installed. Run: pip install lpips") from e
        import torchvision.transforms as T

        self.model = lpips.LPIPS(net=net).float().to(device)
        self.model.eval()
        self.device = device

        ops = [T.ToTensor()]
        if resize:
            ops.insert(0, T.Resize((resize, resize), antialias=True))
        self.transform = T.Compose(ops)

    def _tensor(self, img):
        if isinstance(img, (str, os.PathLike)):
            img = Image.open(img).convert("RGB")
        return self.transform(img).unsqueeze(0).float().to(self.device)

    def __call__(self, a, b) -> float:
        ta = self._tensor(a)
        tb = self._tensor(b)
        with torch.no_grad():
            return float(self.model(ta, tb, normalize=True).item())

    # explicit alias so call sites can read `scorer.distance(...)`
    distance = __call__


def mse_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-squared error between two uint8 RGB arrays, computed in float64.

    The shared core of both PSNR drivers; each keeps its own dB formula and
    identical-image sentinel on top of this value (they are mathematically equal
    for mse>0 but not bit-identical, so the log line stays in the driver).
    """
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
