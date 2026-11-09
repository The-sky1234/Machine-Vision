"""CPU PatchCore: ImageNet ResNet18 memory bank, trained on good parts only.

This is a reduced PatchCore (single layer, random coreset, no anomalib).
It is not the paper's full WR50 ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(slots=True)
class VisionResult:
    score: float
    heatmap: NDArray[np.float32]
    is_anomaly: bool


class LightweightPatchCore:
    def __init__(self, image_size: int = 128, coreset_ratio: float = 0.2, threshold: float | None = None):
        if torch is None:
            raise ImportError("torch and torchvision are required for PatchCore")
        self.image_size = image_size
        self.coreset_ratio = coreset_ratio
        self.threshold = threshold
        self.device = torch.device("cpu")
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.eval()
        self.stem = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        ).to(self.device)
        for p in self.stem.parameters():
            p.requires_grad_(False)
        self.memory_bank: NDArray[np.float32] | None = None

    def _preprocess(self, image: NDArray[np.uint8]) -> "torch.Tensor":
        img = Image.fromarray(image).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def _embed(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        x = self._preprocess(image)
        feat = self.stem(x)  # (1, C, H, W)
        feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        feat = F.normalize(feat, p=2, dim=1)
        _, c, h, w = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy().astype(np.float32)
        return patches

    def fit(self, good_images: list[NDArray[np.uint8]], rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        banks = [self._embed(img) for img in good_images]
        bank = np.concatenate(banks, axis=0)
        n_keep = max(32, int(len(bank) * self.coreset_ratio))
        n_keep = min(n_keep, len(bank))
        idx = rng.choice(len(bank), size=n_keep, replace=False)
        self.memory_bank = bank[idx]

    def _patch_scores(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], int, int]:
        if self.memory_bank is None:
            raise RuntimeError("call fit() before predict")
        patches = self._embed(image)
        # Features are L2-normalised, so min Euclidean is sqrt(2-2*max cosine).
        dots = patches @ self.memory_bank.T
        mins = np.sqrt(np.clip(2.0 - 2.0 * dots.max(axis=1), 0.0, None)).astype(np.float32)
        side = int(np.sqrt(len(mins)))
        heat = mins.reshape(side, side)
        return heat, side, side

    def predict(self, image: NDArray[np.uint8]) -> VisionResult:
        heat, _, _ = self._patch_scores(image)
        score = float(np.max(heat))
        is_anomaly = bool(self.threshold is not None and score >= self.threshold)
        return VisionResult(score=score, heatmap=heat, is_anomaly=is_anomaly)

    def calibrate_threshold(self, good_images: list[NDArray[np.uint8]], quantile: float = 0.95) -> float:
        scores = [self.predict(img).score for img in good_images]
        self.threshold = float(np.quantile(scores, quantile))
        return self.threshold
