"""Frozen DINOv2 + SigLIP image encoder shared by retrieval and the selector score.

    phi(I) = L2-norm( concat( L2-norm(DINOv2(I)), L2-norm(SigLIP(I)) ) )   -> 1536-d

Models are loaded lazily on the first call.
"""
from __future__ import annotations
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

DINO_ID = "facebook/dinov2-base"
SIG_ID = "google/siglip-base-patch16-224"


class DinoSigLIPEncoder:
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self._dino = self._dino_p = self._sig = self._sig_p = None

    def _load(self):
        if self._dino is None:
            self._dino_p = AutoImageProcessor.from_pretrained(DINO_ID, use_fast=True)
            self._dino = AutoModel.from_pretrained(DINO_ID).to(self.device, dtype=self.dtype).eval()
            self._sig_p = AutoImageProcessor.from_pretrained(SIG_ID, use_fast=True)
            self._sig = SiglipVisionModel.from_pretrained(SIG_ID).to(self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def __call__(self, imgs: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """imgs: tensor (B,3,H,W) in [0,1], or a list / array of (H,W,3) uint8 images.
        Returns (B, 1536) L2-normalized embeddings on self.device."""
        self._load()
        if isinstance(imgs, torch.Tensor):
            if imgs.dtype == torch.float32 and imgs.max() <= 1.5:   # [0,1] -> uint8
                imgs = (imgs * 255).clamp(0, 255).byte()
            if imgs.ndim == 4 and imgs.shape[1] == 3:               # B,C,H,W -> B,H,W,C
                imgs = imgs.permute(0, 2, 3, 1)
            imgs = [im.cpu().numpy() for im in imgs]
        elif isinstance(imgs, np.ndarray):
            imgs = [imgs] if imgs.ndim == 3 else [imgs[i] for i in range(imgs.shape[0])]
        di = self._dino_p(images=imgs, return_tensors="pt").to(self.device)
        d = self._dino(**di).pooler_output                         # (B, 768)
        si = self._sig_p(images=imgs, return_tensors="pt").to(self.device)
        s = self._sig(**si).pooler_output                          # (B, 768)
        emb = torch.cat([F.normalize(d.float(), dim=1), F.normalize(s.float(), dim=1)], dim=1)
        return F.normalize(emb, dim=1)                             # (B, 1536)
