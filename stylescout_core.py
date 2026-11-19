"""Shared model code for StyleScout: CLIP embeddings, trained heads, labels."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
ROOT = Path(__file__).resolve().parent
CATALOG_DIR = ROOT / "catalog"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

_model = None
_proc = None


def load_clip():
    global _model, _proc
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL_ID)
    return _model, _proc


def _normalize(t: torch.Tensor) -> np.ndarray:
    t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy()


@torch.no_grad()
def embed_images(images, batch_size: int = 64) -> np.ndarray:
    model, proc = load_clip()
    out = []
    for i in range(0, len(images), batch_size):
        chunk = [im.convert("RGB") for im in images[i : i + batch_size]]
        inputs = proc(images=chunk, return_tensors="pt").to(DEVICE)
        out.append(_normalize(model.get_image_features(**inputs)))
    return np.concatenate(out, axis=0)


@torch.no_grad()
def embed_texts(texts: List[str]) -> np.ndarray:
    model, proc = load_clip()
    inputs = proc(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    return _normalize(model.get_text_features(**inputs))


class LinearHead(nn.Module):
    """Small PyTorch classifier trained on top of frozen CLIP image embeddings."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, n_classes))

    def forward(self, x):
        return self.net(x)


GENDER_WORDS = {"men", "women", "unisex", "boys", "girls", "kids", "men's", "women's"}


def brand_from_name(name: str) -> str:
    """Product names look like 'Puma Men Black Running Shoes' -> 'Puma'."""
    tokens = str(name).split()
    for i, tok in enumerate(tokens[:5]):
        if tok.lower() in GENDER_WORDS and i > 0:
            return " ".join(tokens[:i])
    return tokens[0] if tokens else "Unknown"


# Zero-shot vocabularies (no training data needed for these).
STYLE_TAGS = [
    "Y2K", "vintage workwear", "streetwear", "minimalist", "boho", "preppy",
    "athleisure", "grunge", "business formal", "90s", "70s retro", "cottagecore",
    "techwear", "punk", "traditional ethnic", "evening party", "casual everyday",
    "outdoor hiking", "old money", "skater",
]
STYLE_PROMPTS = [f"a photo of {s} style fashion" for s in STYLE_TAGS]

FASHION_PROMPTS = [
    "a photo of clothing", "a photo of a person wearing an outfit", "a photo of shoes",
    "a photo of a handbag", "a photo of jewelry", "a photo of a watch",
    "a photo of a hat", "a photo of sunglasses", "a product photo of a garment",
    "a photo of a belt", "a photo of socks",
]
NON_FASHION_PROMPTS = [
    "a photo of a dog", "a photo of a cat", "a photo of food", "a landscape photo",
    "a photo of a car", "a photo of a building", "a screenshot of text",
    "a photo of furniture", "a photo of a phone or laptop", "a photo of a plant",
    "a photo of a toy", "a photo of a kitchen utensil",
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
