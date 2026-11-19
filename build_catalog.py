"""Build the StyleScout catalog and train the classifier heads.

Steps:
  1. Download the Fashion Product Images (small) dataset from Hugging Face.
  2. Embed every product photo with CLIP and save thumbnails.
  3. Train two small PyTorch heads (item type, colour) on the embeddings
     and report accuracy on held-out products.

Usage:  python build_catalog.py [--limit 15000]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from torch import nn

from stylescout_core import (
    CATALOG_DIR, DEVICE, LinearHead, brand_from_name, embed_images,
)

KEEP_MASTER = {"Apparel", "Footwear", "Accessories"}
MIN_PER_CLASS = 25


def train_head(X: np.ndarray, labels: list[str], name: str, epochs: int = 40):
    counts = Counter(labels)
    classes = sorted(c for c, n in counts.items() if n >= MIN_PER_CLASS)
    idx_of = {c: i for i, c in enumerate(classes)}
    rows = [i for i, l in enumerate(labels) if l in idx_of]
    random.Random(0).shuffle(rows)
    split = int(len(rows) * 0.8)
    tr, te = rows[:split], rows[split:]

    Xt = torch.tensor(X)
    y = torch.tensor([idx_of.get(l, -1) for l in labels])
    head = LinearHead(X.shape[1], len(classes))
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    tr_t = torch.tensor(tr)
    for ep in range(epochs):
        head.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for b in range(0, len(perm), 256):
            bi = perm[b : b + 256]
            # CLIP features are unit-norm; scale up so logits aren't tiny.
            loss = loss_fn(head(Xt[bi] * 10), y[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(Xt[te] * 10).argmax(1)
        acc = (pred == y[te]).float().mean().item()
        top3 = (head(Xt[te] * 10).topk(3, 1).indices == y[te][:, None]).any(1).float().mean().item()
    print(f"  {name}: {len(classes)} classes, train {len(tr)}, held-out {len(te)} "
          f"-> top-1 {acc:.1%}, top-3 {top3:.1%}")
    return {"state_dict": head.state_dict(), "classes": classes,
            "top1": acc, "top3": top3, "n_test": len(te)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15000,
                    help="max products to index (0 = all ~42k)")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading dataset ashraq/fashion-product-images-small ...")
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    ds = ds.filter(lambda r: r["masterCategory"] in KEEP_MASTER)
    ds = ds.shuffle(seed=42)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Indexing {len(ds)} products")

    img_dir = CATALOG_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    meta, embs = [], []
    t0 = time.time()
    B = 256
    for start in range(0, len(ds), B):
        batch = ds[start : start + B]
        images = batch["image"]
        embs.append(embed_images(images))
        for j, im in enumerate(images):
            pid = int(batch["id"][j])
            path = img_dir / f"{pid}.jpg"
            if not path.exists():
                im.convert("RGB").save(path, quality=88)
            name = batch["productDisplayName"][j] or ""
            meta.append({
                "id": pid,
                "name": name,
                "brand": brand_from_name(name),
                "gender": batch["gender"][j],
                "master": batch["masterCategory"][j],
                "sub": batch["subCategory"][j],
                "type": batch["articleType"][j],
                "colour": batch["baseColour"][j] or "Unknown",
                "season": batch["season"][j],
                "usage": batch["usage"][j],
            })
        done = start + len(images)
        rate = done / (time.time() - t0)
        print(f"  embedded {done}/{len(ds)}  ({rate:.0f} img/s)", flush=True)

    X = np.concatenate(embs).astype(np.float32)
    np.save(CATALOG_DIR / "embeddings.npy", X)
    (CATALOG_DIR / "meta.json").write_text(json.dumps(meta))

    print("Training PyTorch heads on CLIP embeddings ...")
    heads = {
        "type": train_head(X, [m["type"] for m in meta], "item type"),
        "colour": train_head(X, [m["colour"] for m in meta], "colour"),
    }
    torch.save(heads, CATALOG_DIR / "heads.pt")
    report = {k: {"top1": v["top1"], "top3": v["top3"], "classes": len(v["classes"]),
                  "n_test": v["n_test"]} for k, v in heads.items()}
    report["catalog_size"] = len(meta)
    (CATALOG_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print("Done. Results on held-out products:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
