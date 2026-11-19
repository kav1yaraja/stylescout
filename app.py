"""StyleScout: upload a clothing photo, find out what it is and where to get it.

Run:  python app.py   then open http://127.0.0.1:7860
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict

import gradio as gr
import numpy as np
import torch

from stylescout_core import (
    CATALOG_DIR, FASHION_PROMPTS, LinearHead, NON_FASHION_PROMPTS, STYLE_PROMPTS,
    STYLE_TAGS, embed_images, embed_texts, load_clip,
)

CONF_THRESHOLD = 0.40
TOP_K = 12

# ---------- load everything once ----------
if not (CATALOG_DIR / "embeddings.npy").exists():
    raise SystemExit("No catalog yet. Run:  python build_catalog.py")

load_clip()
CAT_X = np.load(CATALOG_DIR / "embeddings.npy")
CAT_META = json.loads((CATALOG_DIR / "meta.json").read_text())
METRICS = json.loads((CATALOG_DIR / "metrics.json").read_text())
_raw_heads = torch.load(CATALOG_DIR / "heads.pt", weights_only=False)
HEADS = {}
for key, h in _raw_heads.items():
    m = LinearHead(CAT_X.shape[1], len(h["classes"]))
    m.load_state_dict(h["state_dict"])
    m.eval()
    HEADS[key] = (m, h["classes"])

STYLE_T = embed_texts(STYLE_PROMPTS)
GATE_T = embed_texts(FASHION_PROMPTS + NON_FASHION_PROMPTS)
N_FASHION = len(FASHION_PROMPTS)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def classify(key, emb):
    model, classes = HEADS[key]
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(emb[None]) * 10), 1)[0].numpy()
    order = p.argsort()[::-1][:3]
    return [(classes[i], float(p[i])) for i in order]


# ---------- optional live eBay results ----------
def ebay_search_by_image(image) -> list[dict]:
    cid, secret = os.getenv("EBAY_CLIENT_ID"), os.getenv("EBAY_CLIENT_SECRET")
    if not (cid and secret):
        return []
    try:
        auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
        req = urllib.request.Request(
            "https://api.ebay.com/identity/v1/oauth2/token",
            data=urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            }).encode(),
            headers={"Authorization": f"Basic {auth}",
                     "Content-Type": "application/x-www-form-urlencoded"})
        token = json.load(urllib.request.urlopen(req, timeout=10))["access_token"]
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        req = urllib.request.Request(
            "https://api.ebay.com/buy/browse/v1/item_summary/search_by_image?limit=8",
            data=json.dumps({"image": base64.b64encode(buf.getvalue()).decode()}).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                     "X-EBAY-C-MARKETPLACE-ID": os.getenv("EBAY_MARKETPLACE", "EBAY_US")})
        items = json.load(urllib.request.urlopen(req, timeout=15)).get("itemSummaries", [])
        return [{"title": it.get("title"), "url": it.get("itemWebUrl"),
                 "price": f'{it.get("price", {}).get("value", "")} {it.get("price", {}).get("currency", "")}',
                 "condition": it.get("condition", "")} for it in items]
    except Exception as e:  # keep the app working if eBay fails
        print("eBay lookup failed:", e)
        return []


def shop_links(query: str) -> str:
    q = urllib.parse.quote_plus(query)
    links = [
        ("Google Shopping (new)", f"https://www.google.com/search?tbm=shop&q={q}"),
        ("eBay (new + used)", f"https://www.ebay.com/sch/i.html?_nkw={q}"),
        ("eBay (used only)", f"https://www.ebay.com/sch/i.html?_nkw={q}&LH_ItemCondition=3000"),
        ("Depop", f"https://www.depop.com/search/?q={q}"),
        ("Poshmark", f"https://poshmark.com/search?query={q}"),
        ("ThredUp", f"https://www.thredup.com/searches?search_text={q}"),
        ("Vinted", f"https://www.vinted.com/catalog?search_text={q}"),
    ]
    return " · ".join(f"[{n}]({u})" for n, u in links)


# ---------- main pipeline ----------
def analyze(image):
    if image is None:
        return "Upload a photo to get started.", [], ""

    emb = embed_images([image])[0]

    # 1. Is this even fashion?
    gate = softmax(100 * GATE_T @ emb)
    fashion_p = float(gate[:N_FASHION].sum())
    if fashion_p < 0.5:
        best = (FASHION_PROMPTS + NON_FASHION_PROMPTS)[int(gate.argmax())]
        return (f"### That doesn't look like clothing\nIt looks more like *{best.replace('a photo of ', '')}* "
                f"({1 - fashion_p:.0%} sure). Try a photo of a single clothing item, shoe or accessory."), [], ""

    # 2. What it is (trained PyTorch heads)
    types = classify("type", emb)
    colours = classify("colour", emb)
    t_name, t_p = types[0]
    c_name, c_p = colours[0]
    type_ok, colour_ok = t_p >= CONF_THRESHOLD, c_p >= CONF_THRESHOLD

    # 3. Similar items from the catalog
    sims = CAT_X @ emb
    top = sims.argsort()[::-1][:TOP_K]
    gallery = [(str(CATALOG_DIR / "images" / f"{CAT_META[i]['id']}.jpg"),
                f"{CAT_META[i]['name']} · {sims[i]:.0%} match") for i in top]

    # 4. Brand guess: similarity-weighted vote among the closest products
    votes = defaultdict(float)
    for i in top:
        votes[CAT_META[i]["brand"]] += float(sims[i])
    brand, score = max(votes.items(), key=lambda kv: kv[1])
    share = score / sum(votes.values())
    brand_count = sum(CAT_META[i]["brand"] == brand for i in top)
    brand_ok = share >= 0.35 and brand_count >= 3

    # 5. Style / era tags (zero-shot CLIP)
    style_p = softmax(100 * STYLE_T @ emb)
    styles = [(STYLE_TAGS[i], float(style_p[i])) for i in style_p.argsort()[::-1][:3]]

    def fmt(name, p, ok):
        return f"**{name}** ({p:.0%})" if ok else f"not sure (best guess: {name}, {p:.0%})"

    alts = ", ".join(f"{n} {p:.0%}" for n, p in types[1:])
    md = [
        "### What it is",
        f"- **Item:** {fmt(t_name, t_p, type_ok)} · also possible: {alts}",
        f"- **Colour:** {fmt(c_name, c_p, colour_ok)}",
        "### Brand (guess)",
        (f"- Looks most like **{brand}** ({brand_count} of the {TOP_K} closest catalog products). "
         "This is a guess based on similar items, not a logo reading."
         if brand_ok else
         f"- Not sure. The closest products come from mixed brands (top: {brand})."),
        "### Style / era",
        "- " + " · ".join(f"**{s}** {p:.0%}" for s, p in styles),
    ]

    # 6. Where to get it
    parts = [c_name if colour_ok else "", brand if brand_ok else "", t_name if type_ok else types[0][0]]
    query = " ".join(p for p in parts if p).strip()
    closest_name = CAT_META[top[0]]["name"]
    where = [
        "### Where to get it",
        f"**Search for \"{query}\":** {shop_links(query)}",
        "",
        f"**Closest catalog product (\"{closest_name}\"):** {shop_links(closest_name)}",
    ]
    ebay = ebay_search_by_image(image)
    if ebay:
        where += ["", "**Live eBay listings (search by image):**"]
        where += [f"- [{e['title']}]({e['url']}) · {e['price']} · {e['condition']}" for e in ebay]
    else:
        where += ["", "_Live eBay listings are off. Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET to turn them on._"]

    return "\n".join(md), gallery, "\n".join(where)


about = (f"Catalog: **{METRICS['catalog_size']:,}** products · classifier accuracy on held-out products: "
         f"item type **{METRICS['type']['top1']:.0%}** (top-3 {METRICS['type']['top3']:.0%}), "
         f"colour **{METRICS['colour']['top1']:.0%}**")

with gr.Blocks(title="StyleScout", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# StyleScout\nUpload a photo of one clothing item, shoe or accessory. "
                "StyleScout tells you what it is, guesses the brand and style, finds similar "
                "products and shows where to buy it new or secondhand.")
    gr.Markdown(about)
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(type="pil", label="Your photo", height=360)
            btn = gr.Button("Scout it", variant="primary")
            gr.Markdown("Tip: crop to a single item for the best matches.")
        with gr.Column(scale=1):
            result = gr.Markdown()
    gallery = gr.Gallery(label="Similar items from the catalog", columns=6, height="auto")
    where = gr.Markdown()
    btn.click(analyze, img, [result, gallery, where])
    img.upload(analyze, img, [result, gallery, where])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", 7860)),
                allowed_paths=[str(CATALOG_DIR)])
