# StyleScout

Upload a photo of a clothing item and StyleScout tells you:

- **What it is**: item type and colour, with confidence scores. It says "not sure" instead of guessing when confidence is low.
- **Brand (guess)**: a similarity-weighted vote among the closest catalog products.
- **Style / era**: zero-shot tags such as Y2K, streetwear or vintage workwear.
- **Similar items**: the 12 closest products in the catalog.
- **Where to get it**: search links for Google Shopping, eBay (new and used), Depop, Poshmark, ThredUp and Vinted. Live eBay search-by-image results appear if you add API keys.

It also rejects photos that aren't fashion.

## How it works

1. **CLIP** (`openai/clip-vit-base-patch32`, PyTorch) turns every image into a 512-d embedding.
2. `build_catalog.py` embeds products from the [Fashion Product Images (small)](https://huggingface.co/datasets/ashraq/fashion-product-images-small) dataset and trains two small **PyTorch** linear heads (item type, colour) on those embeddings. It reports accuracy on held-out products.
3. `app.py` (Gradio) embeds your photo, runs the heads, does a cosine-similarity search over the catalog, and runs zero-shot CLIP prompts for style tags and the "is this fashion?" check.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python build_catalog.py --limit 15000   # use --limit 0 for all ~42k products
.venv/bin/python app.py                            # open http://127.0.0.1:7860
```

Optional live eBay results (free keys from developer.ebay.com):

```bash
EBAY_CLIENT_ID=... EBAY_CLIENT_SECRET=... .venv/bin/python app.py
```

## Limitations

- The catalog is clean studio product shots from an Indian retailer, so brand guesses lean toward those brands and real-world photos match less well. Cropping to one item helps.
- The dataset page does not state a license, so check it before hosting the app publicly.
