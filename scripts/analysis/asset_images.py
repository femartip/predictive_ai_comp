"""What the assets (images, audio, video) look like inside the converter's item text: how many items carry them, and how.

    python scripts/analysis/asset_images.py --jsonl-dir <asset run folder> --out <dir> [--max-items 5000]

In the asset runs, an item's `question` keeps the text and lists its media under "multimedia_elements" ({"content_type":
"image/...", "location": <path or URL>}); each image, audio clip or video also appears in the text as a written
description, <image_description media_type="image/...">...</image_description> (likewise <audio_description>,
<video_description>). The media themselves are not in the text: a text-only annotator (the ADeLe v1 annotator) sees
only the description. Per benchmark, on a uniform sample of at most --max-items items (as token_lengths.py): share of
items with at least one description, descriptions per item, their length (characters) and share of the item's text,
media types, and how the images are located.
Writes <out>/asset_images.csv and prints it. No item text is written or printed.
"""

import argparse
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_lengths import items_of  # noqa: E402  (same item sample)

LOCATION = re.compile(r'"content_type":\s*"image/[^"]*"\s*,\s*"location":\s*"([^"]{0,300})"')
DESC = re.compile(r'<(image|audio|video)_description media_type="([a-z]+/[^"]+)"[^>]*>(.*?)(?:</\1_description>|$)', re.S)


def location_kind(loc):
    if loc.startswith(("http://", "https://")):
        return "url"
    if "assets/" in loc or re.search(r"[0-9a-f]{40,}", loc):
        return "asset file"
    return "other path"


def measure(args):
    path, max_items, seed = args
    df, n = items_of(path, max_items, seed)
    n_media, desc_chars, desc_share, types, kinds = [], [], [], Counter(), Counter()
    for q in df["question"]:
        q = q if isinstance(q, str) else str(q)
        d = DESC.findall(q)                                   # (kind, media type, description) per asset
        n_media.append(len(d))
        types.update(t for _, t, _ in d)
        kinds.update(location_kind(x) for x in LOCATION.findall(q))
        if d:
            chars = [len(x.strip()) for _, _, x in d]
            desc_chars += chars
            desc_share.append(sum(chars) / max(len(q), 1))
    n_media = np.array(n_media)
    return {"benchmark": Path(path).stem, "items": n, "sampled": len(n_media),
            "share_with_media": float((n_media > 0).mean()) if len(n_media) else 0.0,
            "media_per_item_mean": float(n_media.mean()) if len(n_media) else 0.0,
            "media_per_item_max": int(n_media.max()) if len(n_media) else 0,
            "desc_chars_median": int(np.median(desc_chars)) if desc_chars else 0,
            "desc_share_of_text": float(np.mean(desc_share)) if desc_share else 0.0,
            "media_types": ";".join(f"{k}:{v}" for k, v in types.most_common()),
            "locations": ";".join(f"{k}:{v}" for k, v in kinds.most_common())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = sorted(Path(a.jsonl_dir).glob("*.jsonl"), key=lambda f: -f.stat().st_size)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(measure, [(str(f), a.max_items, a.seed) for f in files]))
    t = pd.DataFrame(rows).sort_values("share_with_media", ascending=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "asset_images.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.drop(columns=["media_types"]).round(3).to_string(index=False))
    print(f"\n{len(t)} benchmarks, {int(t['items'].sum()):,} items; items with a media description (weighted by "
          f"benchmark size): {np.average(t['share_with_media'], weights=t['items']):.1%}")
    print(f"wrote {out / 'asset_images.csv'}")


if __name__ == "__main__":
    main()
