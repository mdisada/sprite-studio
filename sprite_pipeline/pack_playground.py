"""Bake a project's finished sprites into ONE self-contained HTML playground.

The studio's play area (viewer.html) needs the server: it fetches /api/state
and every frame as its own PNG. This packs the same sprites into a single file
that runs from a URL with no server at all - one atlas image per character
(inlined as a data URI), plus the small amount of spec/meta the play loop
actually reads. Nothing here can edit the sprites; it is a toy, not a tool.

    python -m sprite_pipeline.pack_playground --project heroes --out playground.html

Size is the whole design constraint (a published page must stay under 16MB,
and base64 costs another third on top of the bytes), hence: frames downscaled
by --factor (the art is 32-bit pixel art living inside a 384px frame, so /3
lands on its own grid and looks sharper, not softer), one shared palette per
character (also stops colours flickering between frames), hard alpha.
"""
import argparse
import base64
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "sprites_out")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "playground.html")

# meta/spec fields the play loop reads - everything else stays home
META_KEYS = ("fps", "rate", "ground_y", "stand_h", "crop_box", "source_size",
             "action_range", "range_manual", "collapsed", "style")
SPEC_KEYS = ("loop", "use_action", "hold_last", "interrupt", "rate",
             "move_speed", "scale")


def frame_paths(cid, anim):
    d = os.path.join(OUT_ROOT, cid, anim, "frames")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(".png")]


def foot_x(frame0, meta):
    """Horizontal anchor: centre of the feet in the first frame, in SOURCE px.

    Same rule the play area uses - without it, clips cropped differently put
    the same body at different screen positions. Computed here on the
    full-resolution frame so the packed copy never has to be measured.
    """
    b = meta.get("crop_box")
    if not b:
        return None
    im = Image.open(frame0).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    sx = (b[2] - b[0]) / float(im.width)
    sy = (b[3] - b[1]) / float(im.height)
    gy = ((meta["ground_y"] - b[1]) / sy) if meta.get("ground_y") is not None \
        else im.height
    band = max(4, int(round(6 / sy)))
    y0 = max(0, int(round(gy - band)))
    y1 = min(im.height, int(round(gy)) + 1)
    cols = np.where((a[y0:y1] > 32).any(axis=0))[0]
    if not len(cols):
        return (b[0] + b[2]) / 2.0
    return b[0] + ((int(cols.min()) + int(cols.max())) / 2.0) * sx


def quantize(atlas, colors):
    """PNG8 with one palette for the whole character and hard edges.

    Beats WebP by 2x on this art (flat colour, lots of empty space) and keeps
    every frame on the same palette, so nothing shifts hue between frames.
    """
    arr = np.asarray(atlas).copy()
    arr[:, :, 3] = np.where(arr[:, :, 3] >= 110, 255, 0)
    hard = Image.fromarray(arr)
    q = hard.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
    idx = np.asarray(q).copy()
    idx[np.asarray(hard)[:, :, 3] == 0] = colors        # last slot = clear
    out = Image.fromarray(idx, "P")
    out.putpalette(q.getpalette()[:colors * 3] + [0, 0, 0])
    out.info["transparency"] = colors
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def pack_character(cid, spec, factor, colors):
    """One row band per animation: frame i sits at (i*fw, row_y)."""
    anims, rows, total_h, width = {}, [], 0, 0
    loaded = []
    for name, aspec in (spec.get("animations") or {}).items():
        paths = frame_paths(cid, name)
        mp = os.path.join(OUT_ROOT, cid, name, "meta.json")
        if not paths or not os.path.isfile(mp):
            continue
        with open(mp) as fh:
            meta = json.load(fh)
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGBA")
            if factor > 1:
                im = im.resize((max(1, im.width // factor),
                                max(1, im.height // factor)), Image.LANCZOS)
            imgs.append(im)
        fw = max(i.width for i in imgs)
        fh_ = max(i.height for i in imgs)
        entry = {k: meta.get(k) for k in META_KEYS}
        entry.update({k: aspec[k] for k in SPEC_KEYS if aspec.get(k) is not None})
        entry["n"] = len(imgs)
        entry["fw"] = fw
        entry["fh"] = fh_
        entry["y"] = total_h
        entry["footx"] = foot_x(paths[0], meta)
        anims[name] = entry
        loaded.append((name, imgs, fw, fh_, total_h))
        rows.append(name)
        total_h += fh_
        width = max(width, fw * len(imgs))
    if "idle" not in anims:
        return None
    atlas = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    for name, imgs, fw, fh_, y in loaded:
        for i, im in enumerate(imgs):
            atlas.paste(im, (i * fw, y))
    png = quantize(atlas, colors)
    return {
        "id": cid,
        "name": spec.get("display_name") or cid,
        "scale": spec.get("sprite_scale") or 1,
        "w": width, "h": total_h,
        "anims": anims,
        "atlas": "data:image/png;base64," + base64.b64encode(png).decode(),
        "_bytes": len(png),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="heroes")
    ap.add_argument("--chars", default="", help="comma-separated ids (overrides --project)")
    ap.add_argument("--factor", type=int, default=3, help="downscale divisor")
    ap.add_argument("--colors", type=int, default=63, help="palette size per character")
    ap.add_argument("--out", default=os.path.join(ROOT, "playground.html"))
    args = ap.parse_args(argv)

    with open(os.path.join(ROOT, "characters.json"), encoding="utf-8") as fh:
        chars = json.load(fh)
    if args.chars:
        want = [c.strip() for c in args.chars.split(",") if c.strip()]
    else:
        want = [c for c, s in chars.items()
                if s.get("project") == args.project and s.get("sprite_type") != "vfx"]

    packed, total = [], 0
    for cid in want:
        p = pack_character(cid, chars[cid], args.factor, args.colors)
        if not p:
            print("[skip] %s - no finished idle" % cid)
            continue
        total += p.pop("_bytes")
        packed.append(p)
        print("[pack] %-14s %2d anims  %5d x %-5d %6.0f KB"
              % (p["id"], len(p["anims"]), p["w"], p["h"],
                 len(p["atlas"]) * 0.75 / 1024))
    if not packed:
        sys.exit("nothing to pack")

    with open(TEMPLATE, encoding="utf-8") as fh:
        html = fh.read()
    data = json.dumps({"chars": packed}, separators=(",", ":"))
    if "/*__DATA__*/" not in html:
        sys.exit("template is missing the /*__DATA__*/ marker")
    html = html.replace("/*__DATA__*/", data)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("\n%d characters  ->  %s\n  atlases %.1f MB, page %.1f MB"
          % (len(packed), args.out, total / 1e6, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
