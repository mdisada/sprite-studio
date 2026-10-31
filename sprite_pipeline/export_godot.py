"""Export finished sprites as a game-ready sheet + Godot 4 resources.

The per-animation sheets the post stage writes are cropped to each clip's own
bounding box, which is right for previews and wrong for a game: switching from
idle to attack would jump the character around the screen. This export fixes
every animation of one character onto ONE cell size with ONE anchor - the feet
in the reference stance - so an AnimatedSprite2D can switch clips and the
character stays planted.

It also bakes in what the studio corrects at play time:

* per-animation size normalization (the video model sometimes renders one clip
  much larger in frame; the studio divides it out, so the export must too),
* the character's own size, so every hero exports at one consistent scale and
  their relative sizes in Godot match what you tuned,
* the playback window (a clip's action range) and any cut frames,
* per-animation speed, folded into the Godot animation's fps.

    python -m sprite_pipeline.export_godot --char ninja
    python -m sprite_pipeline.export_godot --project heroes --height 128

Writes sprites_out/_export/<char>/ plus <char>.zip beside it.
"""
import argparse
import json
import math
import os
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "sprites_out")
EXPORT_DIR = os.path.join(OUT_ROOT, "_export")
PAD = 2                      # breathing room around the widest frame


def _meta(cid, anim):
    p = os.path.join(OUT_ROOT, cid, anim, "meta.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _played(cid, anim, meta):
    """The frames that actually play: cuts are already gone from frames/,
    the action window is applied here (it is stored in that same space)."""
    d = os.path.join(OUT_ROOT, cid, anim, "frames")
    if not os.path.isdir(d):
        return []
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]
    r = meta.get("action_range")
    if r and 0 <= r[0] < r[1] <= len(fs) and r[1] - r[0] >= 2:
        fs = fs[r[0]:r[1]]
    return fs


def _foot_x(path, meta):
    """Centre of the feet in the first frame, in source pixels - the same
    horizontal anchor the studio's play area uses."""
    b = meta.get("crop_box")
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    sx = (b[2] - b[0]) / float(im.width)
    sy = (b[3] - b[1]) / float(im.height)
    gy = ((meta["ground_y"] - b[1]) / sy) if meta.get("ground_y") is not None else im.height
    band = max(4, int(round(6 / sy)))
    y0, y1 = max(0, int(round(gy - band))), min(im.height, int(round(gy)) + 1)
    cols = np.where((a[y0:y1] > 32).any(axis=0))[0]
    if not len(cols):
        return (b[0] + b[2]) / 2.0
    return b[0] + ((int(cols.min()) + int(cols.max())) / 2.0) * sx


def plan(cid, spec, height):
    """Measure every animation, then derive the shared cell and anchor."""
    idle = _meta(cid, "idle")
    if not idle or not idle.get("stand_h"):
        raise ValueError("%s has no finished idle to anchor the export on" % cid)
    out_scale = (float(height) / idle["stand_h"]) * (spec.get("sprite_scale") or 1)

    anims, lo_x, hi_x, lo_y, hi_y = [], 0.0, 0.0, 0.0, 0.0
    for name, aspec in (spec.get("animations") or {}).items():
        meta = _meta(cid, name)
        if not meta:
            continue
        files = _played(cid, name, meta)
        if not files:
            continue
        # explicit size wins, else normalise this clip against idle
        s = aspec.get("scale")
        if s is None:
            s = (max(0.4, min(2.5, idle["stand_h"] / meta["stand_h"]))
                 if meta.get("stand_h") else 1.0)
        b = meta["crop_box"]
        foot = _foot_x(files[0], meta)
        gy = meta.get("ground_y")
        gy = min(max(gy, b[1]), b[3]) if gy is not None else b[3]
        k = out_scale * s
        left, top = (b[0] - foot) * k, (b[1] - gy) * k
        right, bottom = (b[2] - foot) * k, (b[3] - gy) * k
        lo_x, hi_x = min(lo_x, left), max(hi_x, right)
        lo_y, hi_y = min(lo_y, top), max(hi_y, bottom)
        anims.append({
            "name": name, "files": files, "meta": meta,
            "scale": s, "left": left, "top": top,
            "w": right - left, "h": bottom - top,
            "fps": round((meta.get("fps") or 12) * (aspec.get("rate") or 1), 3),
            "loop": bool(aspec.get("loop")),
        })
    if not anims:
        raise ValueError("%s has no finished animations" % cid)
    cell_w = int(math.ceil(hi_x - lo_x)) + PAD * 2
    cell_h = int(math.ceil(hi_y - lo_y)) + PAD * 2
    return {
        "anims": anims, "out_scale": out_scale,
        "cell": (cell_w, cell_h),
        "anchor": (-lo_x + PAD, -lo_y + PAD),   # the feet, inside a cell
        "cols": max(len(a["files"]) for a in anims),
    }


def build_sheet(p):
    cw, ch = p["cell"]
    ax, ay = p["anchor"]
    sheet = Image.new("RGBA", (p["cols"] * cw, len(p["anims"]) * ch), (0, 0, 0, 0))
    for row, a in enumerate(p["anims"]):
        w, h = max(1, int(round(a["w"]))), max(1, int(round(a["h"])))
        x0 = int(round(ax + a["left"]))
        y0 = int(round(ay + a["top"]))
        for col, f in enumerate(a["files"]):
            im = Image.open(f).convert("RGBA").resize((w, h), Image.LANCZOS)
            sheet.paste(im, (col * cw + x0, row * ch + y0))
        a["row"] = row
    return sheet


def write_tres(p, cid, res_path):
    """Godot 4 SpriteFrames: one AtlasTexture per frame over the one sheet."""
    cw, ch = p["cell"]
    subs, blocks, anim_entries = 0, [], []
    for a in p["anims"]:
        frames = []
        for col in range(len(a["files"])):
            sid = "A%d_%d" % (a["row"], col)
            blocks.append('[sub_resource type="AtlasTexture" id="%s"]\n'
                          'atlas = ExtResource("1_sheet")\n'
                          'region = Rect2(%d, %d, %d, %d)\n'
                          % (sid, col * cw, a["row"] * ch, cw, ch))
            frames.append('{\n"duration": 1.0,\n"texture": SubResource("%s")\n}' % sid)
            subs += 1
        anim_entries.append('{\n"frames": [%s],\n"loop": %s,\n"name": &"%s",\n"speed": %s\n}'
                            % (", ".join(frames), "true" if a["loop"] else "false",
                               a["name"], a["fps"]))
    head = ('[gd_resource type="SpriteFrames" load_steps=%d format=3]\n\n'
            '[ext_resource type="Texture2D" path="%s/%s.png" id="1_sheet"]\n\n'
            % (subs + 2, res_path, cid))
    return head + "\n".join(blocks) + "\n[resource]\nanimations = [%s]\n" % ", ".join(anim_entries)


def write_tscn(p, cid, res_path, first):
    ax, ay = p["anchor"]
    name = "".join(w.capitalize() for w in cid.replace("-", "_").split("_"))
    return ('[gd_scene load_steps=2 format=3]\n\n'
            '[ext_resource type="SpriteFrames" path="%s/%s.tres" id="1_frames"]\n\n'
            '[node name="%s" type="AnimatedSprite2D"]\n'
            'texture_filter = 1\n'          # nearest: keep the pixels hard
            'sprite_frames = ExtResource("1_frames")\n'
            'animation = &"%s"\n'
            'autoplay = "%s"\n'
            'centered = false\n'
            # origin of the node = the character's feet, so position is where
            # they stand and every animation stays planted
            'offset = Vector2(%s, %s)\n'
            % (res_path, cid, name, first, first, -round(ax, 2), -round(ay, 2)))


README = """{title}

WHAT IS IN HERE
  {cid}.png    one sheet, {cols} columns x {rows} rows - one row per animation
  {cid}.tres   Godot 4 SpriteFrames (every frame is a region of that sheet)
  {cid}.tscn   an AnimatedSprite2D already wired to the .tres
  {cid}.json   the same layout as plain data, for any other engine

USING IT IN GODOT 4
  1. Copy this whole folder into your project so it lands at
     {res_path}/
     (the .tres refers to the sheet by that path - if you put it somewhere
     else, Godot will ask you to locate the texture once)
  2. Drag {cid}.tscn into your scene, or add an AnimatedSprite2D and set its
     Sprite Frames to {cid}.tres.
  3. Play an animation by name: $ {node}.play("walk")

ANCHOR
  Every animation shares one cell of {cw} x {ch} px and one anchor: the
  character's feet in their standing pose. The scene sets centered = false and
  offset = ({ox}, {oy}), which puts the node's origin exactly at those feet -
  so the node's position is where the character stands, and switching
  animations never makes them jump.

SIZE
  Exported at {height} px standing height x this character's own size
  ({sprite_scale}), so every character exported from the same project keeps
  the relative sizes you tuned in the studio. Leave the node scale at 1.

PIXELS
  The scene sets texture_filter = Nearest so the art stays crisp. If you use
  the .tres directly instead, set the same on your node (or set Filter to
  Nearest on the texture import).

ANIMATIONS
{anim_lines}
"""


def export_character(cid, spec, height=128, res_dir="sprites", out_root=EXPORT_DIR):
    p = plan(cid, spec, height)
    d = os.path.join(out_root, cid)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    res_path = "res://%s/%s" % (res_dir.strip("/"), cid)

    build_sheet(p).save(os.path.join(d, cid + ".png"), optimize=True)
    first = "idle" if any(a["name"] == "idle" for a in p["anims"]) else p["anims"][0]["name"]
    with open(os.path.join(d, cid + ".tres"), "w", encoding="utf-8") as fh:
        fh.write(write_tres(p, cid, res_path))
    with open(os.path.join(d, cid + ".tscn"), "w", encoding="utf-8") as fh:
        fh.write(write_tscn(p, cid, res_path, first))

    cw, ch = p["cell"]
    ax, ay = p["anchor"]
    data = {
        "character": cid,
        "display_name": spec.get("display_name") or cid,
        "sheet": cid + ".png",
        "cell": [cw, ch],
        "anchor": [round(ax, 2), round(ay, 2)],
        "columns": p["cols"],
        "stand_height_px": height,
        "sprite_scale": spec.get("sprite_scale") or 1,
        "animations": {a["name"]: {"row": a["row"], "frames": len(a["files"]),
                                   "fps": a["fps"], "loop": a["loop"],
                                   "hold_last": bool((spec["animations"][a["name"]]).get("hold_last")),
                                   "move_speed": (spec["animations"][a["name"]]).get("move_speed")}
                       for a in p["anims"]},
    }
    with open(os.path.join(d, cid + ".json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    lines = "\n".join("  %-12s %2d frames  %5.1f fps  %s"
                      % (a["name"], len(a["files"]), a["fps"],
                         "loops" if a["loop"] else "plays once")
                      for a in p["anims"])
    node = "".join(w.capitalize() for w in cid.replace("-", "_").split("_"))
    with open(os.path.join(d, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(README.format(
            title=(spec.get("display_name") or cid) + " - sprite export",
            cid=cid, cols=p["cols"], rows=len(p["anims"]), res_path=res_path,
            node=node, cw=cw, ch=ch, ox=-round(ax, 2), oy=-round(ay, 2),
            height=height, sprite_scale=spec.get("sprite_scale") or 1,
            anim_lines=lines))

    zip_path = os.path.join(out_root, cid + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(d)):
            z.write(os.path.join(d, f), os.path.join(cid, f))
    return {
        "id": cid, "dir": d, "zip": zip_path,
        "cell": [cw, ch], "anchor": [round(ax, 2), round(ay, 2)],
        "animations": len(p["anims"]),
        "frames": sum(len(a["files"]) for a in p["anims"]),
        "bytes": os.path.getsize(zip_path),
        "files": sorted(os.listdir(d)),
    }


def export_many(ids, chars, height=128, res_dir="sprites", project=None):
    """Export several characters and bundle them into one zip you can drop
    straight into a Godot project."""
    results, errors = [], []
    for cid in ids:
        try:
            results.append(export_character(cid, chars[cid], height, res_dir))
        except (ValueError, KeyError) as exc:
            errors.append("%s: %s" % (cid, exc))
    bundle = None
    if len(results) > 1:
        bundle = os.path.join(EXPORT_DIR, (project or "sprites") + ".zip")
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            for r in results:
                for f in sorted(os.listdir(r["dir"])):
                    z.write(os.path.join(r["dir"], f), os.path.join(r["id"], f))
    return {"exports": results, "errors": errors, "bundle": bundle,
            "bundle_bytes": os.path.getsize(bundle) if bundle else 0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--char", default="")
    ap.add_argument("--project", default="")
    ap.add_argument("--height", type=int, default=128,
                    help="standing height in pixels before the character's own size")
    ap.add_argument("--res-dir", default="sprites",
                    help="folder inside your Godot project (res://<this>/<char>/)")
    args = ap.parse_args(argv)
    with open(os.path.join(ROOT, "characters.json"), encoding="utf-8") as fh:
        chars = json.load(fh)
    if args.char:
        want = [args.char]
    elif args.project:
        want = [c for c, s in chars.items() if s.get("project") == args.project
                and s.get("sprite_type") != "vfx"]
    else:
        sys.exit("pass --char <id> or --project <id>")
    for cid in want:
        try:
            r = export_character(cid, chars[cid], args.height, args.res_dir)
            print("[export] %-14s %2d anims %3d frames  cell %dx%d  %.0f KB  -> %s"
                  % (r["id"], r["animations"], r["frames"], r["cell"][0], r["cell"][1],
                     r["bytes"] / 1024., r["zip"]))
        except ValueError as e:
            print("[skip] %s" % e)


if __name__ == "__main__":
    main()
