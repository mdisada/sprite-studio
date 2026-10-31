"""Sprite generation pipeline orchestrator.

Stages (grouped so each model family loads only once per run):
  ref   Krea-2 Turbo t2i -> character reference sheet on green screen
  anim  MiniMax H3 ref-to-video -> animation clips (frames saved as PNGs)
  post  chroma key + pixelize + sprite sheet + GIF/APNG previews

Usage examples:
  python -m sprite_pipeline.pipeline --char stormshield --mode test --stage ref
  python -m sprite_pipeline.pipeline --char stormshield --mode test --anim idle
  python -m sprite_pipeline.pipeline --char stormshield --mode final --anim idle cast walk
  python -m sprite_pipeline.pipeline --char stormshield --stage post --anim idle
"""
import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import time

from .comfy_client import ComfyClient, ComfyError
from . import characters as templates
from . import graphs, postprocess, store

def _comfy_alive(host, timeout=1.5):
    try:
        import urllib.request
        urllib.request.urlopen("http://%s/system_stats" % host, timeout=timeout)
        return True
    except Exception:
        return False


def resolve_comfy_host(preferred=None):
    """Find ComfyUI even when it isn't on 8188.

    ComfyUI picks the next free port when 8188 is still held (a previous
    instance shutting down, a second copy running), so a hardcoded host makes
    a perfectly healthy server look 'not reachable'. An explicit COMFY_HOST
    env var always wins; otherwise probe the ports it auto-increments through.
    """
    env = os.environ.get("COMFY_HOST")
    if env:
        return env
    for host in ([preferred] if preferred else []) + \
            ["127.0.0.1:%d" % p for p in range(8188, 8200)]:
        if host and _comfy_alive(host):
            return host
    return preferred or "127.0.0.1:8188"


COMFY_HOST = resolve_comfy_host()
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", r"F:\Documents\ComfyUI\output")
PROJECT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "sprites_out")

# test = fast feedback (small, short, turbo). final = quality settings.
MODES = {
    "test": {
        "ref_size": 512,
        "vid_size": 384,          # multiple of 32
        "seconds_cap": 1.0,       # -> 22 frames
        "turbo_lora": True,
        "video_steps": None,      # default for lora choice (4)
        "sprite_h": 64,
        "colors": 32,
        "ref_margin": 0.18,       # headroom so capes/effects don't clip
        "timeout": 1800,
    },
    "final": {
        "ref_size": 1024,
        "vid_size": 640,          # 0.41MP, matches the proven 0.4MP from the user's H3 workflow
        "seconds_cap": None,      # character durations (>=5.2s = 124-frame trained floor)
        "turbo_lora": True,
        "video_steps": 8,         # turbo at 6-8 steps restores micro-detail
        "sprite_h": 128,          # 640/128 = 5x, matches the art's pixel-cluster size
        "colors": 64,
        "ref_margin": 0.18,       # headroom so capes/effects don't clip
        "timeout": 7200,
    },
}


def run_dir(char):
    d = os.path.join(PROJECT_OUT, char)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------- takes
# Every render of an animation is a TAKE. The active take lives directly in
# <char>/<anim>/ (so the viewer, atlas export and post all work unchanged);
# earlier ones are moved aside into <char>/<anim>/takes/<id>/. Switching is
# a move, not a copy, so keeping history costs no extra disk beyond the
# frames themselves - and those are what make free re-processing possible.
def _seed(seed):
    """0/None means "fresh": a pinned seed reproduces a render, an unpinned one
    must NOT silently reuse the old noise or re-rendering returns the same clip."""
    return int(seed) if seed else random.randint(1000, 900000)


TAKE_PARTS = ("raw_frames", "frames", "sheet.png", "meta.json", "atlas.json",
              "preview.webp", "preview.gif", "preview_transparent.png",
              "render.json")


def takes_dir(char, anim):
    return os.path.join(run_dir(char), anim, "takes")


def _move_take(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in TAKE_PARTS:
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.move(s, os.path.join(dst, name))
    for clip in glob.glob(os.path.join(src, "*.mp4")):
        shutil.move(clip, os.path.join(dst, os.path.basename(clip)))


def archive_take(char, anim):
    """Move the current take into takes/ so a new render can't destroy it."""
    live = os.path.join(run_dir(char), anim)
    has = (glob.glob(os.path.join(live, "raw_frames", "*.png"))
           or os.path.isfile(os.path.join(live, "sheet.png")))
    if not has:
        return None
    take_id = time.strftime("%Y%m%d-%H%M%S")
    _move_take(live, os.path.join(takes_dir(char, anim), take_id))
    print("[takes] archived %s/%s as take %s" % (char, anim, take_id))
    return take_id


def list_takes(char, anim):
    """Every take of one animation, newest first; the live one is 'active'."""
    out = []
    live = os.path.join(run_dir(char), anim)
    if os.path.isfile(os.path.join(live, "sheet.png")) or \
            glob.glob(os.path.join(live, "raw_frames", "*.png")):
        out.append(_take_info(live, "active", True))
    d = takes_dir(char, anim)
    if os.path.isdir(d):
        for tid in sorted(os.listdir(d), reverse=True):
            p = os.path.join(d, tid)
            if os.path.isdir(p):
                out.append(_take_info(p, tid, False))
    return out


def _take_info(path, take_id, active):
    info = {"id": take_id, "active": active,
            "raw": len(glob.glob(os.path.join(path, "raw_frames", "*.png"))),
            "frames": None, "fps": None, "when": None, "seed": None}
    try:
        with open(os.path.join(path, "render.json")) as fh:
            info["seed"] = json.load(fh).get("seed")
    except (OSError, ValueError):
        pass
    try:
        info["when"] = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(
                os.path.join(path, "meta.json"))))
    except OSError:
        pass
    try:
        with open(os.path.join(path, "meta.json")) as fh:
            m = json.load(fh)
        info["frames"] = m.get("frames")
        info["fps"] = m.get("fps")
        src = m.get("source_size") or [0, 0]
        info["quality"] = "final" if min(src) >= 512 else "test"
    except (OSError, ValueError):
        pass
    return info


def activate_take(char, anim, take_id):
    """Make an archived take the live one, keeping the current as a take."""
    src = os.path.join(takes_dir(char, anim), take_id)
    if not os.path.isdir(src):
        raise ValueError("no such take: %s" % take_id)
    archive_take(char, anim)                     # current one steps aside
    _move_take(src, os.path.join(run_dir(char), anim))
    try:
        os.rmdir(src)
    except OSError:
        pass
    print("[takes] %s/%s now using take %s" % (char, anim, take_id))


def delete_take(char, anim, take_id):
    p = os.path.join(takes_dir(char, anim), take_id)
    if not os.path.isdir(p):
        raise ValueError("no such take: %s" % take_id)
    shutil.rmtree(p)


def spec_cfg(spec, cfg):
    """Per-sprite overrides on the mode config: a pixel sprite's density
    preset ("pixel_bits") fixes sprite height and palette size so every
    sprite in a project shares one consistent pixel-square size."""
    pr = templates.PIXEL_PRESETS.get(str((spec or {}).get("pixel_bits") or ""))
    if pr and (spec or {}).get("style", "pixel") == "pixel":
        cfg = dict(cfg)
        cfg["sprite_h"] = pr["sprite_h"]
        cfg["colors"] = pr["colors"]
    return cfg


def ref_paths(char):
    d = run_dir(char)
    return os.path.join(d, "ref.png"), os.path.join(d, "ref_sprite.png")


def anim_canvas(base_size, canvas="square"):
    """Per-animation canvas: same pixel area as base_size^2, ratio per shape.
    'portrait' (2:3) gives jump-style animations headroom; 'wide' (3:2) suits
    dashes and ground-slides. Dimensions snap to multiples of 32."""
    if canvas == "square":
        return base_size, base_size
    ratio = (2, 3) if canvas == "portrait" else (3, 2)
    area = float(base_size * base_size)
    w = int(round(math.sqrt(area * ratio[0] / ratio[1]) / 32)) * 32
    h = int(round(w * ratio[1] / float(ratio[0]) / 32)) * 32
    return max(32, w), max(32, h)


# ------------------------------------------------------------------ stage: ref
# Krea-2 Turbo is conditioned by Qwen3-VL with a 512-token budget (~475 for
# the prompt, roughly 350 words) and ComfyUI does NOT truncate: past the
# budget the image silently comes out black or as noise (ComfyUI #14782).
KREA_MAX_WORDS = 300


def check_ref_prompt(prompt):
    words = len(prompt.split())
    if words > KREA_MAX_WORDS:
        raise SystemExit("reference prompt is %d words; Krea-2's token budget "
                         "is ~350 words and ComfyUI fails silently past it - "
                         "shorten it below %d words" % (words, KREA_MAX_WORDS))
    return words


def stage_ref(client, char, spec, cfg, seed):
    cfg = spec_cfg(spec, cfg)
    check_ref_prompt(spec["ref_prompt"])
    ref_png, ref_sprite = ref_paths(char)
    seed = _seed(seed)
    print("[ref] generating reference for %s (%dpx, seed %d)"
          % (spec.get("display_name", char), cfg["ref_size"], seed))
    g = graphs.krea_t2i(spec["ref_prompt"], width=cfg["ref_size"],
                        height=cfg["ref_size"], seed=seed,
                        filename_prefix="sprites/%s/ref" % char)
    entry = client.run(g, label="krea ref", timeout=cfg["timeout"])
    files = client.output_files(entry, kind="images")
    if not files:
        raise ComfyError("krea job produced no images")
    path = client.fetch(files[0], run_dir(char), dest_name="ref.png")
    print("[ref] saved %s" % path)
    postprocess.process_ref_image(path, ref_sprite, sprite_h=cfg["sprite_h"] * 2,
                                  colors=cfg["colors"],
                                  style=spec.get("style", "pixel"),
                                  outline=spec.get("outline"))
    print("[ref] pixelized preview %s" % ref_sprite)
    return path


# ---------------------------------------------------------------- stage: views
def _harvest_pose(char, vname, source):
    """Build a pose reference by copying a real rendered frame.

    A pose generated from scratch sits wherever the image model put it, so
    using one as an END reference makes the character slide across the frame
    to reach it. A frame lifted from an existing clip already lives in the
    same coordinate space as the animation, so it can be used as a start or
    end pose without any drift. Uses raw (uncropped) frames so framing is
    preserved exactly.
    """
    anim = source["from_anim"]
    frames = sorted(glob.glob(os.path.join(run_dir(char), anim, "raw_frames",
                                           "*.png")))
    if not frames:
        raise SystemExit("cannot harvest pose '%s': %s/%s has no rendered "
                         "frames" % (vname, char, anim))
    idx = source.get("frame", -1)
    idx = len(frames) + idx if idx < 0 else min(idx, len(frames) - 1)
    dest = os.path.join(run_dir(char), "ref_%s.png" % vname)
    shutil.copy2(frames[idx], dest)
    print("[views] %s <- %s frame %d of %d (harvested, no GPU)"
          % (vname, anim, idx, len(frames)))
    return dest


def stage_views(client, char, spec, cfg, seed):
    """Build extra reference poses: either harvested from a rendered clip
    (spatially consistent - safe for end poses) or generated from the base
    reference with Qwen Image Edit (safe for start poses)."""
    views = spec.get("views") or {}
    seed = _seed(seed)
    if not views:
        print("[views] no views defined for %s" % char)
        return
    for vname, v in list(views.items()):
        if v.get("from_anim"):
            dest = os.path.join(run_dir(char), "ref_%s.png" % vname)
            if os.path.exists(dest) and not v.get("regenerate"):
                print("[views] %s already exists - skipping" % vname)
                continue
            _harvest_pose(char, vname, v)
    views = {k: v for k, v in views.items() if not v.get("from_anim")}
    if not views:
        return
    ref_png = ref_paths(char)[0]
    if not os.path.exists(ref_png):
        raise SystemExit("reference image missing (%s) - run --stage ref first" % ref_png)
    uploaded = client.upload_image(ref_png, name="%s_ref.png" % char)
    for vname, v in views.items():
        dest = os.path.join(run_dir(char), "ref_%s.png" % vname)
        if os.path.exists(dest) and not v.get("regenerate"):
            print("[views] %s/%s already exists - skipping" % (char, vname))
            continue
        print("[views] %s/%s via qwen edit" % (char, vname))
        g = graphs.qwen_edit(v["edit_prompt"], uploaded, seed=seed,
                             filename_prefix="sprites/%s/view_%s" % (char, vname))
        entry = client.run(g, label="view %s" % vname, timeout=cfg["timeout"])
        files = [f for f in client.output_files(entry, kind="images")
                 if f["filename"].lower().endswith(".png")]
        if not files:
            raise ComfyError("view edit for %s produced no image" % vname)
        path = client.fetch(files[0], run_dir(char), dest_name="ref_%s.png" % vname)
        print("[views] saved %s" % path)


_IDENTITY_SENTENCE = re.compile(r"^<Picture 1> defines[^.]*\.\s*")


# --- H3 official six-section Ref2VA brief -----------------------------------
# MiniMax's own prompt-writing guide (huggingface.co/MiniMaxAI/MiniMax-H3,
# docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) says H3 was trained on this
# labelled brief and that free text underperforms it. Rules baked in here:
# references defined as subjects (fully_preserved) AND named as the first-
# frame anchor; every exclusion phrased as the positive state it wants
# (there is no negative prompt and no CFG - naming a thing attracts it);
# the camera lock restated inside the shot in H3's own vocabulary; a
# timeline whose beats fit the clip length (H3 reads durations literally -
# one beat stretched over 5s plays as slow motion); audio fields filled so
# the model does not invent a soundtrack that competes with the visuals.
# "2D-animated" is official H3 style vocabulary; "pixel art" is not, so the
# rendering is described physically and anchored to <Picture 1>
_STYLE_LINE = {
    "pixel": "2D-animated game-sprite animation in the pixel-art style of "
             "<Picture 1>.",
    "painted": "2D-animated hand-painted game-sprite animation in the style "
               "of <Picture 1>.",
}


def h3_brief(spec, anim, aspec, refs_info, seconds):
    """Compose the six-section brief for one clip.

    refs_info: (has_end, has_identity) as returned by _anim_refs; pictures
    are numbered in upload order: start pose, [end pose], [identity anchor].
    """
    has_end, has_identity = refs_info
    # the style line describes the ART the model must match, which is not
    # the post chain: a pixel-art character on the smooth "painted" post
    # path (style=painted + pixel_bits) is still pixel art to H3
    style = ("pixel" if (spec.get("pixel_bits")
                         or spec.get("style", "pixel") != "painted")
             else "painted")
    identity = spec.get("identity") or spec.get("display_name") or anim
    # Behaviour is DATA on the animation, never a hardcoded name list: any
    # animation a user invents behaves however its spec says.
    looping = bool(aspec.get("loop", False))
    facing = aspec.get("facing", "right")
    action = aspec["action"].strip()
    dur = "%.1f" % seconds
    label = aspec.get("label") or anim.replace("_", " ")

    n = 2
    retention = ["<Picture 1> is the first frame and composition anchor: the "
                 "clip begins exactly from this image, and the character's "
                 "identity, outfit, colors, proportions, scale and position "
                 "in frame stay fully preserved throughout."]
    end_clause = ""
    if has_end:
        retention.append("<Picture %d> is the exact final pose: the last "
                         "frame matches it precisely and the motion between "
                         "the two is one continuous natural action." % n)
        end_clause = (" The clip ends exactly in the pose of <Picture %d>."
                      % n)
        n += 1
    # pronouns come from the spec: a male character described as "she" in the
    # shot block contradicts his own action text mid-prompt
    subj, poss = {"he": ("He", "his"), "they": ("They", "their")}.get(
        (spec.get("pronoun") or "she").lower(), ("She", "her"))
    if has_identity:
        retention.append("<Picture %d> is an identity reference only: %s "
                         "face, hair, outfit and colors match it; its "
                         "pose and framing are not used." % (n, poss))
    loop_clause = (" The cycle repeats evenly and the final frame returns to "
                   "exactly the starting pose so the clip loops seamlessly."
                   if looping and not has_end else "")
    face_clause = {
        "right": "%s faces the right side of the frame the whole time - "
                 "face, eyes, feet and body all pointing right." % subj,
        "left": "%s faces the left side of the frame the whole time - face, "
                "eyes, feet and body all pointing left." % subj,
        "viewer": "%s faces straight toward the viewer the whole time." % subj,
        "away": "%s faces directly away from the viewer the whole time, "
                "back to the camera." % subj,
    }.get(facing, "")

    sections = [
        "subject_definitions: <Subject 1> is the character in <Picture 1>: "
        "%s. fully_preserved." % identity,
        "summary: A %s-second 2D-animated game-sprite clip of <Subject 1> "
        "performing %s on a flat green screen, shot with a locked static "
        "camera." % (dur, label),
        "retention_analysis: " + " ".join(retention),
        # Deliberately lean (user's call): keep only what the model needs to
        # match the reference and what the PIPELINE needs (locked camera so
        # the sprite anchors, flat green so the key works). Everything else -
        # framing, lighting, timing, mood - is left to the model.
        # No assumptions about what the body does - the action timeline says
        # that. The shot only states what the PIPELINE needs: one subject,
        # whole body in frame, flat green, locked camera.
        "detailed_description: " + _STYLE_LINE[style] + " [Shot 1] (0.0s-"
        + dur + "s) A full-body Static Shot of <Subject 1> on a flat bright "
        "green screen, centered with " + poss + " whole body inside the "
        "frame. " + face_clause + " " + action
        + end_clause + loop_clause + " The camera holds the static shot for "
        "the entire clip and the frame stays perfectly fixed [static].",
        "overall_soundscape: faint room tone only.",
        "non_diegetic_music: none.",
    ]
    return "\n".join(sections)


def _ref_preamble(has_end, has_identity, loop=False):
    """Tell H3 which reference drives what. References are addressed by the
    order they are connected, so this must match the upload order used below."""
    p = ("<Picture 1> is the EXACT pose the clip starts in - frame one "
         "matches it precisely. ")
    n = 2
    if has_end:
        p += ("<Picture %d> is the EXACT pose the clip ends in - the final "
              "frame matches it precisely, and the motion between the two is "
              "one continuous natural action. " % n)
        n += 1
    elif loop:
        # closing the cycle on its own opening pose is what makes a loop cut
        # cleanly - the same trick commercial sprite tools use
        p += ("The clip is a seamless LOOP: the final frame returns to the "
              "exact pose of <Picture 1> so it can repeat forever without a "
              "visible jump. ")
    if has_identity:
        p += ("<Picture %d> defines her identity, outfit, colors, hairstyle "
              "and art style: match it exactly in every frame and never "
              "change her hairstyle or costume. " % n)
    return p


def _anim_refs(char, spec, anim):
    """Reference image paths for one animation, in <Picture N> order:
    start pose, [end pose], [identity anchor]."""
    base = ref_paths(char)[0]
    aspec = spec["animations"][anim]

    def view_path(v):
        p = os.path.join(run_dir(char), "ref_%s.png" % v)
        if not os.path.exists(p):
            raise SystemExit("view reference missing (%s) - run --stage views "
                             "first" % p)
        return p

    start = view_path(aspec["view"]) if aspec.get("view") else base
    refs = [start]
    end = view_path(aspec["end_view"]) if aspec.get("end_view") else None
    if end:
        # an end pose must share the start's coordinate space, or the model
        # slides the character across the frame to reach it
        src = (spec.get("views") or {}).get(aspec["end_view"], {})
        if not src.get("from_anim"):
            print("[anim] WARNING: end pose '%s' was generated from scratch, "
                  "not harvested from a rendered clip - the character may "
                  "drift toward its framing" % aspec["end_view"])
        refs.append(end)
    # identity anchor: the canonical reference, unless it IS the start pose
    if start != base:
        refs.append(base)
    return refs, bool(end), start != base


# ----------------------------------------------------------------- stage: anim
def stage_anim(client, char, spec, cfg, seed, anims, ref_override=None):
    cfg = spec_cfg(spec, cfg)
    ref_png = ref_override or ref_paths(char)[0]
    if not os.path.exists(ref_png):
        raise SystemExit("reference image missing (%s) - run --stage ref first" % ref_png)

    margin = float(cfg.get("ref_margin") or 0)
    pad_dir = os.path.join(run_dir(char), "_padded")
    # Poses harvested from a rendered clip are ALREADY in the video's framing
    # (that clip was itself made from a padded reference). Padding them again
    # would shrink the character, so the clip that starts from a harvested
    # pose would come out smaller than every other animation.
    harvested = {os.path.abspath(os.path.join(run_dir(char), "ref_%s.png" % v))
                 for v, spec_v in (spec.get("views") or {}).items()
                 if spec_v.get("from_anim")}
    # pixel sprites: enforce the pixel grid on the REFERENCE the animator
    # copies (and again in post) instead of demanding it per-frame in prompts
    conform_px = (spec.get("style", "pixel") == "pixel"
                  and spec.get("sprite_type") != "vfx")

    def upload(path):
        """Upload a reference (padded for headroom, grid-conformed for
        pixel sprites), once."""
        key = os.path.abspath(path)
        if key in upload.cache:
            return upload.cache[key]
        send = path
        if margin > 0 and key not in harvested:
            send = postprocess.pad_reference(
                path, os.path.join(pad_dir, os.path.basename(path)), margin)
        if conform_px:
            send = postprocess.pixel_conform_reference(
                send, os.path.join(pad_dir, "px_" + os.path.basename(path)),
                sprite_h=cfg["sprite_h"], colors=cfg["colors"])
        name = "%s_%s" % (char, os.path.basename(path))
        upload.cache[key] = client.upload_image(send, name=name)
        return upload.cache[key]
    upload.cache = {}

    results = {}
    for anim in anims:
        aspec = spec["animations"][anim]
        ref_paths_for_anim, has_end, has_identity = _anim_refs(char, spec, anim)
        uploaded = [upload(p) for p in ref_paths_for_anim]
        width, height = anim_canvas(cfg["vid_size"], aspec.get("canvas", "square"))
        seconds = aspec["seconds"] if cfg["seconds_cap"] is None \
            else min(aspec["seconds"], cfg["seconds_cap"])
        length = graphs.h3_valid_length(seconds)
        if aspec.get("action"):
            # new-style animation: the official six-section brief, timeline
            # matched to the actual clip length
            prompt = h3_brief(spec, anim, aspec, (has_end, has_identity),
                              length / 24.0)
        else:
            # legacy free-text prompt: the preamble states which <Picture N>
            # drives what; drop the template's single-reference identity
            # sentence so they can't conflict
            prompt = _IDENTITY_SENTENCE.sub("", aspec["prompt"])
            looping = aspec.get("loop", anim in ("idle", "walk", "run", "guard"))
            prompt = _ref_preamble(has_end, has_identity, looping) + prompt
        aseed = _seed(seed)
        print("[anim] %s/%s: %dx%d, %d frames (~%.1fs), seed %d, turbo=%s, refs=%s%s"
              % (char, anim, width, height, length, length / 24.0, aseed,
                 cfg["turbo_lora"],
                 [os.path.basename(p) for p in ref_paths_for_anim],
                 (", margin %.0f%%" % (margin * 100)) if margin else ""))
        g = graphs.minimax_h3_r2v(
            prompt, uploaded, width=width, height=height, length=length,
            seed=aseed, use_turbo_lora=cfg["turbo_lora"], steps=cfg["video_steps"],
            frames_prefix="sprites/%s/%s/frame" % (char, anim),
            video_prefix="sprites/%s/%s_clip" % (char, anim))
        entry = client.run(g, label="h3 %s" % anim, timeout=cfg["timeout"])
        # the previous render becomes a take instead of being deleted, so
        # earlier attempts stay available to compare and switch back to
        archive_take(char, anim)
        frames_dir = os.path.join(run_dir(char), anim, "raw_frames")
        if os.path.isdir(frames_dir):
            for old in glob.glob(os.path.join(frames_dir, "*.png")):
                os.remove(old)
        # SaveVideo also registers its file under the "images" key in history,
        # so split by extension rather than trusting the key alone.
        all_files = client.output_files(entry)
        img_files = [f for f in all_files if f["filename"].lower().endswith(".png")]
        vid_files = [f for f in all_files if f["filename"].lower().endswith((".mp4", ".webm", ".mov"))]
        paths = [client.fetch(f, frames_dir) for f in img_files]
        vids = [client.fetch(f, os.path.join(run_dir(char), anim))
                for f in vid_files]
        with open(os.path.join(run_dir(char), anim, "render.json"), "w") as fh:
            json.dump({"seed": aseed, "when": time.strftime("%Y-%m-%d %H:%M"),
                       "length": length, "steps": cfg["video_steps"],
                       "refs": [os.path.basename(p) for p in ref_paths_for_anim]},
                      fh, indent=1)
        print("[anim] %s: %d frames (seed %d) -> %s%s"
              % (anim, len(paths), aseed, frames_dir,
                 (", clip: " + vids[0]) if vids else ""))
        results[anim] = paths
    return results


# ----------------------------------------------------------------- stage: post
def stage_post(char, cfg, anims, take_every=2, trim_start=0, trim_end=0,
               spec=None):
    cfg = spec_cfg(spec, cfg)
    key_mode = "glow" if (spec or {}).get("sprite_type") == "vfx" else "auto"
    style = (spec or {}).get("style", "pixel")
    out = {}
    for anim in anims:
        frames_dir = os.path.join(run_dir(char), anim, "raw_frames")
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not paths:
            print("[post] no frames for %s/%s - skipping" % (char, anim))
            continue
        dest = os.path.join(run_dir(char), anim)
        aspec = (spec or {}).get("animations", {}).get(anim, {})
        pixel_kw = {}
        # classic keyframe counts + playback speed by pixel-density era.
        # pixel sprites always collapse; painted sprites opt in with
        # "keyframes": true (the painted-pixel chain: the video's painted
        # look, played as a few held keyframes like hand animation)
        wants_keyframes = (style == "pixel"
                           or (style == "painted"
                               and (spec or {}).get("keyframes")))
        if wants_keyframes and key_mode != "glow":
            pr = templates.PIXEL_PRESETS.get(
                str((spec or {}).get("pixel_bits") or ""), {})
            looping = bool(aspec.get("loop", False))
            px = aspec.get("px_frames")
            pixel_kw = {
                "loop": looping,
                # px_frames: None = preset default; 0 = keep every 2nd
                # source frame (no collapse, 12fps); N = N keyframes
                "pixel_frames": (int(px) if px is not None else
                                 (pr.get("frames_loop", 8) if looping
                                  else pr.get("frames_once", 12))),
                "pixel_fps": float(pr.get("fps", 10)),
                "loop_mode": aspec.get("loop_mode", "cycle"),
                "quiet": float(aspec.get("quiet", 0.35)),
                "drop": aspec.get("drop") or None,
                # a two-beat cycle (left+right stride) must loop both halves
                "gait": bool(aspec.get("gait", False)),
            }
        meta = postprocess.process_animation(
            paths, dest, sprite_h=cfg["sprite_h"], colors=cfg["colors"],
            take_every=take_every, trim_start=trim_start, trim_end=trim_end,
            key_mode=key_mode, style=style,
            range_override=aspec.get("range"),
            ends_at_end=bool(aspec.get("end_view")),
            clean=bool((spec or {}).get("cel_clean")),
            outline=(spec or {}).get("outline"),
            **pixel_kw)
        print("[post] %s/%s: %d sprites %sx%s @ %.0ffps -> %s"
              % (char, anim, meta["frames"], meta["frame_size"][0],
                 meta["frame_size"][1], meta["fps"], dest))
        out[anim] = meta
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="ComfyUI game-sprite pipeline")
    ap.add_argument("--char", default="stormshield",
                    help="sprite id from characters.json")
    ap.add_argument("--mode", default="test", choices=sorted(MODES))
    ap.add_argument("--stage", default="all",
                    choices=["ref", "views", "anim", "post", "all"])
    ap.add_argument("--anim", nargs="*", default=["idle"],
                    help="animation names (see characters.py)")
    ap.add_argument("--seed", type=int, default=0,
                help="0 = a fresh seed per render; any other value pins it")
    ap.add_argument("--ref-image", default=None,
                    help="use an existing reference image instead of generating one")
    ap.add_argument("--vid-size", type=int, default=None,
                    help="override video size px (multiple of 32)")
    ap.add_argument("--seconds-cap", type=float, default=None,
                    help="override clip length cap in seconds")
    ap.add_argument("--take-every", type=int, default=2,
                    help="frame subsampling for post (2 -> 12fps)")
    ap.add_argument("--trim-start", type=int, default=0)
    ap.add_argument("--trim-end", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = dict(MODES[args.mode])
    if args.vid_size:
        cfg["vid_size"] = args.vid_size
    if args.seconds_cap is not None:
        cfg["seconds_cap"] = None if args.seconds_cap <= 0 else args.seconds_cap
    chars = store.load()
    if args.char not in chars:
        raise SystemExit("unknown sprite '%s' (have: %s)"
                         % (args.char, ", ".join(sorted(chars))))
    spec = chars[args.char]
    if args.stage in ("anim", "post", "all"):
        unknown = [a for a in args.anim if a not in spec.get("animations", {})]
        if unknown:
            raise SystemExit("unknown animation(s): %s" % ", ".join(unknown))

    client = ComfyClient(COMFY_HOST, output_dir=COMFY_OUTPUT_DIR)
    try:
        client.stats()
    except Exception as exc:
        raise SystemExit("ComfyUI not reachable at %s (%s)" % (COMFY_HOST, exc))

    t0 = time.time()
    if args.stage in ("ref", "all") and not args.ref_image:
        stage_ref(client, args.char, spec, cfg, args.seed)
        if args.stage == "all":
            client.free()  # release Krea before loading the H3 stack
    if args.stage == "views":
        stage_views(client, args.char, spec, cfg, args.seed)
    if args.stage in ("anim", "all"):
        stage_anim(client, args.char, spec, cfg, args.seed, args.anim,
                   ref_override=args.ref_image)
    if args.stage in ("post", "all"):
        stage_post(args.char, cfg, args.anim, take_every=args.take_every,
                   trim_start=args.trim_start, trim_end=args.trim_end, spec=spec)
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
