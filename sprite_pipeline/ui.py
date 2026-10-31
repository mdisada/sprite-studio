"""Local web UI for the sprite pipeline.

Run:
    python -m sprite_pipeline.ui            # serves http://127.0.0.1:8899
    python -m sprite_pipeline.ui --open     # also opens your browser

Zero extra dependencies: stdlib http.server plus one worker thread that runs
GPU jobs sequentially (ComfyUI executes serially anyway). Sprite definitions
live in characters.json (shared with the CLI); job state is in-memory.
"""
import base64
import io
import json
import mimetypes
import os
import queue
import random
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from . import characters as templates
from . import export_godot, pipeline, postprocess, store
from .comfy_client import ComfyClient

PORT = int(os.environ.get("SPRITE_UI_PORT", "8899"))
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.html")

_QUEUE = queue.Queue()
_JOBS = {}
_JOBS_LOCK = threading.Lock()
_COMFY_CACHE = {"t": 0.0, "ok": False}


# Rough wall-clock estimates in seconds, measured on this machine (RTX 4060 Ti).
ESTIMATES = {
    "ref_test": 60, "ref_final": 100,        # Krea t2i, model warm
    "clip_test": 130, "clip_final": 600,     # H3 clip, model warm
    "view_edit": 120,                        # Qwen edit per view, model warm
    "cold_load": 300,                        # first heavy model load of a session
}


JOURNAL = os.path.join(store.PROJECT_ROOT, "jobs_pending.json")


def _persist_jobs():
    """Journal unfinished jobs so a server restart can recover them."""
    with _JOBS_LOCK:
        live = [{k: j[k] for k in ("type", "char", "anims", "params", "spec")}
                for j in _JOBS.values() if j["status"] in ("queued", "running")]
    try:
        with open(JOURNAL, "w") as fh:
            json.dump(live, fh)
    except OSError as exc:
        print("journal write failed:", exc)


def _recover_jobs():
    """Re-enqueue jobs that were queued/running when the server last stopped."""
    if not os.path.isfile(JOURNAL):
        return
    try:
        with open(JOURNAL) as fh:
            lost = json.load(fh)
        os.remove(JOURNAL)
    except Exception as exc:
        print("journal read failed:", exc)
        return
    for j in lost:
        job = _new_job(j["type"], j["char"], j.get("anims"), j.get("params"),
                       j.get("spec"))
        job["_log"].write("(recovered after server restart - re-running "
                          "from the start)\n")
        print("recovered job:", j["type"], j["char"], j.get("anims"))


# ------------------------------------------------------------------- job model
def _new_job(jtype, char, anims, params, spec):
    job = {
        "id": uuid.uuid4().hex[:8],
        "type": jtype,
        "char": char,
        "anims": anims or [],
        "params": params or {},
        "spec": spec,  # snapshot at queue time: later edits can't change the job
        "status": "queued",
        "error": None,
        "created": time.time(),
        "started": None,
        "finished": None,
        "_log": io.StringIO(),
    }
    with _JOBS_LOCK:
        _JOBS[job["id"]] = job
    _QUEUE.put(job)
    _persist_jobs()
    return job


def _serialize_job(job):
    log = job["_log"].getvalue()
    return {k: job[k] for k in ("id", "type", "char", "anims", "params", "status",
                                "error", "created", "started", "finished")} | \
        {"log": log[-6000:]}


def _cfg_from_params(p):
    mode = p.get("mode", "test")
    cfg = dict(pipeline.MODES.get(mode, pipeline.MODES["test"]))
    for key in ("ref_size", "vid_size", "sprite_h", "colors", "video_steps"):
        val = p.get(key)
        if val not in (None, ""):
            cfg[key] = int(val)
    if "turbo_lora" in p:
        cfg["turbo_lora"] = bool(p["turbo_lora"])
    if p.get("mode") == "final":
        cfg["seconds_cap"] = None
    return cfg


def _post_kwargs(p):
    return {
        "take_every": int(p.get("take_every") or 2),
        "trim_start": int(p.get("trim_start") or 0),
        "trim_end": int(p.get("trim_end") or 0),
    }


def _run_job(client, job):
    char = job["char"]
    spec = job.get("spec") or store.get(char)
    p = job["params"]
    cfg = _cfg_from_params(p)
    # 0 = let the stage roll a fresh seed; a pinned value reproduces a render
    seed = int(p.get("seed") or 0)
    jtype = job["type"]
    anims = [a for a in job["anims"] if a in spec.get("animations", {})]

    if jtype == "ref":
        pipeline.stage_ref(client, char, spec, cfg, seed)
    elif jtype == "views":
        pipeline.stage_views(client, char, spec, cfg, seed)
    elif jtype == "anim":
        if not anims:
            raise ValueError("no valid animations selected")
        pipeline.stage_anim(client, char, spec, cfg, seed, anims)
        pipeline.stage_post(char, cfg, anims, spec=spec, **_post_kwargs(p))
    elif jtype == "post":
        if not anims:
            raise ValueError("no valid animations selected")
        pipeline.stage_post(char, cfg, anims, spec=spec, **_post_kwargs(p))
    elif jtype == "all":
        pipeline.stage_ref(client, char, spec, cfg, seed)
        client.free()
        if anims:
            pipeline.stage_anim(client, char, spec, cfg, seed, anims)
            pipeline.stage_post(char, cfg, anims, spec=spec, **_post_kwargs(p))
    elif jtype == "variations":
        _run_variations(client, char, spec, p)
    else:
        raise ValueError("unknown job type %r" % jtype)


def _run_variations(client, char, spec, p):
    """Render reference candidates: seed noise or art-style presets."""
    from . import graphs
    kind = p.get("kind", "seeds")
    pipeline.check_ref_prompt(spec["ref_prompt"])  # Krea-2 fails silently past its budget
    out = os.path.join(pipeline.PROJECT_OUT, char,
                       "style_candidates" if kind == "styles" else "ref_candidates")
    os.makedirs(out, exist_ok=True)
    jobs = []
    if kind == "styles":
        is_pixel = spec.get("style", "pixel") == "pixel"
        presets = (templates.PIXEL_STYLE_PRESETS if is_pixel
                   else templates.STYLE_PRESETS)
        base = spec["ref_prompt"]
        # strip a leading style block if it matches a known preset
        strippable = (list(templates.STYLE_PRESETS.values())
                      + list(templates.PIXEL_STYLE_PRESETS.values())
                      + [templates.PAINTED_STYLE, templates.PIXEL_STYLE]
                      + [p["style"] for p in templates.PIXEL_PRESETS.values()])
        for s in strippable:
            if base.startswith(s):
                base = base[len(s):]
                break
        if is_pixel:
            # keep the era/density block in front so every variation stays
            # true pixel art at the project's pixel size
            era = templates.PIXEL_PRESETS.get(
                str(spec.get("pixel_bits") or ""), {"style": templates.PIXEL_STYLE})
            base = era["style"] + base
        for name, style in presets.items():
            g = graphs.krea_t2i(style + base, width=1024, height=1024,
                                # unpinned = fresh, so re-rolling a style batch
                                # returns new images instead of the same ones
                                seed=int(p.get("seed") or random.randint(1000, 900000)),
                                filename_prefix="sprites/%s/style_%s" % (char, name))
            jobs.append((name + ".png", client.submit(g)))
            print("queued style", name, flush=True)
    else:
        # a fresh batch every time: an unset seed must NOT fall back to a
        # constant, or every re-roll returns the same six images
        base_seed = int(p.get("seed") or random.randint(1000, 900000))
        seeds = [base_seed + i * 37 for i in range(6)]
        for s in seeds:
            g = graphs.krea_t2i(spec["ref_prompt"], width=1024, height=1024,
                                seed=s,
                                filename_prefix="sprites/%s/cand_s%d" % (char, s))
            jobs.append(("seed_%d.png" % s, client.submit(g)))
            print("queued seed", s, flush=True)
    for fname, pid in jobs:
        entry = client.wait(pid, timeout=3600, label=fname)
        files = [f for f in client.output_files(entry, kind="images")
                 if f["filename"].lower().endswith(".png")]
        client.fetch(files[0], out, dest_name=fname)
        print("done", fname, flush=True)


def _worker():
    client = ComfyClient(pipeline.COMFY_HOST, output_dir=pipeline.COMFY_OUTPUT_DIR)
    while True:
        job = _QUEUE.get()
        job["status"] = "running"
        job["started"] = time.time()
        try:
            # worker is the only thread that prints; capture into the job log
            with redirect_stdout(job["_log"]):
                _run_job(client, job)
            job["status"] = "done"
        except (Exception, SystemExit) as exc:  # SystemExit: pipeline guard rails
            job["_log"].write("\nERROR: %s\n%s" % (exc, traceback.format_exc()))
            job["status"] = "error"
            job["error"] = str(exc)[:500]
        finally:
            job["finished"] = time.time()
            _persist_jobs()


# ------------------------------------------------------------------ http layer
def _comfy_ok():
    now = time.time()
    if now - _COMFY_CACHE["t"] > 15:
        try:
            q = requests.get("http://%s/queue" % pipeline.COMFY_HOST, timeout=2).json()
            _COMFY_CACHE["ok"] = True
            _COMFY_CACHE["busy"] = (len(q.get("queue_running", []))
                                    + len(q.get("queue_pending", [])))
        except Exception:
            # ComfyUI may simply have moved to another port (it auto-
            # increments past a held 8188) - look for it before declaring
            # it offline, so a healthy server is never reported as down
            found = pipeline.resolve_comfy_host(pipeline.COMFY_HOST)
            if found != pipeline.COMFY_HOST and pipeline._comfy_alive(found):
                print("ComfyUI moved to %s" % found)
                pipeline.COMFY_HOST = found
                try:
                    q = requests.get("http://%s/queue" % found, timeout=2).json()
                    _COMFY_CACHE["ok"] = True
                    _COMFY_CACHE["busy"] = (len(q.get("queue_running", []))
                                            + len(q.get("queue_pending", [])))
                    _COMFY_CACHE["t"] = now
                    return True
                except Exception:
                    pass
            _COMFY_CACHE["ok"] = False
            _COMFY_CACHE["busy"] = 0
        _COMFY_CACHE["t"] = now
    return _COMFY_CACHE["ok"]


def _assets():
    """Per-sprite on-disk status: does a reference exist, which animations
    have raw frames / finished sheets. Drives the UI's guardrails."""
    out = {}
    root = pipeline.PROJECT_OUT
    for cid in store.load():
        d = os.path.join(root, cid)
        info = {"ref": os.path.isfile(os.path.join(d, "ref.png")),
                "views": [], "anims": {}}
        if os.path.isdir(d):
            for name in os.listdir(d):
                if (name.startswith("ref_") and name.endswith(".png")
                        and name != "ref_sprite.png"):
                    info["views"].append(name[4:-4])
                sub = os.path.join(d, name)
                if os.path.isdir(sub):
                    rf = os.path.join(sub, "raw_frames")
                    frames = len([f for f in os.listdir(rf)
                                  if f.lower().endswith(".png")]) if os.path.isdir(rf) else 0
                    tdir = os.path.join(sub, "takes")
                    entry = {
                        "frames": frames,
                        "done": os.path.isfile(os.path.join(sub, "sheet.png")),
                        # how many earlier renders of this animation are kept
                        "takes": (len([t for t in os.listdir(tdir)
                                       if os.path.isdir(os.path.join(tdir, t))])
                                  if os.path.isdir(tdir) else 0),
                    }
                    if entry["done"]:
                        # test renders use small source videos (<512px short
                        # edge); finals are 640+ - lets the UI gate approvals
                        try:
                            with open(os.path.join(sub, "meta.json")) as fh:
                                meta = json.load(fh)
                            src = meta.get("source_size") or [640, 640]
                            entry["quality"] = ("final" if min(src) >= 512
                                                else "test")
                        except Exception:
                            entry["quality"] = "final"
                    info["anims"][name] = entry
        out[cid] = info
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    # helpers ---------------------------------------------------------------
    def _send(self, data, ctype="application/json", code=200):
        body = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._send({"error": msg}, code=code)

    # GET -------------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                with open(HTML_PATH, "rb") as fh:
                    self._send(fh.read(), "text/html; charset=utf-8")
            elif path == "/viewer":
                viewer = os.path.join(os.path.dirname(HTML_PATH), "viewer.html")
                with open(viewer, "rb") as fh:
                    self._send(fh.read(), "text/html; charset=utf-8")
            elif path == "/api/state":
                with _JOBS_LOCK:
                    jobs = [_serialize_job(j) for j in
                            sorted(_JOBS.values(), key=lambda j: j["created"],
                                   reverse=True)][:40]
                self._send({
                    "characters": store.load(),
                    "projects": store.load_projects(),
                    "jobs": jobs,
                    "modes": {k: {kk: vv for kk, vv in v.items()}
                              for k, v in pipeline.MODES.items()},
                    "templates": {
                        "pixel_style": templates.PIXEL_STYLE,
                        "green_stage": templates.GREEN_STAGE,
                        "anim_rules": templates.ANIM_RULES,
                        "contained_fx": templates.CONTAINED_FX,
                        "vfx_pixel": templates.VFX_PIXEL,
                        "vfx_stage": templates.VFX_STAGE,
                        "vfx_rules": templates.VFX_RULES,
                        "painted_style": templates.PAINTED_STYLE,
                        "painted_anim_rules": templates.PAINTED_ANIM_RULES,
                        "pixel_presets": templates.PIXEL_PRESETS,
                        "game_kits": templates.GAME_KITS,
                    },
                    "comfy_ok": _comfy_ok(),
                    "comfy_busy": _COMFY_CACHE.get("busy", 0),
                    "out_root": pipeline.PROJECT_OUT,
                    "assets": _assets(),
                    "estimates": ESTIMATES,
                })
            elif path.startswith("/api/takes/"):
                # /api/takes/<char>/<anim> - every render of that animation
                parts = path[len("/api/takes/"):].split("/")
                if len(parts) != 2:
                    return self._err("expected /api/takes/<char>/<anim>")
                cid = store.sanitize_id(parts[0])
                spec = store.get(cid)
                if parts[1] not in spec.get("animations", {}):
                    return self._err("unknown animation")
                self._send({"takes": pipeline.list_takes(cid, parts[1])})
            elif path.startswith("/api/frames/"):
                # /api/frames/<char>/<anim> - the RAW render frames of the
                # active take (full-res green screen) plus the post meta, for
                # the Pixel Lab, which re-pixelizes them live in the browser
                parts = path[len("/api/frames/"):].split("/")
                if len(parts) != 2:
                    return self._err("expected /api/frames/<char>/<anim>")
                cid = store.sanitize_id(parts[0])
                anim = parts[1]
                if anim not in store.get(cid).get("animations", {}):
                    return self._err("unknown animation")
                adir = os.path.join(pipeline.PROJECT_OUT, cid, anim)
                rf = os.path.join(adir, "raw_frames")
                names = sorted(f for f in os.listdir(rf)
                               if f.lower().endswith(".png"))                     if os.path.isdir(rf) else []
                meta = None
                mp = os.path.join(adir, "meta.json")
                if os.path.isfile(mp):
                    try:
                        with open(mp) as fh:
                            meta = json.load(fh)
                    except Exception:
                        meta = None
                self._send({"dir": "%s/%s/raw_frames" % (cid, anim),
                            "frames": names, "meta": meta})
            elif path.startswith("/api/candidates/"):
                cid = store.sanitize_id(path.rsplit("/", 1)[-1])
                out = {}
                for kind, sub in (("seeds", "ref_candidates"),
                                  ("styles", "style_candidates"),
                                  ("designs", "design_candidates")):
                    d = os.path.join(pipeline.PROJECT_OUT, cid, sub)
                    out[kind] = sorted(
                        f for f in os.listdir(d)
                        if f.endswith(".png") and "sheet" not in f) \
                        if os.path.isdir(d) else []
                self._send(out)
            elif path.startswith("/files/"):
                rel = path[len("/files/"):].replace("%20", " ")
                full = os.path.normpath(os.path.join(pipeline.PROJECT_OUT, rel))
                if not full.startswith(os.path.normpath(pipeline.PROJECT_OUT)):
                    return self._err("bad path", 403)
                if not os.path.isfile(full):
                    return self._err("not found", 404)
                ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
                with open(full, "rb") as fh:
                    self._send(fh.read(), ctype)
            else:
                self._err("not found", 404)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._err("server error: %s" % exc, 500)

    # POST ------------------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if path == "/api/char/save":
                cid = store.sanitize_id(body.get("id", ""))
                spec = body.get("spec") or {}
                if not spec.get("ref_prompt"):
                    return self._err("ref_prompt is required")
                spec.setdefault("display_name", cid)
                spec.setdefault("animations", {})
                store.upsert(cid, spec)
                self._send({"ok": True, "id": cid})
            elif path == "/api/char/delete":
                store.delete(body.get("id", ""))
                self._send({"ok": True})
            elif path == "/api/project/save":
                pid = store.sanitize_id(body.get("id", ""))
                proj = body.get("proj") or {}
                if not proj.get("display_name"):
                    return self._err("display_name is required")
                if proj.get("style") not in ("pixel", "painted"):
                    return self._err("style must be pixel or painted")
                if proj.get("style") == "pixel" and \
                        str(proj.get("pixel_bits")) not in templates.PIXEL_PRESETS:
                    return self._err("pixel_bits must be one of %s"
                                     % ", ".join(sorted(templates.PIXEL_PRESETS)))
                if proj.get("game_type") not in templates.GAME_KITS:
                    return self._err("unknown game_type")
                existing = store.load_projects().get(pid) or {}
                proj.setdefault("created", existing.get("created", time.time()))
                store.upsert_project(pid, proj)
                self._send({"ok": True, "id": pid})
            elif path == "/api/project/delete":
                try:
                    store.delete_project(body.get("id", ""))
                except ValueError as exc:
                    return self._err(str(exc))
                self._send({"ok": True})
            elif path == "/api/char/pick_ref":
                cid = store.sanitize_id(body.get("id", ""))
                spec = store.get(cid)
                sub = {"styles": "style_candidates",
                       "designs": "design_candidates"}.get(
                    body.get("kind"), "ref_candidates")
                fname = os.path.basename(body.get("file", ""))
                src = os.path.join(pipeline.PROJECT_OUT, cid, sub, fname)
                if not os.path.isfile(src):
                    return self._err("candidate not found")
                import shutil as _sh
                dst_dir = os.path.join(pipeline.PROJECT_OUT, cid)
                os.makedirs(dst_dir, exist_ok=True)
                _sh.copy2(src, os.path.join(dst_dir, "ref.png"))
                try:
                    from . import postprocess
                    postprocess.process_ref_image(
                        os.path.join(dst_dir, "ref.png"),
                        os.path.join(dst_dir, "ref_sprite.png"),
                        sprite_h=256, style=spec.get("style", "pixel"),
                        **pipeline.outline_opts(spec))
                except Exception as exc:
                    print("ref preview failed:", exc)
                spec.setdefault("approved", {})["ref"] = True
                store.upsert(cid, spec)
                self._send({"ok": True})
            elif path == "/api/char/upload_ref":
                cid = store.sanitize_id(body.get("id", ""))
                spec = store.get(cid)
                data = body.get("data", "")
                if data.startswith("data:"):
                    data = data.split(",", 1)[-1]
                try:
                    raw = base64.b64decode(data, validate=True)
                except Exception:
                    return self._err("bad image data (not valid base64)")
                if len(raw) > 20 * 1024 * 1024:
                    return self._err("image too large (max 20MB)")
                try:
                    img = Image.open(io.BytesIO(raw))
                    img.load()
                except (UnidentifiedImageError, OSError):
                    return self._err("not a readable image file")
                dst_dir = os.path.join(pipeline.PROJECT_OUT, cid)
                os.makedirs(dst_dir, exist_ok=True)
                img.convert("RGB").save(os.path.join(dst_dir, "ref.png"))
                try:
                    postprocess.process_ref_image(
                        os.path.join(dst_dir, "ref.png"),
                        os.path.join(dst_dir, "ref_sprite.png"),
                        sprite_h=256, style=spec.get("style", "pixel"),
                        **pipeline.outline_opts(spec))
                except Exception as exc:
                    print("ref preview failed:", exc)
                spec.setdefault("approved", {})["ref"] = True
                store.upsert(cid, spec)
                self._send({"ok": True})
            elif path == "/api/char/approve":
                cid = store.sanitize_id(body.get("id", ""))
                spec = store.get(cid)
                ap = spec.setdefault("approved", {})
                if body.get("anim"):
                    ap.setdefault("anims", {})[body["anim"]] = bool(
                        body.get("approved", True))
                else:
                    ap["ref"] = bool(body.get("approved", True))
                store.upsert(cid, spec)
                self._send({"ok": True})
            elif path == "/api/view/harvest":
                cid = store.sanitize_id(body.get("id", ""))
                spec = store.get(cid)
                vname = store.sanitize_id(body.get("name", ""))
                src = {"from_anim": body.get("from_anim", ""),
                       "frame": int(body.get("frame", -1)),
                       "regenerate": True}
                if src["from_anim"] not in spec.get("animations", {}):
                    return self._err("unknown source animation")
                try:
                    pipeline._harvest_pose(cid, vname, src)
                except SystemExit as exc:
                    return self._err(str(exc))
                src.pop("regenerate")
                spec.setdefault("views", {})[vname] = src
                store.upsert(cid, spec)
                self._send({"ok": True})
            elif path == "/api/take/activate":
                cid = store.sanitize_id(body.get("id", ""))
                spec = store.get(cid)
                anim = body.get("anim", "")
                if anim not in spec.get("animations", {}):
                    return self._err("unknown animation")
                try:
                    pipeline.activate_take(cid, anim, body.get("take", ""))
                except ValueError as exc:
                    return self._err(str(exc))
                self._send({"ok": True})
            elif path == "/api/take/delete":
                cid = store.sanitize_id(body.get("id", ""))
                spec = store.get(cid)
                anim = body.get("anim", "")
                if anim not in spec.get("animations", {}):
                    return self._err("unknown animation")
                try:
                    pipeline.delete_take(cid, anim, body.get("take", ""))
                except ValueError as exc:
                    return self._err(str(exc))
                self._send({"ok": True})
            elif path == "/api/anim/tune":
                cid = store.sanitize_id(body.get("id", ""))
                anim = body.get("anim", "")
                spec = store.get(cid)
                if not anim:
                    # character-level only (sprite size applies to every
                    # animation, so it is edited outside any single one)
                    if "sprite_scale" in body:
                        v = body["sprite_scale"]
                        if v in (None, ""):
                            spec.pop("sprite_scale", None)
                        else:
                            spec["sprite_scale"] = max(0.2, min(4.0, float(v)))
                    store.upsert(cid, spec)
                    return self._send({"ok": True, "job_id": None})
                if anim not in spec.get("animations", {}):
                    return self._err("unknown animation")
                a = spec["animations"][anim]
                if body.get("rate") is not None:
                    a["rate"] = max(0.25, min(5.0, float(body["rate"])))
                if body.get("range"):
                    r = body["range"]
                    a["range"] = [int(r[0]), int(r[1])]
                # frame-shaping options: these change which frames make up
                # the sprite sequence, so they trigger a free sheet rebuild
                rebuild = False
                if "px_frames" in body:          # 0 = all frames @12fps
                    v = body["px_frames"]
                    new = None if v in (None, "") else max(0, min(64, int(v)))
                    rebuild |= a.get("px_frames") != new
                    if new is None:
                        a.pop("px_frames", None)
                    else:
                        a["px_frames"] = new
                if body.get("loop_mode") in ("cycle", "pingpong"):
                    rebuild |= a.get("loop_mode", "cycle") != body["loop_mode"]
                    a["loop_mode"] = body["loop_mode"]
                if body.get("quiet") is not None:
                    q = max(0.1, min(1.0, float(body["quiet"])))
                    rebuild |= abs(float(a.get("quiet", 0.35)) - q) > 1e-6
                    a["quiet"] = q
                if "drop" in body:
                    d = sorted({int(x) for x in (body.get("drop") or [])})
                    rebuild |= sorted(a.get("drop") or []) != d
                    if d:
                        a["drop"] = d
                    else:
                        a.pop("drop", None)
                # play-area tuning: these change how the sprite is USED, not
                # which frames exist, so they never trigger a rebuild
                if "move_speed" in body:
                    v = body["move_speed"]
                    if v in (None, ""):
                        a.pop("move_speed", None)
                    else:
                        a["move_speed"] = max(0, min(2000, float(v)))
                if "anim_scale" in body:        # this animation's own size
                    v = body["anim_scale"]
                    if v in (None, ""):
                        a.pop("scale", None)
                    else:
                        a["scale"] = max(0.2, min(4.0, float(v)))
                if "sprite_scale" in body:      # character-level, not per-anim
                    v = body["sprite_scale"]
                    if v in (None, ""):
                        spec.pop("sprite_scale", None)
                    else:
                        spec["sprite_scale"] = max(0.2, min(4.0, float(v)))
                if "fx" in body:
                    fx = []
                    for e in (body.get("fx") or [])[:8]:
                        src = store.load().get(e.get("char"))
                        if not src or e.get("anim") not in (src.get("animations") or {}):
                            continue
                        fx.append({
                            "char": e["char"], "anim": e["anim"],
                            "at": max(0.0, min(1.0, float(e.get("at", 0.5)))),
                            "dx": max(-4.0, min(4.0, float(e.get("dx", 0)))),
                            "dy": max(-4.0, min(4.0, float(e.get("dy", 0)))),
                            "scale": max(0.1, min(4.0, float(e.get("scale", 1)))),
                        })
                    if fx:
                        a["fx"] = fx
                    else:
                        a["fx"] = []       # explicit empty = "no effects here"
                store.upsert(cid, spec)
                job_id = None
                if rebuild:
                    job = _new_job("post", cid, [anim], {"mode": "final"}, spec)
                    job_id = job["id"]
                meta_path = os.path.join(pipeline.PROJECT_OUT, cid, anim,
                                         "meta.json")
                if os.path.isfile(meta_path):
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                    if a.get("rate") is not None:
                        meta["rate"] = a["rate"]
                    if a.get("range"):
                        meta["action_range"] = a["range"]
                        meta["range_manual"] = True  # user-set: applies to loops too
                    with open(meta_path, "w") as fh:
                        json.dump(meta, fh, indent=2)
                atlas_path = os.path.join(pipeline.PROJECT_OUT, cid, anim,
                                          "atlas.json")
                if os.path.isfile(atlas_path):  # keep engine export in sync
                    with open(atlas_path) as fh:
                        atlas = json.load(fh)
                    if a.get("rate") is not None:
                        atlas["meta"]["speedMultiplier"] = a["rate"]
                    if a.get("range"):
                        atlas["meta"]["actionRange"] = a["range"]
                    with open(atlas_path, "w") as fh:
                        json.dump(atlas, fh, indent=1)
                self._send({"ok": True, "job_id": job_id})
            elif path == "/api/export":
                # Free and instant: re-packs finished sprites into one
                # game-ready sheet per character (shared cell + feet anchor)
                # plus Godot 4 resources. Never touches the GPU.
                cid = body.get("id", "")
                proj = body.get("project", "")
                chars = store.load()
                if cid:
                    ids = [store.sanitize_id(cid)]
                elif proj:
                    ids = [c for c, sp in chars.items()
                           if sp.get("project") == proj
                           and sp.get("sprite_type") != "vfx"]
                else:
                    return self._err("pass a sprite id or a project")
                if not ids:
                    return self._err("nothing to export")
                height = max(16, min(512, int(body.get("height") or 128)))
                res_dir = (body.get("res_dir") or "sprites").strip("/ ") or "sprites"
                out = export_godot.export_many(ids, chars, height, res_dir,
                                               project=proj or None)
                # paths the browser can fetch, relative to /files/
                for r in out["exports"]:
                    r["zip_url"] = "/files/_export/%s.zip" % r["id"]
                    r["sheet_url"] = "/files/_export/%s/%s.png" % (r["id"], r["id"])
                if out.get("bundle"):
                    out["bundle_url"] = "/files/_export/%s" % os.path.basename(out["bundle"])
                out["ok"] = True
                out["height"] = height
                out["res_dir"] = res_dir
                self._send(out)
            elif path == "/api/job":
                jtype = body.get("type")
                char = body.get("char", "")
                if jtype not in ("ref", "views", "anim", "post", "all",
                                 "variations"):
                    return self._err("bad job type")
                spec = store.get(char)  # validates existence, snapshot for the job
                anims = body.get("anims") or []
                unknown = [a for a in anims if a not in spec.get("animations", {})]
                if unknown:
                    return self._err("unknown animations: %s (save first?)"
                                     % ", ".join(unknown))
                if jtype in ("anim", "post") and not anims:
                    return self._err("select at least one animation")
                if jtype in ("anim", "all", "ref", "variations") and not _comfy_ok():
                    return self._err("ComfyUI is not reachable - start it first")
                with _JOBS_LOCK:
                    for j in _JOBS.values():
                        if (j["status"] in ("queued", "running")
                                and j["type"] == jtype and j["char"] == char
                                and sorted(j["anims"]) == sorted(anims)):
                            return self._err(
                                "an identical job is already queued/running "
                                "(job %s) - not queuing a duplicate" % j["id"])
                job = _new_job(jtype, char, anims, body.get("params"), spec)
                self._send({"ok": True, "job_id": job["id"]})
            else:
                self._err("not found", 404)
        except (KeyError, ValueError) as exc:
            self._err(str(exc))
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._err("server error: %s" % exc, 500)


def main():
    store.load()  # seed characters.json on first run
    threading.Thread(target=_worker, daemon=True).start()
    _recover_jobs()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:%d" % PORT
    print("Sprite Studio running at %s  (Ctrl+C to stop)" % url, flush=True)
    if "--open" in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
