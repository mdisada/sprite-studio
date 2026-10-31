"""Can this PC run the studio? GPU, memory, disk, model files and the engine,
each judged against what the pipeline actually needs - so a user learns it
from one screen instead of from a 60 GB download or a crashed render.

No extra dependencies: the GPU comes from nvidia-smi (installed with every
NVIDIA driver), memory from the OS, model files from ComfyUI itself when it
is running (it knows every folder it reads) or from its folders when not.
"""
import ctypes
import os
import shutil
import subprocess
import time

import requests

from . import engine, pipeline

# (file, ComfyUI models subfolder, size in GB, what it is for, Hugging Face
# source "repo:path"). Names must match exactly (graphs.py asks for them).
# Every source was checked to exist and is not gated (no login needed).
REQUIRED = [
    ("krea2_turbo_fp8_scaled.safetensors", "diffusion_models", 13.1, "image",
     "Comfy-Org/Krea-2:diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
    ("qwen3vl_4b_fp8_scaled.safetensors", "text_encoders", 5.2, "image",
     "Comfy-Org/Krea-2:text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
    ("qwen_image_vae.safetensors", "vae", 0.3, "image",
     "Comfy-Org/Krea-2:vae/qwen_image_vae.safetensors"),
    ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models", 21.0, "animation",
     "Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", 15.7, "animation",
     "Comfy-Org/MiniMax-H3:text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("minimax_h3_video_vae_fp16.safetensors", "vae", 5.2, "animation",
     "Comfy-Org/MiniMax-H3:vae/minimax_h3_video_vae_fp16.safetensors"),
    ("minimax_h3_audio_vae_fp32.safetensors", "vae", 0.6, "animation",
     "Comfy-Org/MiniMax-H3:vae/minimax_h3_audio_vae_fp32.safetensors"),
    ("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "loras", 2.0, "animation",
     "Comfy-Org/MiniMax-H3:loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
]
OPTIONAL = [   # only the "New pose" image edit uses these
    ("qwen_image_edit_2509_fp8_e4m3fn.safetensors", "diffusion_models", 20.4, "pose",
     "Comfy-Org/Qwen-Image-Edit_ComfyUI:split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
    ("Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "loras", 0.8, "pose",
     "lightx2v/Qwen-Image-Lightning:Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"),
    ("qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders", 9.4, "pose",
     "Comfy-Org/Qwen-Image_ComfyUI:split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
]


def hf_page(src):
    repo, path = src.split(":", 1)
    return "https://huggingface.co/%s/blob/main/%s" % (repo, path)

_CACHE = {"t": 0.0, "hw": None}


def gpu():
    """Largest NVIDIA GPU as {name, vram_gb, driver}, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    best = None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram = float(parts[1]) / 1024.0           # MiB -> GiB
        except ValueError:
            continue
        g = {"name": parts[0], "vram_gb": round(vram, 1),
             "driver": parts[2] if len(parts) > 2 else ""}
        if not best or g["vram_gb"] > best["vram_gb"]:
            best = g
    return best


def ram_gb():
    if os.name == "nt":
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MemStatus()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / 1024 ** 3, 1)
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _hardware():
    """GPU and RAM don't change while the studio runs; nvidia-smi is slow."""
    if not _CACHE["hw"] or time.time() - _CACHE["t"] > 300:
        _CACHE.update(t=time.time(), hw={"gpu": gpu(), "ram_gb": ram_gb()})
    return _CACHE["hw"]


def models(comfy_ok):
    """Which model files ComfyUI can load. A file ComfyUI lists inside a
    subfolder does NOT count: the pipeline asks for the bare file name."""
    files = REQUIRED + OPTIONAL
    seen, how = {}, None
    if comfy_ok:
        try:
            for folder in sorted({f[1] for f in files}):
                r = requests.get("http://%s/models/%s" % (pipeline.COMFY_HOST, folder),
                                 timeout=10)
                seen[folder] = set(r.json())
            how = "engine"
        except (requests.RequestException, ValueError):
            seen, how = {}, None
    if how is None:
        dirs = engine.model_dirs()
        if dirs:
            how = "folders"
            for folder in {f[1] for f in files}:
                seen[folder] = {n for d in dirs
                                for n in (os.listdir(os.path.join(d, folder))
                                          if os.path.isdir(os.path.join(d, folder)) else [])}
    out = []
    for i, (name, folder, gb, use, src) in enumerate(files):
        out.append({"file": name, "folder": folder, "gb": gb, "use": use,
                    "url": hf_page(src), "optional": i >= len(REQUIRED),
                    "found": None if how is None else name in seen.get(folder, ())})
    return out, how


def _free_gb(path):
    try:
        return round(shutil.disk_usage(path).free / 1024 ** 3, 1)
    except OSError:
        return None


def report(comfy_ok):
    """Everything the System page shows, with a verdict per row."""
    hw = _hardware()
    g, ram = hw["gpu"], hw["ram_gb"]
    est = engine.state(comfy_ok)
    mods, how = models(comfy_ok)
    req = [m for m in mods if not m["optional"]]
    missing = [m for m in req if m["found"] is False]
    missing_gb = round(sum(m["gb"] for m in missing), 1)
    inst = engine.find()
    dirs = engine.model_dirs(inst)
    disk_path = dirs[0] if dirs else pipeline.PROJECT_OUT
    free = _free_gb(disk_path)
    rows = []

    if not g:
        rows.append(("gpu", "bad", "No NVIDIA GPU found", "The AI models need an NVIDIA graphics card."))
    else:
        v = g["vram_gb"]
        label = "%s · %g GB" % (g["name"].replace("NVIDIA GeForce ", ""), round(v))
        if v >= 15.5:
            rows.append(("gpu", "ok", label, "Same as the tested PC."))
        elif v >= 11.5:
            rows.append(("gpu", "warn", label, "Should work. Renders take longer."))
        elif v >= 7.5:
            rows.append(("gpu", "warn", label, "Drafts should work. High quality may run out of memory."))
        else:
            rows.append(("gpu", "bad", label, "Too little graphics memory. About 12 GB is the minimum."))

    if ram is None:
        rows.append(("ram", "unknown", "Unknown", ""))
    elif ram >= 47:
        rows.append(("ram", "ok", "%g GB" % round(ram), "Plenty."))
    elif ram >= 31:
        rows.append(("ram", "ok", "%g GB" % round(ram), "Enough. Close other big programs while animating."))
    elif ram >= 23:
        rows.append(("ram", "warn", "%g GB" % round(ram), "Animating will be slow. 32 GB or more is recommended."))
    else:
        rows.append(("ram", "bad", "%g GB" % round(ram), "Not enough to animate. It needs 32 GB."))

    if how is None:
        rows.append(("models", "unknown", "Can't check yet", "Shown once the engine is found."))
    elif missing:
        rows.append(("models", "bad", "%d of %d missing" % (len(missing), len(req)),
                     "Missing files: about %g GB to download." % missing_gb))
    else:
        rows.append(("models", "ok", "All %d found" % len(req), ""))

    if free is None:
        rows.append(("disk", "unknown", "Unknown", ""))
    elif missing and free < missing_gb + 10:
        rows.append(("disk", "bad", "%g GB free" % free,
                     "Not enough room for the missing models (about %g GB)." % missing_gb))
    elif free < 20:
        rows.append(("disk", "warn", "%g GB free" % free, "Low. Renders need a few GB each."))
    else:
        rows.append(("disk", "ok", "%g GB free" % free, ""))

    rows.append(("engine", {"ready": "ok", "starting": "warn", "off": "warn",
                            "failed": "bad", "missing": "bad"}[est],
                 {"ready": "Running", "starting": "Starting…", "off": "Off",
                  "failed": "Couldn't start", "missing": "ComfyUI not found"}[est],
                 {"ready": "", "starting": "The first start takes about a minute.",
                  "off": "", "failed": "See the log below.",
                  "missing": "Install Comfy Desktop, or choose your ComfyUI folder."}[est]))

    worst = ("bad" if any(r[1] == "bad" for r in rows)
             else "warn" if any(r[1] == "warn" and r[0] != "engine" for r in rows) else "ok")
    return {
        "verdict": worst,
        "rows": [{"id": i, "status": s, "value": v, "note": n} for i, s, v, n in rows],
        "models": mods,
        "engine": {"state": est, "managed": engine.managed(),
                   "autostart": engine.autostart(),
                   "install": inst and {"kind": inst.get("kind"),
                                        "folder": inst.get("folder") or os.path.dirname(inst["main"])},
                   "log": engine.log_tail() if est in ("failed", "starting") else ""},
        "gpu": g, "ram_gb": ram, "free_gb": free,
    }
