"""Find, start and stop the ComfyUI engine for the studio.

The studio drives ComfyUI over HTTP. Most people already have ComfyUI - from
Comfy Desktop, the Windows portable build or a git checkout - so this module
finds that install and runs it headless (no window, no browser tab): starting
the studio is the only thing a user ever has to do.

An engine the studio started is tied to the studio's lifetime (on Windows a
job object kills it when the studio exits, however it exits), so closing the
studio never leaves 40 GB of model weights sitting in RAM. A ComfyUI that was
already running is someone else's and is left alone.
"""
import glob
import json
import os
import subprocess
import threading
import time
from urllib.parse import urlparse

from . import pipeline, store

SETTINGS = os.path.join(store.PROJECT_ROOT, "studio_settings.json")
LOG = os.path.join(pipeline.PROJECT_OUT, "_engine.log")

_LOCK = threading.Lock()
_PROC = None          # the ComfyUI process WE started (None = not ours)
_JOB = None           # Windows job handle that kills _PROC with us
_FOUND = {"t": 0.0, "inst": None}

# launch flags worth carrying over from however ComfyUI was started before;
# everything else (ports, telemetry, manager) is the launcher's business
_KEEP = ("--extra-model-paths-config", "--input-directory", "--output-directory",
         "--base-directory", "--user-directory", "--temp-directory")


# --------------------------------------------------------------- settings
def load_settings():
    try:
        with open(SETTINGS, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_settings(**changes):
    s = load_settings()
    s.update(changes)
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=2)
    return s


def autostart():
    return load_settings().get("autostart", True)


# ---------------------------------------------------------------- finding
def _valid(inst):
    return bool(inst and os.path.isfile(inst.get("python", ""))
                and os.path.isfile(inst.get("main", "")))


def _desktop_installs():
    """Comfy Desktop lists its installs, and the model folders it gives each
    one, under %APPDATA%\\Comfy Desktop."""
    root = os.path.join(os.environ.get("APPDATA", ""), "Comfy Desktop")
    try:
        with open(os.path.join(root, "installations.json"), encoding="utf-8") as fh:
            insts = json.load(fh)
    except (OSError, ValueError):
        return []
    out = []
    for inst in insts if isinstance(insts, list) else []:
        base = inst.get("installPath") or ""
        cand = {"kind": "Comfy Desktop", "folder": base,
                "python": os.path.join(base, "ComfyUI", ".venv", "Scripts", "python.exe"),
                "main": os.path.join(base, "ComfyUI", "main.py"), "args": []}
        paths = os.path.join(root, "instance-model-paths", "%s.yaml" % inst.get("id"))
        if os.path.isfile(paths):
            cand["args"] += ["--extra-model-paths-config", paths]
        if _valid(cand):
            out.append(cand)
    return out


def _legacy_desktop():
    """The older Electron "ComfyUI Desktop" app."""
    try:
        with open(os.path.join(os.environ.get("APPDATA", ""), "ComfyUI",
                               "config.json"), encoding="utf-8") as fh:
            base = json.load(fh).get("basePath") or ""
    except (OSError, ValueError):
        return None
    local = os.environ.get("LOCALAPPDATA", "")
    for main in (os.path.join(local, "Programs", "@comfyorgcomfyui-electron",
                              "resources", "ComfyUI", "main.py"),
                 os.path.join(local, "Programs", "ComfyUI", "resources",
                              "ComfyUI", "main.py")):
        cand = {"kind": "ComfyUI Desktop", "folder": base, "main": main,
                "python": os.path.join(base, ".venv", "Scripts", "python.exe"),
                "args": ["--base-directory", base]}
        yaml = os.path.join(os.environ.get("APPDATA", ""), "ComfyUI",
                            "extra_models_config.yaml")
        if os.path.isfile(yaml):
            cand["args"] += ["--extra-model-paths-config", yaml]
        if _valid(cand):
            return cand
    return None


def folder_install(folder):
    """A portable build (python_embeded next to ComfyUI\\) or a checkout with
    its own venv, given the folder a user would point at."""
    folder = os.path.abspath(os.path.expanduser(folder or ""))
    for main in (os.path.join(folder, "ComfyUI", "main.py"),
                 os.path.join(folder, "main.py")):
        if not os.path.isfile(main):
            continue
        code = os.path.dirname(main)
        for py in (os.path.join(folder, "python_embeded", "python.exe"),
                   os.path.join(code, ".venv", "Scripts", "python.exe"),
                   os.path.join(code, "venv", "Scripts", "python.exe"),
                   os.path.join(code, ".venv", "bin", "python"),
                   os.path.join(code, "venv", "bin", "python")):
            if os.path.isfile(py):
                return {"kind": "ComfyUI portable" if "python_embeded" in py
                        else "ComfyUI", "folder": folder, "python": py,
                        "main": main, "args": []}
    return None


def _search_folders():
    home = os.path.expanduser("~")
    roots = [home] + [os.path.join(home, d) for d in ("Desktop", "Downloads",
                                                      "Documents")]
    if os.name == "nt":
        roots += ["%s:\\" % c for c in "CDEFGH" if os.path.isdir("%s:\\" % c)]
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "ComfyUI*"))):
            inst = folder_install(d)
            if inst:
                return inst
    return None


def _split_cmdline(cmd):
    if os.name != "nt":
        import shlex
        return shlex.split(cmd)
    import ctypes
    from ctypes import wintypes
    f = ctypes.windll.shell32.CommandLineToArgvW
    f.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    f.restype = ctypes.POINTER(wintypes.LPWSTR)
    n = ctypes.c_int()
    argv = f(cmd, ctypes.byref(n))
    try:
        return [argv[i] for i in range(n.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv)


def recipe_from_running():
    """How the ComfyUI running right now was started - the most reliable
    recipe there is, whatever installed it. Windows only."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'main\\.py' } | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # several can run at once (a second install, a test copy): take the one
    # serving the port the studio talks to
    want = _local_port()
    argv = None
    for line in out.splitlines():
        cand = _split_cmdline(line.strip()) if line.strip() else []
        if not cand:
            continue
        port = cand[cand.index("--port") + 1] if "--port" in cand[:-1] else "8188"
        if str(want) == port:
            argv = cand
            break
    if not argv:
        return None
    py = argv[0]
    mi = next((i for i, a in enumerate(argv) if a.endswith("main.py")), None)
    if mi is None:
        return None
    main = argv[mi]
    if not os.path.isabs(main):
        # the process's working folder isn't visible, but a venv python
        # lives a few folders below the install, so walk up from it
        d = os.path.dirname(py)
        for _ in range(5):
            if os.path.isfile(os.path.join(d, main)):
                main = os.path.join(d, main)
                break
            d = os.path.dirname(d)
    args, rest = [], argv[mi + 1:]
    for i, a in enumerate(rest):
        if a in _KEEP and i + 1 < len(rest):
            args += [a, rest[i + 1]]
    inst = {"kind": "ComfyUI", "folder": os.path.dirname(main), "python": py,
            "main": main, "args": args}
    return inst if _valid(inst) else None


def find(refresh=False):
    """The ComfyUI install to start, best source first; cached a minute."""
    now = time.time()
    if not refresh and now - _FOUND["t"] < 60:
        return _FOUND["inst"]
    inst = load_settings().get("engine")
    if not _valid(inst):
        inst = (next(iter(_desktop_installs()), None) or _legacy_desktop()
                or _search_folders())
    _FOUND.update(t=now, inst=inst)
    return inst


def remember(inst):
    """Use this install from now on (a folder the user chose, or the recipe
    of a ComfyUI we saw running)."""
    save_settings(engine=inst)
    _FOUND.update(t=time.time(), inst=inst)


def pick_folder():
    """Windows' own folder picker, opened on the user's desktop (a web page
    can't hand the server a folder path). Returns the path or ""."""
    if os.name != "nt":
        return ""
    script = ("Add-Type -AssemblyName System.Windows.Forms;"
              "$f=New-Object System.Windows.Forms.FolderBrowserDialog;"
              "$f.Description='Choose your ComfyUI folder';"
              "$o=New-Object System.Windows.Forms.Form -Property @{TopMost=$true};"
              "if($f.ShowDialog($o) -eq 'OK'){$f.SelectedPath}")
    try:
        return subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script],
                              capture_output=True, text=True, timeout=600,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                              ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def model_dirs(inst=None):
    """Folders ComfyUI reads models from, for checking files while it is off:
    its own models\\ plus every base_path in an extra-model-paths file."""
    inst = inst or find()
    if not inst:
        return []
    # configured folders first: the default one is where new models go, so
    # it is the disk whose free space matters
    dirs = []
    a = inst.get("args") or []
    yamls = [a[i + 1] for i, x in enumerate(a[:-1]) if x == "--extra-model-paths-config"]
    yamls.append(os.path.join(os.path.dirname(inst["main"]), "extra_model_paths.yaml"))
    for y in yamls:
        try:
            with open(y, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if s.startswith("base_path:") and "..." not in s:
                        dirs.append(s.split(":", 1)[1].strip().strip("'\""))
        except OSError:
            pass
    dirs.append(os.path.join(os.path.dirname(inst["main"]), "models"))
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


# ------------------------------------------------------------ start / stop
def _local_port():
    """Only a ComfyUI on this machine can be started from here."""
    u = urlparse("http://" + pipeline.COMFY_HOST)
    if u.hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    return u.port or 8188


def _tie_to_us(proc):
    """Put a suspended process into a job that dies with this one, then let it
    run. Being in the job from its first instruction matters: a venv's
    python.exe immediately launches the real interpreter as a child, and only
    a child born inside the job is caught by it."""
    import ctypes
    from ctypes import wintypes
    k32, ntdll = ctypes.windll.kernel32, ctypes.windll.ntdll
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic),
                    ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    handle = int(proc._handle)
    job = None
    try:
        job = k32.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000   # KILL_ON_JOB_CLOSE
        if not (job and k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                                    ctypes.sizeof(info))
                and k32.AssignProcessToJobObject(job, handle)):
            job = None
    finally:
        ntdll.NtResumeProcess(handle)      # never leave it frozen
    return job


def start(inst=None):
    """Start ComfyUI headless. Returns an error string, or None."""
    global _PROC, _JOB
    port = _local_port()
    if port is None:
        return "ComfyUI is set to another computer (%s)" % pipeline.COMFY_HOST
    inst = inst or find()
    if not _valid(inst):
        return "ComfyUI wasn't found on this computer"
    with _LOCK:
        if _PROC and _PROC.poll() is None:
            return None
        args = [inst["python"], "-s", inst["main"], "--listen", "127.0.0.1",
                "--port", str(port), "--disable-auto-launch"] + list(inst.get("args") or [])
        # keep fetching renders straight off disk, as before
        if "--output-directory" not in args and os.path.isdir(pipeline.COMFY_OUTPUT_DIR):
            args += ["--output-directory", pipeline.COMFY_OUTPUT_DIR]
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        log = open(LOG, "ab")
        log.write(("\n==== %s  starting %s\n%s\n" % (
            time.strftime("%Y-%m-%d %H:%M:%S"), inst.get("kind", "ComfyUI"),
            subprocess.list2cmdline(args))).encode("utf-8"))
        log.flush()
        win = os.name == "nt"
        flags = (0x08000000 | 0x00000004) if win else 0   # NO_WINDOW | SUSPENDED
        try:
            proc = subprocess.Popen(args, cwd=os.path.dirname(inst["main"]),
                                    stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            log.close()
            return "couldn't start ComfyUI: %s" % exc
        if win:
            try:
                _JOB = _tie_to_us(proc)
            except Exception as exc:          # still running, just not tied
                print("engine: job object failed:", exc, flush=True)
        _PROC = proc
    print("engine: started %s (pid %d)" % (inst.get("kind"), proc.pid), flush=True)
    return None


def stop():
    """Stop the engine - only one this studio started."""
    global _PROC
    with _LOCK:
        proc, _PROC = _PROC, None
    if not proc or proc.poll() is not None:
        return False
    if os.name == "nt":       # the whole tree: the venv launcher + interpreter
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    return True


def managed():
    return bool(_PROC and _PROC.poll() is None)


def state(comfy_ok):
    """ready | starting | failed | off | missing - one word for the UI."""
    if comfy_ok:
        return "ready"
    if _PROC is not None:
        return "starting" if _PROC.poll() is None else "failed"
    return "off" if find() else "missing"


def log_tail(lines=25):
    try:
        with open(LOG, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16000))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def boot(is_up):
    """On studio start: learn from a ComfyUI that is already running, or
    start one. Runs in a background thread - finding can take a few seconds."""
    if is_up():
        if not _valid(load_settings().get("engine")):
            inst = recipe_from_running()
            if inst:
                remember(inst)
        return
    if autostart():
        err = start()
        if err:
            print("engine:", err, flush=True)
