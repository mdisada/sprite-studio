"""Persistent sprite/character spec store (characters.json at project root).

The JSON store is the single source of truth shared by the CLI and the web UI.
On first load it is seeded with the built-in examples from characters.py.
Spec shape:
  {
    "display_name": "Stormshield Arcanist",
    "ref_prompt": "...",                       # t2i prompt for the reference
    "animations": {
      "idle": {"prompt": "...", "seconds": 5.2},
      ...
    }
  }
"""
import json
import os
import re
import threading
import time

from .characters import CHARACTERS as _BUILTINS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_PATH = os.path.join(PROJECT_ROOT, "characters.json")
PROJECTS_PATH = os.path.join(PROJECT_ROOT, "projects.json")
_LOCK = threading.Lock()


def sanitize_id(raw):
    s = re.sub(r"[^a-z0-9_-]+", "_", (raw or "").strip().lower()).strip("_")
    if not s:
        raise ValueError("empty sprite id")
    return s


def load():
    with _LOCK:
        if not os.path.exists(STORE_PATH):
            _write(dict(_BUILTINS))
        with open(STORE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)


def _write(chars):
    tmp = STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(chars, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, STORE_PATH)


def get(char_id):
    chars = load()
    if char_id not in chars:
        raise KeyError("unknown sprite '%s' (have: %s)" % (char_id, ", ".join(sorted(chars))))
    return chars[char_id]


def upsert(char_id, spec):
    char_id = sanitize_id(char_id)
    with _LOCK:
        with open(STORE_PATH, "r", encoding="utf-8") as fh:
            chars = json.load(fh)
        chars[char_id] = spec
        _write(chars)
    return char_id


def delete(char_id):
    with _LOCK:
        with open(STORE_PATH, "r", encoding="utf-8") as fh:
            chars = json.load(fh)
        chars.pop(char_id, None)
        _write(chars)


# ------------------------------------------------------------------- projects
# A project groups sprites and fixes their shared art direction: style
# (pixel / painted), pixel density for pixel projects, and the game type
# (which determines the default movement kit). Sprites carry a "project"
# field pointing at their project id.

def _write_projects(projects):
    tmp = PROJECTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(projects, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, PROJECTS_PATH)


def load_projects():
    """Load projects.json, creating it (and adopting orphan sprites) on
    first run so pre-project installs upgrade in place."""
    with _LOCK:
        if not os.path.exists(PROJECTS_PATH):
            _write_projects({
                "spirtes": {
                    "display_name": "Spirtes (painted)",
                    "style": "painted",
                    "game_type": "side_arena",
                    "created": time.time(),
                },
            })
        with open(PROJECTS_PATH, "r", encoding="utf-8") as fh:
            projects = json.load(fh)
    # adopt sprites that predate projects into the first project
    chars = load()
    default_pid = next(iter(projects), None)
    orphans = [cid for cid, spec in chars.items() if not spec.get("project")]
    if default_pid and orphans:
        for cid in orphans:
            chars[cid]["project"] = default_pid
            upsert(cid, chars[cid])
    return projects


def upsert_project(pid, proj):
    pid = sanitize_id(pid)
    with _LOCK:
        projects = {}
        if os.path.exists(PROJECTS_PATH):
            with open(PROJECTS_PATH, "r", encoding="utf-8") as fh:
                projects = json.load(fh)
        projects[pid] = proj
        _write_projects(projects)
    return pid


def delete_project(pid):
    """Remove a project. Refuses while sprites still point at it."""
    used = [cid for cid, spec in load().items() if spec.get("project") == pid]
    if used:
        raise ValueError("project still contains sprites: %s" % ", ".join(sorted(used)))
    with _LOCK:
        if not os.path.exists(PROJECTS_PATH):
            return
        with open(PROJECTS_PATH, "r", encoding="utf-8") as fh:
            projects = json.load(fh)
        projects.pop(pid, None)
        _write_projects(projects)
