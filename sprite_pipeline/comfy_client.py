"""Minimal ComfyUI HTTP API client.

Uses only `requests` (no websocket dependency): jobs are submitted to
POST /prompt and completion is detected by polling GET /history/<id>.
Output files are read straight off the ComfyUI output directory when it is
on the same machine, with GET /view as a fallback.
"""
import json
import os
import shutil
import time
import uuid

import requests


class ComfyError(RuntimeError):
    pass


class JobCancelled(BaseException):
    """Raised inside a running job when the user cancels it. A BaseException
    so no broad `except Exception` in a pipeline stage can swallow it."""


def _fmt_error(entry):
    """Extract a readable error out of a history entry's status messages."""
    lines = []
    for msg in entry.get("status", {}).get("messages", []):
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            continue
        kind, data = msg[0], msg[1]
        if kind == "execution_error" and isinstance(data, dict):
            lines.append("node %s (%s): %s: %s" % (
                data.get("node_id"), data.get("node_type"),
                data.get("exception_type"), data.get("exception_message")))
            for tb_line in (data.get("traceback") or [])[-4:]:
                lines.append("    " + tb_line.strip())
    return "\n".join(lines) or json.dumps(entry.get("status", {}))[:2000]


class ComfyClient:
    def __init__(self, host="127.0.0.1:8188", output_dir=None):
        self.base = "http://" + host
        self.client_id = str(uuid.uuid4())
        self.output_dir = output_dir
        # set by the UI worker per job: a callable that returns True once the
        # user pressed Cancel; checked before submitting and while waiting
        self.should_cancel = None
        self._submitted = []    # this job's prompt ids, so cancel removes only ours

    def _check_cancel(self):
        if not (self.should_cancel and self.should_cancel()):
            return
        mine = set(self._submitted)
        try:
            q = requests.get(self.base + "/queue", timeout=10).json()
            running = {item[1] for item in q.get("queue_running", [])}
            if mine:
                requests.post(self.base + "/queue",
                              json={"delete": sorted(mine)}, timeout=10)
            if running & mine:    # never interrupt someone else's prompt
                requests.post(self.base + "/interrupt", timeout=10)
        except requests.RequestException as exc:
            print("  (cancel cleanup failed: %s)" % exc, flush=True)
        raise JobCancelled("cancelled")

    # ------------------------------------------------------------- server info
    def stats(self):
        return requests.get(self.base + "/system_stats", timeout=10).json()

    def queue_state(self):
        q = requests.get(self.base + "/queue", timeout=10).json()
        return len(q.get("queue_running", [])), len(q.get("queue_pending", []))

    # ------------------------------------------------------------------- jobs
    def submit(self, graph):
        self._check_cancel()
        payload = {"prompt": graph, "client_id": self.client_id}
        r = requests.post(self.base + "/prompt", json=payload, timeout=60)
        if r.status_code != 200:
            try:
                detail = json.dumps(r.json(), indent=2)
            except Exception:
                detail = r.text
            raise ComfyError("submit failed (HTTP %s):\n%s" % (r.status_code, detail[:4000]))
        data = r.json()
        if data.get("node_errors"):
            raise ComfyError("validation errors:\n" + json.dumps(data["node_errors"], indent=2)[:4000])
        self._submitted.append(data["prompt_id"])
        return data["prompt_id"]

    def wait(self, prompt_id, timeout=3600, poll=2.0, label=""):
        t0 = time.time()
        last_report = 0.0
        while True:
            self._check_cancel()
            hist = requests.get(self.base + "/history/" + prompt_id, timeout=30).json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError("execution failed for %s:\n%s" % (label or prompt_id, _fmt_error(entry)))
                if status.get("completed"):
                    return entry
            elapsed = time.time() - t0
            if elapsed > timeout:
                raise ComfyError("timed out after %.0fs waiting for %s" % (elapsed, label or prompt_id))
            if elapsed - last_report >= 20:
                try:
                    running, pending = self.queue_state()
                    print("  ... %s running %.0fs (queue: %d active / %d pending)"
                          % (label, elapsed, running, pending), flush=True)
                except requests.RequestException:
                    pass
                last_report = elapsed
            time.sleep(poll)

    def run(self, graph, label="job", timeout=3600):
        pid = self.submit(graph)
        t0 = time.time()
        print("  queued %s (prompt_id %s)" % (label, pid), flush=True)
        entry = self.wait(pid, timeout=timeout, label=label)
        print("  done %s in %.1fs" % (label, time.time() - t0), flush=True)
        return entry

    # ---------------------------------------------------------------- outputs
    @staticmethod
    def output_files(entry, kind=None):
        """List output file records (filename/subfolder/type) from a history entry."""
        files = []
        for _node_id, node_out in sorted((entry.get("outputs") or {}).items()):
            for key in ("images", "videos", "gifs", "audio"):
                if kind is not None and key != kind:
                    continue
                for f in node_out.get(key) or []:
                    if f.get("type") == "output":
                        files.append(f)
        return files

    def fetch(self, f, dest_dir, dest_name=None):
        """Copy one output file into dest_dir; returns the local path."""
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, dest_name or f["filename"])
        if self.output_dir:
            src = os.path.join(self.output_dir, f.get("subfolder", ""), f["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dest)
                return dest
        r = requests.get(self.base + "/view", params={
            "filename": f["filename"],
            "subfolder": f.get("subfolder", ""),
            "type": f.get("type", "output"),
        }, timeout=300)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(r.content)
        return dest

    def fetch_all(self, entry, dest_dir, kind=None):
        return [self.fetch(f, dest_dir) for f in self.output_files(entry, kind=kind)]

    # ----------------------------------------------------------------- inputs
    def upload_image(self, path, name=None):
        """Upload an image into the ComfyUI input dir; returns the LoadImage name."""
        name = name or os.path.basename(path)
        with open(path, "rb") as fh:
            r = requests.post(self.base + "/upload/image",
                              files={"image": (name, fh, "image/png")},
                              data={"overwrite": "true"}, timeout=300)
        r.raise_for_status()
        d = r.json()
        sub = d.get("subfolder")
        return (sub + "/" + d["name"]) if sub else d["name"]

    # ----------------------------------------------------------------- memory
    def free(self, unload_models=True):
        """Ask ComfyUI to release memory (call between model-family switches)."""
        try:
            requests.post(self.base + "/free",
                          json={"unload_models": bool(unload_models), "free_memory": True},
                          timeout=30)
        except requests.RequestException as exc:
            print("  (free failed: %s)" % exc, flush=True)
