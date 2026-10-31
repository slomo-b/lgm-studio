"""Splatforge — lokale Web-App: Bild -> 3D-Gaussian-Splat.

Backend ohne Web-Framework (nur Standardbibliothek). Zwei Renderer:

* Browser:  WebGL2-Viewer (web/splat-viewer.js) fuer freies Drehen
* Server:   aceone `splat-render.exe` (wgpu) fuer Thumbnails und PNG-Export

    .venv\\Scripts\\python.exe webui.py     ->  http://127.0.0.1:7860
"""

import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
LGM = ROOT / "LGM"

# LGM-Module importierbar machen und den HF-Cache vor dem torch-Import setzen
sys.path.insert(0, str(LGM))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
INFER = LGM / "infer.py"
CKPT = LGM / "pretrained" / "model_fp16_fixrot.safetensors"
WORK = LGM / "workspace_test"
WEB = ROOT / "web"
STATE = ROOT / ".webui"
UPLOADS = STATE / "uploads"
THUMBS = STATE / "thumbs"
RENDERS = STATE / "renders"
CANDIDATES = STATE / "candidates"

HOST = "127.0.0.1"
PORT = 7860
MAX_UPLOAD = 25 * 1024 * 1024
ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# aceone-Renderer (wgpu). Ueberschreibbar per Umgebungsvariable.
ENGINE = Path(os.environ.get(
    "SPLAT_RENDER",
    r"C:\Users\Mo\Documents\aceone\target\release\splat-render.exe",
))

JOB = {
    "id": None, "state": "idle", "stage": 0, "progress": 0.0,
    "message": "", "msg_key": None, "msg_params": {},
    "elapsed": 0.0, "name": None, "download": None, "ply": None,
    "total": 20, "count": 1, "cand": 0, "stem": None, "candidates": [],
    "error": None, "error_key": None, "error_detail": None,
}
JOB_LOCK = threading.Lock()
ENGINE_INFO = {"available": ENGINE.is_file(), "path": str(ENGINE), "adapter": None}
ASSET_CACHE = {}


def set_job(**kw):
    with JOB_LOCK:
        JOB.update(kw)


# ---------------------------------------------------------------- PLY-Metadaten

def ply_meta(path: Path):
    """Anzahl Splats und Bounding-Box aus einem 3DGS-PLY lesen (gecacht)."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    cached = ASSET_CACHE.get(key)
    if cached and cached["mtime"] == st.st_mtime:
        return cached

    try:
        import numpy as np
    except ImportError:
        np = None

    with path.open("rb") as fh:
        head = fh.read(65536)
        end = head.find(b"end_header")
        if end < 0:
            return None
        end = head.find(b"\n", end) + 1
        text = head[:end].decode("latin1")

    count = 0
    props = []
    for line in text.splitlines():
        t = line.strip()
        if t.startswith("element vertex"):
            count = int(t.split()[2])
        elif t.startswith("property"):
            p = t.split()
            props.append((p[-1], p[1]))

    bbox = None
    extent = 1.0
    if np is not None and count:
        names = {"x": 0, "y": 1, "z": 2}
        types = {"float": "<f4", "double": "<f8"}
        fmt = []
        for name, typ in props:
            fmt.append(types.get(typ, "<f4"))
        try:
            dt = np.dtype([(n, t) for (n, _), t in zip(props, fmt)])
            arr = np.fromfile(path, dtype=dt, count=count, offset=end)
            xs, ys, zs = arr["x"], arr["y"], arr["z"]
            bbox = [float(xs.min()), float(ys.min()), float(zs.min()),
                    float(xs.max()), float(ys.max()), float(zs.max())]
            extent = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
        except Exception:
            bbox = None

    meta = {
        "mtime": st.st_mtime,
        "bytes": st.st_size,
        "count": count,
        "created": st.st_mtime,
        "extent": extent,
        "bbox": bbox,
    }
    ASSET_CACHE[key] = meta
    return meta


def pretty_name(stem: str) -> str:
    """Aus Dateinamen einen lesbaren Titel machen."""
    s = re.sub(r"_[0-9a-f]{6}$", "", stem)          # Job-Suffix entfernen
    s = re.sub(r"[_\-\s]*(rgba|rgb|alpha|transparent)$", "", s, flags=re.I)
    s = re.sub(r"[_\-]+", " ", s).strip()
    if not s:
        return stem
    return s[0].upper() + s[1:]


def scan_assets():
    if not WORK.is_dir():
        return []
    out = []
    for ply in sorted(WORK.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = ply_meta(ply)
        if not meta:
            continue
        out.append({
            "id": ply.stem,
            "name": pretty_name(ply.stem),
            "gaussians": meta["count"],
            "bytes": meta["bytes"],
            "created": meta["created"],
            "extent": meta["extent"],
            "bbox": meta["bbox"],
            "ply": f"/ply/{ply.stem}.ply",
            "thumb": f"/thumb/{ply.stem}.png",
        })
    return out


# ---------------------------------------------------------------- aceone

def engine_render(src: Path, dst: Path, *, width=512, height=512, yaw=200.0, pitch=8.0,
                  distance=None, timeout=240):
    """Ein Bild mit der aceone-Engine rendern. Gibt Metadaten zurueck oder None."""
    if not ENGINE.is_file():
        return None
    if distance is None:
        meta = ply_meta(src)
        extent = (meta or {}).get("extent") or 1.0
        distance = max(0.6, 1.0 * extent)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ENGINE), "image", str(src), str(dst),
        "--width", str(width), "--height", str(height),
        "--yaw", f"{yaw:.3f}", "--pitch", f"{pitch:.3f}", "--distance", f"{distance:.4f}",
        "--json",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0 or not dst.is_file():
        return None
    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError:
        info = {}
    ad = info.get("adapter") or {}
    if ad.get("name"):
        ENGINE_INFO["adapter"] = f"{ad.get('name')} · {ad.get('backend')}"
    return info or {"ok": True}


def ensure_thumb(asset) -> Path:
    ply = WORK / f"{asset['id']}.ply"
    thumb = THUMBS / f"{asset['id']}.png"
    meta = ply_meta(ply)
    if thumb.is_file() and meta and thumb.stat().st_mtime >= meta["mtime"]:
        return thumb
    ok = engine_render(ply, thumb, width=512, height=512)
    if ok:
        return thumb
    # Fallback: eine der Multi-View-Ansichten
    mv = WORK / "mv" / f"{asset['id']}_view0.png"
    if mv.is_file():
        return mv
    return None


# ---------------------------------------------------------------- Inferenz

RUNNER = None
RUNNER_LOCK = threading.Lock()
# Modelle zwischen Auftraegen im Speicher halten (spart ~26 s pro Bild).
# Abschaltbar mit SPLAT_WORKER=0 -> dann laedt jeder Auftrag neu.
RESIDENT = os.environ.get("SPLAT_WORKER", "1") != "0"


def get_runner():
    """Modelle einmal laden und im Prozess halten (~7 GB)."""
    global RUNNER
    with RUNNER_LOCK:
        if RUNNER is None:
            from core.options import config_defaults
            from core.pipeline import LgmPipeline

            RUNNER = LgmPipeline(config_defaults["big"], str(CKPT))
        return RUNNER


def drop_runner():
    """Modelle freigeben (z. B. nach einem Fehler oder auf Wunsch der UI)."""
    global RUNNER
    with RUNNER_LOCK:
        RUNNER = None
    import gc

    gc.collect()
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        pass


def candidate_thumb(ply: Path, job_id: str, index: int):
    """Thumbnail eines Kandidaten über aceone rendern (Fallback: Multi-View)."""
    out = THUMBS / f"cand_{job_id}_{index}.png"
    if not engine_render(ply, out, width=512, height=512):
        mv = ply.parent / "mv" / f"{ply.stem}_view0.png"
        if mv.is_file():
            shutil.copyfile(mv, out)
        else:
            return None
    return out


def run_inference(job_id: str, image: Path, steps: int = 16, count: int = 1) -> bool:
    t0 = time.time()
    stem = image.stem
    set_job(state="running", stage=0, progress=0.02, message="Lade Modelle…",
            msg_key="job.loading", msg_params={}, started=t0, total=steps,
            count=count, cand=0, stem=stem, candidates=[])

    # Bei genau einer Variante direkt in die Bibliothek schreiben -- sonst
    # landet das Modell in einem Kandidatenordner und ist nirgends sichtbar.
    if count == 1:
        target_dir = WORK
        names = [stem]
    else:
        target_dir = CANDIDATES / job_id
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "meta.json").write_text(
            json.dumps({"stem": stem, "count": count}), encoding="utf-8")
        names = [f"c{i}" for i in range(count)]

    def on_progress(stage, frac, key, params=None):
        set_job(state="running", stage=stage, progress=frac, msg_key=key,
                msg_params=params or {}, message=key, elapsed=time.time() - t0)

    try:
        runner = get_runner()
        runner.set_progress(on_progress)
        for i in range(count):
            if count > 1:
                set_job(cand=i, msg_key="job.candidate", msg_params={"i": i + 1, "n": count})
            runner.generate(str(image), workspace=str(target_dir), steps=steps,
                            seed=(1000 + i) if count > 1 else None, name=names[i])
    except Exception as exc:  # noqa: BLE001
        set_job(state="error", error_key="err.lgm",
                error_detail=f"{type(exc).__name__}: {exc}", elapsed=time.time() - t0)
        drop_runner()
        if count > 1:
            shutil.rmtree(target_dir, ignore_errors=True)
        return False

    if count == 1:
        ply = WORK / f"{stem}.ply"
        if not ply.is_file():
            set_job(state="error", error_key="err.no_ply", elapsed=time.time() - t0)
            return False
        ASSET_CACHE.pop(str(ply), None)
        set_job(stage=3, progress=0.94, message="Rendere Vorschau…",
                msg_key="job.preview", elapsed=time.time() - t0)
        ensure_thumb({"id": stem})
        set_job(state="done", stage=4, progress=1.0, message="Fertig",
                msg_key="job.done", elapsed=time.time() - t0, id=job_id, name=stem,
                download=f"{stem}.ply", ply=f"/ply/{stem}.ply", candidates=[])
        if not RESIDENT:
            drop_runner()
        return True

    # --- mehrere Varianten
    cands = []
    for i in range(count):
        ply = target_dir / f"c{i}.ply"
        if not ply.is_file():
            continue
        meta = ply_meta(ply) or {}
        thumb = candidate_thumb(ply, job_id, i)
        cands.append({
            "index": i,
            "ply": f"/candidate/{job_id}/{i}.ply",
            "thumb": f"/candthumb/{job_id}/{i}.png" if thumb else "",
            "gaussians": meta.get("count"),
            "bytes": meta.get("bytes"),
        })
    set_job(candidates=cands)

    if not cands:
        set_job(state="error", error_key="err.no_ply", elapsed=time.time() - t0)
        shutil.rmtree(target_dir, ignore_errors=True)
        return False

    set_job(state="done", stage=4, progress=1.0, message="Fertig", msg_key="job.done",
            elapsed=time.time() - t0, id=job_id, name=stem,
            download=f"{stem}.ply", ply=cands[0]["ply"])
    if not RESIDENT:
        drop_runner()
    return True


def start_job(job_id: str, image: Path, steps: int = 16, count: int = 1):
    set_job(id=job_id, state="running", stage=0, progress=0.0, message="Starte…",
            started=time.time(), elapsed=0.0, name=image.stem, total=steps, count=count,
            cand=0, candidates=[], download=f"{image.stem}.ply", error=None)

    def worker():
        try:
            run_inference(job_id, image, steps, count)
        except Exception as exc:  # noqa: BLE001
            set_job(state="error", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- HTTP

# Fehlerantworten: Key fuer die UI, englischer Text fuer CLI-Nutzer.
ERR = {
    "err.not_found": {"error_key": "err.not_found", "error": "Not found"},
    "err.unknown": {"error_key": "err.unknown", "error": "Unknown request"},
    "err.format": {"error_key": "err.format", "error": "Format not supported"},
    "err.empty": {"error_key": "err.empty", "error": "Empty file"},
    "err.too_big": {"error_key": "err.too_big", "error": "File too large (max. 25 MB)"},
    "err.upload": {"error_key": "err.upload", "error": "Upload incomplete"},
    "err.no_engine": {"error_key": "err.no_engine", "error": "aceone splat-render.exe not found"},
    "err.no_model": {"error_key": "err.no_model", "error": "Model not found"},
    "err.render": {"error_key": "err.render", "error": "Render failed"},
    "err.busy": {"error_key": "err.busy", "error": "A job is already running"},
    "err.delete_running": {"error_key": "err.delete_running", "error": "Model is being generated"},
}


class Handler(BaseHTTPRequestHandler):
    server_version = "Splatforge/2.0"

    def log_message(self, *_args):
        pass

    # -- helpers
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _stream(self, path: Path, ctype: str, cache="no-store", download_name=None):
        if not path.is_file():
            return self._json(404, ERR["err.not_found"])
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", cache)
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        try:
            with path.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile, 131072)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    # -- GET
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path

        if p in ("/", "/index.html"):
            return self._stream(WEB / "index.html", "text/html; charset=utf-8", "no-store")

        if p.startswith("/static/"):
            rel = p[len("/static/"):].replace("..", "")
            f = WEB / rel
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if f.suffix == ".js":
                ctype = "text/javascript; charset=utf-8"
            return self._stream(f, ctype, "no-store")

        if p == "/api/job":
            with JOB_LOCK:
                return self._json(200, dict(JOB))

        if p == "/api/assets":
            return self._json(200, {"assets": scan_assets(), "engine": ENGINE_INFO})

        if p.startswith("/ply/"):
            stem = Path(p[len("/ply/"):]).stem
            return self._stream(WORK / f"{stem}.ply", "application/octet-stream",
                                "max-age=600", f"{stem}.ply")

        if p.startswith("/thumb/"):
            stem = Path(p[len("/thumb/"):]).stem
            t = ensure_thumb({"id": stem})
            if not t:
                return self._json(404, ERR["err.not_found"])
            return self._stream(t, "image/png", "max-age=300")

        if p.startswith("/candidate/"):
            parts = p.strip("/").split("/")
            if len(parts) != 3:
                return self._json(400, {"error": "bad path"})
            try:
                idx = int(parts[2].replace(".ply", ""))
            except ValueError:
                return self._json(400, {"error": "bad index"})
            return self._stream(CANDIDATES / parts[1] / f"c{idx}.ply",
                                "application/octet-stream", "no-store")

        if p.startswith("/candthumb/"):
            parts = p.strip("/").split("/")
            if len(parts) != 3:
                return self._json(400, {"error": "bad path"})
            try:
                idx = int(parts[2].replace(".png", ""))
            except ValueError:
                return self._json(400, {"error": "bad index"})
            return self._stream(THUMBS / f"cand_{parts[1]}_{idx}.png", "image/png", "no-store")

        if p.startswith("/cache/"):
            rel = Path(p[len("/cache/"):].replace("..", ""))
            return self._stream(STATE / rel, "image/png", "max-age=600")

        return self._json(404, ERR["err.unknown"])

    # -- POST / DELETE
    def do_POST(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)

        if u.path == "/api/sample":
            name = (qs.get("name") or [""])[0]
            src = LGM / "data_test" / name
            if not src.is_file():
                return self._json(404, ERR["err.not_found"])
            return self._start(src, uuid.uuid4().hex[:12])

        if u.path == "/api/generate":
            raw_name = (qs.get("name") or ["bild.png"])[0]
            ext = Path(raw_name).suffix.lower()
            if ext not in ALLOWED:
                return self._json(400, {**ERR["err.format"], "error_params": {"ext": ext or "?"}})

            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self._json(400, ERR["err.empty"])
            if length > MAX_UPLOAD:
                return self._json(413, ERR["err.too_big"])

            data = b""
            while len(data) < length:
                chunk = self.rfile.read(min(1 << 16, length - len(data)))
                if not chunk:
                    break
                data += chunk
            if len(data) != length:
                return self._json(400, ERR["err.upload"])

            job_id = uuid.uuid4().hex[:12]
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(raw_name).stem)[:40] or "bild"
            UPLOADS.mkdir(parents=True, exist_ok=True)
            path = UPLOADS / f"{stem}_{job_id[:6]}{ext}"
            path.write_bytes(data)
            return self._start(path, job_id)

        if u.path == "/api/select":
            # gewaehlten Kandidaten in die Bibliothek uebernehmen
            job_id = (qs.get("job") or [""])[0]
            try:
                idx = int((qs.get("index") or ["0"])[0])
            except ValueError:
                idx = 0
            src = CANDIDATES / job_id / f"c{idx}.ply"
            if not src.is_file():
                return self._json(404, ERR["err.no_model"])
            stem = job_id
            meta_file = CANDIDATES / job_id / "meta.json"
            if meta_file.is_file():
                try:
                    stem = json.loads(meta_file.read_text(encoding="utf-8")).get("stem") or job_id
                except (OSError, ValueError):
                    stem = job_id
            dst = WORK / f"{stem}.ply"
            shutil.copyfile(src, dst)
            mv_src = src.parent / "mv"
            if mv_src.is_dir():
                for f in mv_src.glob(f"c{idx}_view*.png"):
                    shutil.copyfile(f, WORK / "mv" / f.name.replace(f"c{idx}_", f"{stem}_"))
            ASSET_CACHE.pop(str(dst), None)
            ensure_thumb({"id": stem})
            shutil.rmtree(CANDIDATES / job_id, ignore_errors=True)
            for i in range(8):
                (THUMBS / f"cand_{job_id}_{i}.png").unlink(missing_ok=True)
            return self._json(200, {"id": stem, "name": pretty_name(stem)})

        if u.path == "/api/discard":
            job_id = (qs.get("job") or [""])[0]
            shutil.rmtree(CANDIDATES / job_id, ignore_errors=True)
            for i in range(8):
                (THUMBS / f"cand_{job_id}_{i}.png").unlink(missing_ok=True)
            return self._json(200, {"discarded": True})

        if u.path == "/api/unload":
            drop_runner()
            return self._json(200, {"unloaded": True})

        if u.path == "/api/render":
            if not ENGINE.is_file():
                return self._json(503, ERR["err.no_engine"])
            stem = (qs.get("id") or [""])[0]
            ply = WORK / f"{stem}.ply"
            if not ply.is_file():
                return self._json(404, ERR["err.no_model"])

            def fnum(key, dflt):
                try:
                    return float((qs.get(key) or [dflt])[0])
                except (TypeError, ValueError):
                    return dflt

            width = int(min(4096, max(64, fnum("width", 1400))))
            height = int(min(4096, max(64, fnum("height", 1050))))
            yaw = fnum("yaw", 200.0)
            pitch = max(-89.0, min(89.0, fnum("pitch", 8.0)))
            meta = ply_meta(ply) or {}
            dist = fnum("distance", max(0.6, 1.0 * (meta.get("extent") or 1.0)))

            out = RENDERS / f"{stem}_{uuid.uuid4().hex[:8]}.png"
            info = engine_render(ply, out, width=width, height=height,
                                 yaw=yaw, pitch=pitch, distance=dist)
            if not info:
                return self._json(500, ERR["err.render"])
            return self._json(200, {"url": f"/cache/renders/{out.name}"})

        return self._json(404, ERR["err.unknown"])

    def _steps(self):
        try:
            n = int((parse_qs(urlparse(self.path).query).get("steps") or ["16"])[0])
        except (TypeError, ValueError):
            n = 16
        return max(4, min(60, n))

    def _count(self):
        try:
            n = int((parse_qs(urlparse(self.path).query).get("count") or ["1"])[0])
        except (TypeError, ValueError):
            n = 1
        return max(1, min(4, n))

    def _start(self, image: Path, job_id: str):
        if JOB.get("state") in ("running",):
            return self._json(409, {**ERR["err.busy"], "running": True})
        start_job(job_id, image, self._steps(), self._count())
        return self._json(200, {"job": job_id})

    def do_DELETE(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/assets/"):
            return self._json(404, ERR["err.unknown"])
        stem = Path(u.path[len("/api/assets/"):]).stem
        ply = WORK / f"{stem}.ply"
        if not ply.is_file():
            return self._json(404, ERR["err.not_found"])
        if JOB.get("state") == "running" and JOB.get("name") == stem:
            return self._json(409, ERR["err.delete_running"])
        try:
            ply.unlink()
        except OSError as exc:
            return self._json(500, {**ERR["err.unknown"], "error_detail": str(exc)})
        for f in (WORK / "mv").glob(f"{stem}_view*.png"):
            f.unlink(missing_ok=True)
        (THUMBS / f"{stem}.png").unlink(missing_ok=True)
        ASSET_CACHE.pop(str(ply), None)
        return self._json(200, {"deleted": stem})


def main():
    if not INFER.is_file():
        print(f"LGM nicht gefunden: {INFER}")
        return 1
    if not CKPT.is_file():
        print(f"Checkpoint fehlt: {CKPT}")
        return 1
    for d in (STATE, UPLOADS, THUMBS, RENDERS, CANDIDATES):
        d.mkdir(parents=True, exist_ok=True)

    if not ENGINE.is_file():
        print(f"[WARN] aceone-Engine nicht gefunden: {ENGINE}")
        print("       Thumbnails fallen auf die Multi-View-Bilder zurück.")
        print("       Pfad setzen: $env:SPLAT_RENDER = 'C:\\...\\splat-render.exe'")
    else:
        print(f"[INFO] Engine: {ENGINE}")

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"[INFO] Splatforge läuft: {url}")
    print("[INFO] Beenden mit Strg+C")
    try:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbeendet")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
