"""
Models for /cutout and /layers: download + checksum, ONNX sessions (CPU), the optional vision description.

    Matte      BiRefNet-lite            -> the subject's alpha (0..1)
    Depth      Depth Anything V2 Small  -> depth, 0 = far .. 1 = near
    Segmenter  SegFormer-B2 ADE20K      -> what every pixel is (150 classes) + confidence
    Inpainter  LaMa                     -> fills the hidden parts (fill=lama)

Everything runs locally on the CPU. The only network use is the one-time model download (Hugging Face).
Settings: config.json -> layers.models.

Self-check on a real photo (writes saliency.png, depth.png, classes.png; prints names, shapes, timings):

    cd src
    ..\\.venv\\Scripts\\python.exe -m photos.layer_models --selfcheck <photo> [--out DIR]
"""

import gc
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from core import config
from core.debuglog import log

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_VERIFIED: set = set()   # model files whose sha256 was checked in this process
_LOADED: dict = {}       # name -> model object
_FAKES: dict = {}        # tests: name -> fake model (never unloaded)


class ModelError(Exception):
    """A model could not be downloaded or loaded (the message is for the user)."""


# ---------------- models on disk ----------------

def repo_dir(entry: dict, lcfg: dict) -> Path:
    """Every repo gets its own folder: several of them name their file onnx/model.onnx."""
    return config.path(lcfg["models"]["dir"]) / entry["repo"].replace("/", "__")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(entry: dict, filename: str, lcfg: dict, progress, name: str) -> Path:
    d = repo_dir(entry, lcfg)
    path = d / filename
    if path.is_file():
        return path
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download
    if progress:
        progress("layers_model_download", name=name, size_mb=entry.get("size_mb", "?"))
    d.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(repo_id=entry["repo"], filename=filename, local_dir=d)
    except Exception as e:
        log("layers_failed", where="model_download", path=str(path), error=repr(e))
        raise ModelError(lcfg["texts"]["model_failed"].format(name=name, error=e)) from e
    return path


def ensure_model(entry: dict, lcfg: dict, progress=None, name: str = "") -> Path:
    """The model file (downloaded once from Hugging Face), sha256-checked. Raises ModelError."""
    t = time.monotonic()
    existed = (repo_dir(entry, lcfg) / entry["file"]).is_file()
    path = _download(entry, entry["file"], lcfg, progress, name or entry["file"])
    key = str(path)
    if key not in _VERIFIED and entry.get("sha256"):
        if _sha256(path) != entry["sha256"].lower():
            path.unlink(missing_ok=True)
            log("layers_models", model=name, path=key, downloaded=not existed, sha_ok=False)
            raise ModelError(lcfg["texts"]["model_bad_hash"].format(file=entry["file"]))
        _VERIFIED.add(key)
    log("layers_models", model=name or entry["file"], path=key, downloaded=not existed, sha_ok=True,
        load_s=round(time.monotonic() - t, 2))
    return path


def entry_for(name: str, lcfg: dict) -> dict:
    m = lcfg["models"]
    if name == "matte":
        return m["mattes"][m["matte"]]
    if name == "segmenter":
        return m["segmenters"][m["segmenter"]]
    return m[{"depth": "depth", "inpaint": "inpaint"}[name]]


# ---------------- sessions ----------------

def _session(path: Path, threads: int, name: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t = time.monotonic()
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    log("layers_models", model=name, path=str(path), load_s=round(time.monotonic() - t, 2),
        inputs=[(i.name, i.shape, i.type) for i in sess.get_inputs()],
        outputs=[(o.name, o.shape, o.type) for o in sess.get_outputs()])
    return sess


def _blob(rgb, w: int, h: int, interp) -> np.ndarray:
    x = cv2.resize(rgb, (w, h), interpolation=interp).astype(np.float32) / 255.0
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1)[None])


class Matte:
    """BiRefNet: predict(rgb uint8 HxWx3) -> float32 HxW 0..1 (1 = subject)."""

    def __init__(self, path: Path, entry: dict, threads: int):
        self.sess = _session(path, threads, "matte")
        self.inp = self.sess.get_inputs()[0].name
        self.size = int(entry.get("input", 1024))
        self.kind = entry.get("output", "logits")
        self.raw_range = None

    def predict(self, rgb) -> np.ndarray:
        h, w = rgb.shape[:2]
        out = self.sess.run(None, {self.inp: _blob(rgb, self.size, self.size, cv2.INTER_LINEAR)})[-1]
        out = np.asarray(out, dtype=np.float32).reshape(out.shape[-2], out.shape[-1])
        self.raw_range = (float(out.min()), float(out.max()))
        already_prob = self.raw_range[0] >= 0.0 and self.raw_range[1] <= 1.0
        if self.kind == "logits" and not already_prob:
            out = 1.0 / (1.0 + np.exp(-np.clip(out, -30, 30)))
        return np.clip(cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1).astype(np.float32)


class Depth:
    """Depth Anything V2: predict(rgb) -> float32 HxW, 0 = far .. 1 = near (normalized per photo)."""

    def __init__(self, path: Path, entry: dict, threads: int):
        self.sess = _session(path, threads, "depth")
        i = self.sess.get_inputs()[0]
        self.inp = i.name
        self.fixed = (i.shape[2], i.shape[3]) if all(isinstance(v, int) for v in i.shape[2:4]) else None
        self.short = int(entry.get("short_side", 518))
        self.mult = int(entry.get("multiple_of", 14))
        self.near_is_larger = bool(entry.get("near_is_larger", True))
        self.raw_range = None

    def predict(self, rgb) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.fixed:
            ih, iw = self.fixed
        else:
            s = self.short / min(h, w)
            ih = max(self.mult, int(round(h * s / self.mult)) * self.mult)
            iw = max(self.mult, int(round(w * s / self.mult)) * self.mult)
        out = self.sess.run(None, {self.inp: _blob(rgb, iw, ih, cv2.INTER_CUBIC)})[0]
        out = np.asarray(out, dtype=np.float32)
        out = out.reshape(out.shape[-2], out.shape[-1])
        self.raw_range = (float(out.min()), float(out.max()))
        d = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - lo) / max(float(hi - lo), 1e-6), 0, 1)
        return (d if self.near_is_larger else 1.0 - d).astype(np.float32)


class Segmenter:
    """SegFormer ADE20K: predict(rgb) -> (classes uint8 HxW, confidence float32 HxW); .labels = 150 names."""

    def __init__(self, path: Path, entry: dict, threads: int, labels: list):
        self.sess = _session(path, threads, "segmenter")
        self.inp = self.sess.get_inputs()[0].name
        self.size = int(entry.get("input", 512))
        self.labels = labels

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        logits = self.sess.run(None, {self.inp: _blob(rgb, self.size, self.size, cv2.INTER_LINEAR)})[0][0]
        s = min(1.0, self.size / max(h, w))        # upsample the logits to ~512 only (never to the work size)
        mh, mw = max(1, round(h * s)), max(1, round(w * s))
        best = np.full((mh, mw), -np.inf, dtype=np.float32)
        arg = np.zeros((mh, mw), dtype=np.uint8)
        total = np.zeros((mh, mw), dtype=np.float64)
        for c in range(logits.shape[0]):           # one channel at a time: a few MB instead of 100+
            up = cv2.resize(np.ascontiguousarray(logits[c], dtype=np.float32), (mw, mh), interpolation=cv2.INTER_LINEAR)
            total += np.exp(np.minimum(up, 80.0), dtype=np.float64)
            m = up > best
            best[m] = up[m]
            arg[m] = c
        conf = (np.exp(np.minimum(best, 80.0), dtype=np.float64) / np.maximum(total, 1e-12)).astype(np.float32)
        return (cv2.resize(arg, (w, h), interpolation=cv2.INTER_NEAREST),
                cv2.resize(conf, (w, h), interpolation=cv2.INTER_NEAREST))


class Inpainter:
    """LaMa: fill(rgb uint8 HxWx3, hole bool HxW) -> rgb uint8 (the whole picture; paste only the hole)."""

    def __init__(self, path: Path, entry: dict, threads: int):
        self.sess = _session(path, threads, "inpaint")
        self.size = int(entry.get("input", 512))
        ins = self.sess.get_inputs()
        mask = next((i for i in ins if "mask" in i.name.lower() or i.shape[1] == 1), ins[-1])
        self.mask = mask.name
        self.image = next(i.name for i in ins if i.name != mask.name)

    def fill(self, rgb, hole) -> np.ndarray:
        h, w = rgb.shape[:2]
        n = self.size
        img = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        m = (cv2.resize(hole.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        out = self.sess.run(None, {self.image: np.ascontiguousarray(img.transpose(2, 0, 1)[None]),
                                   self.mask: m[None, None]})[0][0]
        out = np.asarray(out, dtype=np.float32).transpose(1, 2, 0)
        if float(out.max()) <= 2.0:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


# ---------------- loading / cache ----------------

def read_labels(entry: dict, lcfg: dict, progress=None) -> list:
    path = _download(entry, entry.get("labels_file", "config.json"), lcfg, progress, "segmenter")
    id2label = json.loads(path.read_text(encoding="utf-8"))["id2label"]
    return [str(id2label[str(i)]).strip() for i in range(len(id2label))]   # the list has "bed " with a space


def install_fakes(fakes: dict | None):
    """Tests: use these objects instead of the real models (None / {} = back to the real ones)."""
    _FAKES.clear()
    _FAKES.update(fakes or {})


def get(name: str, cfg: dict, progress=None):
    """One model by name (matte | depth | segmenter | inpaint), loaded on first use."""
    if name in _FAKES:
        return _FAKES[name]
    if name in _LOADED:
        return _LOADED[name]
    lcfg = cfg["layers"]
    entry = entry_for(name, lcfg)
    path = ensure_model(entry, lcfg, progress, name)
    if progress:
        progress("layers_loading", name=name)
    try:
        if name == "matte":
            model = Matte(path, entry, lcfg["threads"])
        elif name == "depth":
            model = Depth(path, entry, lcfg["threads"])
        elif name == "segmenter":
            model = Segmenter(path, entry, lcfg["threads"], read_labels(entry, lcfg, progress))
        else:
            model = Inpainter(path, entry, lcfg["threads"])
    except ModelError:
        raise
    except Exception as e:
        log("layers_failed", where="model_load", model=name, error=repr(e))
        raise ModelError(lcfg["texts"]["model_failed"].format(name=name, error=e)) from e
    _LOADED[name] = model
    return model


def models(cfg: dict, progress=None, need=("matte", "depth", "segmenter")) -> dict:
    return {n: get(n, cfg, progress) for n in need}


def unload(name: str | None = None) -> None:
    """Free the ONNX sessions (one, or all)."""
    for n in ([name] if name else list(_LOADED)):
        _LOADED.pop(n, None)
    gc.collect()


def download_all(cfg: dict) -> None:
    """Download + check every model and run it once on a dummy image. Prints to stderr."""
    def say(key, **v):
        print("  " + cfg["ui"]["status"][key].format(**v), file=sys.stderr, flush=True)

    dummy = np.full((256, 320, 3), 127, dtype=np.uint8)
    for name in ("matte", "depth", "segmenter", "inpaint"):
        t = time.monotonic()
        m = get(name, cfg, say)
        if name == "inpaint":
            hole = np.zeros(dummy.shape[:2], bool)
            hole[100:150, 100:150] = True
            m.fill(dummy, hole)
        else:
            m.predict(dummy)
        print(f"{name}: ok ({time.monotonic() - t:.1f} s)", file=sys.stderr, flush=True)
        unload(name)


# ---------------- the optional description ----------------

_ANSWER = re.compile(r"Name:\s*(.+?)\s*Description:\s*(.+)", re.I | re.S)


def parse_description(answer: str, max_name_words: int) -> dict | None:
    m = _ANSWER.search(answer or "")
    if not m:
        return None
    name = m.group(1).strip(" .,:;-*\"'")
    desc = " ".join(m.group(2).split()).strip(" *\"'")
    if not desc:
        return None
    if not name or len(name.split()) > max_name_words:
        name = ""
    return {"name": name, "description": desc}


def describe_focus(rgb_full, bbox, label: str, cfg: dict, progress=None) -> dict | None:
    """One sentence about the object of focus from the local vision model (slow: 20-60 s on a small CPU).
    -> {"name", "description"} ('name' may be empty = keep the class label), or None if the answer is garbled.
    The vision model is always closed again (frees ~3 GB)."""
    from PIL import Image
    from media import video
    dc = cfg["layers"]["describe"]
    h, w = rgb_full.shape[:2]
    x, y, bw, bh = bbox
    mx, my = int(bw * dc["crop_margin_pct"] / 100), int(bh * dc["crop_margin_pct"] / 100)
    crop = rgb_full[max(0, y - my):min(h, y + bh + my), max(0, x - mx):min(w, x + bw + mx)]
    img = Image.fromarray(crop)
    img.thumbnail((dc["max_side"], dc["max_side"]))
    f = Path(tempfile.gettempdir()) / f"local_llm_layers_{os.getpid()}.jpg"
    img.save(f, "JPEG", quality=90)
    vc = dict(cfg["video"], vision=dict(cfg["video"]["vision"], max_tokens=dc["max_tokens"],
                                        temperature=dc["temperature"]))
    t = time.monotonic()
    vis = video.Vision(vc, (lambda msg: progress("layers_vision", msg=msg)) if progress else (lambda msg: None))
    try:
        if progress:
            progress("layers_describe", label=label)
        answer = vis.describe(f, config.text(dc["prompt"]).format(label=label))
    finally:
        vis.close()
        f.unlink(missing_ok=True)
    res = parse_description(answer, dc["max_name_words"])
    log("layers_describe", seconds=round(time.monotonic() - t, 1), answer=answer,
        name=(res or {}).get("name"), description=(res or {}).get("description"))
    return res


# ---------------- self-check ----------------

def _selfcheck(photo: str, out_dir: str | None):
    from PIL import Image, ImageOps
    cfg = config.load()
    lcfg = cfg["layers"]
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="layers_selfcheck_"))
    out.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    full = np.asarray(img)
    s = min(1.0, lcfg["work_max_side"] / max(full.shape[:2]))
    work = cv2.resize(full, (round(full.shape[1] * s), round(full.shape[0] * s)), interpolation=cv2.INTER_AREA)
    print(f"photo {full.shape[1]}x{full.shape[0]} -> work {work.shape[1]}x{work.shape[0]}  (decode {time.monotonic() - t:.2f} s)")

    def say(key, **v):
        print("  " + cfg["ui"]["status"][key].format(**v))

    for name in ("matte", "depth", "segmenter"):
        t = time.monotonic()
        m = get(name, cfg, say)
        load_s = time.monotonic() - t
        print(f"{name}: inputs {[(i.name, i.shape) for i in m.sess.get_inputs()]}")
        print(f"{name}: outputs {[(o.name, o.shape) for o in m.sess.get_outputs()]}")
        t = time.monotonic()
        res = m.predict(work)
        run_s = time.monotonic() - t
        if name == "matte":
            print(f"matte: raw output range {m.raw_range} -> {'logits' if m.raw_range[0] < 0 or m.raw_range[1] > 1 else 'prob'}"
                  f" (config says {m.kind}); subject covers {100 * float((res > 0.5).mean()):.1f}%")
            Image.fromarray((res * 255).astype(np.uint8)).save(out / "saliency.png")
            sal = res
        elif name == "depth":
            near = float(res[sal > 0.5].mean()) if (sal > 0.5).any() else float("nan")
            far = float(res[sal <= 0.5].mean())
            print(f"depth: raw range {m.raw_range}; mean depth of the subject {near:.2f} vs the rest {far:.2f} "
                  f"-> near = larger: {'YES' if near > far else 'NO - set near_is_larger false'}")
            Image.fromarray((res * 255).astype(np.uint8)).save(out / "depth.png")
        else:
            cls, conf = res
            ids, counts = np.unique(cls, return_counts=True)
            top = sorted(zip(counts, ids), reverse=True)[:8]
            print("classes: " + ", ".join(f"{m.labels[i]} {100 * c / cls.size:.0f}%" for c, i in top)
                  + f"; mean confidence {float(conf.mean()):.2f}")
            rng = np.random.default_rng(1)
            palette = rng.integers(40, 255, size=(256, 3), dtype=np.uint8)
            Image.fromarray(palette[cls]).save(out / "classes.png")
        print(f"{name}: load {load_s:.1f} s, run {run_s:.1f} s")
        unload(name)
    print(f"maps written to {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Self-check of the /layers models on a real photo")
    ap.add_argument("--selfcheck", metavar="PHOTO", required=True)
    ap.add_argument("--out", metavar="DIR")
    a = ap.parse_args()
    _selfcheck(a.selfcheck, a.out)
