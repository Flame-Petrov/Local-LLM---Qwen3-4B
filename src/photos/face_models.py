"""
Face models for /faces: download + checksum, image loading, face detection, alignment, fingerprints, quality.

    YuNet (OpenCV FaceDetectorYN)   -> boxes + 5 face points (eyes, nose tip, mouth corners)
    ArcFace R50 (onnxruntime)        -> 512-number fingerprint (default, most accurate)
    SFace (OpenCV FaceRecognizerSF)  -> 128-number fingerprint (permissive license, faster)

Everything runs locally on the CPU. The only network use is the one-time model download (Hugging Face).
Settings: config.json -> faces.models / faces.detection / faces.quality.

Self-check on a real photo:

    cd src
    ..\\.venv\\Scripts\\python.exe -m photos.face_models <photo> [more photos...]

prints every face (box, score, size, sharpness, yaw, quality), the timings and the similarity between the
faces found, and saves the aligned 112x112 faces to the temp folder.
"""

import hashlib
import io
import os
import time
from pathlib import Path

import numpy as np

from core import config
from core.debuglog import log

# ArcFace's 5-point template for a 112x112 aligned face (InsightFace)
ARCFACE_5PT = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

ONLINE_ONLY = 0x400000 | 0x40000   # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS | FILE_ATTRIBUTE_RECALL_ON_OPEN

_MODELS: dict = {}       # (detector file, embedder name, threads) -> (Detector, Embedder)
_VERIFIED: set = set()   # model files whose sha256 was checked in this process


class ModelError(Exception):
    """A model could not be downloaded or is damaged (the message is for the user)."""


class ImageError(Exception):
    """A photo could not be read (the message is the reason shown in the report).
    kind: unreadable (corrupt, no access) | skipped (too small / too big) | unsupported (e.g. HEIC without pillow-heif)."""

    def __init__(self, reason: str, kind: str = "unreadable"):
        super().__init__(reason)
        self.kind = kind


# ---------------- models on disk ----------------

def model_dir(fcfg: dict) -> Path:
    return config.path(fcfg["models"]["dir"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(entry: dict, fcfg: dict, spin, name: str = "", hf_cfg: dict | None = None) -> Path:
    """The model file (downloaded once from Hugging Face), sha256-checked. Raises ModelError."""
    d = model_dir(fcfg)
    path = d / entry["file"]
    downloaded = False
    t = time.monotonic()
    if not path.is_file():
        if (hf_cfg or {}).get("disable_hf_telemetry", True):
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from huggingface_hub import hf_hub_download
        if spin:
            spin.status("faces_model_download", size_mb=entry.get("size_mb", "?"))
        d.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(repo_id=entry["repo"], filename=entry["file"], local_dir=d)
        except Exception as e:
            log("faces_error", where="model_download", path=str(path), error=repr(e))
            raise ModelError(f"could not download the face model {entry['repo']}/{entry['file']}: {e}") from e
        finally:
            if spin:
                spin.stop()
        downloaded = True
    key = str(path)
    sha_ok = True
    if key not in _VERIFIED and entry.get("sha256"):
        sha_ok = _sha256(path) == entry["sha256"].lower()
        if not sha_ok:
            path.unlink(missing_ok=True)
            log("faces_models", path=key, downloaded=downloaded, sha_ok=False)
            raise ModelError(fcfg["texts"]["model_bad_hash"].format(file=entry["file"]))
        _VERIFIED.add(key)
    log("faces_models", model=name or entry["file"], path=key, downloaded=downloaded, sha_ok=sha_ok,
        load_s=round(time.monotonic() - t, 2))
    return path


# ---------------- photos ----------------

def is_online_only(path, st=None) -> bool:
    """OneDrive (and other cloud) 'online-only' file: reading it would download it."""
    try:
        st = st or os.stat(path)
    except OSError:
        return False
    return bool(getattr(st, "st_file_attributes", 0) & ONLINE_ONLY)


def heic_supported() -> bool:
    try:
        import pillow_heif  # noqa: F401
        return True
    except ImportError:
        return False


_HEIF_REGISTERED = False


def load_image(path, fcfg: dict, data: bytes | None = None):
    """A photo -> (BGR uint8 array, info). Unicode paths work (bytes are read by Python, never cv2.imread),
    phone photos are turned upright (EXIF), big JPEGs are decoded at reduced size (draft).
    info: w, h (upright full size), scale (decoded / full), sha1, orientation, format. Raises ImageError."""
    global _HEIF_REGISTERED
    from PIL import Image, ImageOps, UnidentifiedImageError
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if ext in ("heic", "heif"):
        if not heic_supported():
            raise ImageError("HEIC needs the pillow-heif package", "unsupported")
        if not _HEIF_REGISTERED:
            import pillow_heif
            pillow_heif.register_heif_opener()
            _HEIF_REGISTERED = True
    try:
        data = p.read_bytes() if data is None else data
    except OSError as e:
        raise ImageError(f"cannot read the file ({e.strerror or e})") from e
    sha1 = hashlib.sha1(data).hexdigest()
    Image.MAX_IMAGE_PIXELS = None  # our own (configurable) guard below
    try:
        img = Image.open(io.BytesIO(data))
        fw, fh = img.size
        if fw * fh > fcfg["max_image_megapixels"] * 1_000_000:
            raise ImageError(f"too big ({fw}x{fh})", "skipped")
        orientation = 1
        try:
            orientation = int(img.getexif().get(0x0112, 1) or 1)
        except Exception:
            pass
        if getattr(img, "is_animated", False):
            img.seek(0)
        side = fcfg["decode_max_side"]
        if img.format == "JPEG" and max(fw, fh) > side:
            # JPEG can decode at 1/2, 1/4, 1/8 directly (much faster): ask for at least half of decode_max_side
            r = side / 2 / max(fw, fh)
            img.draft("RGB", (max(1, int(fw * r)), max(1, int(fh * r))))
        img.load()
        dw, dh = img.size  # decoded (maybe reduced) size, before rotation
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
        raise ImageError(f"not a readable image ({type(e).__name__})") from e
    if orientation in (5, 6, 7, 8):
        fw, fh = fh, fw
    w, h = img.size
    if min(fw, fh) < fcfg["min_image_side_px"]:
        raise ImageError(f"too small ({fw}x{fh})", "skipped")
    if max(w, h) > side:  # not a JPEG (no draft): shrink here so detection and crops are fast
        img.thumbnail((side, side))
        w, h = img.size
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    return bgr, {"w": fw, "h": fh, "scale": w / fw if fw else 1.0, "sha1": sha1, "orientation": orientation,
                 "format": img.format or ext, "decoded": (dw, dh)}


def crop(bgr, box_px, size: int, margin: float):
    """A square crop around a face box (decoded-image pixels) with margin, resized to size x size (BGR)."""
    import cv2
    x, y, w, h = box_px
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + 2 * margin) / 2
    H, W = bgr.shape[:2]
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def save_jpeg(bgr, path, quality: int = 90) -> None:
    """Write a BGR array as JPEG (Unicode-safe: encoded in memory, written by Python)."""
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise OSError("JPEG encoding failed")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(buf.tobytes())


def read_bgr(path):
    """Any small image file (e.g. a crop) -> BGR array, Unicode-safe. None when unreadable."""
    import cv2
    try:
        arr = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ---------------- detection ----------------

class Detector:
    """YuNet: faces with box + 5 points + score, left to right."""

    name = "yunet"

    def __init__(self, model_path: Path, dcfg: dict):
        import cv2
        self.cfg = dcfg
        self.net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), float(dcfg["score_threshold"]),
                                             float(dcfg["nms_threshold"]), int(dcfg["top_k"]))

    def detect(self, bgr) -> list[dict]:
        import cv2
        H, W = bgr.shape[:2]
        s = min(1.0, self.cfg["max_side"] / max(H, W))
        img = cv2.resize(bgr, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        h, w = img.shape[:2]
        self.net.setInputSize((w, h))
        _, rows = self.net.detect(img)
        out = []
        for r in (rows if rows is not None else []):
            score = float(r[14])
            if score < self.cfg["score_threshold"]:
                continue
            box = (float(r[0]) / s, float(r[1]) / s, float(r[2]) / s, float(r[3]) / s)
            pts = [(float(r[4 + 2 * i]) / s, float(r[5 + 2 * i]) / s) for i in range(5)]
            out.append({"box": box, "points": pts, "score": score})
        out.sort(key=lambda d: (d["box"][0] + d["box"][2] / 2, d["box"][1]))
        return out


# ---------------- fingerprints ----------------

def _normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(len(v) if np.ndim(v) > 1 else 1, -1)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return v / n


def align_arcface(bgr, points):
    """5 points -> 112x112 face aligned to the ArcFace template (similarity transform)."""
    import cv2
    src = np.asarray(points, dtype=np.float32).reshape(5, 2)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_5PT, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)


class ArcFace:
    name, dim = "arcface", 512

    def __init__(self, model_path: Path, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.batch_ok = not isinstance(shape[0], int)
        self.dim = int(self.sess.get_outputs()[0].shape[-1] or 512)

    def aligned(self, bgr, det: dict):
        return align_arcface(bgr, det["points"])

    def embed(self, faces: list) -> np.ndarray:
        import cv2
        if not faces:
            return np.zeros((0, self.dim), np.float32)
        blob = cv2.dnn.blobFromImages(faces, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        if self.batch_ok:
            out = self.sess.run(None, {self.input: blob})[0]
        else:
            out = np.concatenate([self.sess.run(None, {self.input: blob[i:i + 1]})[0] for i in range(len(blob))])
        return _normalize(out)


class SFace:
    name, dim = "sface", 128

    def __init__(self, model_path: Path, threads: int):
        import cv2
        self.rec = cv2.FaceRecognizerSF.create(str(model_path), "")

    def aligned(self, bgr, det: dict):
        x, y, w, h = det["box"]
        row = np.array([x, y, w, h, *[c for p in det["points"] for c in p], det.get("score", 1.0)], dtype=np.float32)
        return self.rec.alignCrop(bgr, row.reshape(1, -1))

    def embed(self, faces: list) -> np.ndarray:
        if not faces:
            return np.zeros((0, self.dim), np.float32)
        return _normalize(np.concatenate([self.rec.feature(f).reshape(1, -1) for f in faces]))


EMBEDDERS = {"arcface": ArcFace, "sface": SFace}


def models(cfg: dict, spin=None):
    """(Detector, Embedder) for the configured models - downloaded once, cached while keep_models_loaded."""
    import cv2
    fcfg = cfg["faces"]
    m = fcfg["models"]
    name = m["embedder"]
    if name not in EMBEDDERS:
        raise ModelError(f"unknown faces.models.embedder {name!r} (use: {', '.join(EMBEDDERS)})")
    key = (m["detector"]["file"], name, fcfg["threads"])
    if key in _MODELS:
        return _MODELS[key]
    hf = cfg.get("model", {})
    det_path = ensure_model(m["detector"], fcfg, spin, "yunet", hf)
    emb_path = ensure_model(m["embedders"][name], fcfg, spin, name, hf)
    if spin:
        spin.status("faces_loading")
    t = time.monotonic()
    try:
        cv2.setNumThreads(int(fcfg["threads"]))
        pair = (Detector(det_path, fcfg["detection"]), EMBEDDERS[name](emb_path, fcfg["threads"]))
    except Exception as e:
        log("faces_error", where="model_load", error=repr(e))
        raise ModelError(f"could not load the face models: {e}") from e
    finally:
        if spin:
            spin.stop()
    log("faces_models", detector=str(det_path), embedder=name, load_s=round(time.monotonic() - t, 2))
    if fcfg["keep_models_loaded"]:
        _MODELS.clear()
        _MODELS[key] = pair
    return pair


def unload() -> None:
    _MODELS.clear()


# ---------------- quality ----------------

def yaw(points) -> float:
    """0 = looking at the camera, about 0.5 = side-on (nose offset from the eye midpoint / eye distance)."""
    (rx, ry), (lx, ly), (nx, _ny) = points[0], points[1], points[2]
    eye_d = float(np.hypot(lx - rx, ly - ry)) or 1.0
    return (nx - (rx + lx) / 2) / eye_d


def sharpness(aligned) -> float:
    import cv2
    if aligned is None:
        return 0.0
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality(det: dict, aligned, qcfg: dict, scale: float = 1.0) -> dict:
    """size_px (in the full-size photo), sharpness, yaw and the grade: poor | ok | good (config faces.quality)."""
    x, y, w, h = det["box"]
    size_px = int(round(min(w, h) / (scale or 1.0)))
    sharp = sharpness(aligned)
    yw = yaw(det["points"])
    score = det.get("score", 1.0)
    if aligned is None or size_px < qcfg["min_face_px"] or score < qcfg.get("min_det_score", 0.8) or sharp < qcfg["min_sharpness"]:
        grade = "poor"
    elif (size_px >= qcfg["good_face_px"] and score >= qcfg["good_det_score"] and sharp >= qcfg["good_sharpness"]
          and abs(yw) <= qcfg["max_yaw"]):
        grade = "good"
    else:
        grade = "ok"
    return {"size_px": size_px, "sharpness": round(sharp, 1), "yaw": round(yw, 3), "quality": grade}


# ---------------- self-check ----------------

def _self_check(paths: list[str]) -> int:
    import tempfile
    cfg = config.load()
    fcfg = cfg["faces"]
    det, emb = models(cfg)
    print(f"models: {det.name} + {emb.name} ({emb.dim} numbers)")
    out_dir = Path(tempfile.gettempdir()) / "faces_selfcheck"
    out_dir.mkdir(exist_ok=True)
    all_embs = []
    for p in paths:
        t0 = time.monotonic()
        try:
            bgr, info = load_image(p, fcfg)
        except ImageError as e:
            print(f"{p}: {e}")
            continue
        t1 = time.monotonic()
        faces = det.detect(bgr)
        t2 = time.monotonic()
        aligned = [emb.aligned(bgr, d) for d in faces]
        vecs = emb.embed([a for a in aligned if a is not None])
        t3 = time.monotonic()
        print(f"\n{p}  {info['w']}x{info['h']} (decoded {bgr.shape[1]}x{bgr.shape[0]}, EXIF {info['orientation']})"
              f"  decode {1000 * (t1 - t0):.0f} ms · detect {1000 * (t2 - t1):.0f} ms · "
              f"embed {1000 * (t3 - t2):.0f} ms ({len(faces)} faces)")
        vi = 0
        for i, (d, a) in enumerate(zip(faces, aligned)):
            q = quality(d, a, fcfg["quality"], info["scale"])
            x, y, w, h = (round(v / info["scale"]) for v in d["box"])
            print(f"  face {i}: box {x},{y} {w}x{h}  score {d['score']:.2f}  size {q['size_px']}px  "
                  f"sharpness {q['sharpness']}  yaw {q['yaw']}  -> {q['quality']}")
            if a is not None:
                f = out_dir / f"{Path(p).stem}__{i}.jpg"
                save_jpeg(a, f)
                # alignment check: the eyes of the aligned face should be level at y ~ 51
                again = det.detect(cv2_pad(a))
                if again:
                    (rx, ry), (lx, ly) = again[0]["points"][:2]
                    print(f"          aligned -> {f}  eyes y {ry - 56:.1f} / {ly - 56:.1f} (expected ~51)")
                all_embs.append((f"{Path(p).name}#{i}", vecs[vi]))
                vi += 1
    if len(all_embs) > 1:
        print("\nsimilarity (1 = same face, > 0.5 usually the same person, < 0.3 different people):")
        for i in range(len(all_embs)):
            for j in range(i + 1, len(all_embs)):
                print(f"  {all_embs[i][0]}  vs  {all_embs[j][0]}:  {float(all_embs[i][1] @ all_embs[j][1]):.3f}")
    return 0


def cv2_pad(face, pad: int = 56):
    """Put an aligned face in the middle of a bigger canvas so the detector can find it again."""
    import cv2
    return cv2.copyMakeBorder(face, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(_self_check(sys.argv[1:]))
