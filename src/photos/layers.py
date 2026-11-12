"""
Background removal + depth layers for one photo: '/cutout <photo|url>', '/layers <photo|url> [options]',
or just write "remove the background from E:\\Photos\\me.jpg" / "split E:\\Photos\\room.jpg into layers".
The same engine serves the MCP tools remove_background / split_layers (mcp_server.py).

    1. finds the object of focus (subject matte + what every region is + depth) and says what it is
    2. cuts it out: cutout.png (transparent, full size, clean edges) + mask.png
    3. splits the rest into depth-ordered layers: 0 = in front of the subject (if any), 1 = the subject,
       2 = what it touches (the couch it sits on), then things further back, last = background
    4. saves every layer as a full-size transparent PNG, cumulative stacks, layers.psd, preview.jpg, layers.json

All layers stacked give exactly the photo again. The input file is never changed; every run writes a new folder.
Everything runs locally on the CPU. All texts and settings: config.json -> layers.
"""

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from core import config
from photos import layer_models
from photos import layer_ops as ops
from core.debuglog import log

_LOCK = threading.Lock()          # one photo at a time (CPU-bound)
_CACHE: OrderedDict = OrderedDict()   # sha1 -> loaded photo + analysis (for 'again' and repeated MCP calls)
_LAST: dict = {}                  # the last run: src, sha1, opts

_URL = re.compile(r"https?://[^\s\"<>]+", re.I)
_KV = re.compile(r'(?<!\S)([A-Za-z]+)=("[^"]*"|\S+)')
_BARE = ("crop", "fill", "describe", "nodescribe", "nostacks", "nopsd")
_PLAIN_SKIP = ("picture",)        # plain chat: "this picture" is the photo, not a painting in it
MIN_W = 1.0 / 510.0               # weights under half an 8-bit step are dropped (clean solid cores)


class LayersError(Exception):
    """The message is shown to the user as is."""


# ---------------- understanding the request ----------------

def parse_options(text: str, lcfg: dict, bare: bool = True):
    """'layers=3 fill=lama crop focus="person on the left"' -> (options, the text without them)."""
    opts = {}

    def kv(m):
        k, v = m.group(1).lower(), m.group(2).strip('"').strip()
        if k == "layers" and v.isdigit():
            opts["layers"] = max(2, min(8, int(v)))
        elif k == "focus" and v:
            opts["focus"] = v
        elif k == "fill" and v.lower() in ("none", "fast", "lama"):
            opts["fill"] = v.lower()
        elif k == "to" and v:
            opts["to"] = v
        else:
            return m.group(0)
        return " "

    text = _KV.sub(kv, text)
    if bare:
        def word(m):
            w = m.group(0).lower()
            if w == "crop":
                opts["crop"] = True
            elif w == "fill":
                opts.setdefault("fill", "lama")
            elif w in ("describe", "nodescribe"):
                opts["describe"] = w == "describe"
            elif w == "nostacks":
                opts["stacks"] = False
            elif w == "nopsd":
                opts["psd"] = False
            return " "
        text = re.sub(r"(?<![\w\\/.:-])(" + "|".join(_BARE) + r")(?![\w\\/.:-])", word, text, flags=re.I)
    return opts, " ".join(text.split())


def _is_image_name(name: str, lcfg: dict) -> bool:
    return Path(name).suffix.lower().lstrip(".") in lcfg["image_extensions"]


def make_request(mode: str, src: str, opts: dict | None = None) -> dict:
    return {"mode": mode, "src": str(src).strip().strip('"'), "again": False,
            "opts": {k: v for k, v in (opts or {}).items() if v is not None}}


def parse_request(message: str, lcfg: dict):
    """-> {"mode": "cutout"|"layers", "src", "opts", "again"} | a usage / not-found text | None (not ours)."""
    msg = message.strip()
    if not lcfg["enabled"] or not msg:
        return None
    T = lcfg["texts"]
    cmd, _, rest = msg.partition(" ")
    cmd = cmd.lower()
    if cmd in ("/cutout", "/layers"):
        mode = cmd[1:]
        rest = rest.strip()
        if not rest:
            return T["usage_" + mode]
        if rest.split()[0].lower() == "again":
            if not _LAST:
                return T["no_last"]
            opts, _ = parse_options(rest[5:], lcfg)
            return {"mode": mode, "src": _LAST["src"], "opts": opts, "again": True}
        opts, left = parse_options(rest, lcfg, bare=False)
        m = _URL.search(left)
        if m:
            src, left = m.group(0), left.replace(m.group(0), " ")
        else:
            from photos.faces import find_path
            found = find_path(left)
            if found:
                src, left = found
            else:
                q = re.match(r'"([^"]+)"\s*(.*)', left) or re.match(r"(\S+)\s*(.*)", left)
                if not q:
                    return T["usage_" + mode]
                tok, left = q.group(1), q.group(2)
                p = Path(tok).expanduser()
                p = p if p.is_absolute() else config.path(tok)
                if not p.exists():
                    return T["not_found"].format(path=tok)
                src = str(p)
        more, _ = parse_options(left, lcfg)
        return {"mode": mode, "src": src, "opts": {**more, **opts}, "again": False}
    if msg.startswith("/"):
        return None

    # plain chat: an existing image path (or an image URL) AND background / cutout / layer words
    opts, text = parse_options(msg, lcfg, bare=False)
    from photos.faces import find_path
    src = words = None
    found = find_path(text)
    if found and os.path.isfile(found[0]) and _is_image_name(found[0], lcfg):
        src, words = found
    else:
        m = _URL.search(text)
        if m and _is_image_name(unquote(urlparse(m.group(0)).path), lcfg):
            src, words = m.group(0).rstrip(".,;:!?)"), text.replace(m.group(0), " ")
    if not src:
        return None
    if not any(re.search(p, words, re.I) for p in lcfg["intent_patterns"]):
        return None
    mode = "layers" if any(re.search(p, words, re.I) for p in lcfg["layer_patterns"]) else "cutout"
    if "focus" not in opts:
        skip = set(_PLAIN_SKIP) | {ops.norm_text(v) for v in lcfg["grouping"]["class_names"].values()}
        f = ops.parse_focus(words, lcfg, skip=skip)
        if f["classes"]:
            opts["focus"] = " ".join([f["classes"][0]] + f["positions"][:1])
    return {"mode": mode, "src": src, "opts": opts, "again": False}


# ---------------- loading the photo ----------------

def _fetch(url: str, cfg: dict) -> bytes:
    import requests
    lcfg = cfg["layers"]
    cap = int(lcfg["max_download_mb"]) * 1024 * 1024
    try:
        r = requests.get(url, timeout=lcfg["timeout_s"], stream=True,
                         headers={"User-Agent": cfg.get("search", {}).get("user_agent", "local-llm")})
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(1 << 16):
            buf.write(chunk)
            if buf.tell() > cap:
                raise ValueError(f"bigger than {lcfg['max_download_mb']} MB")
        return buf.getvalue()
    except Exception as e:
        raise LayersError(lcfg["texts"]["url_failed"].format(error=e)) from e


def _decode(data: bytes, shown: str, ext: str, lcfg: dict):
    """bytes -> (rgb uint8 upright, alpha_in uint8 | None, icc | None)"""
    from PIL import Image, ImageOps, UnidentifiedImageError
    T = lcfg["texts"]
    if ext in ("heic", "heif"):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise LayersError(T["not_image"].format(path=shown, reason="HEIC needs the pillow-heif package"))
    Image.MAX_IMAGE_PIXELS = None   # our own (configurable) guard below
    try:
        img = Image.open(io.BytesIO(data))
        fw, fh = img.size
        if fw * fh > lcfg["max_image_megapixels"] * 1_000_000:
            raise LayersError(T["too_big"].format(mp=round(fw * fh / 1e6), max=lcfg["max_image_megapixels"]))
        if getattr(img, "is_animated", False):
            img.seek(0)
        img.load()
        icc = img.info.get("icc_profile")
        transparent = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        img = ImageOps.exif_transpose(img)
        alpha = None
        if img.mode.startswith("I") or img.mode == "F":          # 16-bit / float greys
            a = np.asarray(img).astype(np.float32)
            top = float(a.max()) or 1.0
            a = a / 257.0 if 255 < top <= 65535 else a * (255.0 / top) if top > 255 or img.mode == "F" else a
            rgb = np.repeat(np.clip(a, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            icc = None
        elif transparent:
            rgba = np.asarray(img.convert("RGBA"))
            alpha = rgba[:, :, 3]
            if int(alpha.min()) == 255:
                alpha, rgb = None, rgba[:, :, :3]
            else:   # analyse the picture on white; the input alpha is multiplied into every output
                a = alpha.astype(np.float32)[:, :, None] / 255.0
                rgb = (rgba[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a) + 0.5).astype(np.uint8)
        else:
            if img.mode not in ("RGB", "L"):
                icc = None if img.mode == "CMYK" else icc
            rgb = np.asarray(img.convert("RGB"))
    except LayersError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
        raise LayersError(T["not_image"].format(path=shown, reason=type(e).__name__)) from e
    return np.ascontiguousarray(rgb), (np.ascontiguousarray(alpha) if alpha is not None else None), icc


def _load(src: str, cfg: dict, progress) -> dict:
    lcfg = cfg["layers"]
    T = lcfg["texts"]
    if _URL.fullmatch(src):
        name = Path(unquote(urlparse(src).path)).name or "image"
        if progress:
            progress("layers_reading", name=name)
        data, shown = _fetch(src, cfg), src
    else:
        p = Path(src.strip().strip('"')).expanduser()
        if not p.is_absolute():
            p = config.path(str(p))
        if not p.is_file():
            raise LayersError(T["not_found"].format(path=src))
        name, shown = p.name, str(p)
        if progress:
            progress("layers_reading", name=name)
        try:
            data = p.read_bytes()
        except OSError as e:
            raise LayersError(T["not_image"].format(path=shown, reason=e.strerror or e)) from e
    sha1 = hashlib.sha1(data).hexdigest()
    key = (sha1, lcfg["work_max_side"], lcfg["output_max_side"])
    if key in _CACHE:
        _CACHE.move_to_end(key)
        entry = _CACHE[key]
        entry["src"] = shown
        return entry
    rgb, alpha, icc = _decode(data, shown, Path(name).suffix.lower().lstrip("."), lcfg)
    h, w = rgb.shape[:2]
    if min(w, h) < lcfg["min_image_side_px"]:
        raise LayersError(T["too_small"].format(w=w, h=h, min=lcfg["min_image_side_px"]))
    side = lcfg["output_max_side"]
    if side and max(w, h) > side:
        s = side / max(w, h)
        size = (max(1, round(w * s)), max(1, round(h * s)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA) if alpha is not None else None
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).stem).strip(" .")[:60] or "image"
    return {"key": key, "sha1": sha1, "src": shown, "stem": stem, "rgb_full": rgb, "alpha_in": alpha, "icc": icc,
            "descriptions": {}}


def _analyse(entry: dict, cfg: dict, progress, check, seconds: dict):
    """The three models at work size (cached per photo)."""
    lcfg = cfg["layers"]
    if "S" in entry:
        return
    full = entry["rgb_full"]
    h, w = full.shape[:2]
    s = min(1.0, lcfg["work_max_side"] / max(h, w))
    work = full if s == 1.0 else cv2.resize(full, (max(1, round(w * s)), max(1, round(h * s))),
                                            interpolation=cv2.INTER_AREA)
    keep = lcfg["keep_models_loaded"]
    res = {}
    for name, key, sec in (("matte", "layers_subject", "subject"), ("depth", "layers_depth", "depth"),
                           ("segmenter", "layers_segment", "objects")):
        check()
        t = time.monotonic()
        try:
            model = layer_models.get(name, cfg, progress)
        except layer_models.ModelError as e:
            raise LayersError(str(e)) from e
        if progress:
            progress(key)
        res[name] = model.predict(work)
        if name == "segmenter":
            res["names"] = list(model.labels)
        if not keep:
            layer_models.unload(name)
        seconds[sec] = round(time.monotonic() - t, 1)
        log("layers_stage", stage=sec, seconds=seconds[sec])
    C, conf = res["segmenter"]
    entry.update(work=work, S=res["matte"], D=res["depth"], C=C, conf=conf, names=res["names"])
    _CACHE[entry["key"]] = entry
    while len(_CACHE) > max(1, int(lcfg["again_cache"])):
        _CACHE.popitem(last=False)


# ---------------- layer names ----------------

def _friendly(cls: str, lcfg: dict) -> str:
    return lcfg["grouping"]["class_names"].get(cls, cls)


def _slug(parts: list) -> str:
    s = "+".join(re.sub(r"[^A-Za-z0-9]+", "-", p).strip("-") for p in parts if p)
    return (s[:40].rstrip("-+") or "layer")


def _group_classes(group: dict, by_id: dict) -> list:
    """The classes of a layer, biggest first."""
    area = {}
    for rid in group["ids"]:
        s = by_id[rid]
        area[s["class"]] = area.get(s["class"], 0) + s["px"]
    return [c for c, _ in sorted(area.items(), key=lambda x: -x[1])]


def _names(classes: list, lcfg: dict) -> str:
    seen = []
    for c in classes:
        n = _friendly(c, lcfg)
        if n not in seen:
            seen.append(n)
    k = lcfg["grouping"]["max_names_per_layer"]
    return ", ".join(seen[:k]) + (f" +{len(seen) - k} more" if len(seen) > k else "")


# ---------------- full-size alphas ----------------

def _full_alphas(entry: dict, owner, alpha_f, n: int, fi: int, lcfg: dict):
    """Work-size ownership -> full-size 8-bit alphas [n, H, W] (front to back), the subject's own alpha
    and every layer's visible share. Edge-aware (fast guided filter), in strips to stay small in memory."""
    ed = lcfg["edges"]
    work, full = entry["work"], entry["rgb_full"]
    H, W = full.shape[:2]
    same = work.shape[:2] == (H, W)
    r = max(1, int(round(ed["guided_radius"] * max(work.shape[:2]) / 1024.0)))
    prep = ops.gf_prepare(work, r, ed["guided_eps"])
    coeffs = [None if j == fi else ops.gf_coeffs(prep, (owner == j).astype(np.float32)) for j in range(n)]
    fa_coeffs = None if same else ops.gf_coeffs(prep, alpha_f)
    alphas = np.zeros((n, H, W), dtype=np.uint8)
    falpha = np.zeros((H, W), dtype=np.uint8)
    share = np.zeros(n, dtype=np.float64)
    step = max(64, int(ed["strip_rows"]))
    for y0 in range(0, H, step):
        y1 = min(H, y0 + step)
        I = prep["I"][y0:y1] if same else full[y0:y1].astype(np.float32) / 255.0
        m = np.zeros((n, y1 - y0, W), dtype=np.float32)
        for j in range(n):
            if j != fi:
                A, b = coeffs[j]
                m[j] = ops.apply_coeffs(ops.upsample_rows(A, W, H, y0, y1), ops.upsample_rows(b, W, H, y0, y1), I)
        if same:
            fa = alpha_f[y0:y1]
        else:
            A, b = fa_coeffs
            fa = np.clip(ops.apply_coeffs(ops.upsample_rows(A, W, H, y0, y1),
                                          ops.upsample_rows(b, W, H, y0, y1), I), 0, 1)
        w = ops.weights_from_soft(m, fi, fa, MIN_W)
        share += w.reshape(n, -1).sum(axis=1)
        alphas[:, y0:y1] = (ops.alphas_from_weights(w) * 255.0 + 0.5).astype(np.uint8)
        falpha[y0:y1] = (w[fi] * 255.0 + 0.5).astype(np.uint8)
    return alphas, falpha, 100.0 * share / (H * W)


def _decontaminated(rgb, falpha, behind: bool = False) -> np.ndarray:
    """The photo with the subject's own colours at its soft edges (inside the subject's box only).
    behind=True: the other way round - what is behind the subject, without the subject's colours on its rim."""
    ys, xs = np.nonzero(falpha)
    if not len(ys):
        return rgb
    H, W = falpha.shape
    pad = 96
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1), max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    out = rgb.copy()
    a = falpha[y0:y1, x0:x1].astype(np.float32) / 255.0
    F = ops.estimate_foreground(rgb[y0:y1, x0:x1].astype(np.float32) / 255.0, 1.0 - a if behind else a)
    out[y0:y1, x0:x1] = (F * 255.0 + 0.5).astype(np.uint8)
    return out


# ---------------- filling hidden parts ----------------

def _inpaint(rgb, hole, method: str, cfg: dict, progress, scale: float) -> np.ndarray:
    """The photo with `hole` filled in (only the hole and a thin feathered rim change)."""
    fc = cfg["layers"]["fill"]
    if not hole.any():
        return rgb
    if progress:
        progress("layers_fill", method=method)
    H, W = hole.shape
    f = max(1, int(round(fc["feather_px"] * scale)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * f + 1, 4 * f + 1))
    grown = cv2.dilate(hole.astype(np.uint8), k)
    if method == "lama":
        try:
            model = layer_models.get("inpaint", cfg, progress)
        except layer_models.ModelError as e:
            raise LayersError(str(e)) from e
        if progress:
            progress("layers_fill", method=method)
        ys, xs = np.nonzero(grown)
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        side = int(max(bh, bw) * (1 + 2 * fc["context_pct"] / 100.0))
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
        y1, x1 = min(H, y0 + side), min(W, x0 + side)
        y0, x0 = max(0, y1 - side), max(0, x1 - side)
        filled = rgb.copy()
        filled[y0:y1, x0:x1] = model.fill(rgb[y0:y1, x0:x1], grown[y0:y1, x0:x1] > 0)
        if not cfg["layers"]["keep_models_loaded"]:
            layer_models.unload("inpaint")
    else:
        s = min(1.0, fc["max_side"] / max(H, W))
        size = (max(1, round(W * s)), max(1, round(H * s)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(grown, size, interpolation=cv2.INTER_NEAREST)
        res = cv2.inpaint(small, (mask > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        filled = cv2.resize(res, (W, H), interpolation=cv2.INTER_CUBIC)
    m = cv2.blur(grown.astype(np.float32), (2 * f + 1, 2 * f + 1))[:, :, None]
    return (rgb.astype(np.float32) * (1 - m) + filled.astype(np.float32) * m + 0.5).astype(np.uint8)


# ---------------- outputs ----------------

def _save_png(path: Path, rgb, alpha, icc, level: int):
    from PIL import Image
    kw = {"compress_level": int(level)}
    if icc:
        kw["icc_profile"] = icc
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path, **kw)   # no EXIF is written


def _mul8(a, b):
    return ((a.astype(np.uint16) * b.astype(np.uint16) + 127) // 255).astype(np.uint8)


def _stack(rgb, rgb_focus, alphas, fi: int, last: int):
    """Layers fi..last composed (normal 'over'): the subject on top of the layers behind it."""
    behind = np.ones(alphas.shape[1:], dtype=np.float32)
    for j in range(fi + 1, last + 1):
        behind *= 1.0 - alphas[j].astype(np.float32) / 255.0
    A = 1.0 - behind
    a = alphas[fi].astype(np.float32) / 255.0
    oa = a + A * (1 - a)
    if rgb_focus is rgb:
        out = rgb
    else:
        wf = (a / np.maximum(oa, 1e-6))[:, :, None]
        out = (rgb_focus.astype(np.float32) * wf + rgb.astype(np.float32) * (1 - wf) + 0.5).astype(np.uint8)
    return out, (oa * 255.0 + 0.5).astype(np.uint8)


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _preview(path: Path, tiles: list, tile: int, max_side: int):
    """Contact sheet: every tile = (rgb uint8, alpha uint8 | None, caption), shown on a checkerboard."""
    from PIL import Image, ImageDraw
    band = 26
    cols = max(1, min(len(tiles), max_side // tile))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + band)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for i, (rgb, alpha, caption) in enumerate(tiles):
        x, y = (i % cols) * tile, (i // cols) * (tile + band)
        h, w = rgb.shape[:2]
        s = min(tile / w, tile / h)
        size = (max(1, round(w * s)), max(1, round(h * s)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            small = ops.on_checker(small, cv2.resize(alpha, size, interpolation=cv2.INTER_AREA))
        sheet.paste(Image.fromarray(ops.checkerboard(tile, tile)), (x, y))
        sheet.paste(Image.fromarray(small), (x + (tile - size[0]) // 2, y + (tile - size[1]) // 2))
        text = caption
        while len(text) > 4 and draw.textlength(text, font=font) > tile - 12:
            text = text[:-2].rstrip() + "…"
        draw.text((x + 6, y + tile + 3), text, fill=(235, 235, 235), font=font)
    sheet.save(path, "JPEG", quality=85)
    return sheet


def _write_psd(path: Path, rgb_full, layers: list):
    """layers: [(name, rgb, alpha)] front to back. Written bottom -> top; every layer is cropped to its content."""
    from PIL import Image
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    psd = PSDImage.frompil(Image.fromarray(rgb_full))
    for name, rgb, alpha in reversed(layers):
        ys, xs = np.nonzero(alpha)
        if not len(ys):
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        pil = Image.fromarray(np.dstack([rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]]), "RGBA")
        before = len(psd)
        layer = PixelLayer.frompil(pil, psd, name[:60], y0, x0)
        if len(psd) == before:
            psd.append(layer)
    psd.save(str(path))


def _new_folder(entry: dict, opts: dict, lcfg: dict) -> Path:
    base = Path(opts["to"]).expanduser() if opts.get("to") else config.path(lcfg["output_dir"])
    if not base.is_absolute():
        base = config.path(str(base))
    name = lcfg["folder_name"].format(stem=entry["stem"], time=datetime.now().strftime(lcfg["folder_time_format"]))
    folder, i = base / name, 1
    while True:
        try:
            folder.mkdir(parents=True, exist_ok=False)
            return folder
        except FileExistsError:
            i += 1
            folder = base / f"{name}_{i}"


def _where(bbox, w: int, h: int) -> str:
    cx, cy = (bbox[0] + bbox[2] / 2) / w, (bbox[1] + bbox[3] / 2) / h
    s = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    return s + (" top" if cy < 1 / 3 else " bottom" if cy > 2 / 3 else "")


# ---------------- the run ----------------

def run(req: dict, cfg: dict, progress=None, should_stop=None) -> dict:
    """Cut out the object of focus (mode 'cutout') or split the photo into layers (mode 'layers').
    progress(key, **values) -> ui.status texts; should_stop() -> bool, checked between the stages.
    Never prints. Raises LayersError (the message is for the user)."""
    with _LOCK:
        try:
            return _run(req, cfg, progress, should_stop)
        finally:
            if not cfg["layers"]["keep_models_loaded"]:
                layer_models.unload()


def _run(req, cfg, progress, should_stop) -> dict:
    lcfg = cfg["layers"]
    T, G, O = lcfg["texts"], lcfg["grouping"], lcfg["outputs"]
    t0 = time.monotonic()
    seconds, warnings = {}, []
    mode = req["mode"]
    opts = dict(req.get("opts") or {})
    if req.get("again") and _LAST:
        opts = {**_LAST["opts"], **opts}

    def check():
        if should_stop and should_stop():
            raise LayersError(T["stopped"])

    # 1. the photo + the three models
    entry = _CACHE.get(_LAST.get("key")) if req.get("again") and _LAST else None
    if entry is None:
        entry = _load(req["src"], cfg, progress)
    _analyse(entry, cfg, progress, check, seconds)
    check()
    if progress:
        progress("layers_grouping")
    t = time.monotonic()
    work, S, D, C, names = entry["work"], entry["S"], entry["D"], entry["C"], entry["names"]
    full = entry["rgb_full"]
    H, W = full.shape[:2]
    wh, ww = work.shape[:2]
    scale = max(H, W) / max(wh, ww)

    # 2. regions and the object of focus
    labels, _ = ops.regions(C, int(G["min_region_pct"] / 100.0 * C.size))
    stats = ops.region_stats(labels, C, S, D, names)
    by_id = {s["id"]: s for s in stats}
    try:
        focus = ops.pick_focus(stats, S, lcfg, opts.get("focus"))
    except ops.NothingFound:
        raise LayersError(T["nothing"])
    warnings += focus["warnings"]
    alpha_f, w2 = ops.focus_alpha(work, S, labels, focus["ids"], lcfg, stats)
    warnings += w2
    fmask = alpha_f >= 0.5
    if not fmask.any():
        fmask = np.isin(labels, focus["ids"])
        alpha_f = fmask.astype(np.float32)
    log("layers_focus", ids=focus["ids"], label=focus["label"], count=focus["count"],
        confidence=focus["confidence"], user_focus=opts.get("focus"), candidates=focus["candidates"])

    # 3. layers
    max_layers = 2 if mode == "cutout" else max(2, min(8, int(opts.get("layers") or G["max_layers"])))
    groups = ops.group_layers(stats, labels, fmask, D, lcfg, max_layers, focus["ids"])
    owner = ops.owner_map(labels, groups, fmask)
    used = [i for i, g in enumerate(groups) if g["kind"] == "focus" or (owner == i).any()]
    if len(used) < len(groups):
        remap = np.full(len(groups), -1, dtype=np.int32)
        remap[used] = np.arange(len(used))
        owner, groups = remap[owner], [groups[i] for i in used]
    n = len(groups)
    fi = next(i for i, g in enumerate(groups) if g["kind"] == "focus")
    d_f = groups[fi]["depth"]
    bgc = set(G["background_classes"])
    fset = set(focus["ids"])
    nearest = not any(s["depth"] >= d_f + G["front_margin"] - 1e-4 for s in stats
                      if s["id"] not in fset and s["class"] not in bgc)
    ys, xs = np.nonzero(fmask)
    bbox = [int(xs.min() * W / ww), int(ys.min() * H / wh),
            int(np.ceil((xs.max() + 1) * W / ww)) - int(xs.min() * W / ww),
            int(np.ceil((ys.max() + 1) * H / wh)) - int(ys.min() * H / wh)]
    label_name = ops.focus_name(focus["label"], focus["count"], G["class_names"])
    seconds["grouping"] = round(time.monotonic() - t, 1)

    # 4. the optional sentence from the local vision model
    describe = opts.get("describe")
    if describe is None:
        describe = lcfg["describe"]["chat"]
    desc = None
    if describe:
        check()
        key = tuple(focus["ids"])
        if key in entry["descriptions"]:
            desc = entry["descriptions"][key]
        else:
            t = time.monotonic()
            try:
                desc = layer_models.describe_focus(full, bbox, label_name, cfg, progress)
                entry["descriptions"][key] = desc
            except KeyboardInterrupt:
                raise
            except Exception as e:
                warnings.append(T["describe_failed"].format(error=e))
                log("layers_failed", where="describe", error=repr(e))
            seconds["description"] = round(time.monotonic() - t, 1)
    focus_info = {"name": (desc or {}).get("name") or label_name, "label": label_name,
                  "description": (desc or {}).get("description"), "classes": [focus["label"]],
                  "count": focus["count"], "bbox": bbox, "area_pct": round(100.0 * float(alpha_f.mean()), 1),
                  "where": _where(bbox, W, H), "nearest": bool(nearest), "confidence": focus["confidence"],
                  "depth": round(float(d_f), 2)}

    # 5. full-size alphas
    check()
    if progress:
        progress("layers_grouping")
    t = time.monotonic()
    alphas, falpha, share = _full_alphas(entry, owner, alpha_f, n, fi, lcfg)
    rgb_dec = _decontaminated(full, falpha) if lcfg["edges"]["decontaminate"] else full
    rgb_focus = rgb_dec if lcfg["edges"]["decontaminate_layers"] else full
    # the layers behind the subject: without the subject's colours on the rim of the hole it leaves
    rgb_behind = (_decontaminated(full, falpha, behind=True)
                  if lcfg["edges"]["decontaminate_layers"] and mode == "layers" else full)
    seconds["grouping"] = round(seconds["grouping"] + time.monotonic() - t, 1)

    has_front = groups[0]["kind"] == "front"
    layers = []
    for i, g in enumerate(groups):
        classes = [focus["label"]] if i == fi else _group_classes(g, by_id)
        if i == fi:
            name, slug = focus_info["name"], _slug([_friendly(focus["label"], lcfg)])
        elif g["kind"] == "background":
            name = T["background_name"].format(names=_names(classes, lcfg)) if classes else "background"
            slug = "background"
        else:
            name = _names(classes, lcfg)
            slug = _slug([_friendly(c, lcfg) for c in classes][:G["max_names_per_layer"]])
        num = i if has_front else i + 1
        layers.append({"n": num, "name": name, "kind": g["kind"], "classes": classes,
                       "area_pct": round(float(share[i]), 1), "depth": round(float(g["depth"]), 2),
                       "touches_focus": g["kind"] == "touching", "in_front": g["kind"] == "front",
                       "file": f"layers/layer_{num}_{slug}.png"})
    log("layers_groups", layers=[{k: l[k] for k in ("n", "kind", "classes", "area_pct", "depth")} for l in layers])

    # 6. fill the hidden parts
    fill = (opts.get("fill") or lcfg["fill"]["default"]).lower()
    rgb_bg, without = rgb_behind, None
    if fill != "none":
        check()
        t = time.monotonic()
        fc = lcfg["fill"]
        d = max(1, int(round(fc["dilate_px"] * scale)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        subj_hole = cv2.dilate((falpha > 5).astype(np.uint8), k) > 0
        bg_hole = alphas[n - 1] == 0 if (mode == "layers" and n > 1 and fi != n - 1) else None
        if bg_hole is not None and n == 2:
            without = rgb_bg = _inpaint(full, subj_hole | bg_hole, fill, cfg, progress, scale)
        else:
            without = _inpaint(full, subj_hole, fill, cfg, progress, scale)
            if bg_hole is not None and bg_hole.any():
                rgb_bg = _inpaint(rgb_behind, cv2.dilate(bg_hole.astype(np.uint8), k) > 0, fill, cfg, progress,
                                  scale)
        if bg_hole is not None:
            alphas[n - 1] = 255
        seconds["fill"] = round(time.monotonic() - t, 1)

    if entry["alpha_in"] is not None:     # pixels that were transparent in the input stay transparent
        ain = entry["alpha_in"]
        for j in range(n):
            alphas[j] = _mul8(alphas[j], ain)
        falpha = _mul8(falpha, ain)

    # 7. write everything into a new folder
    check()
    want_layers = mode == "layers" and O["layers"]
    want_stacks = mode == "layers" and O["stacks"] and opts.get("stacks", True)
    want_psd = mode == "layers" and O["psd"] and opts.get("psd", True)
    stack_ends = list(range(fi + 1, n - 1)) if want_stacks else []
    planned = (int(O["cutout"]) + int(O["mask"]) + (n if want_layers else 0) + len(stack_ends) + int(want_psd)
               + int(O["preview"]) + int(without is not None) + 1)
    if progress:
        progress("layers_saving", n=planned)
    t = time.monotonic()
    icc = entry["icc"] if O["keep_icc_profile"] else None
    level = O["png_compress_level"]
    folder = _new_folder(entry, opts, lcfg)
    files = {"cutout": None, "mask": None, "psd": None, "preview": None, "without_subject": None}
    stacks = []
    preview_small = None
    try:
        from PIL import Image
        if O["cutout"]:
            crgb, calpha = rgb_dec, falpha
            if opts.get("crop", O["crop_cutout"]):
                ys, xs = np.nonzero(falpha)
                if len(ys):
                    my, mx = int(H * O["crop_margin_pct"] / 100), int(W * O["crop_margin_pct"] / 100)
                    y0, y1 = max(0, ys.min() - my), min(H, ys.max() + 1 + my)
                    x0, x1 = max(0, xs.min() - mx), min(W, xs.max() + 1 + mx)
                    crgb, calpha = rgb_dec[y0:y1, x0:x1], falpha[y0:y1, x0:x1]
            _save_png(folder / "cutout.png", np.ascontiguousarray(crgb), np.ascontiguousarray(calpha), icc, level)
            files["cutout"] = "cutout.png"
        if O["mask"]:
            Image.fromarray(falpha, "L").save(folder / "mask.png", compress_level=int(level))
            files["mask"] = "mask.png"

        def colours(i):
            return rgb_focus if i == fi else rgb_bg if i == n - 1 else rgb_behind if i > fi else full

        if want_layers:
            (folder / "layers").mkdir()
            for i, l in enumerate(layers):
                _save_png(folder / l["file"], colours(i), alphas[i], icc, level)
        else:
            for l in layers:
                l["file"] = None
        if stack_ends:
            (folder / "stacks").mkdir()
            for last in stack_ends:
                srgb, salpha = _stack(rgb_behind, rgb_focus, alphas, fi, last)
                rel = f"stacks/stack_{layers[fi]['n']}-{layers[last]['n']}.png"
                _save_png(folder / rel, srgb, salpha, icc, level)
                stacks.append({"layers": [layers[j]["n"] for j in range(fi, last + 1)], "file": rel})
        if want_psd:
            try:
                psd_layers = []
                for i, l in enumerate(layers):
                    nm = f"{l['n']} {l['name']}"
                    if i == fi and focus_info["description"]:
                        nm += " - " + focus_info["description"]
                    psd_layers.append((nm, colours(i), alphas[i]))
                _write_psd(folder / "layers.psd", full, psd_layers)
                files["psd"] = "layers.psd"
            except ImportError:
                warnings.append(T["psd_missing"])
        if without is not None:
            _save_png(folder / "without_subject.png", without,
                      entry["alpha_in"] if entry["alpha_in"] is not None else np.full((H, W), 255, np.uint8),
                      icc, level)
            files["without_subject"] = "without_subject.png"
        if O["preview"]:
            cap = focus_info["name"]
            if mode == "layers":
                tile = int(O["preview_tile_px"])
                tiles = [(full, None, "original"), (rgb_dec, falpha, f"cutout: {cap}")]
                tiles += [(colours(i), alphas[i], f"{l['n']} {l['name']}") for i, l in enumerate(layers)]
            else:
                tile = min(512, int(O["preview_max_side"]) // 2)
                tiles = [(full, None, "original"), (rgb_dec, falpha, f"cutout: {cap}")]
            sheet = _preview(folder / "preview.jpg", tiles, tile, int(O["preview_max_side"]))
            files["preview"] = "preview.jpg"
            preview_small = str(folder / "preview.jpg")
            side = int(cfg.get("mcp", {}).get("preview_max_side", 1024))
            if max(sheet.size) > side:
                sheet.thumbnail((side, side))
                small = Path(tempfile.gettempdir()) / f"local_llm_layers_preview_{entry['sha1'][:12]}.jpg"
                sheet.save(small, "JPEG", quality=82)
                preview_small = str(small)
        if O["debug_maps"]:
            dbg = folder / "debug"
            dbg.mkdir()
            pal = np.random.default_rng(1).integers(40, 255, size=(4096, 3), dtype=np.uint8)
            Image.fromarray((D * 255).astype(np.uint8)).save(dbg / "depth.png")
            Image.fromarray((S * 255).astype(np.uint8)).save(dbg / "saliency.png")
            Image.fromarray(pal[C]).save(dbg / "classes.png")
            Image.fromarray(pal[labels % 4096]).save(dbg / "regions.png")
        seconds["saving"] = round(time.monotonic() - t, 1)
        seconds["total"] = round(time.monotonic() - t0, 1)
        m = lcfg["models"]
        manifest = {
            "source": entry["src"], "size": [W, H], "focus": focus_info,
            "layers": [{k: l[k] for k in ("n", "name", "classes", "area_pct", "depth", "touches_focus",
                                          "in_front", "file")} for l in layers] if mode == "layers" else [],
            "stacks": stacks, "files": files,
            "models": {"matte": m["matte"], "depth": m["depth"]["repo"].split("/")[-1], "segmenter": m["segmenter"],
                       "describe": cfg["video"]["vision"].get("file") if describe else None},
            "options": {"mode": mode, "layers": max_layers, "focus": opts.get("focus"), "fill": fill,
                        "crop": bool(opts.get("crop", O["crop_cutout"])), "describe": bool(describe)},
            "seconds": seconds, "warnings": warnings}
        (folder / "layers.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)   # our own new folder: never leave half a result
        raise

    _LAST.clear()
    _LAST.update(src=entry["src"], key=entry["key"], opts=opts)
    written = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    log("layers_saved", folder=str(folder), files=written, seconds=seconds, warnings=warnings)
    return {**manifest, "mode": mode, "folder": str(folder), "fill": fill, "kinds": [l["kind"] for l in layers],
            "all_layers": layers, "preview_small": preview_small}


# ---------------- report / history / summary ----------------

def _abs(result: dict, rel):
    return str(Path(result["folder"]) / rel) if rel else None


def _focus_lines(result: dict, T: dict, confidence: bool = False):
    f = result["focus"]
    desc = f' - "{f["description"]}"' if f.get("description") else ""
    same = f["name"].strip().lower() == f["label"].lower()
    name = f["label"] if same else f["name"] if not f.get("description") else f"{f['label']} ({f['name']})"
    where = T["focus_where"].format(where=f["where"], pct=round(f["area_pct"]),
                                    nearest=T["nearest"] if f["nearest"] else "")
    if confidence:
        where += f" (confidence: {f['confidence']})"
    return T["focus"].format(name=name, description=desc), where


def report(result: dict, lcfg: dict, color: bool = True) -> str:
    """What is printed after a run (color=False: plain text for the MCP client / tests)."""
    T, col = lcfg["texts"], lcfg["colors_256"]
    if color:
        from core import ui
        paint = ui.paint
        link = ui.file_link if lcfg["clickable_paths"] else str
    else:
        paint, link = (lambda s, c: s), str
    head, where = _focus_lines(result, T, confidence=not color)
    out = [paint(head, col["focus"]), paint(where, col["dim"])]
    out += [paint(w, col["warn"]) for w in result["warnings"]]
    files = result["files"]
    if files["cutout"]:
        out.append(T["cutout_saved"].format(path=link(_abs(result, files["cutout"]))))
    if files["mask"] and (result["mode"] == "cutout" or not color):
        out.append(T["mask_saved"].format(path=link(_abs(result, files["mask"]))))
    if result["mode"] == "layers":
        out.append(T["layers_header"])
        layers = result["all_layers"]
        for i, l in enumerate(layers):
            note = ""
            if l["touches_focus"]:
                note = T["note_touches"]
            elif l["in_front"]:
                note = T["note_front"]
            elif i == len(layers) - 1 and l["kind"] != "focus":
                note = (T["note_hidden"] if result["fill"] == "none"
                        else T["note_filled"].format(method=result["fill"]))
            line = T["layer_line"].format(n=l["n"], name=l["name"], pct=round(l["area_pct"]), note=note)
            out.append(paint(line, col["focus"] if l["kind"] == "focus" else col["layer"]))
        if result["stacks"]:
            out.append(T["stacks"].format(path=link(_abs(result, "stacks"))))
        if files["psd"]:
            out.append(T["psd"].format(path=link(_abs(result, files["psd"]))))
    if files["without_subject"]:
        out.append(T["without_subject"].format(path=link(_abs(result, files["without_subject"]))))
    if files["preview"] and (result["mode"] == "layers" or not color):
        out.append(T["preview"].format(path=link(_abs(result, files["preview"]))))
    sec = result["seconds"]
    parts = " · ".join(f"{label} {round(sec[k])} s" for k, label in
                       (("subject", "subject"), ("depth", "depth"), ("objects", "objects"),
                        ("description", "description"), ("fill", "fill"), ("saving", "saving"))
                       if sec.get(k, 0) >= 0.5)
    took = T["took"].format(s=round(sec["total"]), parts=parts)
    out.append(paint(took if parts else took.replace(" ()", ""), col["dim"]))
    return "\n".join(out)


def history(result: dict, lcfg: dict) -> str:
    """The run as plain text for the chat history (follow-up questions, /write)."""
    f = result["focus"]
    desc = f' - "{f["description"]}"' if f.get("description") else ""
    w, h = result["size"]
    out = [f"Photo: {result['source']} ({w}x{h})",
           f"Object of focus: {f['name']}{desc} ({f['where']}, {round(f['area_pct'])}%)"]
    out += result["warnings"]
    files = result["files"]
    if files["cutout"]:
        out.append(f"Background removed: {_abs(result, files['cutout'])}")
    if result["mode"] == "layers":
        out.append("Layers, front to back: " + "; ".join(
            f"{l['n']} {l['name']} ({round(l['area_pct'])}%)" for l in result["all_layers"]))
    if files["without_subject"]:
        out.append(f"Photo without the subject: {_abs(result, files['without_subject'])}")
    extra = [x for x in (files["psd"], "stacks" if result["stacks"] else None, files["preview"]) if x]
    out.append(f"Folder: {result['folder']}" + (f" ({', '.join(extra)})" if extra else ""))
    return "\n".join(out)


def summary(result: dict) -> dict:
    """JSON-safe result with absolute paths (the MCP structured content)."""
    files = result["files"]
    return {
        "focus": result["focus"],
        "cutout": _abs(result, files["cutout"]), "mask": _abs(result, files["mask"]),
        "layers": [{**l, "file": _abs(result, l["file"])} for l in result["layers"]],
        "stacks": [{**s, "file": _abs(result, s["file"])} for s in result["stacks"]],
        "psd": _abs(result, files["psd"]), "preview": _abs(result, files["preview"]),
        "without_subject": _abs(result, files["without_subject"]),
        "folder": result["folder"], "source": result["source"], "size": result["size"],
        "seconds": result["seconds"]["total"], "warnings": list(result["warnings"])}
