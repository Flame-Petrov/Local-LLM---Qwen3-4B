"""
Image maths for /cutout and /layers (layers.py): pure numpy / OpenCV, no models, no file I/O.

    guided filter (colour guide) + edge-aware upsampling   -> soft layer edges that follow the photo's edges
    blur-fusion foreground estimation                      -> no colour fringe of the old background on hair
    regions / focus scoring / depth grouping               -> which pixels belong to which layer
    weights -> alphas, compositing                         -> layers that stack back to exactly the photo

Conventions: images are HxWx3, masks / maps HxW float32 0..1; depth 0 = far .. 1 = near;
layer lists are ordered front (index 0) to back (last = background).
"""

import re
import unicodedata

import cv2
import numpy as np


class NothingFound(Exception):
    """There is nothing to separate in the image (one flat region, no subject)."""


# ---------------- small helpers ----------------

def as_float(img) -> np.ndarray:
    """uint8 / float image (HxW or HxWxC) -> float32 HxWx3, 0..1."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return np.ascontiguousarray(a[:, :, :3])


def _box(x, r: int):
    return cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT_101)


def norm_text(s) -> str:
    return unicodedata.normalize("NFKC", str(s or "")).lower().replace("’", "'").replace("‘", "'")


def words_rx(words):
    """Regex that finds any of these words / phrases as whole words; 'abc*' = any word starting with abc
    (the same rule as music._words_rx, kept here so the MCP server never imports the music module)."""
    parts = []
    for w in sorted({norm_text(w).strip() for w in words if w and w.strip()}, key=len, reverse=True):
        prefix = w.endswith("*")
        stem = w[:-1] if prefix else w
        p = re.escape(stem) + (r"\w*" if prefix else "")
        if re.match(r"\w", stem):
            p = r"(?<!\w)" + p
        if not prefix and re.search(r"\w$", stem):
            p += r"(?!\w)"
        parts.append(p)
    return re.compile("|".join(parts) if parts else r"(?!x)x", re.I)


# ---------------- guided filter ----------------

def gf_prepare(guide, r: int, eps: float) -> dict:
    """Everything of the guided filter that depends only on the guide (reused for every layer)."""
    I = as_float(guide)
    I64 = I.astype(np.float64)
    mean = _box(I64, r)
    m = [mean[:, :, i] for i in range(3)]
    ch = [I64[:, :, i] for i in range(3)]

    def var(i, j):
        return _box(ch[i] * ch[j], r) - m[i] * m[j]

    a, b, c = var(0, 0) + eps, var(0, 1), var(0, 2)
    d, e, f = var(1, 1) + eps, var(1, 2), var(2, 2) + eps
    i_rr, i_rg, i_rb = d * f - e * e, c * e - b * f, b * e - c * d
    i_gg, i_gb, i_bb = a * f - c * c, b * c - a * e, a * d - b * b
    det = a * i_rr + b * i_rg + c * i_rb
    det[det == 0] = 1e-12
    inv = [(x / det).astype(np.float32) for x in (i_rr, i_rg, i_rb, i_gg, i_gb, i_bb)]
    return {"I": I, "mean": mean.astype(np.float32), "inv": inv, "r": int(r)}


def gf_coeffs(prep: dict, src):
    """Linear coefficients (A HxWx3, b HxW) with  filtered = sum(A * guide, axis=2) + b."""
    I, mean, r = prep["I"], prep["mean"], prep["r"]
    p = np.ascontiguousarray(src, dtype=np.float32)
    mean_p = _box(p, r)
    cov = _box(I * p[:, :, None], r) - mean * mean_p[:, :, None]
    rr, rg, rb, gg, gb, bb = prep["inv"]
    c0, c1, c2 = cov[:, :, 0], cov[:, :, 1], cov[:, :, 2]
    a = np.stack([rr * c0 + rg * c1 + rb * c2, rg * c0 + gg * c1 + gb * c2, rb * c0 + gb * c1 + bb * c2], axis=2)
    b = mean_p - (a * mean).sum(axis=2)
    return _box(a, r), _box(b, r)


def apply_coeffs(A, b, guide_float) -> np.ndarray:
    return (A * guide_float).sum(axis=2) + b


def guided_filter(guide, src, r: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing of `src` that follows the edges of `guide`. Linear in `src`."""
    prep = gf_prepare(guide, r, eps)
    A, b = gf_coeffs(prep, src)
    return apply_coeffs(A, b, prep["I"]).astype(np.float32)


def upsample_rows(arr, full_w: int, full_h: int, y0: int = 0, y1: int | None = None) -> np.ndarray:
    """Bilinear upsampling of a small map (HxW or HxWxC, C <= 4) to rows y0..y1 of the full size
    (pixel-centre aligned, so strips join without seams)."""
    y1 = full_h if y1 is None else y1
    sh, sw = arr.shape[:2]
    if (sh, sw) == (full_h, full_w):
        return arr[y0:y1]
    xs = ((np.arange(full_w, dtype=np.float32) + 0.5) * (sw / full_w) - 0.5)
    ys = ((np.arange(y0, y1, dtype=np.float32) + 0.5) * (sh / full_h) - 0.5)
    map_x = np.repeat(xs[None, :], y1 - y0, axis=0)
    map_y = np.repeat(ys[:, None], full_w, axis=1)
    return cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def guided_upsample(guide_small, src_small, guide_full, r: int, eps: float) -> np.ndarray:
    """Fast guided filter (He & Sun 2015): coefficients at the small size, applied to the full-size guide."""
    A, b = gf_coeffs(gf_prepare(guide_small, r, eps), src_small)
    G = as_float(guide_full)
    h, w = G.shape[:2]
    return apply_coeffs(upsample_rows(A, w, h), upsample_rows(b, w, h), G).astype(np.float32)


# ---------------- clean subject colours ----------------

def _fusion(image, F, B, alpha, k: int):
    a = alpha[:, :, None]
    ba = cv2.blur(alpha, (k, k))[:, :, None]
    bF = cv2.blur(F * a, (k, k)) / (ba + 1e-5)
    bB = cv2.blur(B * (1 - a), (k, k)) / ((1 - ba) + 1e-5)
    F2 = bF + a * (image - a * bF - (1 - a) * bB)
    return np.clip(F2, 0, 1), bB


def estimate_foreground(rgb_float, alpha, k1: int = 91, k2: int = 7) -> np.ndarray:
    """The subject's own colour at soft edges (blur-fusion, Forte & Pitie 2021), so the old background
    does not bleed into hair on a new background. rgb_float HxWx3 0..1, alpha HxW 0..1."""
    image = np.ascontiguousarray(rgb_float, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)
    F, B = _fusion(image, image, image, alpha, k1)
    F, _ = _fusion(image, F, B, alpha, k2)
    solid = (alpha >= 0.999) | (alpha <= 0.001)
    F[solid] = image[solid]
    return F


# ---------------- regions ----------------

def _nearest_fill(values, known):
    """Every pixel that is not `known` gets the value of the nearest known pixel."""
    src = np.where(known, 0, 1).astype(np.uint8)
    _, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(lab.max()) + 1, dtype=values.dtype)
    lut[lab[known]] = values[known]
    return lut[lab]


def regions(classes, min_px: int):
    """Connected components (8-connected) of every class; components under min_px are dissolved into the
    nearest remaining region. -> (labels int32 HxW with ids 1..n, n)"""
    classes = np.asarray(classes)
    labels = np.zeros(classes.shape, dtype=np.int32)
    n, big = 0, []
    for c in np.unique(classes):
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats((classes == c).astype(np.uint8), connectivity=8)
        for i in range(1, cnt):
            n += 1
            big.append(stats[i, cv2.CC_STAT_AREA] >= min_px)
        m = lab > 0
        labels[m] = lab[m] + (n - (cnt - 1))
    keep = np.array([False] + big)
    if not keep.any():          # everything is tiny: keep the regions as they are
        return labels, n
    if not keep[1:].all():
        known = keep[labels]
        labels = _nearest_fill(labels, known)
        remap = np.zeros(n + 1, dtype=np.int32)
        remap[keep] = np.arange(1, int(keep.sum()) + 1)
        labels = remap[labels]
        n = int(keep.sum())
    return labels, n


def region_stats(labels, classes, S, D, label_names) -> list:
    """Per region: id, class (name), px, area_pct, bbox (x, y, w, h), centroid, depth (median, 1 = near),
    sal (mean saliency), sal_share (its part of all saliency), center (0 = image centre .. 1), border."""
    h, w = labels.shape
    total_s = float(S.sum()) or 1.0
    out = []
    n = int(labels.max())
    px = np.bincount(labels.ravel(), minlength=n + 1)
    for rid in range(1, n + 1):
        if not px[rid]:
            continue
        ys, xs = np.nonzero(labels == rid)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cls = int(np.bincount(classes[ys, xs].astype(np.int64)).argmax())
        s = S[ys, xs]
        cx, cy = float(xs.mean()), float(ys.mean())
        out.append({
            "id": rid, "class": label_names[cls] if cls < len(label_names) else str(cls), "px": int(px[rid]),
            "area_pct": 100.0 * px[rid] / (h * w), "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
            "centroid": [cx, cy], "depth": float(np.median(D[ys, xs])),
            "sal": float(s.mean()), "sal_share": float(s.sum()) / total_s,
            "center": float(np.hypot((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2)) / np.sqrt(2)),
            "border": bool(x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1), "img": [w, h]})
    return out


# ---------------- the object of focus ----------------

_ARTICLES = re.compile(r"(?<!\w)(the|a|an|my|this|that|on|in|at|of|to)(?!\w)", re.I)
PLURALS = {"person": "people", "animal": "animals"}


def parse_focus(text: str, lcfg: dict, label_names=(), skip=()) -> dict:
    """'the dog', 'person on the left', 'диван' -> {"classes": [ADE names], "positions": [left|right|...]}."""
    t = norm_text(text)
    fc = lcfg["focus"]
    pairs = [(norm_text(w), cls) for cls, ws in fc["synonyms"].items() for w in ws]
    pairs += [(norm_text(n), n) for n in label_names]
    pairs += [(norm_text(n), cls) for cls, n in lcfg["grouping"]["class_names"].items()]
    classes = []
    for word, cls in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        if not word or word in skip:
            continue
        rx = words_rx((word,))
        if rx.search(t):
            t = rx.sub(" ", t)
            if cls not in classes:
                classes.append(cls)
    positions = [k for k, ws in fc["position_words"].items() if words_rx(tuple(ws)).search(t)]
    return {"classes": classes, "positions": positions}


def _by_position(cands: list, positions: list) -> dict:
    pick = {"left": lambda s: s["centroid"][0], "right": lambda s: -s["centroid"][0],
            "middle": lambda s: s["center"], "front": lambda s: -s["depth"], "back": lambda s: s["depth"],
            "biggest": lambda s: -s["px"]}
    return min(cands, key=pick[positions[0]])


def focus_name(label: str, count: int, class_names: dict) -> str:
    name = class_names.get(label, label)
    if count <= 1:
        return name
    return f"{count} {PLURALS.get(name, name + 's')}"


def pick_focus(stats: list, S, lcfg: dict, focus_req=None) -> dict:
    """Which region(s) are the object of focus.
    -> {"ids", "label", "count", "confidence": high|medium|low, "warnings": [text], "user": bool, "candidates"}"""
    fc, T = lcfg["focus"], lcfg["texts"]
    bgc = set(lcfg["grouping"]["background_classes"])
    weights = fc["class_weights"]
    warnings = []
    things = [s for s in stats if s["class"] not in bgc]

    def score(s):
        return s["sal_share"] * weights.get(s["class"], weights["default"]) * (0.75 + 0.25 * s["depth"])

    def done(chosen, confidence, user=False):
        share = sum(s["sal_share"] for s in chosen)
        if confidence is None:
            confidence = "high" if share >= 0.6 else "medium" if share >= 0.3 else "low"
        top = sorted(stats, key=score, reverse=True)[:8]
        return {"ids": [s["id"] for s in chosen], "label": chosen[0]["class"], "count": len(chosen),
                "confidence": confidence, "warnings": warnings, "user": user,
                "candidates": [{"class": s["class"], "area_pct": round(s["area_pct"], 1), "sal": round(s["sal"], 2),
                                "depth": round(s["depth"], 2), "score": round(score(s), 3)} for s in top]}

    # 1. what the user asked for
    if focus_req and str(focus_req).strip():
        req = parse_focus(focus_req, lcfg, sorted({s["class"] for s in stats}))
        cands = [s for s in stats if s["class"] in req["classes"]]
        if not req["classes"] and req["positions"]:
            cands = [s for s in things if s["area_pct"] >= 1.0]
        if cands:
            if req["positions"]:
                return done([_by_position(cands, req["positions"])], "high", user=True)
            best = max(cands, key=lambda s: (s["px"] * (0.5 + s["sal"])))
            group = [best] + [s for s in cands if s is not best and s["class"] == best["class"]
                              and s["px"] >= fc["same_class_share"] * best["px"]]
            return done(group, "high", user=True)
        warnings.append(T["focus_not_found"].format(focus=str(focus_req).strip()))

    # 2. automatic
    if float(S.mean()) * 100.0 >= fc["min_saliency_pct"]:
        salient = [s for s in things if s["sal"] >= fc["region_min_saliency"]]
        if salient:
            best = max(salient, key=score)
            group = [best] + [s for s in salient if s is not best and s["class"] == best["class"]
                              and s["sal_share"] >= fc["same_class_share"] * best["sal_share"]]
            return done(group, None)
    # no clear subject: the nearest sizeable thing
    sizeable = [s for s in things if s["area_pct"] >= 1.0]
    if sizeable:
        best = max(sizeable, key=lambda s: s["area_pct"] * s["depth"] ** 2)
        warnings.append(T["focus_low"].format(name=lcfg["grouping"]["class_names"].get(best["class"], best["class"])))
        return done([best], "low")
    # only background classes: the most salient region, else the nearest one - if there is a choice at all
    if float(S.mean()) * 100.0 >= fc["min_saliency_pct"] and stats:
        return done([max(stats, key=lambda s: s["sal_share"])], "low")
    if len(stats) < 2:
        raise NothingFound()
    best = max(stats, key=lambda s: min(s["area_pct"], 30.0) * s["depth"] ** 2)
    warnings.append(T["focus_low"].format(name=lcfg["grouping"]["class_names"].get(best["class"], best["class"])))
    return done([best], "low")


def focus_alpha(work, S, labels, focus_ids, lcfg: dict, stats=None):
    """The subject's alpha at work size: the matte gives the edges against the background, the region
    border cuts where the matte also covers a neighbour (the sofa under the person). -> (alpha, warnings)"""
    fc, ed = lcfg["focus"], lcfg["edges"]
    F = np.isin(labels, list(focus_ids))
    scale = max(labels.shape) / 1024.0
    d = max(1, int(round(fc["dilate_px"] * scale)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    grown = cv2.dilate(F.astype(np.uint8), k) > 0
    free = np.ones(F.shape, bool)
    if stats:   # hair and fingers may reach into the background, never into another salient thing
        fset = set(focus_ids)
        blocked = [s["id"] for s in stats if s["id"] not in fset and s["sal"] >= fc["region_min_saliency"]]
        if blocked:
            free = ~np.isin(labels, blocked)
            grown &= free
    grown |= F
    # what the matte marks and that hangs together with the subject belongs to it, whatever the segmenter
    # called it (shoes on the floor, hair against the wall, a bag in the hand)
    cnt, comp = cv2.connectedComponents((((S > 0.5) & free) | F).astype(np.uint8), connectivity=8)
    if cnt > 1:
        keep = np.zeros(cnt, bool)
        keep[np.unique(comp[F])] = True
        keep[0] = False
        grown |= keep[comp]
    r = max(1, int(round(ed["guided_radius"] * scale)))
    soft = np.clip(guided_filter(work, grown.astype(np.float32), r, ed["guided_eps"]), 0, 1)
    agree = float(S[F].mean()) if F.any() else 0.0
    if agree >= fc["min_matte_agreement"]:
        # levels: an almost-solid matte is solid (no see-through subject), faint haze is nothing
        return np.clip((S * soft - 0.05) / 0.85, 0, 1).astype(np.float32), []
    soft = np.clip(guided_filter(work, F.astype(np.float32), r, ed["guided_eps"]), 0, 1)
    return soft.astype(np.float32), [lcfg["texts"]["rough_edges"]]


# ---------------- groups -> layers ----------------

def group_layers(stats: list, labels, focus_mask, D, lcfg: dict, max_layers: int, focus_ids=None) -> list:
    """Regions -> layers, front to back: [front] focus [touching] depth groups... background.
    -> [{"ids", "kind": front|focus|touching|depth|background, "depth"}]"""
    g = lcfg["grouping"]
    eps = 1e-4
    px = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    if focus_ids is None:
        inside = np.bincount(labels[focus_mask], minlength=len(px))
        focus_ids = [s["id"] for s in stats if inside[s["id"]] > 0.5 * px[s["id"]]]
    fset = set(focus_ids)
    fstats = [s for s in stats if s["id"] in fset]
    d_f = float(np.median(D[focus_mask])) if focus_mask.any() else max([s["depth"] for s in fstats] or [1.0])
    bgc = set(g["background_classes"])
    others = [s for s in stats if s["id"] not in fset]
    bg = [s for s in others if s["class"] in bgc]
    things = [s for s in others if s["class"] not in bgc]

    front = [s for s in things if g["foreground_layer"] and s["depth"] >= d_f + g["front_margin"] - eps]
    front_ids = {s["id"] for s in front}
    ring = (cv2.dilate(focus_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0) & ~focus_mask
    ring_total = int(ring.sum()) or 1
    on_ring = np.bincount(labels[ring], minlength=len(px))
    touching = [s for s in things if s["id"] not in front_ids
                and on_ring[s["id"]] / ring_total >= g["touch_min_share"]
                and abs(s["depth"] - d_f) <= g["touch_depth_margin"] + eps]
    taken = front_ids | {s["id"] for s in touching}
    rest = sorted([s for s in things if s["id"] not in taken], key=lambda s: -s["depth"])

    def depth_of(rs):
        tot = sum(s["px"] for s in rs)
        return sum(s["depth"] * s["px"] for s in rs) / tot if tot else 0.0

    def make(kind, rs):
        return {"kind": kind, "regions": list(rs), "depth": depth_of(rs)}

    slots = max(0, int(max_layers) - 2)
    pool, g_touch, g_front, g_depth = [], None, None, []
    if touching:
        if slots:
            g_touch, slots = make("touching", touching), slots - 1
        else:
            pool += touching
    if front:
        if slots:
            g_front, slots = make("front", front), slots - 1
        else:
            pool += front
    if rest:
        if slots:
            gaps = [(rest[i]["depth"] - rest[i + 1]["depth"], i) for i in range(len(rest) - 1)]
            cuts = sorted(i for gap, i in sorted([x for x in gaps if x[0] >= g["min_depth_gap"] - eps],
                                                 reverse=True)[:slots - 1])
            start = 0
            for i in cuts + [len(rest) - 1]:
                g_depth.append(make("depth", rest[start:i + 1]))
                start = i + 1
        else:
            pool += rest
    g_bg = make("background", bg)

    def targets(skip=None):
        t = [x for x in [g_front, g_touch] + g_depth if x is not None and x is not skip]
        if g_bg is not skip:
            t.append(g_bg)
        return t

    for s in pool:          # no layer of its own: joins the layer with the nearest depth (never the focus)
        min(targets(), key=lambda x: abs(x["depth"] - s["depth"]))["regions"].append(s)
    total = float(labels.size)
    while True:             # layers that are too small join their nearest-depth neighbour
        small = [x for x in targets(skip=g_bg)
                 if 100.0 * sum(s["px"] for s in x["regions"]) / total < g["min_layer_pct"]]
        if not small:
            break
        x = small[0]
        dest = min(targets(skip=x), key=lambda y: abs(y["depth"] - x["depth"]))
        dest["regions"] += x["regions"]
        if x is g_front:
            g_front = None
        elif x is g_touch:
            g_touch = None
        else:
            g_depth = [y for y in g_depth if y is not x]

    order = ([g_front] if g_front else []) + [make("focus", fstats)] + ([g_touch] if g_touch else []) + g_depth
    if g_bg["regions"] or len(order) == 1:
        order.append(g_bg)
    return [{"kind": x["kind"], "ids": [s["id"] for s in x["regions"]],
             "depth": d_f if x["kind"] == "focus" else x["depth"]} for x in order]


def owner_map(labels, groups: list, focus_mask) -> np.ndarray:
    """Hard ownership: layer index per pixel (the focus where its alpha >= 0.5, else the pixel's region's
    layer; pixels of the focus regions outside the alpha go to the nearest other layer)."""
    lut = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    fi = next(i for i, g in enumerate(groups) if g["kind"] == "focus")
    for i, g in enumerate(groups):
        if i != fi:
            lut[g["ids"]] = i
    owner = lut[labels]
    owner[focus_mask] = fi
    unknown = owner < 0
    if unknown.any():
        known = (owner >= 0) & (owner != fi)
        if known.any():
            owner = np.where(unknown, _nearest_fill(owner, known), owner)
        else:
            owner[unknown] = len(groups) - 1 if len(groups) > 1 else fi
    return owner.astype(np.int32)


# ---------------- weights -> alphas ----------------

def weights_from_soft(m, focus_index: int, focus_alpha, min_w: float = 0.0) -> np.ndarray:
    """Soft ownership maps m [n, H, W] (the focus row is ignored) + the focus alpha -> weights that sum to 1:
    the focus gets its alpha, the others share the rest; where nobody claims a pixel the last layer gets it."""
    m = np.clip(np.asarray(m, dtype=np.float32), 0, None)
    n = m.shape[0]
    fa = np.clip(focus_alpha, 0, 1).astype(np.float32)
    if n == 1:
        return np.ones_like(m)
    m[focus_index] = 0
    rest_index = n - 1 if focus_index != n - 1 else n - 2
    total = m.sum(axis=0)
    empty = total <= 1e-6
    m[rest_index][empty] = 1.0
    total[empty] = 1.0
    w = m * ((1.0 - fa) / total)[None]
    w[focus_index] = fa
    if min_w > 0:
        w[w < min_w] = 0
        s = w.sum(axis=0)
        gone = s <= 0
        if gone.any():
            w[rest_index][gone] = 1.0
            s[gone] = 1.0
        w /= s[None]
    return w


def soft_weights(owner, n_layers: int, focus_index: int, focus_alpha, guide, r: int, eps: float) -> np.ndarray:
    """Hard owner map -> soft weights [n, H, W] that sum to 1 (edges follow the guide image)."""
    prep = gf_prepare(guide, r, eps)
    m = np.zeros((n_layers,) + owner.shape, dtype=np.float32)
    for j in range(n_layers):
        if j != focus_index:
            A, b = gf_coeffs(prep, (owner == j).astype(np.float32))
            m[j] = apply_coeffs(A, b, prep["I"])
    return weights_from_soft(m, focus_index, focus_alpha)


def alphas_from_weights(w) -> np.ndarray:
    """Alphas for normal 'over' stacking, front (0) to back: a_j = w_j / sum(w_i for i >= j).
    All layers stacked give exactly 1; a layer is opaque under the soft edge of the layer in front of it."""
    w = np.asarray(w, dtype=np.float32)
    tail = np.cumsum(w[::-1], axis=0)[::-1]
    a = np.zeros_like(w)
    np.divide(w, tail, out=a, where=tail > 0)
    return np.clip(a, 0, 1)


def compose(rgba_list_back_to_front) -> np.ndarray:
    """Normal 'over' compositing of straight-alpha RGBA float images (0..1), back to front."""
    out = np.array(rgba_list_back_to_front[0], dtype=np.float32, copy=True)
    for top in rgba_list_back_to_front[1:]:
        top = np.asarray(top, dtype=np.float32)
        a, A = top[:, :, 3:4], out[:, :, 3:4]
        oa = a + A * (1 - a)
        rgb = (top[:, :, :3] * a + out[:, :, :3] * A * (1 - a)) / np.maximum(oa, 1e-6)
        out = np.concatenate([rgb, oa], axis=2)
    return out


# ---------------- previews ----------------

def checkerboard(h: int, w: int, cell: int = 12, light: int = 236, dark: int = 200) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    board = np.where(((ys // cell) + (xs // cell)) % 2 == 0, light, dark).astype(np.uint8)
    return np.repeat(board[:, :, None], 3, axis=2)


def on_checker(rgb, alpha, cell: int = 12) -> np.ndarray:
    """RGB uint8 + alpha uint8 shown on a grey checkerboard -> RGB uint8."""
    h, w = alpha.shape
    a = alpha.astype(np.float32)[:, :, None] / 255.0
    return (rgb.astype(np.float32) * a + checkerboard(h, w, cell).astype(np.float32) * (1 - a) + 0.5).astype(np.uint8)
