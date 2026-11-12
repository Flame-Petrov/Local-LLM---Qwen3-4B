"""
Recognition math for /faces - numpy only (no files, no models), so every rule is unit-testable.

    person_score     mean of the top-k similarities between a face and a person's examples
    assign           auto / maybe / unknown / ignored for each face (margins, never_match, negatives,
                     one person per photo)
    cluster          unknown good faces -> new people (leader pass, refine, merge; cannot-link: two faces of
                     one photo never end up in one person; deterministic)
    build_examples   what a person "knows": confirmed faces + strong automatic matches, capped with
                     farthest-point sampling (diverse looks), negatives, centroid
    suggest_threshold  a per-person match threshold from the confirmed / rejected faces

Fingerprints are L2-normalized float32 vectors; similarity = dot product (cosine, -1..1).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPerson:
    """A person as the matcher sees it."""
    id: int
    examples: np.ndarray                      # (n, dim)
    negatives: np.ndarray | None = None       # (m, dim) - faces confirmed NOT them
    status: str = "active"                    # active | ignored
    match_threshold: float | None = None      # person.json override (None = global)
    auto_assign: bool = True
    never: set = field(default_factory=set)   # person ids from never_match
    centroid: np.ndarray | None = None

    def __post_init__(self):
        self.examples = _as2d(self.examples)
        self.negatives = _as2d(self.negatives) if self.negatives is not None else None
        if self.centroid is None:
            self.centroid = centroid(self.examples)


def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1) if a.size else a.reshape(0, 0)
    return a


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1
    return v / n


def centroid(examples: np.ndarray):
    if examples is None or len(examples) == 0:
        return None
    return normalize(examples.mean(axis=0))


def topk_mean(sims: np.ndarray, k: int) -> float:
    if sims.size == 0:
        return -1.0
    k = max(1, min(int(k), sims.size))
    if k == sims.size:
        return float(sims.mean())
    return float(np.partition(sims, -k)[-k:].mean())


def person_score(emb: np.ndarray, person: MPerson, k: int) -> float:
    """Mean of the top-k similarities to the person's examples. The face itself (when it is one of the
    examples, similarity ~1) is left out, so re-checking a known face is not inflated by itself."""
    if person.examples.size == 0:
        return -1.0
    sims = person.examples @ emb
    keep = sims < 0.9999
    if not keep.all() and keep.any():
        sims = sims[keep]
    return topk_mean(sims, k)


def thresholds(person: MPerson | None, mcfg: dict) -> tuple[float, float]:
    """(match, maybe) for a person: the person's override or the global value, + run/session delta."""
    d = float(mcfg.get("delta", 0.0))
    match = (person.match_threshold if person is not None and person.match_threshold is not None
             else mcfg["match_threshold"]) + d
    maybe = min(mcfg["maybe_threshold"] + d, match)
    return match, maybe


# ---------------- assigning faces to known people ----------------

def assign(faces: list[dict], people: list[MPerson], mcfg: dict, rejected: dict | None = None,
           taken: dict | None = None) -> list[dict]:
    """faces: [{id, image_id, embedding}] (quality ok or good). rejected: {face_id: {person ids}}.
    taken: {image_id: {person ids already fixed in that photo}} (confirmed faces).
    Returns [{face_id, status, person_id, score, candidates: [(pid, score), (pid, score)]}] in input order.
    status: auto | maybe | unknown | ignored."""
    rejected = rejected or {}
    taken = {k: set(v) for k, v in (taken or {}).items()}
    people = [p for p in people if p.examples.size and p.centroid is not None]
    results = {f["id"]: {"face_id": f["id"], "status": "unknown", "person_id": None, "score": None, "candidates": []}
               for f in faces}
    if not people or not faces:
        return [results[f["id"]] for f in faces]
    k = int(mcfg["top_k"])
    slack = float(mcfg["prefilter_slack"])
    by_id = {p.id: p for p in people}
    C = np.stack([p.centroid for p in people])
    E = np.stack([np.asarray(f["embedding"], dtype=np.float32) for f in faces])
    pre = E @ C.T                                    # (faces, people) centroid similarity: fast prefilter

    maybes = np.array([thresholds(p, mcfg)[1] for p in people], dtype=np.float32)
    scores: dict = {}                                # face_id -> {pid: score}
    emb_of = {}
    for i, f in enumerate(faces):
        cand = {}
        rej = rejected.get(f["id"], set())
        emb_of[f["id"]] = E[i]
        for j in np.nonzero(pre[i] >= maybes - slack)[0]:
            p = people[int(j)]
            if p.id in rej:
                continue
            s = person_score(E[i], p, k)
            if s >= maybes[int(j)]:
                cand[p.id] = s
        scores[f["id"]] = cand

    # one person per photo: the pairing of a photo's faces with people that has the best total score
    by_image: dict = {}
    for i, f in enumerate(faces):
        by_image.setdefault(f["image_id"], []).append(i)
    for image_id, idxs in by_image.items():
        fixed = taken.get(image_id, set())
        fids = [faces[i]["id"] for i in idxs]
        pairing = _pair(fids, scores, fixed)
        for fid, pid in pairing.items():
            s = scores[fid][pid]
            used_by_others = fixed | {q for f2, q in pairing.items() if f2 != fid}
            others = sorted(((pp, ss) for pp, ss in scores[fid].items() if pp != pid and pp not in used_by_others),
                            key=lambda x: -x[1])
            P = by_id[pid]
            r = results[fid]
            r["person_id"], r["score"] = pid, round(s, 4)
            r["candidates"] = [(pid, round(s, 4))] + [(pp, round(ss, 4)) for pp, ss in others[:1]]
            if P.status == "ignored":
                r["status"] = "ignored"
                continue
            match, _ = thresholds(P, mcfg)
            q = others[0] if others else None
            need = mcfg["margin"]
            if q and (q[0] in P.never or pid in by_id[q[0]].never):
                need = mcfg["sibling_margin"]
            neg_close = bool(P.negatives is not None and P.negatives.size
                             and float((P.negatives @ emb_of[fid]).max()) >= s)
            if s >= match and P.auto_assign and (q is None or s - q[1] >= need) and not neg_close:
                r["status"] = "auto"
            else:
                r["status"] = "maybe"
    return [results[f["id"]] for f in faces]


def _pair(fids: list, scores: dict, fixed: set, exact_max: int = 6) -> dict:
    """{face_id: person_id} - each person at most once per photo (and not one already fixed there).
    Up to exact_max faces with candidates: the pairing with the highest total score (exact search);
    more (big group photos): greedy, best pair first. Ties are broken by ids, so it is deterministic."""
    cand = {fid: sorted(((pid, s) for pid, s in scores[fid].items() if pid not in fixed), key=lambda x: (-x[1], x[0]))[:4]
            for fid in fids}
    active = sorted((f for f in fids if cand[f]), key=lambda f: (-cand[f][0][1], f))
    if not active:
        return {}
    if len(active) <= exact_max:
        best = [-1.0, {}]

        def dfs(i: int, used: set, total: float, cur: dict):
            if i == len(active):
                if total > best[0] + 1e-9:
                    best[0], best[1] = total, dict(cur)
                return
            # upper bound: everything left takes its best candidate
            if total + sum(cand[f][0][1] for f in active[i:]) <= best[0] + 1e-9:
                return
            fid = active[i]
            for pid, s in cand[fid]:
                if pid in used:
                    continue
                cur[fid] = pid
                dfs(i + 1, used | {pid}, total + s, cur)
                del cur[fid]
            dfs(i + 1, used, total, cur)

        dfs(0, set(), 0.0, {})
        return best[1]
    pairs = sorted(((s, fid, pid) for fid in active for pid, s in cand[fid]), key=lambda x: (-x[0], x[1], x[2]))
    out, used = {}, set()
    for s, fid, pid in pairs:
        if fid in out or pid in used:
            continue
        out[fid] = pid
        used.add(pid)
    return out


# ---------------- new people ----------------

def cluster(faces: list[dict], mcfg: dict) -> list[list[int]]:
    """Unknown good faces -> groups of face ids, one per new person (only groups with at least
    min_faces_for_new_person faces). faces: [{id, image_id, embedding, rank}] - rank orders the faces
    (bigger/sharper first; ties by path), so the result does not depend on the input order."""
    if not faces:
        return []
    join = float(mcfg["join_threshold"])
    k = int(mcfg["top_k"])
    order = sorted(faces, key=lambda f: (f.get("rank", (0,)), f["id"]))
    E = np.stack([np.asarray(f["embedding"], dtype=np.float32) for f in order])
    n = len(order)
    label = np.full(n, -1)
    members: list[list[int]] = []
    images: list[set] = []
    sums: list[np.ndarray] = []

    def score(i: int, c: int, skip: int | None = None) -> float:
        m = [j for j in members[c] if j != skip]
        if not m:
            return -1.0
        return topk_mean(E[m] @ E[i], k)

    # 1. leader pass
    for i in range(n):
        best, best_s = -1, -2.0
        if members:
            cents = np.stack([normalize(s) for s in sums]) @ E[i]
            for c in np.nonzero(cents >= join - 0.2)[0]:
                c = int(c)
                if order[i]["image_id"] in images[c]:
                    continue
                s = score(i, c)
                if s > best_s:
                    best, best_s = c, s
        if best >= 0 and best_s >= join:
            members[best].append(i)
            images[best].add(order[i]["image_id"])
            sums[best] = sums[best] + E[i]
            label[i] = best
        else:
            members.append([i])
            images.append({order[i]["image_id"]})
            sums.append(E[i].copy())
            label[i] = len(members) - 1

    # 2. refine pass: move a face to a clearly better group
    for i in range(n):
        cur = int(label[i])
        cur_s = score(i, cur, skip=i)
        best, best_s = cur, cur_s
        for c in range(len(members)):
            if c == cur or not members[c] or order[i]["image_id"] in images[c]:
                continue
            s = score(i, c)
            if s > best_s:
                best, best_s = c, s
        if best != cur and best_s >= join and best_s - cur_s >= 0.02:
            members[cur].remove(i)
            images[cur] = {order[j]["image_id"] for j in members[cur]}
            sums[cur] = sums[cur] - E[i]
            members[best].append(i)
            images[best].add(order[i]["image_id"])
            sums[best] = sums[best] + E[i]
            label[i] = best

    # 3. merge pass: average-link similarity (= mean vector dot mean vector) >= join, cannot-link kept
    live = [c for c in range(len(members)) if members[c]]
    changed = True
    while changed:
        changed = False
        best = None
        for a_i, a in enumerate(live):
            for b in live[a_i + 1:]:
                if images[a] & images[b]:
                    continue
                avg = float((sums[a] / len(members[a])) @ (sums[b] / len(members[b])))
                if avg >= join and (best is None or avg > best[0]):
                    best = (avg, a, b)
        if best:
            _, a, b = best
            members[a] += members[b]
            images[a] |= images[b]
            sums[a] = sums[a] + sums[b]
            for j in members[b]:
                label[j] = a
            members[b] = []
            live.remove(b)
            changed = True

    need = max(1, int(mcfg.get("min_faces_for_new_person", 1)))
    big = sorted((c for c in live if len(members[c]) >= need), key=lambda c: min(members[c]))
    return [sorted(order[j]["id"] for j in members[c]) for c in big]


# ---------------- learning ----------------

def farthest_point(cands: np.ndarray, selected: np.ndarray | None, n: int) -> list[int]:
    """Indexes of n candidates chosen so each new one is the least similar to everything already chosen."""
    if n <= 0 or len(cands) == 0:
        return []
    chosen: list[int] = []
    if selected is not None and len(selected):
        closest = (cands @ selected.T).max(axis=1)
    else:
        chosen.append(0)
        closest = cands @ cands[0]
    closest = closest.astype(np.float64)
    if chosen:
        closest[0] = np.inf
    while len(chosen) < min(n, len(cands)):
        i = int(np.argmin(closest))
        if not np.isfinite(closest[i]):
            break
        chosen.append(i)
        closest = np.maximum(closest, cands @ cands[i])
        closest[chosen] = np.inf
    return chosen


def build_examples(confirmed, auto, rejected, mcfg: dict, learn_from_auto: bool = True):
    """confirmed: (n, dim) array. auto: [(embedding, similarity, quality)]. rejected: (m, dim) array or None.
    Returns (examples, negatives, centroid). Confirmed faces are always kept (first); automatic matches only
    when strong (similarity >= learn_min_similarity) and good - otherwise the person drifts step by step."""
    cap = int(mcfg["max_examples_per_person"])
    conf = _as2d(confirmed) if confirmed is not None and len(confirmed) else None
    strong = []
    if learn_from_auto:
        strong = [(np.asarray(e, dtype=np.float32), s) for e, s, q in auto
                  if e is not None and s is not None and s >= mcfg["learn_min_similarity"] and q == "good"]
    if conf is not None and len(conf) > cap:
        keep = farthest_point(conf, None, cap)
        examples = conf[keep]
    else:
        examples = conf if conf is not None else None
        room = cap - (0 if examples is None else len(examples))
        if strong and room > 0:
            strong.sort(key=lambda x: -x[1])               # strongest first: deterministic start
            cand = np.stack([e for e, _ in strong])
            pick = farthest_point(cand, examples, room)
            add = cand[pick]
            examples = add if examples is None else np.concatenate([examples, add])
    if examples is None:
        examples = np.zeros((0, 0), np.float32)
    negs = _as2d(rejected) if rejected is not None and len(rejected) else None
    return examples, negs, centroid(examples)


def suggest_threshold(confirmed: np.ndarray, examples: np.ndarray, rejected, others: list, mcfg: dict):
    """A match threshold for one person from what the user confirmed and rejected.
    confirmed: (n, dim) the person's confirmed faces; examples: their example matrix; rejected: (m, dim) or None;
    others: [(person_id, confirmed (k, dim))] for the other people.
    Returns (value or None, most_similar {person_id, similarity} or None, overlap: bool)."""
    k = int(mcfg["top_k"])
    most = None
    for pid, conf in others:
        if conf is None or not len(conf) or examples is None or not examples.size:
            continue
        sims = [topk_mean(examples @ e, k) for e in _as2d(conf)]
        top = float(np.mean(sims)) if sims else -1
        if most is None or top > most["similarity"]:
            most = {"person_id": pid, "similarity": round(top, 3)}
    conf = _as2d(confirmed) if confirmed is not None and len(confirmed) else None
    if conf is None or len(conf) < int(mcfg["suggest_min_confirmed"]) or not examples.size:
        return None, most, False
    pos = []
    for e in conf:
        sims = examples @ e
        keep = sims < 0.9999                               # leave the face itself out
        pos.append(topk_mean(sims[keep], k) if keep.any() else -1.0)
    neg = []
    for _, oc in others:
        if oc is not None and len(oc):
            neg += [topk_mean(examples @ e, k) for e in _as2d(oc)]
    if rejected is not None and len(rejected):
        neg += [topk_mean(examples @ e, k) for e in _as2d(rejected)]
    p10 = float(np.percentile(pos, 10))
    if not neg:
        return round(float(np.clip(p10 - 0.05, 0.25, 0.70)), 2), most, False
    n99 = float(np.percentile(neg, 99))
    value = round(float(np.clip((p10 + n99) / 2, 0.25, 0.70)), 2)
    return value, most, p10 <= n99
