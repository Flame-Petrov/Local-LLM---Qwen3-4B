"""
The people folders of /faces (downloads/faces/people/<name>/) and everything that touches them:

    Lib            one open faces library: config, DB (face_store), models (loaded on first use), screen lines
    ingest         one photo -> its faces in the DB (cached by path + size + mtime, or by sha1)
    sync_disk      make the folders match the DB: photo copies / hard links, face crops (_faces, _maybe)
    learn          read the user's edits (renamed / deleted / moved photos and crops, person.json, new photos,
                   hand-made folders, deleted folders) and turn them into answers ("feed the AI")
    reevaluate     re-check the faces near some people after they learned something
    person ops     create, rename, merge, ignore, unignore, forget, confirm / reject a face

The app only deletes or renames files it created inside the people folders (copies, links, crops, its own
person.json); source photos and files the user added are never deleted.
All texts and settings: config.json -> faces.
"""

import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from core import config
from photos import face_match
from photos import face_models
from core import ui
from core.debuglog import log
from photos.face_store import Store, now

SESSION = {"strictness": 0.0, "min_faces_for_new_person": None}   # chat: "be stricter", "only 2+ photos"

FACE_ID_RX = re.compile(r"__f(\d+)(?:\D|$)")
_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ---------------- screen lines ----------------

class Out:
    """Lines above the spinner (coloured, paths clickable) + the same lines as plain text for the history."""

    def __init__(self, spin, fcfg: dict, echo: bool = True):
        self.spin, self.fcfg, self.colors = spin, fcfg, fcfg["colors_256"]
        self.echo = echo
        self.lines: list[str] = []

    def line(self, text: str = "", color: str | None = None):
        self.lines.append(text)
        if self.echo and self.spin is not None:
            self.spin.log(ui.paint(text, self.colors.get(color)) if color else text)

    def path(self, p):
        s = str(p)
        self.lines.append(self.fcfg["texts"]["path"].format(path=s))
        if self.echo and self.spin is not None:
            shown = ui.file_link(s) if self.fcfg["clickable_paths"] else s
            self.spin.log(self.fcfg["texts"]["path"].format(path=ui.paint(shown, self.colors["path"])))


# ---------------- the library ----------------

def library_dir(fcfg: dict) -> Path:
    return config.path(fcfg["library_dir"])


def matching_cfg(fcfg: dict, embedder: str, delta: float = 0.0) -> dict:
    m = {k: v for k, v in fcfg["matching"].items() if not k.startswith("_") and not isinstance(v, dict)}
    m.update(fcfg["matching"][embedder])
    m["delta"] = round(float(delta) + SESSION["strictness"], 4)
    if SESSION["min_faces_for_new_person"]:
        m["min_faces_for_new_person"] = SESSION["min_faces_for_new_person"]
    return m


class Lib:
    def __init__(self, cfg: dict, spin, out: Out | None = None, delta: float = 0.0, echo: bool = True):
        self.cfg = cfg
        self.fcfg = cfg["faces"]
        self.T = self.fcfg["texts"]
        self.spin = spin
        self.dir = library_dir(self.fcfg)
        self.people_dir = self.dir / self.fcfg["people_folder"]
        self.store = Store.open(self.dir, self.fcfg)
        self.embedder = self.fcfg["models"]["embedder"]
        self.delta = delta
        self.mcfg = matching_cfg(self.fcfg, self.embedder, delta)
        self.out = out or Out(spin, self.fcfg, echo=echo)
        self._people = None
        self._models = None
        self.copy_fallback_noted = False

    def close(self):
        self.store.close()

    # models are loaded only when a photo really has to be read
    def models(self):
        if self._models is None:
            self._models = face_models.models(self.cfg, self.spin)
        return self._models

    def folder(self, p: dict) -> Path:
        return self.people_dir / (p.get("folder") or safe_name(p["name"]))

    def crop_cache(self, face_id: int) -> Path:
        return self.dir / self.fcfg["crop_cache_folder"] / f"{face_id // 1000:03d}" / f"f{face_id}.jpg"

    def is_default_name(self, name: str) -> bool:
        pat = re.escape(self.fcfg["new_person_name"]).replace(re.escape("{n}"), r"\d+")
        return bool(re.fullmatch(pat, name or "", re.I))

    def dirty(self):
        self._people = None

    # ---------------- people as the matcher sees them ----------------

    def people(self) -> dict:
        """{pid: MPerson} for active and ignored people (examples rebuilt from the DB, cached until dirty())."""
        if self._people is not None:
            return self._people
        st = self.store
        persons = {p["id"]: p for p in st.persons(status=("active", "ignored"))}
        rows = st.faces(status=("confirmed", "auto", "ignored", "maybe"),
                        where="person_id IS NOT NULL AND embedding IS NOT NULL")
        yes = {k for k, v in st.verdicts().items() if v == "yes"}
        conf, auto, maybe = {}, {}, {}
        for f in rows:
            pid = f["person_id"]
            if pid not in persons:
                continue
            if f["status"] == "maybe":
                maybe.setdefault(pid, []).append((f["embedding"], f["similarity"], f["quality"]))
            elif f["status"] == "confirmed" or (f["id"], pid) in yes:
                conf.setdefault(pid, []).append(f["embedding"])
            else:
                auto.setdefault(pid, []).append((f["embedding"], f["similarity"], f["quality"]))
        # a person whose matches all went to _maybe (e.g. a much stricter threshold) keeps their strong
        # matches as examples - otherwise they would forget what they look like
        for pid, rows_ in maybe.items():
            if not conf.get(pid) and not any(s is not None and s >= self.mcfg["learn_min_similarity"]
                                             for _, s, _ in auto.get(pid, [])):
                auto.setdefault(pid, []).extend(rows_)
        rej = {}
        rej_ids = {}
        for (fid, pid), v in st.verdicts().items():
            if v == "no" and pid in persons:
                rej_ids.setdefault(pid, []).append(fid)
        all_rej = sorted({fid for ids in rej_ids.values() for fid in ids})
        emb_by_id = {f["id"]: f["embedding"] for f in st.faces(ids=all_rej) if f["embedding"] is not None} if all_rej else {}
        for pid, ids in rej_ids.items():
            embs = [emb_by_id[i] for i in ids if i in emb_by_id]
            if embs:
                rej[pid] = np.stack(embs)
        names = name_index(persons.values())
        out = {}
        for pid, p in persons.items():
            s = p["settings"] or {}
            c = np.stack(conf[pid]) if conf.get(pid) else None
            ex, negs, cen = face_match.build_examples(c, auto.get(pid, []), rej.get(pid), self.mcfg,
                                                      learn_from_auto=s.get("learn_from_auto", True))
            never = set()
            for ref in s.get("never_match") or []:
                q = resolve_ref(ref, persons, names)
                if q and q != pid:
                    never.add(q)
            out[pid] = face_match.MPerson(id=pid, examples=ex, negatives=negs, status=p["status"],
                                          match_threshold=s.get("match_threshold"), auto_assign=s.get("auto_assign", True),
                                          never=never, centroid=cen)
        # never_match works both ways
        for pid, mp in out.items():
            for q in list(mp.never):
                if q in out:
                    out[q].never.add(pid)
        self._people = out
        return out

    def person_examples(self, pid: int) -> int:
        mp = self.people().get(pid)
        return 0 if mp is None else len(mp.examples)


def name_index(persons) -> dict:
    """normalized name / alias -> [person ids]."""
    idx: dict = {}
    for p in persons:
        for n in [p["name"], *(p.get("aliases") or [])]:
            k = norm_name(n)
            if k:
                idx.setdefault(k, [])
                if p["id"] not in idx[k]:
                    idx[k].append(p["id"])
    return idx


def norm_name(s) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", str(s or "")).lower().split())


def resolve_ref(ref, persons: dict, names: dict | None = None):
    """'12', '#12', 12 or a name / alias -> person id (merged people -> who they were merged into)."""
    names = names if names is not None else name_index(persons.values())
    pid = None
    if isinstance(ref, int) or re.fullmatch(r"\s*#?\d+\s*", str(ref)):
        pid = int(str(ref).strip().lstrip("#"))
    else:
        hits = names.get(norm_name(ref)) or []
        pid = hits[0] if len(hits) == 1 else None
    seen = set()
    while pid in persons and persons[pid].get("status") == "merged" and persons[pid].get("merged_into") and pid not in seen:
        seen.add(pid)
        pid = persons[pid]["merged_into"]
    return pid if pid in persons else None


def safe_name(name: str) -> str:
    s = _BAD_NAME.sub(" ", str(name or "")).strip().rstrip(". ")
    s = " ".join(s.split())[:80]
    if s.upper() in ("CON", "PRN", "AUX", "NUL") or re.fullmatch(r"(?i)(com|lpt)\d", s):
        s += "_"
    return s or "Person"


def unique_folder(parent: Path, name: str, own: Path | None = None) -> Path:
    base = safe_name(name)
    if base.startswith("_"):
        base = base.lstrip("_") or "Person"
    cand = parent / base
    i = 2
    while cand.exists() and (own is None or cand.resolve() != own.resolve()):
        cand = parent / f"{base} ({i})"
        i += 1
    return cand


def unique_file(folder: Path, name: str) -> Path:
    p = folder / name
    stem, ext = p.stem, p.suffix
    i = 2
    while p.exists():
        p = folder / f"{stem} ({i}){ext}"
        i += 1
    return p


def sha1_file(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_exts(fcfg: dict) -> set:
    return {"." + e.lower().lstrip(".") for e in fcfg["image_extensions"]}


# ---------------- reading one photo ----------------

def ingest(lib: Lib, path, st=None, force: bool = False, source: str = "scan", data: bytes | None = None) -> dict:
    """A photo -> its faces in the DB. Returns {image_id, how, faces, sha1, error, kind, ms}.
    how: unchanged (same path, size, mtime, model) | reused (same bytes known elsewhere) | new | error."""
    st_ = lib.store
    path = str(path)
    st = st or os.stat(path)
    row = st_.image_by_path(path)
    if (row and not force and row["size"] == st.st_size and abs((row["mtime"] or 0) - st.st_mtime) < 0.01
            and (row["model"] in (lib.embedder, None) or row["image_status"] != "ok")):
        st_.touch_path(path)
        return {"image_id": row["image_id"], "how": "unchanged", "faces": row["faces"] or 0, "sha1": row["sha1"],
                "error": row["error"], "status": row["image_status"]}
    t0 = time.monotonic()
    try:
        data = Path(path).read_bytes() if data is None else data
    except OSError as e:
        return {"image_id": None, "how": "error", "faces": 0, "sha1": None, "error": f"cannot read the file ({e.strerror or e})",
                "kind": "unreadable", "status": "unreadable"}
    sha1 = hashlib.sha1(data).hexdigest()
    img = st_.image_by_sha1(sha1)
    if img and not force and (img["model"] in (lib.embedder, None) or img["status"] != "ok"):
        st_.add_path(path, img["id"], st.st_size, st.st_mtime, source)
        how = "unchanged" if row and row["image_id"] == img["id"] else "reused"
        return {"image_id": img["id"], "how": how, "faces": img["faces"] or 0, "sha1": sha1, "error": img["error"],
                "status": img["status"]}
    try:
        bgr, info = face_models.load_image(path, lib.fcfg, data)
    except face_models.ImageError as e:
        status = "skipped" if e.kind in ("skipped", "unsupported") else "unreadable"
        image_id = st_.add_image(sha1, 0, 0, 0, status, str(e), None)
        st_.add_path(path, image_id, st.st_size, st.st_mtime, source)
        return {"image_id": image_id, "how": "error", "faces": 0, "sha1": sha1, "error": str(e), "kind": e.kind,
                "status": status}
    t1 = time.monotonic()
    det, emb = lib.models()
    dets = det.detect(bgr)
    t2 = time.monotonic()
    faces, aligned = [], []
    s = info["scale"] or 1.0
    for idx, d in enumerate(dets):
        a = emb.aligned(bgr, d)
        q = face_models.quality(d, a, lib.fcfg["quality"], s)
        x, y, w, h = d["box"]
        faces.append({"idx": idx, "box": (x / s, y / s, w / s, h / s), "points": [(px / s, py / s) for px, py in d["points"]],
                      "det_score": d["score"], **q, "model": emb.name, "embedding": None, "_det": d})
        aligned.append(a)
    todo = [i for i, f in enumerate(faces) if f["quality"] != "poor"]
    if todo:
        vecs = emb.embed([aligned[i] for i in todo])
        for i, v in zip(todo, vecs):
            faces[i]["embedding"] = v
    t3 = time.monotonic()
    old = st_.faces(image_id=img["id"]) if img else []
    image_id = st_.add_image(sha1, info["w"], info["h"], len(faces), "ok" if faces else "no_faces", None, emb.name)
    ids = _store_faces(st_, image_id, faces, old)
    for o in old:  # F6: a dropped face's cached crop must not outlive it (ids can be reused by SQLite)
        if o["id"] not in ids:
            try:
                lib.crop_cache(o["id"]).unlink(missing_ok=True)
            except OSError:
                pass
    st_.add_path(path, image_id, st.st_size, st.st_mtime, source)
    ccfg = lib.fcfg["crops"]
    for fid, f in zip(ids, faces):
        c = face_models.crop(bgr, f["_det"]["box"], ccfg["size_px"], ccfg["margin"])
        if c is not None:
            try:
                face_models.save_jpeg(c, lib.crop_cache(fid), ccfg["jpeg_quality"])
            except OSError as e:
                log("faces_error", where="crop_cache", path=path, error=repr(e))
    ms = {"decode": round(1000 * (t1 - t0)), "detect": round(1000 * (t2 - t1)), "embed": round(1000 * (t3 - t2))}
    if lib.fcfg["log_each_photo"]:
        log("faces_photo", path=path, ms=ms, faces=[{k: f[k] for k in ("box", "det_score", "size_px", "sharpness", "yaw",
                                                                         "quality")} for f in faces])
    return {"image_id": image_id, "how": "new", "faces": len(faces), "sha1": sha1, "error": None,
            "status": "ok" if faces else "no_faces", "ms": ms, "face_ids": ids}


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def _store_faces(st_: Store, image_id: int, faces: list, old: list) -> list[int]:
    """New faces of a photo; with `old` (re-read with force) faces in the same place keep their id, status and answers."""
    if not old:
        return st_.add_faces(image_id, faces)
    ids, used = [], set()
    for f in faces:
        best = max(((_iou(f["box"], o["box"]), o) for o in old if o["id"] not in used), key=lambda x: x[0], default=(0, None))
        if best[1] is not None and best[0] >= 0.5:
            o = best[1]
            used.add(o["id"])
            keep_status = o["status"] if f["quality"] != "poor" else "poor"
            if keep_status == "poor" and f["quality"] != "poor":
                keep_status = "unknown"
            st_.set_face(o["id"], idx=f["idx"], x=f["box"][0], y=f["box"][1], w=f["box"][2], h=f["box"][3],
                         points=[list(map(float, p)) for p in f["points"]], det_score=float(f["det_score"]),
                         size_px=int(f["size_px"]), sharpness=float(f["sharpness"]), yaw=float(f["yaw"]),
                         quality=f["quality"], model=f["model"], embedding=f["embedding"], status=keep_status,
                         person_id=o["person_id"] if keep_status != "poor" else None)
            ids.append(o["id"])
        else:
            ids.append(st_.add_faces(image_id, [f])[0])
    for o in old:
        if o["id"] not in used:
            st_.db.execute("DELETE FROM faces WHERE id=?", (o["id"],))
            st_.db.execute("DELETE FROM crops WHERE face_id=?", (o["id"],))
    return ids


# ---------------- person.json ----------------

# err value of read_person_file for a file that exists but can't be read now (locked, not UTF-8): compare with `is`
UNREADABLE = (0, 0, "")


def read_person_file(path: Path):
    """(data, None); (None, (line, col, raw)) for broken JSON; (None, None) when missing;
    (None, UNREADABLE) when it exists but can't be read (locked / not UTF-8) - callers must not treat it as missing."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeDecodeError):
        return None, UNREADABLE
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, (e.lineno, e.colno, raw)
    return (data if isinstance(data, dict) else None), (None if isinstance(data, dict) else (1, 1, raw))


def code_from_raw(raw: str):
    m = re.search(r'"id"\s*:\s*"(p\d+)"', raw or "")
    return m.group(1) if m else None


def validate_person_file(data: dict, lib: Lib, path: Path) -> tuple[dict, list]:
    """person.json values -> clean fields {name, aliases, notes, status, merge_into, settings} + warnings."""
    T = lib.T
    warns = []
    out = {}

    def bad(key, value, used):
        warns.append(T["bad_value"].format(path=path, key=key, value=json.dumps(value, ensure_ascii=False), used=used))

    name = data.get("name")
    if isinstance(name, str) and name.strip():
        out["name"] = " ".join(name.split())
    elif "name" in data:
        bad("name", name, "the old name")
    al = data.get("aliases", [])
    if isinstance(al, str):
        al = [al]
    out["aliases"] = [" ".join(a.split()) for a in al if isinstance(a, str) and a.strip()] if isinstance(al, list) else []
    notes = data.get("notes", "")
    out["notes"] = notes if isinstance(notes, str) else str(notes or "")
    status = str(data.get("status", "active") or "active").lower().strip()
    if status not in ("active", "ignored"):
        bad("status", data.get("status"), "active")
        status = "active"
    out["status"] = status
    mi = data.get("merge_into")
    out["merge_into"] = mi if isinstance(mi, (str, int)) and str(mi).strip() else None
    rec = data.get("recognition") if isinstance(data.get("recognition"), dict) else {}
    s = {}
    mt = rec.get("match_threshold")
    lo, hi = lib.fcfg["matching"]["threshold_range"]
    if mt is None:
        s["match_threshold"] = None
    elif isinstance(mt, (int, float)) and not isinstance(mt, bool):
        v = float(mt)
        if v < lo or v > hi:
            v = min(hi, max(lo, v))
            bad("recognition.match_threshold", mt, v)
        s["match_threshold"] = round(v, 3)
    else:
        bad("recognition.match_threshold", mt, "null")
        s["match_threshold"] = None
    for key in ("auto_assign", "learn_from_auto"):
        v = rec.get(key, True)
        if not isinstance(v, bool):
            bad(f"recognition.{key}", v, "true")
            v = True
        s[key] = v
    nm = rec.get("never_match", [])
    if isinstance(nm, (str, int)):
        nm = [nm]
    s["never_match"] = [x for x in nm if isinstance(x, (str, int)) and str(x).strip()] if isinstance(nm, list) else []
    out["settings"] = s
    return out, warns


def person_doc(lib: Lib, p: dict, existing: dict | None = None) -> dict:
    """The person.json content: the template order with the DB values; extra keys the user added are kept."""
    C = lib.T["person_file"]
    s = p.get("settings") or {}
    default = lib.fcfg["matching"][lib.embedder]["match_threshold"]
    learned = {"_about": C["learned_about"], **(p.get("learned") or {})}
    rec = {
        "match_threshold": s.get("match_threshold"),
        "_match_threshold": C["_match_threshold"].format(default=default, model=lib.embedder),
        "auto_assign": s.get("auto_assign", True), "_auto_assign": C["_auto_assign"],
        "learn_from_auto": s.get("learn_from_auto", True), "_learn_from_auto": C["_learn_from_auto"],
        "never_match": list(s.get("never_match") or []), "_never_match": C["_never_match"],
    }
    doc = {
        "_about": C["_about"],
        "id": p["code"], "_id": C["_id"],
        "number": p["id"],
        "name": p["name"], "_name": C["_name"],
        "aliases": list(p.get("aliases") or []), "_aliases": C["_aliases"],
        "notes": p.get("notes") or "", "_notes": C["_notes"],
        "status": p["status"] if p["status"] in ("active", "ignored") else "active", "_status": C["_status"],
        "merge_into": None, "_merge_into": C["_merge_into"],
        "recognition": rec,
        "learned": learned,
    }
    if existing:
        for k, v in existing.items():
            if k not in doc:
                doc[k] = v
        if isinstance(existing.get("recognition"), dict):
            for k, v in existing["recognition"].items():
                if k not in rec:
                    rec[k] = v
    return doc


def write_person_file(lib: Lib, p: dict, force: bool = False) -> bool:
    """Write / refresh person.json. A broken file is never overwritten (unless force). Returns True when written."""
    folder = lib.folder(p)
    path = folder / lib.fcfg["person_file"]
    existing = None
    if path.is_file():
        existing, err = read_person_file(path)
        if err and not force:
            return False
    doc = person_doc(lib, p, existing)
    txt = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_text(encoding="utf-8-sig") == txt:
            return False
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(txt, encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as e:
        log("faces_error", where="person_file", path=str(path), error=repr(e))
        return False


# ---------------- people: create / rename / settings ----------------

def create_person(lib: Lib, name: str | None = None, folder: Path | None = None, status: str = "active",
                  code: str | None = None) -> dict:
    """A new person (+ folder + person.json). code: keep the id of a person.json from before (lost faces.db)."""
    st_ = lib.store
    pid = st_.next_person_id()
    if code and re.fullmatch(r"p\d+", code) and not st_.person_by_code(code) and not st_.person(int(code[1:])) and code not in st_.forgotten_codes():
        pid = int(code[1:])
    else:
        code = None
    name = name or lib.fcfg["new_person_name"].format(n=pid)
    if folder is None:
        folder = unique_folder(lib.people_dir, name)
    folder.mkdir(parents=True, exist_ok=True)
    st_.add_person(name, folder=folder.name, status=status,
                   settings={"match_threshold": None, "auto_assign": True, "learn_from_auto": True, "never_match": []},
                   pid=pid, code=code)
    p = st_.person(pid)
    write_person_file(lib, p)
    log("faces_person", what="created", id=pid, name=name, folder=str(folder))
    lib.dirty()
    return p


def rename_person(lib: Lib, p: dict, new_name: str, add_alias: bool = True) -> dict:
    """Name + folder + person.json. The old real name becomes an alias (default names like 'Person 12' don't)."""
    st_ = lib.store
    new_name = " ".join(str(new_name).split())
    old = p["name"]
    aliases = list(p.get("aliases") or [])
    if add_alias and old and old != new_name and not lib.is_default_name(old) and norm_name(old) not in map(norm_name, aliases):
        aliases.append(old)
    aliases = [a for a in aliases if norm_name(a) != norm_name(new_name)]
    folder = lib.folder(p)
    target = unique_folder(lib.people_dir, new_name, own=folder)
    fname = folder.name
    last_name = new_name
    if folder.exists() and target.name != folder.name:
        try:
            os.replace(folder, target)
            fname = target.name
        except OSError:
            lib.out.line(lib.T["locked_folder"].format(path=folder), "error")
            last_name = p.get("last_name") or old   # the next learn sees the new name again and retries
    elif not folder.exists():
        fname = target.name
    st_.update_person(p["id"], name=new_name, last_name=last_name, aliases=aliases, folder=fname,
                      last_folder_name=fname)
    p2 = st_.person(p["id"])
    write_person_file(lib, p2)
    log("faces_person", what="renamed", id=p["id"], old=old, name=new_name)
    lib.dirty()
    return p2


# ---------------- faces: confirm / reject ----------------

def confirm_face(lib: Lib, face_id: int, pid: int, source: str, photo: str | None = None) -> None:
    """This face IS that person. A different person it was given to gets a 'no'."""
    st_ = lib.store
    f = st_.face(face_id)
    if not f:
        return
    if f["person_id"] and f["person_id"] != pid and f["status"] in ("auto", "confirmed", "maybe", "ignored"):
        st_.feedback(face_id, f["person_id"], "no", source, photo)
    st_.feedback(face_id, pid, "yes", source, photo)
    score = None
    mp = lib.people().get(pid)
    if mp is not None and f["embedding"] is not None and mp.examples.size:
        score = round(face_match.person_score(f["embedding"], mp, lib.mcfg["top_k"]), 4)
    st_.set_face(face_id, person_id=pid, status="confirmed", similarity=score, candidates=[])
    lib.dirty()


def reject_face(lib: Lib, face_id: int, pid: int, source: str, photo: str | None = None) -> None:
    """This face is NOT that person: a negative example; the face is re-assigned later (reevaluate)."""
    st_ = lib.store
    f = st_.face(face_id)
    if not f:
        return
    st_.feedback(face_id, pid, "no", source, photo)
    if f["person_id"] == pid:
        st_.set_face(face_id, person_id=None, status="unknown" if f["quality"] != "poor" else "poor",
                     similarity=None, candidates=[])
    lib.dirty()


# ---------------- making the folders match the DB ----------------

def _source_for(lib: Lib, image_id: int):
    for p in lib.store.paths(image_id):
        if os.path.isfile(p):
            return p
    for pl in lib.store.placements(image_id=image_id):
        q = lib.store.person(pl["person_id"])
        if q:
            f = lib.folder(q) / pl["file"]
            if f.is_file():
                return str(f)
    return None


def _photo_name(lib: Lib, image_id: int) -> str:
    paths = lib.store.paths(image_id)
    if paths:
        return Path(paths[0]).name
    pl = lib.store.placements(image_id=image_id)
    return pl[0]["file"] if pl else f"photo_{image_id}.jpg"


def _place_file(lib: Lib, src: str, dst: Path) -> str:
    """Copy or hard-link. Returns 'copy' | 'link'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if (lib.placement_mode or lib.fcfg["placement"]) == "link":
        try:
            os.link(src, dst)
            return "link"
        except OSError as e:
            if not lib.copy_fallback_noted:
                lib.out.line(lib.T["link_fallback"].format(error=e.strerror or e), "skip")
                lib.copy_fallback_noted = True
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copy"


Lib.placement_mode = None   # run option copy / link (None = faces.placement)


def _crop_bgr(lib: Lib, f: dict):
    """The face crop: from the cache, else cut again from the photo."""
    c = lib.crop_cache(f["id"])
    if c.is_file():
        return None, c
    src = _source_for(lib, f["image_id"])
    if not src:
        return None, None
    try:
        bgr, info = face_models.load_image(src, lib.fcfg)
    except face_models.ImageError:
        return None, None
    s = info["scale"]
    x, y, w, h = f["box"]
    ccfg = lib.fcfg["crops"]
    img = face_models.crop(bgr, (x * s, y * s, w * s, h * s), ccfg["size_px"], ccfg["margin"])
    if img is not None:
        try:
            face_models.save_jpeg(img, c, ccfg["jpeg_quality"])
            return None, c
        except OSError:
            return img, None
    return None, None


def crop_name(lib: Lib, f: dict) -> str:
    stem = Path(_photo_name(lib, f["image_id"])).stem
    return f"{stem}__f{f['id']}.jpg"


def sync_disk(lib: Lib, pids=None, dry: bool = False) -> dict:
    """Make the person folders match the DB. Returns {pid: {added, removed, maybe, already: [(photo, path)]},
    "_mb": MB written}. Photos: one copy per photo a person is in (auto / confirmed faces); crops: _faces for
    their faces, _maybe for faces that might be them."""
    st_ = lib.store
    T = lib.T
    persons = {p["id"]: p for p in st_.persons()}
    pids = sorted(persons) if pids is None else sorted({p for p in pids if p in persons})
    stats = {"_mb": 0.0}
    crops_dir, maybe_dir = lib.fcfg["crops_folder"], lib.fcfg["maybe_folder"]
    for pid in pids:
        p = persons[pid]
        s = stats.setdefault(pid, {"added": 0, "removed": 0, "maybe": 0, "already": [], "photos": 0})
        active = p["status"] == "active"
        folder = lib.folder(p)
        faces = st_.faces(person_id=pid, with_embedding=False) if active else []
        want_imgs = {f["image_id"] for f in faces if f["status"] in ("auto", "confirmed")}
        want_crops = {f["id"]: ("faces" if f["status"] in ("auto", "confirmed") else "maybe")
                      for f in faces if f["status"] in ("auto", "confirmed", "maybe")}
        face_by_id = {f["id"]: f for f in faces}
        placed = {pl["image_id"]: pl for pl in st_.placements(person_id=pid)}
        if active and not folder.exists() and not dry and (want_imgs or want_crops):
            folder.mkdir(parents=True, exist_ok=True)
            write_person_file(lib, p)
        # photos
        for image_id in sorted(want_imgs - set(placed)):
            img = st_.image(image_id)
            name = _photo_name(lib, image_id)
            dst = folder / name
            if dst.is_file():
                try:
                    same = sha1_file(dst) == img["sha1"]
                except OSError:
                    same = False
                if same:
                    s["already"].append((name, str(dst)))
                    if not dry:
                        stt = dst.stat()
                        st_.add_placement(pid, image_id, dst.name, img["sha1"], "user", stt.st_size, stt.st_mtime)
                    continue
                dst = unique_file(folder, name)
            if dry:
                s["added"] += 1
                continue
            src = _source_for(lib, image_id)
            if not src:
                lib.out.line(T["source_missing"].format(photo=name), "error")
                continue
            try:
                _place_file(lib, src, dst)
                stt = dst.stat()
                st_.add_placement(pid, image_id, dst.name, img["sha1"], "auto", stt.st_size, stt.st_mtime)
                stats["_mb"] += stt.st_size / 1e6
                s["added"] += 1
            except OSError as e:
                log("faces_error", where="place", path=str(dst), error=repr(e))
                lib.out.line(T["place_failed"].format(photo=name, error=e.strerror or e), "error")
        for image_id, pl in placed.items():
            if image_id in want_imgs or pl["kind"] != "auto":
                continue
            f = folder / pl["file"]
            if dry:
                s["removed"] += 1
                continue
            try:
                if f.is_file():
                    stt = f.stat()
                    if pl["size"] is not None and stt.st_size != pl["size"]:
                        # the user edited our copy: it is theirs now
                        st_.add_placement(pid, image_id, pl["file"], pl["sha1"], "user", stt.st_size, stt.st_mtime)
                        continue
                    f.unlink()
                st_.remove_placement(pid, image_id)
                s["removed"] += 1
            except OSError as e:
                lib.out.line(T["locked_file"].format(path=f, error=e.strerror or e), "error")
        # crops
        have = {c["face_id"]: c for c in st_.crops(person_id=pid)}
        for fid, kind in want_crops.items():
            cur = have.get(fid)
            sub = crops_dir if kind == "faces" else maybe_dir
            if cur and cur["kind"] == kind and (folder / sub / cur["file"]).is_file():
                continue
            if kind == "maybe":
                s["maybe"] += 1
            if dry:
                continue
            dst = folder / sub / (cur["file"] if cur else crop_name(lib, face_by_id[fid]))
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if cur:
                    old = folder / (crops_dir if cur["kind"] == "faces" else maybe_dir) / cur["file"]
                    if old.is_file() and old != dst:
                        os.replace(old, dst)
                if not dst.is_file():
                    img, cached = _crop_bgr(lib, face_by_id[fid])
                    if cached:
                        shutil.copyfile(cached, dst)
                    elif img is not None:
                        face_models.save_jpeg(img, dst, lib.fcfg["crops"]["jpeg_quality"])
                    else:
                        continue
                st_.add_crop(fid, pid, dst.name, kind)
            except OSError as e:
                log("faces_error", where="crop", path=str(dst), error=repr(e))
        for fid, c in have.items():
            if fid in want_crops:
                continue
            if dry:
                continue
            f = folder / (crops_dir if c["kind"] == "faces" else maybe_dir) / c["file"]
            try:
                f.unlink(missing_ok=True)
                st_.remove_crop(fid, pid)
            except OSError as e:
                lib.out.line(T["locked_file"].format(path=f, error=e.strerror or e), "error")
        if not dry:
            for sub in (crops_dir, maybe_dir):
                d = folder / sub
                try:
                    if d.is_dir() and not any(d.iterdir()) and sub == maybe_dir:
                        d.rmdir()
                except OSError:
                    pass
        s["photos"] = st_.photo_count(pid) if not dry else len(want_imgs | set(placed))
    if not dry:
        st_.commit()
    return stats


# ---------------- re-evaluation ----------------

def reevaluate(lib: Lib, pids, include_ids=None, statuses=("auto", "maybe", "unknown", "ignored")) -> dict:
    """Re-check auto / maybe / unknown / ignored faces near these people (confirmed faces never change by
    themselves). Returns {"changes": {pid: Counter(added=, to_maybe=, removed=)}, "touched": set of pids}."""
    st_ = lib.store
    lib.dirty()
    people = lib.people()
    pids = {p for p in pids if p in people}
    include_ids = set(include_ids or ())
    if not pids and not include_ids:
        return {"changes": {}, "touched": set()}
    rows = st_.faces(status=tuple(statuses), where="embedding IS NOT NULL")
    if not rows:
        return {"changes": {}, "touched": set()}
    E = np.stack([f["embedding"] for f in rows])
    near = np.zeros(len(rows), bool)
    if pids:
        C = np.stack([people[p].centroid for p in pids if people[p].centroid is not None] or [np.zeros(E.shape[1], np.float32)])
        near = (E @ C.T).max(axis=1) >= lib.mcfg["maybe_threshold"] + lib.mcfg["delta"] - lib.mcfg["prefilter_slack"] - 0.05
    sel = [f for i, f in enumerate(rows)
           if near[i] or f["person_id"] in pids or f["id"] in include_ids or any(c[0] in pids for c in f["candidates"])]
    if not sel:
        return {"changes": {}, "touched": set()}
    sel_ids = {f["id"] for f in sel}
    images = {f["image_id"] for f in sel}
    taken: dict = {}
    for f in st_.faces(status=("confirmed", "auto", "ignored"), with_embedding=False,
                       where=f"image_id IN ({','.join('?' * len(images))})", args=tuple(images)):
        if f["id"] not in sel_ids and f["person_id"]:
            taken.setdefault(f["image_id"], set()).add(f["person_id"])
    # faces of a photo the user dropped into a folder that wait for "which one is them?" stay as they are
    user_pl = {(r[0], r[1]) for r in st_.db.execute("SELECT person_id, image_id FROM placements WHERE kind='user'")}
    sel = [f for f in sel if not (f["status"] == "maybe" and (f["person_id"], f["image_id"]) in user_pl)]
    if not sel:
        return {"changes": {}, "touched": set()}
    sel_ids = {f["id"] for f in sel}
    rejected = st_.rejected(sel_ids)
    res = face_match.assign(sel, list(people.values()), lib.mcfg, rejected, taken)
    by_id = {f["id"]: f for f in sel}
    changes: dict = {}
    touched = set()
    for r in res:
        f = by_id[r["face_id"]]
        new_status, new_pid = r["status"], r["person_id"]
        if new_status == "unknown":
            new_pid = None
        if new_status == f["status"] and new_pid == f["person_id"]:
            # same answer: the similarity from when it was assigned is kept (it decides whether the face may
            # teach the person - re-measuring it against a changed example set would make examples drift away)
            if new_status == "maybe" and r["candidates"] != f["candidates"]:
                st_.set_face(f["id"], candidates=r["candidates"])
            continue
        old_pid, old_status = f["person_id"], f["status"]
        st_.set_face(f["id"], person_id=new_pid, status=new_status, similarity=r["score"], candidates=r["candidates"])
        if old_pid and old_status == "auto":
            c = changes.setdefault(old_pid, Counter())
            c["to_maybe" if (new_status == "maybe" and new_pid == old_pid) else "removed"] += 1
            touched.add(old_pid)
        if new_pid and new_status == "auto":
            changes.setdefault(new_pid, Counter())["added"] += 1
            touched.add(new_pid)
        if new_pid and new_status == "maybe":
            touched.add(new_pid)
            if old_pid != new_pid:
                changes.setdefault(new_pid, Counter())["maybe"] += 1
        if old_pid and old_status == "maybe":
            touched.add(old_pid)
    st_.commit()
    lib.dirty()
    log("faces_reevaluate", people=sorted(pids), faces=len(sel),
        changes={k: dict(v) for k, v in changes.items()})
    return {"changes": changes, "touched": touched}


def describe_changes(lib: Lib, changes: dict) -> str:
    """'4 photos added to Maria, 2 moved from Maria to _maybe'."""
    T = lib.T
    parts = []
    for pid, c in sorted(changes.items()):
        p = lib.store.person(pid)
        if not p:
            continue
        if c.get("added"):
            parts.append(T["change_added"].format(n=c["added"], name=p["name"]))
        if c.get("to_maybe"):
            parts.append(T["change_to_maybe"].format(n=c["to_maybe"], name=p["name"]))
        if c.get("removed"):
            parts.append(T["change_removed"].format(n=c["removed"], name=p["name"]))
        if c.get("maybe"):
            parts.append(T["change_maybe"].format(n=c["maybe"], name=p["name"]))
    return ", ".join(parts)


# ---------------- learned stats ----------------

def update_learned(lib: Lib, pids) -> None:
    """person.json 'learned': photos, faces, examples, self-similarity, most similar person, suggested threshold."""
    st_ = lib.store
    people = lib.people()
    persons = {p["id"]: p for p in st_.persons()}
    pids = [p for p in pids if p in persons and persons[p]["status"] in ("active", "ignored")]
    if not pids:
        return
    verdicts = st_.verdicts()
    confirmed_all: dict = {}
    for f in st_.faces(status="confirmed", where="embedding IS NOT NULL"):
        confirmed_all.setdefault(f["person_id"], []).append(f["embedding"])
    for pid in pids:
        p = persons[pid]
        mp = people.get(pid)
        faces = st_.faces(person_id=pid, with_embedding=False)
        conf = [f for f in faces if f["status"] == "confirmed"]
        auto = [f for f in faces if f["status"] in ("auto", "ignored")]
        rejected = sum(1 for (fid, q), v in verdicts.items() if q == pid and v == "no")
        examples = mp.examples if mp is not None else np.zeros((0, 0), np.float32)
        self_sim = None
        if examples is not None and len(examples) > 1:
            S = examples @ examples.T
            n = len(examples)
            self_sim = round(float((S.sum() - n) / (n * (n - 1))), 3)
        rej_embs = mp.negatives if mp is not None else None
        others = [(q, np.stack(e)) for q, e in confirmed_all.items() if q != pid and q in persons
                  and persons[q]["status"] == "active"]
        if not others and examples is not None and examples.size:
            others = [(q, people[q].examples) for q in people if q != pid and people[q].examples.size]
        conf_e = np.stack(confirmed_all[pid]) if confirmed_all.get(pid) else None
        value, most, overlap = (None, None, False)
        if examples is not None and examples.size:
            value, most, overlap = face_match.suggest_threshold(conf_e, examples, rej_embs, others, lib.mcfg)
        if most and most["person_id"] in persons:
            q = persons[most["person_id"]]
            most = {"number": q["id"], "name": q["name"], "similarity": most["similarity"]}
        learned = {"photos": st_.photo_count(pid), "faces_confirmed": len(conf), "faces_auto": len(auto),
                   "faces_rejected": rejected, "examples_used": 0 if examples is None else len(examples),
                   "self_similarity": self_sim, "most_similar_person": most, "suggested_threshold": value,
                   "look_alike_warning": bool(overlap), "updated": now()[:16].replace("T", " ")}
        st_.update_person(pid, learned=learned)
        p = st_.person(pid)
        if p["status"] == "active":
            write_person_file(lib, p)
    st_.commit()


# ---------------- ignore / unignore / merge / forget ----------------

def _only_ours(lib: Lib, folder: Path, pid: int) -> list[Path]:
    """Files in a person folder that the app did NOT create (user files)."""
    ours = {lib.fcfg["person_file"]}
    ours |= {pl["file"] for pl in lib.store.placements(person_id=pid) if pl["kind"] == "auto"}
    out = []
    if not folder.is_dir():
        return out
    for f in folder.iterdir():
        if f.is_dir() and f.name in (lib.fcfg["crops_folder"], lib.fcfg["maybe_folder"]):
            crops = {c["file"] for c in lib.store.crops(person_id=pid)}
            out += [g for g in f.iterdir() if g.name not in crops]
        elif f.name not in ours:
            out.append(f)
    return out


def _remove_folder_if_ours(lib: Lib, folder: Path, pid: int) -> bool:
    if not folder.is_dir():
        return True
    if _only_ours(lib, folder, pid):
        return False
    try:
        shutil.rmtree(folder)
        return True
    except OSError as e:
        lib.out.line(lib.T["locked_folder"].format(path=folder), "error")
        log("faces_error", where="remove_folder", path=str(folder), error=repr(e))
        return False


def ignore_person(lib: Lib, pid: int, source: str = "command") -> dict:
    st_ = lib.store
    p = st_.person(pid)
    for f in st_.faces(person_id=pid, with_embedding=False):
        if f["status"] == "maybe":
            st_.set_face(f["id"], person_id=None, status="unknown", candidates=[])
        elif f["status"] in ("auto", "confirmed"):
            st_.set_face(f["id"], status="ignored")
    st_.update_person(pid, status="ignored")
    lib.dirty()
    sync_disk(lib, [pid])           # our copies and crops go
    folder = lib.folder(p)
    if folder.is_dir():
        if not _remove_folder_if_ours(lib, folder, pid):
            write_person_file(lib, st_.person(pid))
    st_.commit()
    log("faces_person", what="ignored", id=pid, name=p["name"], source=source)
    return st_.person(pid)


def unignore_person(lib: Lib, pid: int) -> dict:
    st_ = lib.store
    p = st_.person(pid)
    verdicts = st_.verdicts(person_id=pid)
    for f in st_.faces(person_id=pid, status="ignored", with_embedding=False):
        st_.set_face(f["id"], status="confirmed" if verdicts.get((f["id"], pid)) == "yes" else "auto")
    folder = lib.folder(p)
    if not folder.exists():
        folder = unique_folder(lib.people_dir, p["name"])
        st_.update_person(pid, folder=folder.name, last_folder_name=folder.name)
    st_.update_person(pid, status="active")
    lib.dirty()
    p = st_.person(pid)
    lib.folder(p).mkdir(parents=True, exist_ok=True)
    write_person_file(lib, p)
    r = reevaluate(lib, [pid])
    sync_disk(lib, {pid} | r["touched"])
    log("faces_person", what="unignored", id=pid, name=p["name"])
    return st_.person(pid)


def merge_person(lib: Lib, a: int, b: int) -> dict:
    """A is the same person as B: A's faces, answers, photos and user files move to B; A becomes 'merged'."""
    st_ = lib.store
    A, B = st_.person(a), st_.person(b)
    fa, fb = lib.folder(A), lib.folder(B)
    for f in st_.faces(person_id=a, with_embedding=False):
        st_.set_face(f["id"], person_id=b)
    st_.move_feedback(a, b)
    # user-added photos move along (same photo already in B -> skipped)
    b_imgs = {pl["image_id"] for pl in st_.placements(person_id=b)}
    for pl in st_.placements(person_id=a):
        src = fa / pl["file"]
        if pl["kind"] == "user" and src.is_file() and pl["image_id"] not in b_imgs:
            fb.mkdir(parents=True, exist_ok=True)
            dst = unique_file(fb, pl["file"])
            try:
                os.replace(src, dst)
                st_.add_placement(b, pl["image_id"], dst.name, pl["sha1"], "user", pl["size"], pl["mtime"])
                b_imgs.add(pl["image_id"])
            except OSError as e:
                lib.out.line(lib.T["locked_file"].format(path=src, error=e.strerror or e), "error")
    names = [n for n in [A["name"], *(A.get("aliases") or [])] if not lib.is_default_name(n)]
    aliases = list(B.get("aliases") or [])
    for n in names:
        if norm_name(n) not in map(norm_name, aliases + [B["name"]]):
            aliases.append(n)
    sb = dict(B.get("settings") or {})
    nm = list(sb.get("never_match") or [])
    for x in (A.get("settings") or {}).get("never_match") or []:
        if x not in nm:
            nm.append(x)
    sb["never_match"] = nm
    notes = "\n".join(x for x in [B.get("notes") or "", A.get("notes") or ""] if x)
    st_.update_person(b, aliases=aliases, settings=sb, notes=notes)
    st_.update_person(a, status="merged", merged_into=b)
    lib.dirty()
    sync_disk(lib, [b])             # copies for B (from the sources, or from A's copies)
    sync_disk(lib, [a])             # A's copies and crops go
    # other user files of A (not photos we know) move to B too
    if fa.is_dir():
        for f in _only_ours(lib, fa, a):
            if f.parent == fa and f.is_file():
                try:
                    os.replace(f, unique_file(fb, f.name))
                except OSError:
                    pass
        for pl in st_.placements(person_id=a):
            st_.remove_placement(a, pl["image_id"])
        _remove_folder_if_ours(lib, fa, a)
    r = reevaluate(lib, [b], statuses=("maybe", "unknown"))   # the user said they are one: nobody is kicked out
    sync_disk(lib, {b} | r["touched"])
    update_learned(lib, [b])
    st_.commit()
    log("faces_person", what="merged", id=a, into=b, name=A["name"], into_name=B["name"])
    return st_.person(b)


def forget_person(lib: Lib, pid: int) -> dict:
    """Everything about a person is deleted; user-added files move to people/_forgotten/<name>/."""
    st_ = lib.store
    p = st_.person(pid)
    folder = lib.folder(p)
    for f in st_.faces(person_id=pid, with_embedding=False):
        st_.set_face(f["id"], person_id=None, status="unknown" if f["quality"] != "poor" else "poor",
                     similarity=None, candidates=[])
    st_.update_person(pid, status="forgotten")
    lib.dirty()
    sync_disk(lib, [pid])
    moved = 0
    if folder.is_dir():
        keep = _only_ours(lib, folder, pid)
        user_pl = {pl["file"] for pl in st_.placements(person_id=pid) if pl["kind"] == "user"}
        keep += [folder / n for n in user_pl if (folder / n).is_file() and folder / n not in keep]
        failed = []
        if keep:
            dest = unique_folder(lib.people_dir / lib.fcfg["forgotten_folder"], p["name"])
            dest.mkdir(parents=True, exist_ok=True)
            for f in keep:
                try:
                    os.replace(f, unique_file(dest, f.name))
                    moved += 1
                except OSError as e:
                    failed.append(f)
                    log("faces_error", where="forget_move", path=str(f), error=repr(e))
        if failed:
            # never delete a user file that could not be moved: park the whole folder under _forgotten instead
            # (folders starting with "_" are not scanned, so it does not come back as a person)
            try:
                (lib.people_dir / lib.fcfg["forgotten_folder"]).mkdir(parents=True, exist_ok=True)
                os.replace(folder, unique_folder(lib.people_dir / lib.fcfg["forgotten_folder"], p["name"]))
            except OSError:
                lib.out.line(lib.T["locked_folder"].format(path=folder), "error")
        else:
            try:
                shutil.rmtree(folder)
            except OSError:
                lib.out.line(lib.T["locked_folder"].format(path=folder), "error")
    st_.db.execute("DELETE FROM placements WHERE person_id=?", (pid,))
    st_.db.execute("DELETE FROM crops WHERE person_id=?", (pid,))
    st_.delete_feedback(pid)
    st_.forget_code(p["code"])
    st_.db.execute("UPDATE faces SET candidates='[]' WHERE candidates LIKE ?", (f"%[{pid},%",))
    st_.delete_person(pid)
    st_.commit()
    lib.dirty()
    log("faces_person", what="forgotten", id=pid, name=p["name"], user_files_moved=moved)
    return p


# ---------------- learning from the user's edits (§8) ----------------

def _listing(lib: Lib, folder: Path) -> dict:
    exts = image_exts(lib.fcfg)
    out = {"photos": {}, "faces": {}, "maybe": {}, "faces_noid": []}
    try:
        entries = list(folder.iterdir())
    except OSError:
        return out
    for f in entries:
        if f.is_file() and f.suffix.lower() in exts:
            out["photos"][f.name] = f
        elif f.is_dir() and f.name in (lib.fcfg["crops_folder"], lib.fcfg["maybe_folder"]):
            kind = "faces" if f.name == lib.fcfg["crops_folder"] else "maybe"
            try:
                for g in f.iterdir():
                    if not g.is_file() or g.suffix.lower() not in exts:
                        continue
                    m = FACE_ID_RX.search(g.stem + "_")
                    if m:
                        out[kind][int(m.group(1))] = g
                    elif kind == "faces":
                        out["faces_noid"].append(g)
            except OSError:
                pass
    return out


def learn(lib: Lib) -> dict:
    """Apply the user's edits in the people folders. Prints a short summary; returns
    {"lines": [...], "changes": {...}, "touched": set, "summary": str}."""
    st_ = lib.store
    T, fcfg, out = lib.T, lib.fcfg, lib.out
    pf = fcfg["person_file"]
    lib.spin and lib.spin.status("faces_learning")
    learned = Counter()          # "Maria +3 examples"
    not_them = Counter()         # 'Maria: 1 "not Maria"'
    events: list[str] = []
    touched: set = set()
    include: set = set()
    if not lib.people_dir.exists():
        lib.people_dir.mkdir(parents=True, exist_ok=True)
    persons = {p["id"]: p for p in st_.persons()}
    by_code = {p["code"]: p for p in persons.values()}

    # 1. find person folders
    found: dict = {}
    handmade = []
    unreadable: set = set()      # folder names whose person.json exists but can't be read now: leave them alone
    for d in sorted(lib.people_dir.iterdir(), key=lambda x: x.name.lower()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        pj = d / pf
        if pj.is_file():
            data, err = read_person_file(pj)
            if err is UNREADABLE:
                unreadable.add(d.name)
                out.line(T.get("unreadable_person_file", "⚠ Could not read {path} (open in another program, or not "
                               "UTF-8?) - this folder is skipped this time.").format(path=pj), "error")
                continue
            code = (data or {}).get("id") if data else code_from_raw(err[2] if err else "")
            if code in by_code and code not in found:
                found[code] = (d, data, err)
                continue
            if code in found:
                out.line(T["duplicate_folder"].format(path=d, other=found[code][0]), "error")
                continue
            if code and code not in by_code and data is not None:
                handmade.append((d, data))  # a person.json from somewhere else (e.g. DB lost): a new person
                continue
            if err:
                out.line(T["bad_person_file"].format(path=pj, line=err[0], col=err[1]), "error")
                continue
            handmade.append((d, data))
        else:
            lst = _listing(lib, d)
            if lst["photos"] or lst["faces_noid"]:
                handmade.append((d, None))

    # 2. missing folders -> ignored
    for p in persons.values():
        if p["status"] == "active" and p["code"] not in found and not ({p["folder"], p["last_folder_name"]} & unreadable):
            for f in st_.faces(person_id=p["id"], with_embedding=False):
                if f["status"] == "maybe":
                    st_.set_face(f["id"], person_id=None, status="unknown", candidates=[])
                elif f["status"] in ("auto", "confirmed"):
                    st_.set_face(f["id"], status="ignored")
            st_.update_person(p["id"], status="ignored")
            st_.db.execute("DELETE FROM placements WHERE person_id=?", (p["id"],))
            st_.db.execute("DELETE FROM crops WHERE person_id=?", (p["id"],))
            events.append(T["folder_deleted"].format(number=p["id"], name=p["name"]))
            log("faces_person", what="ignored", id=p["id"], name=p["name"], source="folder_deleted")
            lib.dirty()

    # 3-5. person.json settings, name vs folder, status / merge_into
    merges = []
    for code, (d, data, err) in found.items():
        p = by_code[code]
        pid = p["id"]
        if err:
            out.line(T["bad_person_file"].format(path=d / pf, line=err[0], col=err[1]), "error")
            fields = {}
        else:
            fields, warns = validate_person_file(data, lib, d / pf)
            for w in warns:
                out.line(w, "error")
        upd = {}
        if "settings" in fields and fields["settings"] != (p["settings"] or {}):
            old_s = p["settings"] or {}
            upd["settings"] = fields["settings"]
            if any(old_s.get(k) != fields["settings"].get(k) for k in ("match_threshold", "auto_assign", "never_match")):
                touched.add(pid)
                events.append(T["settings_changed"].format(name=p["name"]))
        for k in ("aliases", "notes"):
            if k in fields and fields[k] != (p.get(k) or ([] if k == "aliases" else "")):
                upd[k] = fields[k]
        if upd:
            st_.update_person(pid, **upd)
            lib.dirty()
        # name vs folder name
        file_name = fields.get("name", p["name"])
        name_changed = norm_name(file_name) != norm_name(p["last_name"] or p["name"])
        folder_changed = d.name != (p["last_folder_name"] or p["folder"])
        if d.name != p["folder"]:
            st_.update_person(pid, folder=d.name)
        p = st_.person(pid)
        if name_changed:
            old = p["name"]
            p = rename_person(lib, p, file_name)
            events.append(T["renamed"].format(number=pid, old=old, name=p["name"]))
        elif folder_changed:
            old = p["name"]
            new = d.name
            st_.update_person(pid, name=new, last_name=new, folder=d.name, last_folder_name=d.name)
            aliases = list(p.get("aliases") or [])
            if not lib.is_default_name(old) and norm_name(old) not in map(norm_name, aliases) and norm_name(old) != norm_name(new):
                aliases.append(old)
                st_.update_person(pid, aliases=aliases)
            p = st_.person(pid)
            write_person_file(lib, p)
            events.append(T["renamed"].format(number=pid, old=old, name=new))
            log("faces_person", what="renamed", id=pid, old=old, name=new, source="folder")
            lib.dirty()
        if not err:
            if fields.get("status") == "ignored" and p["status"] == "active":
                ignore_person(lib, pid, source="person_file")
                events.append(T["ignored"].format(number=pid, name=p["name"]))
                continue
            if fields.get("status") == "active" and p["status"] == "ignored":
                unignore_person(lib, pid)
                events.append(T["unignored"].format(number=pid, name=p["name"], photos=st_.photo_count(pid)))
            if fields.get("merge_into") is not None:
                merges.append((pid, fields["merge_into"]))
        persons[pid] = st_.person(pid)

    # hand-made folders -> new people (a person.json from before a lost faces.db keeps its id: answers replay)
    hand_pids = []
    restored: list = []
    for d, data in handmade:
        name = (data or {}).get("name") if isinstance((data or {}).get("name"), str) and data.get("name").strip() else d.name
        old_code = (data or {}).get("id") if isinstance((data or {}).get("id"), str) else None
        p = create_person(lib, name=" ".join(str(name).split()), folder=d, code=old_code)
        if old_code and p["code"] == old_code:
            restored.append(p["id"])
        if data:
            fields, _ = validate_person_file(data, lib, d / pf)
            st_.update_person(p["id"], **{k: v for k, v in fields.items() if k in ("aliases", "notes", "settings")})
            p = st_.person(p["id"])
            write_person_file(lib, p, force=True)
        persons[p["id"]] = p
        found[p["code"]] = (d, None, None)
        hand_pids.append(p["id"])
        events.append(T["handmade"].format(number=p["id"], name=p["name"]))

    # global view of the files in every active person folder
    active = {pid: p for pid, p in ((q["id"], q) for q in st_.persons(status="active"))}
    listings = {pid: _listing(lib, lib.folder(p)) for pid, p in active.items()}
    crop_seen: dict = {}  # face id -> [(pid, kind)]
    for pid, lst in listings.items():
        for kind in ("faces", "maybe"):
            for fid in lst[kind]:
                crop_seen.setdefault(fid, []).append((pid, kind))

    # 8-9. crops: deleted / moved / added
    confirms: list = []   # (face_id, pid, source)
    rejects: list = []
    for pid, lst in listings.items():
        rows = st_.crops(person_id=pid)
        expected = {(c["face_id"], c["kind"]): c for c in rows}
        for c in rows:
            fid, kind = c["face_id"], c["kind"]
            here = lst[kind].get(fid)
            if here is not None:
                if here.name != c["file"]:
                    st_.add_crop(fid, pid, here.name, kind)
                continue
            other = [x for x in crop_seen.get(fid, []) if x != (pid, kind)]
            st_.remove_crop(fid, pid)
            if kind == "maybe" and (pid, "faces") in other:
                continue  # moved from _maybe to _faces of the same person: confirmed below
            if kind == "faces" and (pid, "maybe") in other and all(q == pid for q, _ in other):
                f = st_.face(fid)  # moved from _faces to _maybe: "not sure" again
                if f and f["status"] == "auto":
                    st_.set_face(fid, status="maybe", candidates=[(pid, f["similarity"] or 0.0)])
                    touched.add(pid)
                continue
            if other and all(k == "maybe" for _, k in other):
                continue
            if st_.face(fid):
                rejects.append((fid, pid, "crop" if kind == "faces" else "maybe"))
        for kind in ("faces",):
            for fid, path in lst[kind].items():
                if (fid, "faces") in expected:
                    continue
                f = st_.face(fid)
                if not f:
                    lst["faces_noid"].append(path)
                    continue
                if f["person_id"] == pid and f["status"] == "confirmed":
                    st_.add_crop(fid, pid, path.name, "faces")
                    continue
                confirms.append((fid, pid, "crop"))
                st_.add_crop(fid, pid, path.name, "faces")
    for fid, pid, src in rejects:
        if not any(c[0] == fid and c[1] == pid for c in confirms):
            reject_face(lib, fid, pid, src)
            not_them[pid] += 1
            touched.add(pid)
            include.add(fid)
    for fid, pid, src in confirms:
        confirm_face(lib, fid, pid, src)
        learned[pid] += 1
        touched.add(pid)

    # 6. removed / edited / moved photos
    new_files: dict = {}   # pid -> [(name, path, stat, sha1)]
    sha_new: dict = {}     # sha1 -> [pid]
    for pid, lst in listings.items():
        placed = {pl["file"]: pl for pl in st_.placements(person_id=pid)}
        for name, path in lst["photos"].items():
            try:
                stt = path.stat()
            except OSError:
                continue
            if name in placed:
                pl = placed[name]
                if pl["size"] == stt.st_size and abs((pl["mtime"] or 0) - stt.st_mtime) < 0.01:
                    continue
                try:
                    sha = sha1_file(path)
                except OSError:
                    continue
                st_.add_placement(pid, pl["image_id"], name, pl["sha1"] if sha == pl["sha1"] else sha, pl["kind"],
                                  stt.st_size, stt.st_mtime)  # edited (rotated, cropped): still the same photo
                continue
            row = st_.image_by_path(str(path))
            if row and row["size"] == stt.st_size and abs((row["mtime"] or 0) - stt.st_mtime) < 0.01:
                sha = row["sha1"]
            else:
                try:
                    sha = sha1_file(path)
                except OSError:
                    continue
            new_files.setdefault(pid, []).append((name, path, stt, sha))
            sha_new.setdefault(sha, []).append(pid)
    for pid, lst in listings.items():
        folder = lib.folder(active[pid])
        for pl in st_.placements(person_id=pid):
            if (folder / pl["file"]).is_file():
                continue
            renamed = next((x for x in new_files.get(pid, []) if x[3] == pl["sha1"]), None)
            if renamed:  # the user renamed the file
                st_.add_placement(pid, pl["image_id"], renamed[0], pl["sha1"], pl["kind"], renamed[2].st_size,
                                  renamed[2].st_mtime)
                new_files[pid].remove(renamed)
                continue
            st_.remove_placement(pid, pl["image_id"])
            moved_to = [q for q in sha_new.get(pl["sha1"], []) if q != pid]
            for f in st_.faces(image_id=pl["image_id"], person_id=pid, with_embedding=False):
                if f["status"] in ("auto", "confirmed"):
                    reject_face(lib, f["id"], pid, "folder")
                    not_them[pid] += 1
                    include.add(f["id"])
            touched.add(pid)
            if not moved_to:
                events.append(T["photo_removed"].format(photo=pl["file"], name=active[pid]["name"]))

    # 7. new photos in person folders (+ images in _faces without a face id)
    pending = []
    for pid, lst in listings.items():
        for name, path, stt, sha in new_files.get(pid, []):
            pending.append((pid, path, stt, "photo"))
        for path in lst["faces_noid"]:
            try:
                pending.append((pid, path, path.stat(), "crop"))
            except OSError:
                pass
    if pending:
        results = []
        for pid, path, stt, kind in pending:
            r = ingest(lib, path, stt, source="person_folder")
            results.append((pid, path, stt, kind, r))
        st_.commit()
        lib.dirty()
        # photos with one face first: they give a hand-made person its first examples
        results.sort(key=lambda x: (x[4]["faces"] != 1, str(x[1]).lower()))
        for pid, path, stt, kind, r in results:
            if r["image_id"] is None:
                continue
            img = st_.image(r["image_id"])
            if kind == "photo":
                st_.add_placement(pid, r["image_id"], path.name, img["sha1"], "user", stt.st_size, stt.st_mtime)
            faces = [f for f in st_.faces(image_id=r["image_id"]) if f["embedding"] is not None]
            if any(f["person_id"] == pid and f["status"] == "confirmed" for f in faces):
                continue
            pick = _pick_face(lib, faces, pid)
            p = active[pid]
            if pick == "need_example":
                out.line(T["need_single_face"].format(name=p["name"]), "maybe")
                for f in faces:
                    if f["status"] not in ("confirmed",) and f["person_id"] in (None, pid):
                        st_.set_face(f["id"], person_id=pid, status="maybe", candidates=[(pid, 0.0)])
                touched.add(pid)
            elif pick is None:
                if faces:
                    out.line(T["unclear_face"].format(photo=path.name, name=p["name"]), "maybe")
                    for f in faces:
                        if f["status"] != "confirmed" and f["person_id"] in (None, pid):
                            st_.set_face(f["id"], person_id=pid, status="maybe", candidates=[(pid, f.get("_score") or 0.0)])
                    touched.add(pid)
                else:
                    out.line(T["no_face_in_photo"].format(photo=path.name, name=p["name"]), "skip")
            else:
                confirm_face(lib, pick["id"], pid, "folder", photo=path.name)
                learned[pid] += 1
                touched.add(pid)
            lib.dirty()

    # faces.db was lost: replay what the user taught it (feedback.jsonl) for the people found again
    if restored:
        n = replay_feedback(lib, restored)
        if n:
            events.append(T["replayed"].format(n=n))
            touched |= set(restored)

    # merges asked for in person.json
    for a, ref in merges:
        persons_now = {p["id"]: p for p in st_.persons()}
        b = resolve_ref(ref, persons_now)
        if b and b != a and persons_now.get(a, {}).get("status") == "active":
            A = persons_now[a]
            B = merge_person(lib, a, b)
            events.append(T["merged"].format(from_number=a, from_name=A["name"], number=b, name=B["name"],
                                             photos=st_.photo_count(b)))
            touched.discard(a)
            touched.add(b)
        else:
            out.line(T["bad_merge"].format(name=persons_now.get(a, {}).get("name", a), ref=ref), "error")

    st_.commit()
    # 10. rebuild examples, re-evaluate, folders, learned stats
    changes, after_touched = {}, set()
    if touched or include:
        r = reevaluate(lib, touched, include_ids=include)
        changes, after_touched = r["changes"], r["touched"]
    all_touched = (touched | after_touched | set(hand_pids)) & {p["id"] for p in st_.persons(status=("active", "ignored"))}
    if all_touched:
        sync_disk(lib, all_touched)
        update_learned(lib, all_touched)
    lib.spin and lib.spin.stop()
    parts = []
    names = {p["id"]: p["name"] for p in st_.persons()}
    for pid, n in sorted(learned.items()):
        parts.append(T["learned_examples"].format(name=names.get(pid, pid), n=n))
    for pid, n in sorted(not_them.items()):
        parts.append(T["learned_not"].format(name=names.get(pid, pid), n=n))
    parts += events
    summary = ""
    if parts:
        summary = T["learned"].format(changes=" · ".join(parts))
        out.line(summary, "known")
    after = describe_changes(lib, changes)
    if after:
        out.line(T["after_changes"].format(changes=after), "known")
        summary += ("\n" if summary else "") + T["after_changes"].format(changes=after)
    log("faces_sync", learned=dict(learned), not_them=dict(not_them), events=events,
        changes={k: dict(v) for k, v in changes.items()})
    return {"summary": summary, "changes": changes, "touched": all_touched, "events": events,
            "learned": dict(learned), "rejected": dict(not_them)}


def replay_feedback(lib: Lib, pids=None, sha1s=None) -> int:
    """Answers from feedback.jsonl (for these people / these photos), matched to faces by photo sha1 + box -
    how a lost faces.db gets back what the user taught it. The latest answer per face wins. Returns how many."""
    st_ = lib.store
    if not st_.feedback_path.is_file():
        return 0
    people = [st_.person(pid) for pid in pids] if pids is not None else st_.persons(status=("active", "ignored"))
    codes = {p["code"]: p["id"] for p in people if p}
    sha1s = set(sha1s) if sha1s is not None else None
    latest: dict = {}
    try:
        with open(st_.feedback_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("forgotten"):  # forget_person marker: earlier answers for that code no longer count
                    latest = {k: v for k, v in latest.items() if k[2] != rec["forgotten"]}
                    continue
                code = (rec.get("person") or {}).get("code")
                if code in codes and rec.get("photo_sha1") and rec.get("box") and (
                        sha1s is None or rec["photo_sha1"] in sha1s):
                    latest[(rec["photo_sha1"], tuple(rec["box"]), code)] = rec
    except OSError:
        return 0
    applied = 0
    for (sha1, box, code), rec in latest.items():
        img = st_.image_by_sha1(sha1)
        if not img:
            continue
        cands = [(_iou(f["box"], box), f) for f in st_.faces(image_id=img["id"], with_embedding=False)]
        cands = [c for c in cands if c[0] >= 0.5]
        if not cands:
            continue
        f = max(cands, key=lambda c: c[0])[1]
        pid = codes[code]
        if rec["verdict"] == "yes":
            confirm_face(lib, f["id"], pid, "replay")
        else:
            reject_face(lib, f["id"], pid, "replay")
        applied += 1
    st_.commit()
    return applied


def _pick_face(lib: Lib, faces: list, pid: int):
    """Which face of a photo the user dropped into P's folder is P: one face -> that one; several -> the one
    clearly closest to P. None = not clear; 'need_example' = P has no example yet and the photo has several faces."""
    good = [f for f in faces if f["quality"] in ("good", "ok")]
    if len(good) == 1:
        return good[0]
    if not good:
        return None
    mp = lib.people().get(pid)
    if mp is None or not mp.examples.size:
        return "need_example"
    k = lib.mcfg["top_k"]
    scored = sorted(((face_match.person_score(f["embedding"], mp, k), f) for f in good), key=lambda x: -x[0])
    for s, f in scored:
        f["_score"] = round(s, 3)
    best, nxt = scored[0], scored[1] if len(scored) > 1 else (-1.0, None)
    if best[0] >= lib.mcfg["pick_min_similarity"] and best[0] - nxt[0] >= lib.mcfg["margin"]:
        return best[1]
    return None
