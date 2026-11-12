"""
Face recognition and sorting photos by person: '/faces <folder|photo> [options]', or just write
"sort the photos in E:\\Photos\\2024 by face" / "who is in E:\\Photos\\IMG_0450.jpg?".

    1. finds every face in every photo (YuNet) and stores each one: box, 5 points, quality, fingerprint
       (ArcFace / SFace), crop                                         -> face_models, face_store
    2. recognizes who it is by comparing it with everyone it knows     -> face_match
    3. every new person gets a folder (downloads/faces/people/Person 12) with a person.json; every photo is
       copied (or hard-linked) into the folder of every person in it; originals are never touched -> face_sync
    4. learns from the user's corrections: edits in the folders, person.json, chat ("12 is Maria")
    5. says when it is unsure: 'not sure' list + _maybe folders; warns about photos already sorted (with path)

Incremental: photos already read are not read again (path + size + mtime, or the same bytes elsewhere).
Ctrl+C or typing "stop" ends the run gracefully - everything so far is saved; the same command continues.
Everything runs locally on the CPU. All texts and settings: config.json -> faces.
"""

import fnmatch
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from photos import face_match
from photos import face_models
from photos import face_sync
from core import live
from core import ui
from core.debuglog import log
from photos.face_store import Busy

SUBCOMMANDS = ("people", "who", "rename", "merge", "maybe", "yes", "no", "ignore", "unignore", "forget", "learn",
               "rebuild")

OPEN_MAYBE: list = []      # the numbered 'not sure' list last shown: [{n, face_id, photo, candidates, crop, done}]
_CLOUD_WARNED = False

_PATH_START = re.compile(r'"|(?<![\w])[A-Za-z]:[\\/]|\\\\[^\\\s]|~[\\/]')
_TRAIL = ".,;:!?)]}'\"»“”"


# ---------------- understanding the request ----------------

def find_path(text: str):
    """The longest existing local path in the text (quoted or not, spaces and Cyrillic allowed) ->
    (path, the text without it), or None."""
    best = None
    for m in _PATH_START.finditer(text):
        i = m.start()
        if text[i] == '"':
            j = text.find('"', i + 1)
            if j < 0:
                continue
            cand = text[i + 1:j]
            if cand and os.path.exists(os.path.expanduser(cand)):
                return os.path.expanduser(cand), (text[:i] + " " + text[j + 1:]).strip()
            continue
        # unquoted: try the longest piece first (ends at a space or the end), trailing punctuation dropped
        ends = [k for k in range(len(text), i, -1) if k == len(text) or text[k] == " "]
        for k in ends:
            cand = text[i:k]
            # Windows ignores trailing dots / spaces ("photo.jpg." exists): try without the punctuation first
            for c in (cand.rstrip(_TRAIL + " "), cand):
                c2 = os.path.expanduser(c)
                if c and os.path.exists(c2):
                    best = (c, i, i + len(c), c2)
                    break
            if best and best[1] == i:
                break
        if best:
            break
    if not best:
        return None
    _, i, j, p = best
    return p, (text[:i] + " " + text[j:]).strip()


def _is_image(path: str, fcfg: dict) -> bool:
    return Path(path).suffix.lower() in face_sync.image_exts(fcfg)


def parse_options(rest: str, fcfg: dict) -> dict:
    opts = {"check": False, "placement": None, "recursive": fcfg["recursive"], "new": None, "delta": 0.0,
            "max": None, "force": False}
    low = rest.lower()
    step = fcfg["matching"]["strict_step"]
    for w in re.findall(r"[\w=]+", low):
        if w in ("check", "dry", "preview"):
            opts["check"] = True
        elif w in ("copy", "link"):
            opts["placement"] = w
        elif w in ("norecurse", "norecursive", "flat"):
            opts["recursive"] = False
        elif w == "strict":
            opts["delta"] = step
        elif w == "loose":
            opts["delta"] = -step
        elif w == "force":
            opts["force"] = True
        elif re.fullmatch(r"new=\d+", w):
            opts["new"] = max(1, int(w.split("=")[1]))
        elif re.fullmatch(r"max=\d+", w):
            opts["max"] = max(1, int(w.split("=")[1]))
    return opts


def parse_request(message: str, fcfg: dict):
    """'/faces <path> [options]', '/faces who <photo>' or a plain chat message with a local path and face
    words -> (mode 'scan'|'who', path, options), or None."""
    msg = message.strip()
    if not fcfg["enabled"] or not msg:
        return None
    cmd, _, rest = msg.partition(" ")
    if cmd.lower() == "/faces":
        rest = rest.strip()
        mode = "scan"
        if rest.split()[:1] and rest.split()[0].lower() == "who":
            mode, rest = "who", rest[3:].strip()
        found = find_path(rest)
        if not found:
            return None
        path, left = found
        if mode == "who" and not os.path.isfile(path):
            return None
        return mode, path, parse_options(left, fcfg)
    if msg.startswith("/"):
        return None
    if re.search(r"https?://", msg, re.I):
        return None
    found = find_path(msg)
    if not found:
        return None
    path, left = found
    is_file = os.path.isfile(path)
    if is_file and not _is_image(path, fcfg):
        return None
    words = msg.replace(path, " ")
    faces_words = any(re.search(p, words, re.I) for p in fcfg["intent_patterns"])
    who_words = any(re.search(p, words, re.I) for p in fcfg["who_patterns"])
    if is_file and who_words:
        return "who", path, parse_options("", fcfg)
    if faces_words:
        return "scan", path, parse_options(left, fcfg)
    return None


# ---------------- small helpers ----------------

def fmt_time(sec) -> str:
    sec = int(round(sec or 0))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _n(x) -> str:
    return f"{x:,}"


def in_library(path, fcfg: dict) -> bool:
    try:
        p = Path(path).resolve()
        lib = face_sync.library_dir(fcfg).resolve()
    except OSError:
        return False
    return p == lib or lib in p.parents


def _cloud_warning(lib, out) -> None:
    global _CLOUD_WARNED
    if _CLOUD_WARNED:
        return
    _CLOUD_WARNED = True
    try:
        d = lib.dir.resolve()
    except OSError:
        return
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        root = os.environ.get(var)
        if root:
            try:
                r = Path(root).resolve()
            except OSError:
                continue
            if d == r or r in d.parents:
                out.line(lib.T["cloud_library"].format(path=d), "error")
                return


def _skip_dir(name: str, full: str, fcfg: dict, lib_dir: Path) -> bool:
    for pat in fcfg["skip_dirs"]:
        if fnmatch.fnmatch(name.lower(), pat.lower()):
            return True
    try:
        st = os.stat(full)
        if getattr(st, "st_file_attributes", 0) & 0x6:  # hidden | system
            return True
        p = Path(full).resolve()
        if p == lib_dir or lib_dir in p.parents:
            return True
    except OSError:
        return True
    return False


def walk(root: str, fcfg: dict, recursive: bool) -> dict:
    """Photos under root (folder by folder, in file-name order). Returns {photos: [(path, stat)], online_only,
    unsupported}."""
    exts = face_sync.image_exts(fcfg)
    heic = face_models.heic_supported()
    lib_dir = face_sync.library_dir(fcfg).resolve()
    out = {"photos": [], "online_only": 0, "unsupported": 0}

    def take(path):
        ext = Path(path).suffix.lower()
        if ext not in exts:
            return
        if ext in (".heic", ".heif") and not heic:
            out["unsupported"] += 1
            return
        try:
            st = os.stat(path)
        except OSError:
            return
        if face_models.is_online_only(path, st) and not fcfg["include_online_only"]:
            out["online_only"] += 1
            return
        out["photos"].append((path, st))

    if os.path.isfile(root):
        take(root)
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted((d for d in dirnames if not _skip_dir(d, os.path.join(dirpath, d), fcfg, lib_dir)),
                             key=str.lower) if recursive else []
        for f in sorted(filenames, key=str.lower):
            take(os.path.join(dirpath, f))
    return out


def _eta(started: float, done: int, total: int, T: dict) -> str:
    if done < 3:
        return ""
    left = (time.monotonic() - started) / done * (total - done)
    return T["eta"].format(time=fmt_time(left)) if left >= 5 else ""


def _person_line(p: dict) -> dict:
    return {"number": p["id"], "name": p["name"]}


# ---------------- the scan ----------------

def _base_result(path: str, mode: str) -> dict:
    return {"source": str(path), "mode": mode, "photos": 0, "new": 0, "unchanged": 0, "reused_copies": 0,
            "no_faces": 0, "unreadable": 0, "skipped_files": 0, "unsupported": 0, "online_only": 0,
            "left_for_next_run": 0, "faces": 0, "poor_faces": 0, "new_people": [], "known_people": [],
            "already_sorted": [], "maybe": [], "errors": [], "placed_mb": 0.0, "stopped": False, "done": 0,
            "seconds": 0.0, "report_path": None, "learned": "", "error": None, "check": mode == "check",
            "photo_rows": []}


def run(path: str, opts: dict, cfg: dict, spin) -> dict:
    """Find and recognize the faces in a folder (or one photo) and sort the photos into person folders."""
    fcfg = cfg["faces"]
    T = fcfg["texts"]
    check = bool(opts.get("check"))
    result = _base_result(path, "check" if check else "scan")
    result["opts"] = {k: v for k, v in opts.items() if k != "check"}
    t0 = time.monotonic()
    if not os.path.exists(path):
        result["error"] = T["not_found"].format(path=path)
        return result
    if os.path.isfile(path) and not _is_image(path, fcfg):
        result["error"] = T["not_a_photo"].format(path=path, ext=Path(path).suffix or "?")
        return result
    if in_library(path, fcfg):
        result["error"] = T["library_refused"]
        return result
    try:
        lib = face_sync.Lib(cfg, spin, delta=opts.get("delta") or 0.0)
    except Busy:
        result["error"] = T["busy"]
        return result
    try:
        return _run(lib, path, opts, result, t0)
    except face_models.ModelError as e:
        result["error"] = str(e)
        return result
    finally:
        lib.close()


def _run(lib, path: str, opts: dict, result: dict, t0: float) -> dict:
    fcfg, T, st, out, spin = lib.fcfg, lib.T, lib.store, lib.out, lib.spin
    check = result["check"]
    if opts.get("new"):
        lib.mcfg["min_faces_for_new_person"] = opts["new"]
    lib.placement_mode = opts.get("placement")
    _cloud_warning(lib, out)
    have = st.meta("embedder")
    if have and have != lib.embedder:
        result["error"] = T["model_mismatch"].format(have=have, want=lib.embedder)
        return result
    if not check:
        st.backup()
        lr = face_sync.learn(lib)
        result["learned"] = lr["summary"]

    # 1. the photos
    spin.status("faces_listing", folder=Path(path).name or path)
    w = walk(path, fcfg, opts.get("recursive", True))
    spin.stop()
    photos = w["photos"]
    result["online_only"], result["unsupported"] = w["online_only"], w["unsupported"]
    if w["online_only"]:
        out.line(T["online_only"].format(n=w["online_only"]), "maybe")
    limit = opts.get("max") or fcfg["max_photos_per_run"]
    todo = photos[:limit]
    result["left_for_next_run"] = len(photos) - len(todo)
    result["photos"] = len(todo)
    if not todo:
        result["error"] = T["no_photos"].format(path=path)
        result["seconds"] = round(time.monotonic() - t0, 1)
        return result

    # 2. read every photo (cached ones cost ~1 ms)
    run_images: dict = {}           # image_id -> photo path in this run
    new_sha1s: set = set()          # photos new to the DB
    reused: list = []
    started = time.monotonic()
    faces_so_far = 0
    total = len(todo)
    i = 0
    try:
        for i, (p, stt) in enumerate(todo, 1):
            if live.take_if(lambda t: live.is_stop([t])):
                result["stopped"] = True
                out.line(T["stop_heard"], "maybe")
                i -= 1
                break
            spin.status("faces_photo", n=_n(i), total=_n(total), name=Path(p).name, faces=_n(faces_so_far),
                        eta=_eta(started, i - 1, total, T))
            r = face_sync.ingest(lib, p, stt, force=opts.get("force", False))
            how = r["how"]
            if how == "new":
                result["new"] += 1
            elif how == "unchanged":
                result["unchanged"] += 1
            elif how == "reused":
                result["reused_copies"] += 1
                reused.append((r["image_id"], p))
            if r["status"] == "unreadable":
                result["unreadable"] += 1
                result["errors"].append({"photo": p, "reason": r["error"]})
                if how == "error":
                    out.line(T["unreadable_line"].format(photo=Path(p).name, reason=r["error"]), "error")
            elif r["status"] == "skipped":
                if (r.get("kind") or "") == "unsupported":
                    result["unsupported"] += 1
                else:
                    result["skipped_files"] += 1
            elif r["status"] == "no_faces":
                result["no_faces"] += 1
            if r["image_id"] is not None:
                run_images.setdefault(r["image_id"], p)
            if how == "new" and r.get("sha1"):
                new_sha1s.add(r["sha1"])
            faces_so_far += r["faces"] or 0
            result["photo_rows"].append({"path": p, "image_id": r["image_id"], "how": how, "status": r["status"],
                                         "error": r["error"]})
            if i % fcfg["commit_every"] == 0:
                st.commit()
        result["done"] = i
    except KeyboardInterrupt:
        ui.note_cancel()
        result["stopped"] = True
        result["done"] = max(0, i - 1)
        st.commit()
        out.line(T["stop_heard"], "maybe")
    finally:
        spin.stop()
    st.commit()
    if new_sha1s and not check:
        # photos new to the DB that feedback.jsonl already knows: the DB was lost - their answers come back
        n = face_sync.replay_feedback(lib, None, new_sha1s)
        if n:
            out.line(T["replayed"].format(n=n), "known")
            lib.dirty()
    if not have and st.db.execute("SELECT 1 FROM faces WHERE model IS NOT NULL LIMIT 1").fetchone():
        st.set_meta("embedder", lib.embedder)
        st.commit()

    # 3. recognize
    ids = sorted(run_images)
    if ids:
        q = f"image_id IN ({','.join('?' * len(ids))})"
        all_faces = st.faces(where=q, args=tuple(ids))
    else:
        all_faces = []
    result["faces"] = len(all_faces)
    result["poor_faces"] = sum(1 for f in all_faces if f["quality"] == "poor")
    unknown = [f for f in all_faces if f["status"] == "unknown" and f["embedding"] is not None]
    spin.status("faces_matching", faces=_n(len(unknown)))
    people = lib.people()
    taken: dict = {}
    for f in all_faces:
        if f["status"] in ("confirmed", "auto", "ignored") and f["person_id"]:
            taken.setdefault(f["image_id"], set()).add(f["person_id"])
    assigned = face_match.assign(unknown, list(people.values()), lib.mcfg, st.rejected([f["id"] for f in unknown]),
                                 taken) if unknown else []
    spin.stop()
    status_now = {f["id"]: {"status": f["status"], "person_id": f["person_id"], "candidates": f["candidates"],
                            "similarity": f["similarity"]} for f in all_faces}
    for r in assigned:
        if r["status"] == "unknown":
            continue
        status_now[r["face_id"]] = {"status": r["status"], "person_id": r["person_id"], "candidates": r["candidates"],
                                    "similarity": r["score"]}
        if not check:
            st.set_face(r["face_id"], status=r["status"], person_id=r["person_id"], similarity=r["score"],
                        candidates=r["candidates"])
    if not check:
        st.commit()
        lib.dirty()

    # 4. new people from the faces nobody knows
    spin.status("faces_clustering")
    pool = [f for f in all_faces if status_now[f["id"]]["status"] == "unknown" and f["quality"] == "good"
            and f["embedding"] is not None]
    if lib.mcfg["min_faces_for_new_person"] > 1:
        seen = {f["id"] for f in pool}
        rej = st.rejected()
        pool += [f for f in st.faces(status="unknown", where="quality='good' AND embedding IS NOT NULL")
                 if f["id"] not in seen and not rej.get(f["id"])]
    if pool:
        rej = st.rejected([f["id"] for f in pool])
        pool = [f for f in pool if not rej.get(f["id"])]   # a face the user said 'no' to doesn't start a person
    paths = {}
    for f in pool:
        paths[f["image_id"]] = run_images.get(f["image_id"]) or (st.paths(f["image_id"]) or [""])[0]
    for f in pool:
        f["rank"] = (-(f["size_px"] or 0) * (f["sharpness"] or 0), paths[f["image_id"]].lower(), f["idx"])
    groups = face_match.cluster(pool, lib.mcfg) if pool else []
    spin.stop()
    by_id = {f["id"]: f for f in pool}
    new_people = []
    for g in groups:
        embs = np.stack([by_id[fid]["embedding"] for fid in g])
        if check:
            new_people.append({"number": None, "name": T["would_be_new"], "photos": len({by_id[x]["image_id"] for x in g}),
                               "folder": None, "faces": g})
            continue
        p = face_sync.create_person(lib)
        for j, fid in enumerate(g):
            if len(g) == 1:
                sim = 1.0
            else:
                others = np.delete(embs, j, axis=0)
                sim = round(face_match.topk_mean(others @ embs[j], lib.mcfg["top_k"]), 4)
            st.set_face(fid, status="auto", person_id=p["id"], similarity=sim, candidates=[])
            status_now[fid] = {"status": "auto", "person_id": p["id"], "candidates": [], "similarity": sim}
        new_people.append({"number": p["id"], "name": p["name"], "photos": len({by_id[x]["image_id"] for x in g}),
                           "folder": str(lib.folder(p)), "faces": g})
    if not check:
        st.commit()
        lib.dirty()

    # 5. folders
    touched = {s["person_id"] for s in status_now.values()
               if s["person_id"] and s["status"] in ("auto", "confirmed", "maybe")}
    touched |= {p["number"] for p in new_people if p["number"]}
    persons = {p["id"]: p for p in st.persons()}
    if check:
        stats = face_sync.sync_disk(lib, touched, dry=True)
        # faces matched in this run are not in the DB in check mode: count their photos
        add = {}
        img_of = {f["id"]: f["image_id"] for f in all_faces}
        for fid, s in status_now.items():
            if s["status"] == "auto" and s["person_id"] in persons:
                img = img_of[fid]
                if not st.placements(person_id=s["person_id"], image_id=img):
                    add.setdefault(s["person_id"], set()).add(img)
        for pid, imgs in add.items():
            stats.setdefault(pid, {"added": 0, "removed": 0, "maybe": 0, "already": [], "photos": 0})
            stats[pid]["added"] = len(imgs)
            stats[pid]["photos"] = st.photo_count(pid) + len(imgs)
    else:
        spin.status("faces_placing", n=len(touched), total=len(touched))
        stats = face_sync.sync_disk(lib, touched)
        spin.stop()
        face_sync.update_learned(lib, touched)
    result["placed_mb"] = round(stats.get("_mb", 0.0), 1)
    persons = {p["id"]: p for p in st.persons()}
    new_ids = {p["number"] for p in new_people}
    for np_ in new_people:
        if np_["number"]:
            np_["photos"] = st.photo_count(np_["number"])
    result["new_people"] = [{k: v for k, v in p.items() if k != "faces"} for p in new_people]
    for pid in sorted(touched - new_ids):
        s = stats.get(pid)
        p = persons.get(pid)
        if not s or not p or p["status"] != "active" or not s["added"]:
            continue
        result["known_people"].append({"number": pid, "name": p["name"], "added": s["added"], "total": s["photos"],
                                       "folder": str(lib.folder(p))})
    result["known_people"].sort(key=lambda x: -x["added"])

    # 6. already sorted: the same photo (other name / place) is already in a person's folder
    seen_img = set()
    for image_id, p in reused:
        if image_id in seen_img:
            continue
        seen_img.add(image_id)
        for pl in st.placements(image_id=image_id):
            q = persons.get(pl["person_id"])
            if not q or q["status"] != "active":
                continue
            dest = lib.folder(q) / pl["file"]
            if dest.is_file():
                result["already_sorted"].append({"photo": Path(p).name, "path": str(dest), "name": q["name"]})
    for pid, s in stats.items():
        if pid == "_mb":
            continue
        for name, dest in s.get("already", []):
            q = persons.get(pid)
            result["already_sorted"].append({"photo": name, "path": dest, "name": q["name"] if q else "?"})
    cap = fcfg["report_list_max"]
    for a in result["already_sorted"][:cap]:
        out.line(T["already_sorted"].format(photo=a["photo"], name=a["name"]), "maybe")
        out.path(a["path"])

    # 7. not sure
    maybe = []
    for f in all_faces:
        s = status_now[f["id"]]
        if s["status"] != "maybe":
            continue
        cands = [{"number": pid, "name": persons[pid]["name"], "score": round(sc, 2)}
                 for pid, sc in (s["candidates"] or []) if pid in persons]
        crop = None
        if not check and s["person_id"] in persons:
            c = st.crops(person_id=s["person_id"], face_id=f["id"])
            if c:
                crop = str(lib.folder(persons[s["person_id"]]) / fcfg["maybe_folder"] / c[0]["file"])
        maybe.append({"face_id": f["id"], "photo": Path(run_images.get(f["image_id"], "")).name,
                      "candidates": cands, "crop": crop})
    for n, m in enumerate(maybe, 1):
        m["n"] = n
    result["maybe"] = maybe
    if not check:
        OPEN_MAYBE[:] = [dict(m, done=False) for m in maybe]
    # faces seen but not clear enough to start a new person (ok quality, or a stranger seen too few times)
    result["unsorted_faces"] = sum(1 for f in all_faces if status_now[f["id"]]["status"] == "unknown"
                                   and f["quality"] != "poor")

    result["seconds"] = round(time.monotonic() - t0, 1)
    result["report_path"] = _write_report(lib, result, all_faces, status_now, persons)
    log("faces_scan_done", source=path, mode=result["mode"], photos=result["photos"], new=result["new"],
        unchanged=result["unchanged"], reused=result["reused_copies"], faces=result["faces"],
        poor=result["poor_faces"], new_people=len(result["new_people"]), known=len(result["known_people"]),
        maybe=len(maybe), placed_mb=result["placed_mb"], seconds=result["seconds"], stopped=result["stopped"])
    return result


def _write_report(lib, result: dict, all_faces: list, status_now: dict, persons: dict):
    fcfg = lib.fcfg
    d = lib.dir / fcfg["reports_folder"]
    try:
        d.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + " " + face_sync.safe_name(Path(result["source"]).name or "photos")
        path = d / f"{name}.json"
        by_img: dict = {}
        for f in all_faces:
            s = status_now.get(f["id"], {})
            pid = s.get("person_id")
            by_img.setdefault(f["image_id"], []).append({
                "face_id": f["id"], "box": [round(v, 1) for v in f["box"]], "quality": f["quality"],
                "size_px": f["size_px"], "sharpness": f["sharpness"], "yaw": f["yaw"],
                "person": persons[pid]["name"] if pid in persons else None, "number": pid,
                "status": s.get("status"), "score": s.get("similarity")})
        photos = [{**{k: v for k, v in r.items() if k != "image_id"}, "faces": by_img.get(r["image_id"], [])}
                  for r in result["photo_rows"]]
        data = {k: v for k, v in result.items() if k != "photo_rows"}
        data["photos_list"] = photos
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return str(path)
    except OSError as e:
        log("faces_error", where="report", error=repr(e))
        return None


# ---------------- who is in this photo ----------------

def positions(n: int, T: dict) -> list[str]:
    names = T["positions"]
    if n == 1:
        return [T["position_one"]]
    if n == 2:
        return [names[0], names[2]]
    if n == 3:
        return list(names)
    return [T["position_nth"].format(n=i) for i in range(1, n + 1)]


def who(path: str, cfg: dict, spin) -> dict:
    """Only say who is in one photo (nothing is sorted or saved)."""
    fcfg = cfg["faces"]
    T = fcfg["texts"]
    res = {"mode": "who", "source": path, "photo": Path(path).name, "faces": [], "error": None}
    if not os.path.isfile(path):
        res["error"] = T["not_found"].format(path=path)
        return res
    try:
        lib = face_sync.Lib(cfg, spin)
    except Busy:
        res["error"] = T["busy"]
        return res
    try:
        st = lib.store
        persons = {p["id"]: p for p in st.persons()}
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            res["error"] = T["unreadable_line"].format(photo=Path(path).name, reason=e.strerror or e)
            return res
        import hashlib
        img = st.image_by_sha1(hashlib.sha1(data).hexdigest())
        faces = []
        if img and img["model"] == lib.embedder:
            for f in st.faces(image_id=img["id"]):
                faces.append({"status": f["status"], "person_id": f["person_id"], "score": f["similarity"],
                              "candidates": f["candidates"], "quality": f["quality"]})
        else:
            try:
                bgr, info = face_models.load_image(path, fcfg, data)
            except face_models.ImageError as e:
                res["error"] = T["unreadable_line"].format(photo=Path(path).name, reason=e)
                return res
            det, emb = lib.models()
            spin.status("faces_matching", faces="")
            dets = det.detect(bgr)
            rows = []
            for i, d in enumerate(dets):
                a = emb.aligned(bgr, d)
                q = face_models.quality(d, a, fcfg["quality"], info["scale"])
                rows.append({"id": i, "image_id": 0, "quality": q["quality"], "a": a})
            ok = [r for r in rows if r["quality"] != "poor"]
            if ok:
                vecs = emb.embed([r["a"] for r in ok])
                for r, v in zip(ok, vecs):
                    r["embedding"] = v
            got = {r["face_id"]: r for r in face_match.assign(ok, list(lib.people().values()), lib.mcfg)} if ok else {}
            spin.stop()
            for r in rows:
                g = got.get(r["id"])
                faces.append({"status": g["status"] if g else "poor", "person_id": g["person_id"] if g else None,
                              "score": g["score"] if g else None, "candidates": g["candidates"] if g else [],
                              "quality": r["quality"]})
        for f in faces:
            f["name"] = persons[f["person_id"]]["name"] if f["person_id"] in persons else None
            f["candidate_names"] = [{"number": pid, "name": persons[pid]["name"], "score": round(s, 2)}
                                    for pid, s in f["candidates"] if pid in persons]
        res["faces"] = faces
        return res
    except face_models.ModelError as e:
        res["error"] = str(e)
        return res
    finally:
        lib.close()


def who_report(res: dict, fcfg: dict, color: bool = True) -> str:
    T = fcfg["texts"]
    C = fcfg["colors_256"]
    paint = (lambda s, c: ui.paint(s, C.get(c))) if color else (lambda s, c: s)
    if res.get("error"):
        return paint(res["error"], "error")
    faces = res["faces"]
    if not faces:
        return T["who_none"].format(photo=res["photo"])
    lines = [T["who_title"].format(photo=res["photo"], n=len(faces))]
    for pos, f in zip(positions(len(faces), T), faces):
        if f["status"] in ("auto", "confirmed") and f["name"]:
            score = f"{f['score']:.2f}" if f["score"] is not None else ""
            lines.append(paint(T["who_known"].format(pos=pos, name=f["name"], number=f["person_id"], score=score).rstrip(),
                               "known"))
        elif f["status"] == "maybe" and f["candidate_names"]:
            lines.append(paint(T["who_maybe"].format(pos=pos, candidates=_cands(f["candidate_names"], T)), "maybe"))
        elif f["status"] == "ignored":
            lines.append(paint(T["who_ignored"].format(pos=pos), "skip"))
        elif f["status"] == "poor" or f["quality"] == "poor":
            lines.append(paint(T["who_poor"].format(pos=pos), "skip"))
        else:
            lines.append(paint(T["who_new"].format(pos=pos), "new"))
    return "\n".join(lines)


def _cands(cands: list, T: dict) -> str:
    return T["or"].join(T["candidate"].format(name=f"{c['name']} (#{c['number']})" if c.get("number") else c["name"],
                                              score=f"{c['score']:.2f}") for c in cands)


# ---------------- report + history ----------------

def report(result: dict, fcfg: dict, color: bool = True) -> str:
    T = fcfg["texts"]
    C = fcfg["colors_256"]
    if result.get("mode") == "who":
        return who_report(result, fcfg, color)
    paint = (lambda s, c: ui.paint(s, C.get(c))) if color else (lambda s, c: s)

    def path_line(p):
        shown = ui.file_link(p) if (color and fcfg["clickable_paths"]) else p
        return T["path"].format(path=paint(shown, "path"))

    if result.get("error"):
        return paint(result["error"], "error")
    cap = fcfg["report_list_max"]
    check = result["check"]
    L = []
    title = T["title_check" if check else "title"].format(
        source=result["source"], photos=_n(result["photos"]), new=_n(result["new"]),
        unchanged=_n(result["unchanged"] + result["reused_copies"]), faces=_n(result["faces"]),
        time=fmt_time(result["seconds"]))
    L.append(title)
    if result["stopped"]:
        L.append(paint(T["stopped"].format(done=_n(result["done"]), total=_n(result["photos"])), "maybe"))
    if result["new_people"]:
        L.append("")
        L.append(paint(T["new_people_check" if check else "new_people"], "new"))
        for p in result["new_people"][:cap]:
            if check:
                L.append(T["person_new_check"].format(photos=p["photos"]))
            else:
                L.append(paint(T["person_new"].format(number=p["number"], name=p["name"], photos=p["photos"]), "new"))
                L.append(path_line(p["folder"]))
        if len(result["new_people"]) > cap:
            L.append(paint(T["more"].format(n=len(result["new_people"]) - cap), "skip"))
    if result["known_people"]:
        L.append(paint(T["known_people_check" if check else "known_people"], "known"))
        for p in result["known_people"][:cap]:
            L.append(paint(T["person_known"].format(number=p["number"], name=p["name"], added=p["added"],
                                                    total=p["total"]), "known"))
            if not check:
                L.append(path_line(p["folder"]))
        if len(result["known_people"]) > cap:
            L.append(paint(T["more"].format(n=len(result["known_people"]) - cap), "skip"))
    if not result["new_people"] and not result["known_people"] and result["faces"]:
        L.append(T["nothing_new_check" if check else "nothing_new"])
    if result["already_sorted"]:
        L.append(paint(T["already_sorted_title"].format(n=len(result["already_sorted"])), "maybe"))
        for a in result["already_sorted"][:min(3, cap)]:
            L.append(f"  {a['photo']}")
            L.append(path_line(a["path"]))
    if result["maybe"]:
        L.append(paint(T["maybe_title"], "maybe"))
        for m in result["maybe"][:cap]:
            L.append(paint(T["maybe_line"].format(n=m["n"], photo=m["photo"], candidates=_cands_plain(m["candidates"], T)),
                           "maybe"))
            if m.get("crop"):
                L.append(path_line(m["crop"]))
        if len(result["maybe"]) > cap:
            L.append(paint(T["more"].format(n=len(result["maybe"]) - cap), "skip"))
    skipped = T["skipped"].format(no_faces=_n(result["no_faces"]), poor=_n(result["poor_faces"]),
                                  unreadable=_n(result["unreadable"]))
    extra = []
    if result.get("unsorted_faces"):
        extra.append(T["skipped_unsorted"].format(n=_n(result["unsorted_faces"])))
    if result["skipped_files"]:
        extra.append(T["skipped_small"].format(n=_n(result["skipped_files"])))
    if result["unsupported"]:
        extra.append(T["skipped_unsupported"].format(n=_n(result["unsupported"])))
    L.append(paint(skipped + "".join(" · " + e for e in extra), "skip"))
    if result["online_only"]:
        L.append(paint(T["online_only"].format(n=result["online_only"]), "maybe"))
    if result["left_for_next_run"]:
        L.append(paint(T["more_next_run"].format(n=_n(result["left_for_next_run"])), "maybe"))
    if result["placed_mb"] and not check:
        L.append(paint(T["placed_mb"].format(mb=result["placed_mb"]), "skip"))
    if result["report_path"]:
        L.append(T["report_file"].format(path=ui.file_link(result["report_path"]) if color else result["report_path"]))
    return "\n".join(L)


def _cands_plain(cands: list, T: dict) -> str:
    return T["or"].join(T["candidate"].format(name=c["name"], score=f"{c['score']:.2f}") for c in cands) or "?"


def history(result: dict, fcfg: dict) -> str:
    """Plain text for the chat history (the model and /write read it)."""
    T = fcfg["texts"]
    if result.get("mode") == "who":
        return who_report(result, fcfg, color=False)
    if result.get("error"):
        return result["error"]
    cap = fcfg["history_list_max"]
    H = T["history"]
    L = [H["source"].format(source=result["source"], photos=_n(result["photos"]), new=_n(result["new"]),
                            unchanged=_n(result["unchanged"] + result["reused_copies"]), faces=_n(result["faces"]))]
    if result["check"]:
        L.append(H["check"])
    if result.get("learned"):
        L.append(result["learned"])
    if result["new_people"]:
        if result["check"]:
            L.append(H["new_people_check"].format(n=len(result["new_people"])))
        else:
            L.append(H["new_people"] + " | ".join(
                f"#{p['number']} {p['name']} ({p['photos']} photos) {p['folder']}" for p in result["new_people"][:cap]))
    if result["known_people"]:
        L.append(H["known_people"] + " | ".join(
            f"#{p['number']} {p['name']} +{p['added']} ({p['total']} now)" for p in result["known_people"][:cap]))
    if result["already_sorted"]:
        L.append(H["already_sorted"].format(n=len(result["already_sorted"])))
    if result["maybe"]:
        L.append(H["maybe"] + " | ".join(f"{m['n']}. {m['photo']} maybe {_cands_plain(m['candidates'], T)}"
                                          for m in result["maybe"][:cap]))
    L.append(H["skipped"].format(no_faces=result["no_faces"], poor=result["poor_faces"], unreadable=result["unreadable"]))
    if result["stopped"]:
        L.append(T["stopped"].format(done=result["done"], total=result["photos"]))
    if result["left_for_next_run"]:
        L.append(T["more_next_run"].format(n=result["left_for_next_run"]))
    L.append(T["nothing_changed_note"])
    if result["report_path"]:
        L.append(T["report_file"].format(path=result["report_path"]))
    return "\n".join(L)


# ---------------- the 'not sure' list ----------------

def open_maybe(lib, pid=None) -> list:
    """All open 'not sure' faces (optionally of one person), numbered, as OPEN_MAYBE."""
    st = lib.store
    persons = {p["id"]: p for p in st.persons()}
    rows = st.faces(status="maybe", with_embedding=False)
    items = []
    for f in rows:
        if pid is not None and f["person_id"] != pid and not any(c[0] == pid for c in f["candidates"]):
            continue
        paths = st.paths(f["image_id"])
        photo = Path(paths[0]).name if paths else face_sync._photo_name(lib, f["image_id"])
        cands = [{"number": q, "name": persons[q]["name"], "score": round(s, 2)} for q, s in f["candidates"] if q in persons]
        if not cands and f["person_id"] in persons:
            cands = [{"number": f["person_id"], "name": persons[f["person_id"]]["name"], "score": round(f["similarity"] or 0, 2)}]
        crop = None
        if f["person_id"] in persons:
            c = st.crops(person_id=f["person_id"], face_id=f["id"])
            if c:
                crop = str(lib.folder(persons[f["person_id"]]) / lib.fcfg["maybe_folder"] / c[0]["file"])
        items.append({"face_id": f["id"], "photo": photo, "candidates": cands, "crop": crop, "done": False})
    items.sort(key=lambda m: (m["photo"].lower(), m["face_id"]))
    for n, m in enumerate(items, 1):
        m["n"] = n
    OPEN_MAYBE[:] = items
    return items


def answer_maybe(lib, answers: list, source: str) -> list[str]:
    """answers: [(face_id, 'yes'|'no'|'assign', person id or None)] -> applied, re-evaluated, folders updated.
    Returns the result lines (also printed)."""
    st, T, out = lib.store, lib.T, lib.out
    persons = {p["id"]: p for p in st.persons()}
    touched = set()
    include = set()
    lines = []
    for fid, verdict, pid in answers:
        f = st.face(fid)
        if not f:
            continue
        if verdict in ("yes", "no") and f["status"] != "maybe":
            # decided since the list was shown (learn / rebuild / reevaluate): the shown candidate is stale (G2)
            for m in OPEN_MAYBE:
                if m["face_id"] == fid:
                    m["done"] = True
            continue
        photo = face_sync._photo_name(lib, f["image_id"])
        cand = f["person_id"] or (f["candidates"][0][0] if f["candidates"] else None)
        if verdict == "yes":
            target = pid or cand
            if not target or target not in persons:
                continue
            face_sync.confirm_face(lib, fid, target, source, photo)
            touched.add(target)
            lines.append(T["confirmed"].format(photo=photo, name=persons[target]["name"]))
        elif verdict == "assign":
            if pid not in persons:
                continue
            if cand and cand != pid:
                touched.add(cand)
            face_sync.confirm_face(lib, fid, pid, source, photo)
            touched.add(pid)
            lines.append(T["confirmed"].format(photo=photo, name=persons[pid]["name"]))
        elif verdict == "no":
            target = pid or cand
            if not target:
                continue
            face_sync.reject_face(lib, fid, target, source, photo)
            touched.add(target)
            include.add(fid)
            lines.append(T["rejected"].format(photo=photo, name=persons[target]["name"]))
        for m in OPEN_MAYBE:
            if m["face_id"] == fid:
                m["done"] = True
    st.commit()
    r = face_sync.reevaluate(lib, touched, include_ids=include)
    all_t = touched | r["touched"]
    face_sync.sync_disk(lib, all_t)
    face_sync.update_learned(lib, all_t)
    for ln in lines:
        out.line(ln, "known" if ln.startswith("✓") else "maybe")
    after = face_sync.describe_changes(lib, r["changes"])
    if after:
        out.line(T["after_changes"].format(changes=after), "known")
        lines.append(T["after_changes"].format(changes=after))
    return lines


# ---------------- sub-commands ----------------

def _split_who(args: str, persons: dict):
    """'12 Maria Petrova' -> (12, 'Maria Petrova'); 'Maria Ivan' -> (Maria's id, 'Ivan') (longest known name)."""
    args = args.strip()
    m = re.match(r"#?(\d+)\b\s*(.*)$", args)
    if m:
        pid = face_sync.resolve_ref(m.group(1), persons)
        return pid, m.group(2).strip()
    words = args.split()
    names = face_sync.name_index(persons.values())
    for k in range(len(words), 0, -1):
        cand = " ".join(words[:k]).strip("\"'")
        pid = face_sync.resolve_ref(cand, persons, names)
        if pid:
            return pid, " ".join(words[k:]).strip()
    return None, args


def _numbers(args: str, total: int) -> list[int]:
    if re.search(r"\b(all|всички)\b", args, re.I):
        return list(range(1, total + 1))
    out = []
    for a, b in re.findall(r"(\d+)\s*(?:-\s*(\d+))?", args):
        lo = int(a)
        hi = int(b) if b else lo
        lo, hi = max(1, min(lo, hi)), min(total, max(lo, hi))   # clamp: "1-999999999" must not build a huge list
        out += list(range(lo, hi + 1))
    return [n for n in dict.fromkeys(out) if 1 <= n <= total]


def people_list(lib, color: bool = True) -> str:
    st, T, fcfg = lib.store, lib.T, lib.fcfg
    persons = [p for p in st.persons() if p["status"] in ("active", "ignored")]
    if not persons:
        return T["people_empty"]
    C = fcfg["colors_256"]
    paint = (lambda s, c: ui.paint(s, C.get(c))) if color else (lambda s, c: s)
    L = [T["people_title"].format(n=len(persons), path=lib.people_dir)]
    for p in persons:
        learned = p.get("learned") or {}
        sug = learned.get("suggested_threshold")
        al = p.get("aliases") or []
        line = T["people_line"].format(
            number=p["id"], name=p["name"], aliases=f" ({', '.join(al)})" if al else "",
            photos=st.photo_count(p["id"]), examples=lib.person_examples(p["id"]),
            suggest=T["people_suggest"].format(value=sug) if sug is not None else "")
        if p["status"] == "ignored":
            L.append(paint(line + T["people_ignored"], "skip"))
            continue
        L.append(paint(line, "known"))
        f = str(lib.folder(p))
        L.append(T["path"].format(path=paint(ui.file_link(f) if color and fcfg["clickable_paths"] else f, "path")))
    return "\n".join(L)


def command(sub: str, args: str, cfg: dict, spin) -> dict:
    """/faces people | rename | merge | maybe | yes | no | ignore | unignore | forget | learn | rebuild.
    Returns {"show": text to print, "history": plain text for the chat or None}."""
    fcfg = cfg["faces"]
    T = fcfg["texts"]
    sub = sub.lower().strip()
    args = (args or "").strip()
    if sub == "who":
        found = find_path(args)
        if not found or not os.path.isfile(found[0]):
            return {"show": T["usage"], "history": None}
        res = who(found[0], cfg, spin)
        return {"show": who_report(res, fcfg), "history": who_report(res, fcfg, color=False)}
    if sub in ("people", "maybe") and not (face_sync.library_dir(fcfg) / fcfg["db_file"]).is_file():
        # read-only with no library yet: don't create faces.db and the folder as a side effect (G8)
        return {"show": T["people_empty"] if sub == "people" else T["nothing_open"], "history": None}
    try:
        lib = face_sync.Lib(cfg, spin)
    except Busy:
        return {"show": T["busy"], "history": None}
    try:
        return _command(lib, sub, args, cfg, spin)
    except face_models.ModelError as e:
        return {"show": str(e), "history": None}
    finally:
        lib.close()


def _command(lib, sub: str, args: str, cfg: dict, spin) -> dict:
    from photos import faces_followup
    st, T, fcfg, out = lib.store, lib.T, lib.fcfg, lib.out
    persons = {p["id"]: p for p in st.persons()}
    hist = lambda: "\n".join(out.lines) or None  # noqa: E731

    if sub == "people":
        return {"show": people_list(lib), "history": people_list(lib, color=False)}
    if sub == "learn":
        st.backup()
        r = face_sync.learn(lib)
        if not r["summary"]:
            return {"show": T["learn_nothing"], "history": None}
        return {"show": "", "history": hist()}
    if sub == "rebuild":
        return rebuild(lib, spin)
    if sub == "maybe":
        pid = None
        if args:
            pid, _ = _split_who(args, persons)
        items = open_maybe(lib, pid)
        if not items:
            return {"show": T["nothing_open"], "history": None}
        faces_followup.open_maybe_context(items, cfg)
        L = [T["maybe_title"]]
        for m in items[: fcfg["report_list_max"] * 4]:
            L.append(T["maybe_line"].format(n=m["n"], photo=m["photo"], candidates=_cands_plain(m["candidates"], T)))
            if m["crop"]:
                L.append(T["path"].format(path=m["crop"]))
        L.append(T["hint_maybe"])
        return {"show": "\n".join(L), "history": None}
    if sub in ("yes", "no"):
        if not OPEN_MAYBE:
            open_maybe(lib)
        if not OPEN_MAYBE:
            return {"show": T["nothing_open"], "history": None}
        target = None
        m = re.search(r"\b(?:as|като)\s+(.+)$", args, re.I)
        if m and sub == "yes":
            target = face_sync.resolve_ref(m.group(1).strip(), persons)
            if not target:
                return {"show": T["unknown_person"].format(who=m.group(1).strip()), "history": None}
            args = args[: m.start()]
        elif m:
            args = args[: m.start()]   # "/faces no 2 as 5": the "as 5" part is not a list number
        nums = _numbers(args, len(OPEN_MAYBE))
        if not nums:
            return {"show": T["usage"], "history": None}
        by_n = {it["n"]: it for it in OPEN_MAYBE}
        answers = [(by_n[n]["face_id"], "assign" if target else sub, target) for n in nums if not by_n[n]["done"]]
        lines = answer_maybe(lib, answers, "command")
        return {"show": "", "history": "\n".join(lines) or None}
    if sub in ("rename", "merge", "ignore", "unignore", "forget"):
        if sub == "merge":
            m = re.match(r"(.+?)\s+(?:into|in|with|to|в|със|с)\s+(.+)$", args, re.I)
            if not m:
                return {"show": T["usage"], "history": None}
            a = face_sync.resolve_ref(m.group(1).strip(), persons) or _split_who(m.group(1), persons)[0]
            b = face_sync.resolve_ref(m.group(2).strip(), persons) or _split_who(m.group(2), persons)[0]
            if not a or not b or a == b:
                return {"show": T["unknown_person"].format(who=args), "history": None}
            A = persons[a]
            B = face_sync.merge_person(lib, a, b)
            out.line(T["merged"].format(from_number=a, from_name=A["name"], number=b, name=B["name"],
                                        photos=st.photo_count(b)), "known")
            return {"show": "", "history": hist()}
        pid, rest = _split_who(args, persons)
        if not pid:
            return {"show": T["unknown_person"].format(who=args or "?"), "history": None}
        p = persons[pid]
        if sub == "rename":
            if not rest:
                return {"show": T["usage"], "history": None}
            old = p["name"]
            p2 = face_sync.rename_person(lib, p, rest.strip("\"'"))
            st.commit()
            out.line(T["renamed"].format(number=pid, old=old, name=p2["name"]), "known")
            out.path(lib.folder(p2))
            return {"show": "", "history": hist()}
        if sub == "ignore":
            face_sync.ignore_person(lib, pid)
            out.line(T["ignored"].format(number=pid, name=p["name"]), "known")
            return {"show": "", "history": hist()}
        if sub == "unignore":
            if p["status"] != "ignored":
                return {"show": T["not_ignored"].format(number=pid, name=p["name"]), "history": None}
            p2 = face_sync.unignore_person(lib, pid)
            out.line(T["unignored"].format(number=pid, name=p2["name"], photos=st.photo_count(pid)), "known")
            out.path(lib.folder(p2))
            return {"show": "", "history": hist()}
        if sub == "forget":
            q = T["ask_forget"].format(number=pid, name=p["name"], photos=st.photo_count(pid))
            faces_followup.ask_confirm({"action": "forget", "person": pid, "source": "command"}, q)
            return {"show": q, "history": None}
    return {"show": T["usage"], "history": None}


# ---------------- /faces rebuild ----------------

def rebuild(lib, spin) -> dict:
    """Re-read every face with the current model (after changing faces.models.embedder). Faces that can't be
    re-read lose their fingerprint (embedding NULL) so sizes never mix. A lost faces.db is NOT recovered here
    (the images table is empty then) - a normal scan does that through learn / feedback replay in _run (G7)."""
    st, T, fcfg, out = lib.store, lib.T, lib.fcfg, lib.out
    st.backup()
    det, emb = lib.models()
    images = [dict(r) for r in st.db.execute("SELECT * FROM images WHERE status='ok' ORDER BY id")]
    done = missing = faces_n = 0
    t0 = time.monotonic()
    try:
        for i, img in enumerate(images, 1):
            spin.status("faces_photo", n=_n(i), total=_n(len(images)), name="", faces=_n(faces_n),
                        eta=_eta(t0, i - 1, len(images), T))
            faces = st.faces(image_id=img["id"], with_embedding=False)
            src = face_sync._source_for(lib, img["id"])
            if src:
                try:
                    bgr, info = face_models.load_image(src, fcfg)
                except face_models.ImageError:
                    bgr = None
            else:
                bgr = None
            if bgr is not None:
                s = info["scale"]
                todo = []
                for f in faces:
                    d = {"box": tuple(v * s for v in f["box"]), "points": [(x * s, y * s) for x, y in f["points"]],
                         "score": f["det_score"]}
                    a = emb.aligned(bgr, d)
                    q = face_models.quality(d, a, fcfg["quality"], s)
                    todo.append((f, a, q))
                ok = [(f, a, q) for f, a, q in todo if q["quality"] != "poor" and a is not None]
                vecs = emb.embed([a for _, a, _ in ok]) if ok else []
                for (f, a, q), v in zip(ok, vecs):
                    st.set_face(f["id"], embedding=v, model=emb.name, quality=q["quality"], sharpness=q["sharpness"])
                    faces_n += 1
                done_ids = {f["id"] for f, _, _ in ok}
                for f, a, q in todo:
                    if f["id"] not in done_ids:
                        # not re-read: drop the old-model fingerprint, never mix sizes (H6)
                        st.set_face(f["id"], embedding=None, model=None, quality=q["quality"],
                                    sharpness=q["sharpness"])
                        if q["quality"] != "poor":
                            missing += 1
                st.update_image(img["id"], model=emb.name)
                done += 1
            else:
                # no photo left anywhere: find the face again in its crop
                for f in faces:
                    c = lib.crop_cache(f["id"])
                    crop = face_models.read_bgr(c) if c.is_file() else None
                    a = None
                    if crop is not None:
                        pad = face_models.cv2_pad(crop)
                        dd = det.detect(pad)
                        a = emb.aligned(pad, dd[0]) if dd else None
                    if a is None:
                        # not re-read: drop the old-model fingerprint, never mix sizes (H6)
                        st.set_face(f["id"], embedding=None, model=None)
                        missing += 1
                        continue
                    st.set_face(f["id"], embedding=emb.embed([a])[0], model=emb.name)
                    faces_n += 1
                st.update_image(img["id"], model=emb.name)
            if i % fcfg["commit_every"] == 0:
                st.commit()
    except KeyboardInterrupt:
        ui.note_cancel()
        st.commit()
        spin.stop()
        return {"show": T["rebuild_stopped"], "history": None}
    finally:
        spin.stop()
    st.set_meta("embedder", emb.name)
    st.set_meta("dim", emb.dim)
    st.commit()
    lib.dirty()
    lib.embedder = emb.name
    lib.mcfg = face_sync.matching_cfg(fcfg, emb.name, lib.delta)
    people = [p["id"] for p in st.persons(status=("active", "ignored"))]
    r = face_sync.reevaluate(lib, people)
    face_sync.sync_disk(lib, set(people) | r["touched"])
    face_sync.update_learned(lib, people)
    msg = T["rebuilt"].format(photos=_n(done), faces=_n(faces_n), model=emb.name, missing=_n(missing),
                              time=fmt_time(time.monotonic() - t0))
    out.line(msg, "known")
    return {"show": "", "history": msg}
