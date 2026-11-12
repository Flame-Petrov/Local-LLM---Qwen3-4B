"""
The faces database: downloads/faces/faces.db (SQLite) + feedback.jsonl (readable log of every answer).

    images       one row per distinct photo (by sha1); image_paths = every place it lives
    faces        one row per detected face: box, 5 points, quality, fingerprint, person, status
    persons      the people (number = id, never reused), settings from person.json
    feedback     every yes/no the user gave for (face, person) - the latest one counts
    placements   photos copied/linked into person folders (kind auto) or put there by the user (kind user)
    crops        face crops in the person folders (_faces / _maybe)

WAL mode, foreign keys on, busy_timeout 5 s, one transaction per batch: Ctrl+C or a crash can't corrupt it.
Backups (SQLite backup API) faces.db.1 (newest) .. faces.db.N before every scan and every learn.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from core.debuglog import log

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS images(
    id INTEGER PRIMARY KEY, sha1 TEXT UNIQUE, width INT, height INT, faces INT,
    status TEXT, error TEXT, scanned_at TEXT, model TEXT);
CREATE TABLE IF NOT EXISTS image_paths(
    path TEXT PRIMARY KEY, image_id INT REFERENCES images(id) ON DELETE CASCADE, size INT, mtime REAL,
    source TEXT, seen_at TEXT);
CREATE TABLE IF NOT EXISTS persons(
    id INTEGER PRIMARY KEY, code TEXT UNIQUE, name TEXT, aliases TEXT, notes TEXT,
    folder TEXT, last_folder_name TEXT, last_name TEXT, status TEXT, merged_into INT,
    settings TEXT, learned TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY, image_id INT REFERENCES images(id) ON DELETE CASCADE, idx INT,
    x REAL, y REAL, w REAL, h REAL, points TEXT, det_score REAL, size_px INT,
    sharpness REAL, yaw REAL, quality TEXT, model TEXT, embedding BLOB,
    person_id INT NULL REFERENCES persons(id), status TEXT, similarity REAL, candidates TEXT,
    updated_at TEXT);
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY, face_id INT, person_id INT, verdict TEXT, source TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS placements(
    person_id INT, image_id INT, file TEXT, sha1 TEXT, kind TEXT, size INT, mtime REAL,
    PRIMARY KEY(person_id, image_id));
CREATE TABLE IF NOT EXISTS crops(
    face_id INT, person_id INT, file TEXT, kind TEXT, PRIMARY KEY(face_id, person_id));
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS faces_status ON faces(status);
CREATE INDEX IF NOT EXISTS faces_image ON faces(image_id);
CREATE INDEX IF NOT EXISTS paths_image ON image_paths(image_id);
CREATE INDEX IF NOT EXISTS feedback_face ON feedback(face_id);
CREATE INDEX IF NOT EXISTS placements_image ON placements(image_id);
"""

FACE_FIELDS = ("image_id", "idx", "x", "y", "w", "h", "points", "det_score", "size_px", "sharpness", "yaw",
               "quality", "model", "embedding", "person_id", "status", "similarity", "candidates", "updated_at")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Busy(Exception):
    """Another app window holds the faces library."""


def _face_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["points"] = json.loads(d["points"]) if d.get("points") else []
    d["candidates"] = [tuple(c) for c in json.loads(d["candidates"])] if d.get("candidates") else []
    emb = d.get("embedding")
    d["embedding"] = np.frombuffer(emb, dtype=np.float32).copy() if emb else None
    d["box"] = (d["x"], d["y"], d["w"], d["h"])
    return d


def _person_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["aliases"] = json.loads(d["aliases"]) if d.get("aliases") else []
    d["settings"] = json.loads(d["settings"]) if d.get("settings") else {}
    d["learned"] = json.loads(d["learned"]) if d.get("learned") else {}
    d["number"] = d["id"]
    return d


class Store:
    def __init__(self, lib_dir, fcfg: dict):
        self.dir = Path(lib_dir)
        self.cfg = fcfg
        self.path = self.dir / fcfg["db_file"]
        self.feedback_path = self.dir / fcfg["feedback_file"]
        self.db: sqlite3.Connection | None = None

    # ---------------- open / close / backup ----------------

    @classmethod
    def open(cls, lib_dir, fcfg: dict) -> "Store":
        s = cls(lib_dir, fcfg)
        s.dir.mkdir(parents=True, exist_ok=True)
        s.db = sqlite3.connect(str(s.path), timeout=5, isolation_level="DEFERRED")
        s.db.row_factory = sqlite3.Row
        # SQLite's lower() folds ASCII only ("Снимка" != "снимка"): a Unicode-aware one for file-name searches
        s.db.create_function("pylower", 1, lambda v: v.lower() if isinstance(v, str) else v, deterministic=True)
        try:
            s.db.execute("PRAGMA busy_timeout = 5000")
            s.db.execute("PRAGMA journal_mode = WAL")
            s.db.execute("PRAGMA foreign_keys = ON")
            s.db.executescript(SCHEMA)
        except sqlite3.OperationalError as e:
            s.db.close()
            if "locked" in str(e) or "busy" in str(e):
                raise Busy(str(e)) from e
            raise
        if s.meta("schema_version") is None:
            s.set_meta("schema_version", str(SCHEMA_VERSION))
            s.set_meta("created", now())
        s.commit()
        return s

    def close(self) -> None:
        if self.db:
            try:
                self.db.commit()
            finally:
                self.db.close()
                self.db = None

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def backup(self) -> Path | None:
        """faces.db -> faces.db.1, older ones shift up to .N (faces.db_backups)."""
        n = int(self.cfg["db_backups"])
        if n <= 0 or not self.path.is_file():
            return None
        self.commit()
        try:
            for i in range(n, 1, -1):
                older, newer = self.path.with_name(f"{self.path.name}.{i}"), self.path.with_name(f"{self.path.name}.{i - 1}")
                if newer.is_file():
                    older.unlink(missing_ok=True)
                    newer.replace(older)
            target = self.path.with_name(f"{self.path.name}.1")
            dst = sqlite3.connect(str(target))
            try:
                self.db.backup(dst)
            finally:
                dst.close()
            return target
        except (OSError, sqlite3.Error) as e:
            log("faces_error", where="backup", path=str(self.path), error=repr(e))
            return None

    # ---------------- meta ----------------

    def meta(self, key: str):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, None if value is None else str(value)))

    # ---------------- images ----------------

    def image_by_path(self, path: str):
        r = self.db.execute("SELECT p.*, i.sha1, i.status AS image_status, i.faces, i.model, i.error, i.width, i.height "
                            "FROM image_paths p JOIN images i ON i.id = p.image_id WHERE p.path=?", (str(path),)).fetchone()
        return dict(r) if r else None

    def image_by_sha1(self, sha1: str):
        r = self.db.execute("SELECT * FROM images WHERE sha1=?", (sha1,)).fetchone()
        return dict(r) if r else None

    def image(self, image_id: int):
        r = self.db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        return dict(r) if r else None

    def add_image(self, sha1: str, width: int, height: int, faces: int, status: str, error: str | None,
                  model: str | None) -> int:
        self.db.execute(
            "INSERT INTO images(sha1, width, height, faces, status, error, scanned_at, model) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(sha1) DO UPDATE SET width=excluded.width, height=excluded.height, faces=excluded.faces, "
            "status=excluded.status, error=excluded.error, scanned_at=excluded.scanned_at, model=excluded.model",
            (sha1, width, height, faces, status, error, now(), model))
        return self.db.execute("SELECT id FROM images WHERE sha1=?", (sha1,)).fetchone()[0]

    def update_image(self, image_id: int, **fields) -> None:
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            self.db.execute(f"UPDATE images SET {cols} WHERE id=?", (*fields.values(), image_id))

    def add_path(self, path: str, image_id: int, size: int, mtime: float, source: str = "") -> None:
        self.db.execute("INSERT INTO image_paths(path, image_id, size, mtime, source, seen_at) VALUES(?,?,?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET image_id=excluded.image_id, size=excluded.size, "
                        "mtime=excluded.mtime, source=excluded.source, seen_at=excluded.seen_at",
                        (str(path), image_id, size, mtime, source, now()))

    def touch_path(self, path: str) -> None:
        self.db.execute("UPDATE image_paths SET seen_at=? WHERE path=?", (now(), str(path)))

    def paths(self, image_id: int) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT path FROM image_paths WHERE image_id=? ORDER BY seen_at DESC",
                                              (image_id,))]

    def remove_path(self, path: str) -> None:
        self.db.execute("DELETE FROM image_paths WHERE path=?", (str(path),))

    def images_by_stem(self, stem: str) -> list[dict]:
        """Photos whose file name (without extension) is `stem` (case-insensitive) - for 'IMG_0450 is not Maria'."""
        like = f"%{stem.lower()}.%"
        out, seen = [], set()
        for r in self.db.execute("SELECT path, image_id FROM image_paths WHERE pylower(path) LIKE ?", (like,)):
            name = Path(r["path"]).name.lower()
            if name.rsplit(".", 1)[0] == stem.lower() and r["image_id"] not in seen:
                seen.add(r["image_id"])
                out.append({"image_id": r["image_id"], "path": r["path"]})
        for r in self.db.execute("SELECT file, image_id FROM placements WHERE pylower(file) LIKE ?", (like,)):
            name = Path(r["file"]).name.lower()
            stem_ = name.rsplit(".", 1)[0]
            if (stem_ == stem.lower() or stem_.split(" (")[0] == stem.lower()) and r["image_id"] not in seen:
                seen.add(r["image_id"])
                out.append({"image_id": r["image_id"], "path": r["file"]})
        return out

    # ---------------- faces ----------------

    def add_faces(self, image_id: int, faces: list[dict]) -> list[int]:
        ids = []
        for f in faces:
            x, y, w, h = f["box"]
            emb = f.get("embedding")
            cur = self.db.execute(
                "INSERT INTO faces(image_id, idx, x, y, w, h, points, det_score, size_px, sharpness, yaw, quality, model, "
                "embedding, person_id, status, similarity, candidates, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (image_id, f["idx"], x, y, w, h, json.dumps([list(map(float, p)) for p in f["points"]]),
                 float(f["det_score"]), int(f["size_px"]), float(f["sharpness"]), float(f["yaw"]), f["quality"],
                 f.get("model"), None if emb is None else np.asarray(emb, dtype=np.float32).tobytes(),
                 f.get("person_id"), f.get("status") or ("poor" if f["quality"] == "poor" else "unknown"),
                 f.get("similarity"), json.dumps(f.get("candidates") or []), now()))
            ids.append(cur.lastrowid)
        return ids

    def faces(self, *, image_id=None, person_id=None, status=None, ids=None, with_embedding=True,
              where: str | None = None, args: tuple = ()) -> list[dict]:
        cols = "*" if with_embedding else ", ".join(c for c in ("id",) + FACE_FIELDS if c != "embedding")
        sql, a = [], []
        if image_id is not None:
            sql.append("image_id=?")
            a.append(image_id)
        if person_id is not None:
            sql.append("person_id=?")
            a.append(person_id)
        if status is not None:
            if isinstance(status, (list, tuple, set)):
                sql.append(f"status IN ({','.join('?' * len(status))})")
                a += list(status)
            else:
                sql.append("status=?")
                a.append(status)
        if ids is not None:
            ids = list(ids)
            if not ids:
                return []
            sql.append(f"id IN ({','.join('?' * len(ids))})")
            a += ids
        if where:
            sql.append(where)
            a += list(args)
        q = f"SELECT {cols} FROM faces" + (" WHERE " + " AND ".join(sql) if sql else "") + " ORDER BY image_id, idx"
        out = []
        for r in self.db.execute(q, a):
            d = dict(r)
            if not with_embedding:
                d["embedding"] = None
                d["points"] = json.loads(d["points"]) if d.get("points") else []
                d["candidates"] = [tuple(c) for c in json.loads(d["candidates"])] if d.get("candidates") else []
                d["box"] = (d["x"], d["y"], d["w"], d["h"])
                out.append(d)
            else:
                out.append(_face_row(r))
        return out

    def face(self, face_id: int):
        r = self.faces(ids=[face_id])
        return r[0] if r else None

    def set_face(self, face_id: int, **fields) -> None:
        if not fields:
            return
        if "candidates" in fields:
            fields["candidates"] = json.dumps([[int(p), round(float(s), 4)] for p, s in (fields["candidates"] or [])])
        if "points" in fields:
            fields["points"] = json.dumps(fields["points"])
        if "embedding" in fields and fields["embedding"] is not None:
            fields["embedding"] = np.asarray(fields["embedding"], dtype=np.float32).tobytes()
        fields["updated_at"] = now()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE faces SET {cols} WHERE id=?", (*fields.values(), face_id))

    # ---------------- persons ----------------

    def persons(self, status=None) -> list[dict]:
        if status is None:
            rows = self.db.execute("SELECT * FROM persons ORDER BY id")
        elif isinstance(status, (list, tuple, set)):
            rows = self.db.execute(f"SELECT * FROM persons WHERE status IN ({','.join('?' * len(status))}) ORDER BY id",
                                   tuple(status))
        else:
            rows = self.db.execute("SELECT * FROM persons WHERE status=? ORDER BY id", (status,))
        return [_person_row(r) for r in rows]

    def person(self, pid: int):
        r = self.db.execute("SELECT * FROM persons WHERE id=?", (pid,)).fetchone()
        return _person_row(r) if r else None

    def person_by_code(self, code: str):
        r = self.db.execute("SELECT * FROM persons WHERE code=?", (code,)).fetchone()
        return _person_row(r) if r else None

    def next_person_id(self) -> int:
        """Numbers are never reused (also not after forget): the highest number ever given + 1."""
        top = self.db.execute("SELECT MAX(id) FROM persons").fetchone()[0] or 0
        top = max(top, int(self.meta("last_person_number") or 0))
        return top + 1

    def add_person(self, name: str, folder: str = "", status: str = "active", settings: dict | None = None,
                   aliases=None, notes: str = "", code: str | None = None, pid: int | None = None) -> int:
        pid = pid or self.next_person_id()
        code = code or f"p{pid:04d}"
        t = now()
        self.db.execute("INSERT INTO persons(id, code, name, aliases, notes, folder, last_folder_name, last_name, status, "
                        "merged_into, settings, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (pid, code, name, json.dumps(list(aliases or []), ensure_ascii=False), notes or "", folder,
                         Path(folder).name if folder else "", name, status, None,
                         json.dumps(settings or {}, ensure_ascii=False), t, t))
        self.set_meta("last_person_number", max(pid, int(self.meta("last_person_number") or 0)))
        return pid

    def update_person(self, pid: int, **fields) -> None:
        if not fields:
            return
        for k in ("aliases", "settings", "learned"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        fields["updated_at"] = now()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE persons SET {cols} WHERE id=?", (*fields.values(), pid))

    def delete_person(self, pid: int) -> None:
        self.db.execute("DELETE FROM persons WHERE id=?", (pid,))

    # ---------------- feedback ----------------

    def feedback(self, face_id: int, person_id: int, verdict: str, source: str, photo: str | None = None) -> None:
        """One answer: 'yes' (this face is that person) or 'no' (it is not). DB row + feedback.jsonl line."""
        assert verdict in ("yes", "no")
        t = now()
        self.db.execute("INSERT INTO feedback(face_id, person_id, verdict, source, at) VALUES(?,?,?,?,?)",
                        (face_id, person_id, verdict, source, t))
        f = self.db.execute("SELECT f.x, f.y, f.w, f.h, i.sha1 FROM faces f JOIN images i ON i.id=f.image_id WHERE f.id=?",
                            (face_id,)).fetchone()
        p = self.person(person_id)
        rec = {"at": t, "face_id": face_id, "photo": photo, "photo_sha1": f["sha1"] if f else None,
               "box": [round(f["x"], 1), round(f["y"], 1), round(f["w"], 1), round(f["h"], 1)] if f else None,
               "person": {"id": person_id, "code": p["code"] if p else None, "name": p["name"] if p else None},
               "verdict": verdict, "source": source}
        try:
            if source != "replay":  # answers replayed from the file are already in it
                with open(self.feedback_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            log("faces_error", where="feedback_file", path=str(self.feedback_path), error=repr(e))
        log("faces_feedback", face_id=face_id, person=person_id, verdict=verdict, source=source)

    def verdicts(self, face_ids=None, person_id=None) -> dict:
        """The latest answer per (face, person): {(face_id, person_id): 'yes'|'no'}."""
        sql, a = "SELECT face_id, person_id, verdict FROM feedback", []
        conds = []
        if face_ids is not None:
            face_ids = list(face_ids)
            if not face_ids:
                return {}
            conds.append(f"face_id IN ({','.join('?' * len(face_ids))})")
            a += face_ids
        if person_id is not None:
            conds.append("person_id=?")
            a.append(person_id)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        out = {}
        for r in self.db.execute(sql + " ORDER BY id", a):
            out[(r["face_id"], r["person_id"])] = r["verdict"]
        return out

    def rejected(self, face_ids=None) -> dict:
        """{face_id: {person ids it was rejected for}} (latest answer 'no')."""
        out: dict = {}
        for (fid, pid), v in self.verdicts(face_ids).items():
            if v == "no":
                out.setdefault(fid, set()).add(pid)
        return out

    def move_feedback(self, from_pid: int, to_pid: int) -> None:
        self.db.execute("UPDATE feedback SET person_id=? WHERE person_id=?", (to_pid, from_pid))

    def delete_feedback(self, person_id: int) -> None:
        self.db.execute("DELETE FROM feedback WHERE person_id=?", (person_id,))

    def forget_code(self, code: str) -> None:
        """A forgotten person's code: never given out again, and its feedback.jsonl lines stop counting
        (a marker line - replay drops every earlier answer for that code)."""
        codes = self.forgotten_codes() | {code}
        self.set_meta("forgotten_codes", json.dumps(sorted(codes)))
        try:
            with open(self.feedback_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"at": now(), "forgotten": code}, ensure_ascii=False) + "\n")
        except OSError as e:
            log("faces_error", where="feedback_file", path=str(self.feedback_path), error=repr(e))

    def forgotten_codes(self) -> set:
        try:
            return set(json.loads(self.meta("forgotten_codes") or "[]"))
        except ValueError:
            return set()

    # ---------------- placements (photos in person folders) ----------------

    def placements(self, person_id=None, image_id=None) -> list[dict]:
        sql, a = "SELECT * FROM placements", []
        conds = []
        if person_id is not None:
            conds.append("person_id=?")
            a.append(person_id)
        if image_id is not None:
            conds.append("image_id=?")
            a.append(image_id)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY person_id, file", a)]

    def add_placement(self, person_id: int, image_id: int, file: str, sha1: str, kind: str,
                      size: int | None = None, mtime: float | None = None) -> None:
        self.db.execute("INSERT INTO placements(person_id, image_id, file, sha1, kind, size, mtime) VALUES(?,?,?,?,?,?,?) "
                        "ON CONFLICT(person_id, image_id) DO UPDATE SET file=excluded.file, sha1=excluded.sha1, "
                        "kind=excluded.kind, size=excluded.size, mtime=excluded.mtime",
                        (person_id, image_id, str(file), sha1, kind, size, mtime))

    def remove_placement(self, person_id: int, image_id: int) -> None:
        self.db.execute("DELETE FROM placements WHERE person_id=? AND image_id=?", (person_id, image_id))

    # ---------------- crops ----------------

    def crops(self, person_id=None, face_id=None) -> list[dict]:
        sql, a = "SELECT * FROM crops", []
        conds = []
        if person_id is not None:
            conds.append("person_id=?")
            a.append(person_id)
        if face_id is not None:
            conds.append("face_id=?")
            a.append(face_id)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY person_id, face_id", a)]

    def add_crop(self, face_id: int, person_id: int, file: str, kind: str) -> None:
        self.db.execute("INSERT INTO crops(face_id, person_id, file, kind) VALUES(?,?,?,?) "
                        "ON CONFLICT(face_id, person_id) DO UPDATE SET file=excluded.file, kind=excluded.kind",
                        (face_id, person_id, str(file), kind))

    def remove_crop(self, face_id: int, person_id: int) -> None:
        self.db.execute("DELETE FROM crops WHERE face_id=? AND person_id=?", (face_id, person_id))

    # ---------------- stats ----------------

    def photo_count(self, person_id: int) -> int:
        return self.db.execute("SELECT COUNT(*) FROM placements WHERE person_id=?", (person_id,)).fetchone()[0]

    def person_image_ids(self, person_id: int) -> set[int]:
        """Photos a person is in (auto / confirmed faces)."""
        return {r[0] for r in self.db.execute(
            "SELECT DISTINCT image_id FROM faces WHERE person_id=? AND status IN ('auto','confirmed')", (person_id,))}
