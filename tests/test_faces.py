"""
Offline tests for /faces (face_models, face_store, face_match, face_sync, faces) with FAKE models - no network,
no real model files. Plain asserts - run:

    .venv\\Scripts\\python.exe tests\\test_faces.py

Test photos are tiny generated PNGs whose colour encodes a key; FakeDetector looks the key up in a side table
{key: [faces]}; FakeEmbedder returns synthetic fingerprints (identity vector + noise).
"""

import copy
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core import config  # noqa: E402
from photos import face_match  # noqa: E402
from photos import face_models  # noqa: E402
from photos import face_store  # noqa: E402
from photos import face_sync  # noqa: E402
from photos import faces  # noqa: E402
from test_music import FakeSpin, run_all  # noqa: E402

BASE = config.load()
DIM = 64
RNG = np.random.default_rng(7)


# ---------------- fakes ----------------

TABLE: dict = {}       # photo key -> [{"emb": vector, "x": left, "size": px, "blur": bool}]
_REAL_MODELS = face_models.models


def identity(seed: int) -> np.ndarray:
    v = np.random.default_rng(1000 + seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def face_of(ident: np.ndarray, seed: int, noise: float = 0.6) -> np.ndarray:
    n = np.random.default_rng(5000 + seed).normal(size=DIM)
    n -= (n @ ident) * ident
    n /= np.linalg.norm(n)
    v = ident + noise * n
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeDetector:
    name = "fake"

    def detect(self, bgr):
        b, g, r = (int(x) for x in bgr[0, 0])
        key = r * 65536 + g * 256 + b
        out = []
        for i, f in enumerate(TABLE.get(key, [])):
            x, s = f.get("x", 20 + 130 * i), f.get("size", 110)
            y = 60
            pts = [(x + 0.3 * s, y + 0.4 * s), (x + 0.7 * s, y + 0.4 * s), (x + 0.5 * s, y + 0.6 * s),
                   (x + 0.35 * s, y + 0.8 * s), (x + 0.65 * s, y + 0.8 * s)]
            out.append({"box": (x, y, s, s), "points": pts, "score": f.get("score", 0.95), "_fake": (key, i)})
        return out


class FakeEmbedder:
    name, dim = "arcface", DIM

    def __init__(self):
        self.by = {}

    def aligned(self, bgr, det):
        if "_fake" in det:
            key, i = det["_fake"]
        else:  # rebuild: boxes from the DB - find the face by position
            b, g, r = (int(x) for x in bgr[0, 0])
            key = r * 65536 + g * 256 + b
            xs = [f.get("x", 20 + 130 * j) for j, f in enumerate(TABLE[key])]
            i = int(np.argmin([abs(x - det["box"][0]) for x in xs]))
        f = TABLE[key][i]
        if f.get("blur"):
            a = np.full((112, 112, 3), 128, np.uint8)
        else:
            a = np.random.default_rng(key * 10 + i).integers(0, 255, (112, 112, 3), dtype=np.uint8)
        a[0, 0, 0] = i  # distinct per face
        self.by[a.tobytes()] = f["emb"]
        return a

    def embed(self, faces_):
        if not faces_:
            return np.zeros((0, DIM), np.float32)
        return np.stack([self.by[a.tobytes()] for a in faces_])


FAKE = (FakeDetector(), FakeEmbedder())


def use_fakes():
    face_models.models = lambda cfg, spin=None: FAKE


def make_photo(path: Path, key: int, size=(300, 300), fmt="PNG"):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    color = ((key >> 16) & 255, (key >> 8) & 255, key & 255)
    Image.new("RGB", size, color).save(path, fmt)
    return path


class Env:
    """A temp folder with a photo source, a library and a cfg pointing at it."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="faces_test_"))
        self.src = self.tmp / "Снимки 2024"
        self.src.mkdir()
        self.cfg = copy.deepcopy(BASE)
        f = self.cfg["faces"]
        f["library_dir"] = str(self.tmp / "lib")
        f["followup"]["use_llm"] = False
        f["clickable_paths"] = False
        self.spin = FakeSpin()
        self.next_key = 100
        TABLE.clear()
        faces.OPEN_MAYBE.clear()
        face_sync.SESSION.update(strictness=0.0, min_faces_for_new_person=None)
        use_fakes()

    def photo(self, name: str, people: list, folder: Path | None = None, **kw) -> Path:
        """people: [(identity seed, face seed)] or dicts."""
        self.next_key += 1
        key = self.next_key
        rows = []
        for i, p in enumerate(people):
            if isinstance(p, dict):
                rows.append(p)
            else:
                ident, fs = p
                rows.append({"emb": face_of(identity(ident), fs), "x": 20 + 130 * i})
        TABLE[key] = rows
        return make_photo((folder or self.src) / name, key, size=kw.get("size", (300 + 140 * max(0, len(rows) - 2), 300)))

    @property
    def lib(self) -> Path:
        return Path(self.cfg["faces"]["library_dir"])

    @property
    def people(self) -> Path:
        return self.lib / "people"

    def run(self, path=None, **opts):
        o = faces.parse_options("", self.cfg["faces"])
        o.update(opts)
        with redirect_stdout(io.StringIO()):
            return faces.run(str(path or self.src), o, self.cfg, self.spin)

    def open(self):
        return face_sync.Lib(self.cfg, self.spin, echo=False)

    def person_dirs(self):
        return sorted(d.name for d in self.people.iterdir() if d.is_dir() and not d.name.startswith("_")) \
            if self.people.exists() else []

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def files(d: Path) -> list:
    return sorted(f.name for f in d.iterdir() if f.is_file()) if d.exists() else []


# ---------------- find_path / parse_request ----------------

def test_parse_request():
    env = Env()
    try:
        fc = env.cfg["faces"]
        sub = env.src / "Summer trip"
        sub.mkdir()
        img = env.photo("IMG_0450.jpg", [(1, 1)], folder=sub)
        s = str(env.src)
        assert faces.find_path(f'sort "{sub}" by face')[0] == str(sub)
        assert faces.find_path(f"sort the photos in {sub} by face")[0] == str(sub)
        assert faces.find_path(f"who is in {img}?")[0] == str(img)
        assert faces.find_path(f"кой е на {img}.")[0] == str(img)
        assert faces.find_path("nothing here") is None
        assert faces.parse_request(f"/faces {s}", fc)[:2] == ("scan", s)
        assert faces.parse_request(f'/faces "{sub}" check', fc)[2]["check"] is True
        assert faces.parse_request(f"/faces who {img}", fc)[:2] == ("who", str(img))
        assert faces.parse_request(f"/faces who {sub}", fc) is None
        assert faces.parse_request(f"sort the photos in {sub} by face", fc)[:2] == ("scan", str(sub))
        assert faces.parse_request(f"find the people in {s}", fc)[0] == "scan"
        assert faces.parse_request(f"сортирай снимките в {s} по лица", fc)[0] == "scan"
        assert faces.parse_request(f"разпознай лицата в {s}", fc)[0] == "scan"
        assert faces.parse_request(f"who is in {img}?", fc)[:2] == ("who", str(img))
        assert faces.parse_request(f"кой е на {img}", fc)[0] == "who"
        assert faces.parse_request(f"what is in {s}?", fc) is None       # no face words: other features
        assert faces.parse_request(f"describe {img}", fc) is None
        assert faces.parse_request("/faces people", fc) is None          # a sub-command, not a path
        o = faces.parse_request(f"/faces {s} check link norecurse new=2 strict max=5 force", fc)[2]
        assert o["check"] and o["placement"] == "link" and o["recursive"] is False and o["new"] == 2
        assert o["delta"] > 0 and o["max"] == 5 and o["force"]
        assert faces.parse_request(f"/faces {s} loose", fc)[2]["delta"] < 0
        # a path inside the faces library is refused (by run)
        env.lib.mkdir(parents=True, exist_ok=True)
        r = faces.run(str(env.lib), faces.parse_options("", fc), env.cfg, env.spin)
        assert r["error"] == fc["texts"]["library_refused"]
    finally:
        env.close()


# ---------------- load_image ----------------

def test_load_image():
    from PIL import Image
    env = Env()
    fc = env.cfg["faces"]
    try:
        p = env.src / "кирилица" / "снимка.jpg"
        p.parent.mkdir()
        im = Image.new("RGB", (400, 300), (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6
        im.save(p, "JPEG", exif=exif.tobytes())
        bgr, info = face_models.load_image(p, fc)
        assert info["w"] == 300 and info["h"] == 400, info       # rotated: width/height swapped
        assert bgr.shape[:2] == (400, 300) and info["orientation"] == 6
        bad = env.src / "bad.jpg"
        bad.write_bytes(b"not an image at all")
        try:
            face_models.load_image(bad, fc)
            raise AssertionError("corrupt file accepted")
        except face_models.ImageError as e:
            assert e.kind == "unreadable"
        small = env.src / "small.png"
        Image.new("RGB", (100, 80)).save(small)
        try:
            face_models.load_image(small, fc)
            raise AssertionError("tiny photo accepted")
        except face_models.ImageError as e:
            assert e.kind == "skipped"
        gif = env.src / "anim.gif"
        frames = [Image.new("RGB", (300, 300), c) for c in ((255, 0, 0), (0, 255, 0))]
        frames[0].save(gif, save_all=True, append_images=frames[1:])
        bgr, _ = face_models.load_image(gif, fc)
        assert tuple(bgr[0, 0])[::-1][0] > 200   # first frame (red)
        heic = env.src / "x.heic"
        heic.write_bytes(b"0000")
        if not face_models.heic_supported():
            try:
                face_models.load_image(heic, fc)
                raise AssertionError("heic accepted")
            except face_models.ImageError as e:
                assert e.kind == "unsupported"
        big = env.src / "big.jpg"
        Image.new("RGB", (4000, 3000), (1, 2, 3)).save(big, "JPEG")
        bgr, info = face_models.load_image(big, fc)
        assert max(bgr.shape[:2]) <= fc["decode_max_side"] and info["w"] == 4000 and 0 < info["scale"] < 1
    finally:
        env.close()


# ---------------- walk ----------------

def test_walk():
    env = Env()
    fc = env.cfg["faces"]
    try:
        env.photo("b.jpg", [])
        env.photo("a.png", [])
        (env.src / "notes.txt").write_text("x")
        env.photo("c.jpg", [], folder=env.src / "sub")
        env.photo("d.jpg", [], folder=env.src / ".hidden")
        env.photo("e.jpg", [], folder=env.src / "_faces")
        env.photo("f.jpg", [], folder=env.src / "$RECYCLE.BIN")
        names = [Path(p).name for p, _ in faces.walk(str(env.src), fc, True)["photos"]]
        assert names == ["a.png", "b.jpg", "c.jpg"], names
        assert [Path(p).name for p, _ in faces.walk(str(env.src), fc, False)["photos"]] == ["a.png", "b.jpg"]
        # online-only files are counted, not read
        real = face_models.is_online_only
        face_models.is_online_only = lambda p, st=None: Path(p).name == "b.jpg"
        try:
            w = faces.walk(str(env.src), fc, True)
            assert w["online_only"] == 1 and "b.jpg" not in [Path(p).name for p, _ in w["photos"]]
        finally:
            face_models.is_online_only = real
        # the library itself is never scanned
        env.photo("g.jpg", [], folder=env.lib / "people" / "Maria")
        lib_parent = env.lib.parent
        names = [Path(p).name for p, _ in faces.walk(str(lib_parent), fc, True)["photos"]]
        assert "g.jpg" not in names
        assert face_models.is_online_only(__file__) is False
    finally:
        env.close()


# ---------------- store ----------------

def test_store():
    env = Env()
    fc = env.cfg["faces"]
    try:
        st = face_store.Store.open(env.lib, fc)
        assert st.meta("schema_version") == str(face_store.SCHEMA_VERSION)
        img = st.add_image("abc", 10, 10, 1, "ok", None, "arcface")
        st.add_path("C:\\x\\IMG_1.jpg", img, 5, 1.0)
        e = face_of(identity(1), 1)
        fid = st.add_faces(img, [{"idx": 0, "box": (1, 2, 3, 4), "points": [(1, 1)] * 5, "det_score": 0.9,
                                  "size_px": 90, "sharpness": 99, "yaw": 0.0, "quality": "good", "embedding": e,
                                  "model": "arcface"}])[0]
        st.commit()
        st.close()
        st = face_store.Store.open(env.lib, fc)
        f = st.face(fid)
        assert f["embedding"].dtype == np.float32 and np.allclose(f["embedding"], e) and f["status"] == "unknown"
        pid = st.add_person("Person 1")
        st.feedback(fid, pid, "no", "chat", "IMG_1.jpg")
        st.feedback(fid, pid, "yes", "chat", "IMG_1.jpg")
        assert st.verdicts() == {(fid, pid): "yes"} and st.rejected() == {}
        lines = (env.lib / fc["feedback_file"]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and json.loads(lines[1])["verdict"] == "yes" and json.loads(lines[1])["photo_sha1"] == "abc"
        for _ in range(5):
            st.backup()
        assert sorted(p.name for p in env.lib.glob("faces.db.*")) == ["faces.db.1", "faces.db.2", "faces.db.3"]
        assert st.images_by_stem("img_1")[0]["image_id"] == img
        # numbers are never reused
        st.delete_person(pid)
        assert st.next_person_id() == pid + 1
        st.close()
    finally:
        env.close()


# ---------------- face_match ----------------

def mp(pid, seeds, **kw):
    ident = identity(pid)
    ex = np.stack([face_of(ident, s) for s in seeds])
    return face_match.MPerson(id=pid, examples=ex, **kw)


def borderline(person: face_match.MPerson, M: dict, other_seed: int = 77) -> np.ndarray:
    """A fingerprint whose score to `person` is between maybe_threshold and match_threshold."""
    ident = face_match.centroid(person.examples)
    lo, hi = M["maybe_threshold"], M["match_threshold"]
    for w in np.linspace(1.0, 0.0, 400):
        e = face_match.normalize(ident * w + identity(other_seed) * (1 - w))
        s = face_match.person_score(e, person, M["top_k"])
        if lo + 0.02 <= s < hi - 0.02:
            return e
    raise AssertionError("no borderline fingerprint found")


def mcfg(**kw):
    m = face_sync.matching_cfg(BASE["faces"], "arcface")
    m.update(kw)
    return m


def test_assign():
    M = mcfg()
    A, B = mp(1, range(5)), mp(2, range(5))
    f = lambda fid, img, emb: {"id": fid, "image_id": img, "embedding": emb}  # noqa: E731
    r = face_match.assign([f(1, 1, face_of(identity(1), 99)), f(2, 2, face_of(identity(3), 1))], [A, B], M)
    assert r[0]["status"] == "auto" and r[0]["person_id"] == 1 and r[0]["score"] >= M["match_threshold"]
    assert r[1]["status"] == "unknown"
    # between maybe and match -> maybe
    mid = borderline(A, M, 7)
    s = face_match.person_score(mid, A, M["top_k"])
    assert M["maybe_threshold"] <= s < M["match_threshold"], s
    assert face_match.assign([f(3, 3, mid)], [A], M)[0]["status"] == "maybe"
    # margin too small -> maybe
    twin = mp(2, range(5))
    twin.examples = face_match.normalize(A.examples + 0.05 * B.examples)
    twin.centroid = face_match.centroid(twin.examples)
    twin.id = 2
    r = face_match.assign([f(4, 4, face_of(identity(1), 50))], [A, twin], M)[0]
    assert r["status"] == "maybe" and len(r["candidates"]) == 2
    # per-person threshold and auto_assign off
    e = face_of(identity(1), 98)
    sc = face_match.person_score(e, A, M["top_k"])
    A2 = mp(1, range(5), match_threshold=min(0.9, sc + 0.05))
    assert face_match.assign([f(5, 5, e)], [A2], M)[0]["status"] == "maybe"
    A3 = mp(1, range(5), auto_assign=False)
    assert face_match.assign([f(6, 6, e)], [A3], M)[0]["status"] == "maybe"
    # rejected person excluded
    assert face_match.assign([f(7, 7, e)], [A], M, rejected={7: {1}})[0]["status"] == "unknown"
    # closer to a negative than to the person -> maybe
    A4 = mp(1, range(5), negatives=np.stack([e]))
    assert face_match.assign([f(8, 8, e)], [A4], M)[0]["status"] == "maybe"
    # ignored person -> ignored
    A5 = mp(1, range(5), status="ignored")
    assert face_match.assign([f(9, 9, e)], [A5], M)[0]["status"] == "ignored"
    # never_match: needs the bigger sibling margin
    close = None
    for w in np.linspace(0.5, 1.0, 300):
        c = face_match.normalize(identity(1) * w + identity(2) * (1 - w))
        sa, sb = face_match.person_score(c, A, 3), face_match.person_score(c, B, 3)
        if M["margin"] + 0.01 <= sa - sb < M["sibling_margin"] - 0.01 and sb >= M["maybe_threshold"]                 and sa >= M["match_threshold"]:
            close = c
            break
    assert close is not None
    assert face_match.assign([f(10, 10, close)], [A, B], M)[0]["status"] == "auto"
    As = mp(1, range(5), never={2})
    assert face_match.assign([f(10, 10, close)], [As, B], M)[0]["status"] == "maybe"
    # one person per photo: the second face goes to its next candidate
    e1, e2 = face_of(identity(1), 70), face_match.normalize(identity(1) * 0.9 + identity(2) * 0.6)
    r = face_match.assign([f(11, 1, e1), f(12, 1, e2)], [A, B], M)
    assert r[0]["person_id"] == 1 and r[1]["person_id"] == 2
    # already taken (confirmed) in that photo
    r = face_match.assign([f(13, 5, e1)], [A, B], M, taken={5: {1}})
    assert r[0]["person_id"] != 1


def test_cluster():
    M = mcfg()
    faces_ = []
    fid = 0
    for ident in (11, 12, 13):
        for k in range(5):
            fid += 1
            faces_.append({"id": fid, "image_id": fid, "embedding": face_of(identity(ident), fid), "rank": (0, f"{fid:03d}")})
    want = None
    for seed in range(5):
        order = list(faces_)
        np.random.default_rng(seed).shuffle(order)
        got = face_match.cluster(order, M)
        assert len(got) == 3 and sorted(len(g) for g in got) == [5, 5, 5], got
        want = want or got
        assert got == want
    # cannot-link: two faces of one photo never together
    a, b = face_of(identity(20), 1), face_of(identity(20), 2)
    got = face_match.cluster([{"id": 1, "image_id": 1, "embedding": a}, {"id": 2, "image_id": 1, "embedding": b}], M)
    assert len(got) == 2
    # min_faces_for_new_person = 2 leaves singles unknown
    M2 = mcfg(min_faces_for_new_person=2)
    got = face_match.cluster([{"id": 1, "image_id": 1, "embedding": a}, {"id": 2, "image_id": 2, "embedding": b},
                              {"id": 3, "image_id": 3, "embedding": face_of(identity(21), 3)}], M2)
    assert got == [[1, 2]], got


def test_examples_and_threshold():
    M = mcfg(max_examples_per_person=10)
    ident = identity(30)
    conf = np.stack([face_of(ident, i) for i in range(4)])
    auto = [(face_of(ident, 100 + i), 0.7, "good") for i in range(30)] + [(face_of(ident, 200), 0.5, "good"),
                                                                          (face_of(ident, 201), 0.9, "ok")]
    ex, negs, cen = face_match.build_examples(conf, auto, None, M)
    assert len(ex) == 10 and np.allclose(ex[:4], conf)              # confirmed kept first
    assert abs(np.linalg.norm(cen) - 1) < 1e-5 and negs is None
    weak = auto[-2][0]
    assert not any(np.allclose(e, weak) for e in ex)                 # below learn_min_similarity: never an example
    ex2, _, _ = face_match.build_examples(conf, auto, None, M, learn_from_auto=False)
    assert len(ex2) == 4
    # farthest-point: picks diverse looks
    dup = [(face_of(ident, 1), 0.9, "good")] * 5 + [(face_of(ident, 300 + i), 0.6, "good") for i in range(5)]
    ex3, _, _ = face_match.build_examples(None, dup, None, mcfg(max_examples_per_person=3))
    sims = ex3 @ ex3.T
    assert (sims[np.triu_indices(3, 1)] < 0.999).all()
    # suggested threshold
    other = np.stack([face_of(identity(31), i) for i in range(6)])
    confA = np.stack([face_of(ident, i) for i in range(8)])
    val, most, overlap = face_match.suggest_threshold(confA, confA, None, [(2, other)], M)
    assert val is not None and 0.25 <= val <= 0.70 and not overlap and most["person_id"] == 2
    val, _, _ = face_match.suggest_threshold(confA[:3], confA[:3], None, [(2, other)], M)
    assert val is None                                              # < suggest_min_confirmed
    look = np.stack([face_of(ident, 500 + i, noise=0.5) for i in range(6)])
    _, most, overlap = face_match.suggest_threshold(confA, confA, None, [(2, look)], M)
    assert overlap and most["person_id"] == 2


# ---------------- the scan ----------------

def scene(env):
    """Maria (1) in 3 photos, Ivan (2) in 2, a group photo of both, a blurry face, a photo without faces."""
    env.photo("IMG_0001.jpg", [(1, 1)])
    env.photo("IMG_0002.jpg", [(1, 2)])
    env.photo("IMG_0003.jpg", [(2, 1)])
    env.photo("IMG_0004.jpg", [(2, 2)])
    env.photo("IMG_0005.jpg", [(1, 3), (2, 3)])
    env.photo("IMG_0006.jpg", [{"emb": face_of(identity(3), 1), "blur": True}])
    env.photo("IMG_0007.jpg", [])
    (env.src / "broken.jpg").write_bytes(b"garbage")


def sha_all(d: Path) -> dict:
    import hashlib
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in d.rglob("*") if p.is_file()}


def test_run_end_to_end():
    env = Env()
    try:
        scene(env)
        before = sha_all(env.src)
        r = env.run()
        assert r["error"] is None, r["error"]
        assert r["photos"] == 8 and r["new"] == 7 and r["unreadable"] == 1 and r["no_faces"] == 1, r
        assert r["faces"] == 7 and r["poor_faces"] == 1
        assert len(r["new_people"]) == 2 and not r["known_people"] and not r["maybe"], r["new_people"]
        dirs = env.person_dirs()
        assert dirs == ["Person 1", "Person 2"], dirs
        counts = sorted(p["photos"] for p in r["new_people"])
        assert counts == [3, 3], counts      # the group photo is in both folders
        for d in dirs:
            assert "IMG_0005.jpg" in files(env.people / d)
            pj = json.loads((env.people / d / "person.json").read_text(encoding="utf-8"))
            assert pj["id"].startswith("p") and pj["learned"]["photos"] == 3
            crops = files(env.people / d / "_faces")
            assert len(crops) == 3 and all("__f" in c for c in crops)
        assert sha_all(env.src) == before                        # originals untouched
        assert r["report_path"] and Path(r["report_path"]).is_file()
        rep = faces.report(r, env.cfg["faces"], color=False)
        assert "New people" in rep and "Person 1" in rep and "people" in rep
        hist = faces.history(r, env.cfg["faces"])
        assert "#1 Person 1" in hist and "Nothing was renamed" in hist

        # second run: everything already read, nothing copied
        r2 = env.run()
        assert r2["unchanged"] == 8 and r2["new"] == 0 and not r2["new_people"] and not r2["known_people"], r2
        assert r2["placed_mb"] == 0

        # new photos of a known person go to their folder; a copy of a known photo -> already sorted (with path)
        env.photo("IMG_0010.jpg", [(1, 10)])
        shutil.copy(env.src / "IMG_0003.jpg", env.src / "copy of 3.jpg")
        r3 = env.run()
        assert [k["added"] for k in r3["known_people"]] == [1], r3["known_people"]
        assert r3["known_people"][0]["total"] == 4
        assert r3["reused_copies"] == 1 and len(r3["already_sorted"]) == 1
        a = r3["already_sorted"][0]
        assert a["photo"] == "copy of 3.jpg" and Path(a["path"]).is_file() and "IMG_0003.jpg" in a["path"]
        assert not (env.people / "Person 2" / "copy of 3.jpg").exists()
        assert any("already in" in ln for ln in env.spin.lines)
    finally:
        env.close()


def test_check_mode_writes_nothing():
    env = Env()
    try:
        scene(env)
        r = env.run(check=True)
        assert r["check"] and len(r["new_people"]) == 2 and not env.person_dirs()
        rep = faces.report(r, env.cfg["faces"], color=False)
        assert "Would create" in rep and "Check only" in rep
        # faces are cached: the real run reads nothing again
        r2 = env.run()
        assert r2["new"] == 0 and r2["unchanged"] == 8 and len(r2["new_people"]) == 2
    finally:
        env.close()


def test_same_name_different_photo_and_link():
    env = Env()
    try:
        env.photo("IMG_0001.jpg", [(1, 1)])
        env.photo("IMG_0001.jpg", [(1, 2)], folder=env.src / "other")
        r = env.run()
        assert len(r["new_people"]) == 1
        assert files(env.people / "Person 1") == ["IMG_0001 (2).jpg", "IMG_0001.jpg", "person.json"]
        # link mode: falls back to copy when os.link fails
        env2 = Env()
        try:
            env2.photo("IMG_0002.jpg", [(4, 1)])
            real = os.link
            os.link = lambda *a: (_ for _ in ()).throw(OSError(18, "cross-device"))
            try:
                r = env2.run(placement="link")
            finally:
                os.link = real
            assert files(env2.people / "Person 1") == ["IMG_0002.jpg", "person.json"]
            assert any("Hard links" in ln for ln in env2.spin.lines)
        finally:
            env2.close()
    finally:
        env.close()


def test_new_needs_two_photos():
    env = Env()
    try:
        env.photo("a.jpg", [(1, 1)])
        env.photo("b.jpg", [(2, 1)])
        env.photo("c.jpg", [(2, 2)])
        r = env.run(new=2)
        assert len(r["new_people"]) == 1 and r["new_people"][0]["photos"] == 2
        # the stranger shows up again later -> now they get a folder
        env.photo("d.jpg", [(1, 2)])
        r = env.run(new=2)
        assert len(r["new_people"]) == 1 and r["new_people"][0]["photos"] == 2, r["new_people"]
    finally:
        env.close()


def test_stop_and_continue():
    env = Env()
    try:
        for i in range(6):
            env.photo(f"p{i}.jpg", [(i + 1, 1)])
        from core import live
        calls = {"n": 0}

        def fake_take_if(pred):
            calls["n"] += 1
            return ["stop"] if calls["n"] == 4 else []
        real_take, real_stop = live.take_if, live.is_stop
        live.take_if, live.is_stop = fake_take_if, lambda t: bool(t)
        try:
            r = env.run()
        finally:
            live.take_if, live.is_stop = real_take, real_stop
        assert r["stopped"] and r["done"] == 3 and r["new"] == 3, r
        rep = faces.report(r, env.cfg["faces"], color=False)
        assert "Stopped after 3 of 6" in rep
        r2 = env.run()
        assert r2["unchanged"] == 3 and r2["new"] == 3 and not r2["stopped"]
        # Ctrl+C after 2 photos
        for i in range(6, 9):
            env.photo(f"p{i}.jpg", [(i + 1, 1)])
        real_ingest = face_sync.ingest
        count = {"n": 0}

        def boom(*a, **k):
            count["n"] += 1
            if count["n"] == 9:
                raise KeyboardInterrupt
            return real_ingest(*a, **k)
        face_sync.ingest = boom
        try:
            r3 = env.run()
        finally:
            face_sync.ingest = real_ingest
        assert r3["stopped"] and r3["new"] == 2, r3
        r4 = env.run()
        assert r4["new"] == 1 and r4["unchanged"] == 8
        # max=N leaves the rest for the next run
        for i in range(9, 12):
            env.photo(f"q{i}.jpg", [(i + 1, 1)])
        r5 = env.run(max=10)
        assert r5["left_for_next_run"] == 2 and "left for the next run" in faces.report(r5, env.cfg["faces"], color=False)
    finally:
        env.close()


def test_who():
    env = Env()
    try:
        scene(env)
        env.run()
        group = env.photo("group.jpg", [(1, 30), (3, 1), (2, 30)])
        res = faces.who(str(group), env.cfg, env.spin)
        txt = faces.who_report(res, env.cfg["faces"], color=False)
        lines = txt.splitlines()
        assert "3 faces" in lines[0]
        assert lines[1].strip().startswith("left:") and "Person 1" in lines[1]
        assert "someone new" in lines[2] and "middle" in lines[2]
        assert "Person 2" in lines[3]
        assert not (env.people / "Person 1" / "group.jpg").exists()     # nothing sorted
        # a photo already in the DB: answered from the DB
        res = faces.who(str(env.src / "IMG_0005.jpg"), env.cfg, env.spin)
        assert [f["name"] for f in res["faces"]] == ["Person 1", "Person 2"]
    finally:
        env.close()


# ---------------- learning from the folders (§3.6 / §8) ----------------

def learned_env():
    env = Env()
    scene(env)
    env.run()
    return env


def do_learn(env):
    lib = env.open()
    try:
        return face_sync.learn(lib)
    finally:
        lib.close()


def test_learn_rename_folder_and_name():
    env = learned_env()
    try:
        os.replace(env.people / "Person 1", env.people / "Maria")
        r = do_learn(env)
        assert "Maria" in r["summary"]
        pj = json.loads((env.people / "Maria" / "person.json").read_text(encoding="utf-8"))
        assert pj["name"] == "Maria" and pj["aliases"] == []          # default names are not kept as aliases
        # name changed in person.json -> the folder follows
        pj["name"] = "Maria Petrova"
        (env.people / "Maria" / "person.json").write_text(json.dumps(pj, ensure_ascii=False), encoding="utf-8")
        do_learn(env)
        assert "Maria Petrova" in env.person_dirs() and "Maria" not in env.person_dirs()
        pj = json.loads((env.people / "Maria Petrova" / "person.json").read_text(encoding="utf-8"))
        assert pj["aliases"] == ["Maria"]
        # both changed: the file wins
        pj["name"] = "Mimi"
        (env.people / "Maria Petrova" / "person.json").write_text(json.dumps(pj), encoding="utf-8")
        os.replace(env.people / "Maria Petrova", env.people / "Something else")
        do_learn(env)
        assert "Mimi" in env.person_dirs()
        # the next scan keeps sorting into the renamed folder
        env.photo("IMG_0020.jpg", [(1, 20)])
        r = env.run()
        assert r["known_people"][0]["name"] == "Mimi" and (env.people / "Mimi" / "IMG_0020.jpg").exists()
    finally:
        env.close()


def test_learn_name_collision_and_locked():
    env = learned_env()
    try:
        pj_path = env.people / "Person 2" / "person.json"
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["name"] = "Person 1"
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        do_learn(env)
        assert "Person 1 (2)" in env.person_dirs()
        # locked folder: warning, retried next time
        pj_path = env.people / "Person 1 (2)" / "person.json"
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["name"] = "Ivan"
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        real = os.replace

        def locked(a, b):
            if Path(a).name == "Person 1 (2)":
                raise PermissionError(13, "in use")
            return real(a, b)
        os.replace = locked
        try:
            lib = env.open()
            try:
                face_sync.learn(lib)
                assert any("Could not rename" in ln for ln in lib.out.lines)
            finally:
                lib.close()
        finally:
            os.replace = real
        do_learn(env)
        assert "Ivan" in env.person_dirs()
    finally:
        env.close()


def test_learn_broken_person_file():
    env = learned_env()
    try:
        pj_path = env.people / "Person 1" / "person.json"
        broken = pj_path.read_text(encoding="utf-8").replace('"name":', 'name":', 1)
        pj_path.write_text(broken, encoding="utf-8")
        lib = env.open()
        try:
            face_sync.learn(lib)
            assert any("not valid JSON" in ln and "line" in ln for ln in lib.out.lines), lib.out.lines
        finally:
            lib.close()
        assert pj_path.read_text(encoding="utf-8") == broken           # never overwritten
        env.photo("IMG_0021.jpg", [(1, 21)])
        r = env.run()                                                  # still found by its id
        assert r["known_people"] and r["known_people"][0]["number"] == 1
        assert pj_path.read_text(encoding="utf-8") == broken
        # out-of-range values are clamped and reported
        pj_path = env.people / "Person 2" / "person.json"
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["recognition"]["match_threshold"] = 5
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        lib = env.open()
        try:
            face_sync.learn(lib)
            assert any("not valid" in ln and "match_threshold" in ln for ln in lib.out.lines)
            assert lib.store.person(2)["settings"]["match_threshold"] == 0.9
        finally:
            lib.close()
    finally:
        env.close()


def test_learn_deleted_photo_is_rejection():
    env = learned_env()
    try:
        (env.people / "Person 1" / "IMG_0002.jpg").unlink()
        r = do_learn(env)
        assert r["rejected"].get(1) == 1, r
        assert not any("IMG_0002" in c for c in files(env.people / "Person 1" / "_faces"))
        fb = (env.lib / "feedback.jsonl").read_text(encoding="utf-8")
        assert '"verdict": "no"' in fb and '"source": "folder"' in fb
        # the next scan does not put it back
        r = env.run(force=True)
        assert not (env.people / "Person 1" / "IMG_0002.jpg").exists()
        # an edited photo (rotated) is not a rejection
        p = env.people / "Person 2" / "IMG_0003.jpg"
        from PIL import Image
        Image.open(p).rotate(90).save(p, "PNG")
        r = do_learn(env)
        assert not r["rejected"] and p.exists()
    finally:
        env.close()


def test_learn_crops():
    env = learned_env()
    try:
        # delete one crop of the group photo -> only that face is rejected
        crop = next(c for c in (env.people / "Person 1" / "_faces").iterdir() if "IMG_0005" in c.name)
        crop.unlink()
        r = do_learn(env)
        assert r["rejected"] == {1: 1}
        assert not (env.people / "Person 1" / "IMG_0005.jpg").exists()    # no other face of Person 1 in it
        assert (env.people / "Person 2" / "IMG_0005.jpg").exists()
        # move a crop from Person 2 to Person 1 -> that face is Person 1 (the photo follows)
        crop = next(c for c in (env.people / "Person 2" / "_faces").iterdir() if "IMG_0003" in c.name)
        os.replace(crop, env.people / "Person 1" / "_faces" / crop.name)
        r = do_learn(env)
        assert r["learned"].get(1) == 1
        assert (env.people / "Person 1" / "IMG_0003.jpg").exists()
        assert not (env.people / "Person 2" / "IMG_0003.jpg").exists()
        lib = env.open()
        try:
            fid = int(face_sync.FACE_ID_RX.search(crop.stem + "_").group(1))
            f = lib.store.face(fid)
            assert f["person_id"] == 1 and f["status"] == "confirmed"
        finally:
            lib.close()
    finally:
        env.close()


def test_learn_new_photos_and_handmade_folder():
    env = learned_env()
    try:
        # drop a new photo of Person 1 into the folder -> confirmed example
        env.photo("mine.jpg", [(1, 40)], folder=env.people / "Person 1")
        r = do_learn(env)
        assert r["learned"].get(1) == 1
        assert (env.people / "Person 1" / "mine.jpg").exists()
        # a group photo: the face closest to Person 1 is taken
        env.photo("group_drop.jpg", [(2, 41), (1, 41)], folder=env.people / "Person 1")
        r = do_learn(env)
        assert r["learned"].get(1) == 1
        # hand-made folder "Grandpa" with 3 single-face photos -> a new person, found in new photos later
        g = env.people / "Grandpa"
        for i in range(3):
            env.photo(f"g{i}.jpg", [(9, 50 + i)], folder=g)
        r = do_learn(env)
        assert "Grandpa" in r["summary"]
        pj = json.loads((g / "person.json").read_text(encoding="utf-8"))
        assert pj["name"] == "Grandpa" and pj["learned"]["faces_confirmed"] == 3
        env.photo("IMG_0099.jpg", [(9, 60)])
        r = env.run()
        assert any(k["name"] == "Grandpa" for k in r["known_people"]), r
        assert (g / "IMG_0099.jpg").exists()
        # user files are never deleted: delete the whole Grandpa folder -> ignored
        shutil.rmtree(g)
        r = do_learn(env)
        assert any("ignored" in e for e in r["events"])
        env.photo("IMG_0100.jpg", [(9, 61)])
        r = env.run()
        assert not r["new_people"] and not (env.people / "Grandpa").exists()
    finally:
        env.close()


def test_learn_handmade_needs_single_face():
    env = Env()
    try:
        g = env.people / "Grandpa"
        env.photo("both.jpg", [(9, 1), (8, 1)], folder=g)
        lib = env.open()
        try:
            face_sync.learn(lib)
            assert any("no photo with only them" in ln for ln in lib.out.lines), lib.out.lines
        finally:
            lib.close()
        assert len(files(g / "_maybe")) == 2
    finally:
        env.close()


def test_learn_settings_and_merge_into():
    env = learned_env()
    try:
        # raise the threshold very high -> Person 1's automatic photos move to _maybe
        pj_path = env.people / "Person 1" / "person.json"
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["recognition"]["match_threshold"] = 0.89
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        r = do_learn(env)
        moved = r["changes"].get(1, {})
        assert moved.get("to_maybe", 0) >= 1, r["changes"]
        assert files(env.people / "Person 1" / "_maybe")
        # lower it again -> promoted
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["recognition"]["match_threshold"] = None
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        r = do_learn(env)
        assert r["changes"].get(1, {}).get("added", 0) >= 1, r["changes"]
        # merge_into
        pj_path = env.people / "Person 2" / "person.json"
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["merge_into"] = 1
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        r = do_learn(env)
        assert env.person_dirs() == ["Person 1"], env.person_dirs()
        assert {"IMG_0003.jpg", "IMG_0004.jpg", "IMG_0005.jpg"} <= set(files(env.people / "Person 1"))
        lib = env.open()
        try:
            assert lib.store.person(2)["status"] == "merged"
            assert face_sync.resolve_ref("2", {p["id"]: p for p in lib.store.persons()}) == 1
        finally:
            lib.close()
    finally:
        env.close()


def test_commands():
    env = learned_env()
    try:
        cfg, spin = env.cfg, env.spin
        with redirect_stdout(io.StringIO()):
            out = faces.command("people", "", cfg, spin)
        assert "#1 Person 1" in out["show"] and "#2 Person 2" in out["show"]
        out = faces.command("rename", "1 Maria", cfg, spin)
        assert "Renamed #1 Person 1 → Maria" in out["history"]
        assert "Maria" in env.person_dirs()
        out = faces.command("rename", "Maria Maria Petrova", cfg, spin)
        assert "Maria Petrova" in env.person_dirs()
        out = faces.command("ignore", "2", cfg, spin)
        assert "ignored" in out["history"] and "Person 2" not in env.person_dirs()
        env.photo("IMG_0030.jpg", [(2, 31)])
        r = env.run()
        assert not r["new_people"] and not r["known_people"]          # a stranger stays silent
        out = faces.command("unignore", "2", cfg, spin)
        assert "active again" in out["history"] and "Person 2" in env.person_dirs()
        assert "IMG_0030.jpg" in files(env.people / "Person 2")
        out = faces.command("merge", "2 into Maria Petrova", cfg, spin)
        assert "Merged #2" in out["history"] and env.person_dirs() == ["Maria Petrova"]
        out = faces.command("forget", "1", cfg, spin)
        assert "yes / no" in out["show"]
        from photos import faces_followup
        dec = faces_followup.interpret("yes", None, cfg, spin)
        assert dec and dec["action"] == "forget"
        with redirect_stdout(io.StringIO()):
            faces_followup.execute(dec, cfg, spin)
        assert env.person_dirs() == []
        out = faces.command("learn", "", cfg, spin)
        assert out["show"]
        # numbers are never reused after forget
        env.photo("IMG_0031.jpg", [(5, 1)])
        r = env.run()
        assert r["new_people"][0]["number"] == 3, r["new_people"]
    finally:
        env.close()


def test_maybe_answers_and_commands():
    env = learned_env()
    try:
        # a borderline face of Person 1 -> 'not sure'
        lib = env.open()
        try:
            P1, P2 = lib.people()[1], lib.people()[2]
            M = lib.mcfg
        finally:
            lib.close()
        env.photo("IMG_0450.jpg", [{"emb": borderline(P1, M, 77)}])
        r = env.run()
        assert len(r["maybe"]) == 1 and r["maybe"][0]["candidates"][0]["number"] == 1, r["maybe"]
        assert r["maybe"][0]["crop"] and Path(r["maybe"][0]["crop"]).is_file()
        rep = faces.report(r, env.cfg["faces"], color=False)
        assert "1. IMG_0450.jpg  maybe Person 1" in rep
        out = faces.command("yes", "1", env.cfg, env.spin)
        assert "✓ IMG_0450.jpg: Person 1" in out["history"]
        assert (env.people / "Person 1" / "IMG_0450.jpg").exists()
        assert not files(env.people / "Person 1" / "_maybe")
        # 'no' on another one; 'yes ... as' assigns to someone else
        env.photo("IMG_0451.jpg", [{"emb": borderline(P1, M, 78)}])
        env.photo("IMG_0452.jpg", [{"emb": borderline(P2, M, 79)}])
        r = env.run()
        assert len(r["maybe"]) == 2
        out = faces.command("no", "1", env.cfg, env.spin)
        assert "✗ IMG_0451.jpg: not Person 1" in out["history"]
        out = faces.command("yes", "2 as 1", env.cfg, env.spin)
        assert (env.people / "Person 1" / "IMG_0452.jpg").exists()
        with redirect_stdout(io.StringIO()):
            out = faces.command("maybe", "", env.cfg, env.spin)
        assert "empty" in out["show"]
    finally:
        env.close()


def test_rebuild_keeps_people_and_answers():
    env = learned_env()
    try:
        faces.command("rename", "1 Maria", env.cfg, env.spin)
        crop = next(c for c in (env.people / "Maria" / "_faces").iterdir() if "IMG_0002" in c.name)
        crop.unlink()
        do_learn(env)
        env.cfg["faces"]["models"]["embedder"] = "sface"
        r = env.run()
        assert "rebuild" in (r["error"] or ""), r["error"]
        FakeEmbedder.name = "sface"
        try:
            out = faces.command("rebuild", "", env.cfg, env.spin)
            assert "Re-read" in out["history"], out
            r = env.run()
            assert r["error"] is None and not r["new_people"]
            assert "Maria" in env.person_dirs()
            assert not (env.people / "Maria" / "IMG_0002.jpg").exists()     # the 'no' survived
        finally:
            FakeEmbedder.name = "arcface"
    finally:
        env.close()


def test_rebuild_new_size_never_mixes():
    """H6: a rebuild to a model with another size; faces it can't re-read lose their old fingerprint."""
    env = learned_env()
    emb = FAKE[1]
    real_aligned, real_embed = FakeEmbedder.aligned, FakeEmbedder.embed
    skipped = []

    def aligned(self, bgr, det):
        if not skipped and "_fake" not in det:      # rebuild: the first face can't be aligned
            skipped.append(1)
            return None
        return real_aligned(self, bgr, det)

    def embed(self, faces_):
        e = real_embed(self, faces_)
        return np.ascontiguousarray(e[:, :32]) if len(e) else np.zeros((0, 32), np.float32)

    FakeEmbedder.aligned, FakeEmbedder.embed = aligned, embed
    FakeEmbedder.name, FakeEmbedder.dim = "sface", 32
    env.cfg["faces"]["models"]["embedder"] = "sface"
    try:
        with redirect_stdout(io.StringIO()):
            out = faces.command("rebuild", "", env.cfg, env.spin)
        assert "Re-read" in out["history"], out
        assert skipped
        lib = env.open()
        try:
            rows = lib.store.faces()
            sizes = {len(f["embedding"]) for f in rows if f["embedding"] is not None}
            assert sizes == {32}, sizes
            assert any(f["embedding"] is None for f in rows)
            face_sync.reevaluate(lib, [p["id"] for p in lib.store.persons()])   # np.stack must not fail
        finally:
            lib.close()
        r = env.run()
        assert r["error"] is None, r["error"]
    finally:
        FakeEmbedder.aligned, FakeEmbedder.embed = real_aligned, real_embed
        FakeEmbedder.name, FakeEmbedder.dim = "arcface", DIM
        env.close()


def test_lost_db_recovers_from_folders():
    env = learned_env()
    try:
        faces.command("rename", "1 Maria", env.cfg, env.spin)
        # a 'no' for Maria's face in the group photo, then the DB is lost
        crop = next(c for c in (env.people / "Maria" / "_faces").iterdir() if "IMG_0005" in c.name)
        crop.unlink()
        do_learn(env)
        assert not (env.people / "Maria" / "IMG_0005.jpg").exists()
        for f in env.lib.glob("faces.db*"):
            f.unlink()
        r_learn = do_learn(env)
        lib = env.open()
        try:
            ps = {p["code"]: p for p in lib.store.persons()}
            assert ps["p0001"]["name"] == "Maria" and ps["p0002"]["name"] == "Person 2", ps.keys()
        finally:
            lib.close()
        # the person folders taught it again: a new scan sorts into the same folders
        env.photo("IMG_0060.jpg", [(1, 60)])
        r = env.run()
        assert any(k["name"] == "Maria" for k in r["known_people"]), r["known_people"]
        assert "replayed" in r_learn["summary"], r_learn["summary"]
        assert not (env.people / "Maria" / "IMG_0005.jpg").exists()    # the old 'no' came back
        assert (env.people / "Person 2" / "IMG_0005.jpg").exists()
    finally:
        env.close()


def test_report_lists_are_cut():
    fc = copy.deepcopy(BASE["faces"])
    fc["report_list_max"] = 2
    r = faces._base_result("E:\\Photos", "scan")
    r.update(photos=10, faces=10, seconds=372,
             new_people=[{"number": i, "name": f"Person {i}", "photos": 1, "folder": f"E:\\lib\\Person {i}"} for i in range(5)],
             maybe=[{"n": i, "face_id": i, "photo": f"x{i}.jpg", "candidates": [{"number": 1, "name": "Maria", "score": 0.36}],
                     "crop": None} for i in range(1, 4)])
    rep = faces.report(r, fc, color=False)
    assert "6:12" in rep and "... and 3 more" in rep and "... and 1 more" in rep
    assert "1. x1.jpg  maybe Maria (0.36)" in rep
    assert "Person 4" not in rep
    h = faces.history(r, fc)
    assert "#4 Person 4" in h and "Not sure (open): 1. x1.jpg maybe Maria (0.36)" in h


def test_config_has_everything():
    fc = BASE["faces"]
    for k in ("library_dir", "people_folder", "models", "detection", "quality", "matching", "crops", "followup",
              "prompts", "texts", "colors_256"):
        assert k in fc, k
    for k in ("faces_model_download", "faces_loading", "faces_listing", "faces_photo", "faces_matching",
              "faces_clustering", "faces_placing", "faces_learning", "faces_understanding"):
        assert k in BASE["ui"]["status"], k


# ---------------- review 2026-11-10 HIGH fixes (H2-H5) ----------------

def test_h3_unreadable_person_file_is_not_missing():
    env = learned_env()
    try:
        pj_path = env.people / "Person 1" / "person.json"
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["notes"] = "бележка"
        pj_path.write_bytes(json.dumps(pj, ensure_ascii=False).encode("cp1251"))
        assert face_sync.read_person_file(pj_path)[1] is face_sync.UNREADABLE
        before = len(env.person_dirs())
        lib = env.open()
        try:
            face_sync.learn(lib)
            assert any("Could not read" in ln for ln in lib.out.lines), lib.out.lines
            assert lib.store.person(1)["status"] == "active"
            assert lib.store.placements(person_id=1)
        finally:
            lib.close()
        assert len(env.person_dirs()) == before
    finally:
        env.close()


def test_h4_existing_identical_file_is_user():
    env = learned_env()
    try:
        lib = env.open()
        try:
            st = lib.store
            pl = st.placements(person_id=1)[0]
            st.remove_placement(1, pl["image_id"])
            face_sync.sync_disk(lib, [1])
            again = [x for x in st.placements(person_id=1) if x["image_id"] == pl["image_id"]]
            assert again and again[0]["kind"] == "user", again
        finally:
            lib.close()
    finally:
        env.close()


def test_h2_h5_forget_keeps_unmovable_files_and_code():
    env = learned_env()
    try:
        folder = env.people / "Person 1"
        mine = folder / "my_own.jpg"
        mine.write_bytes((env.src / "IMG_0007.jpg").read_bytes())
        lib = env.open()
        real = face_sync.os.replace

        def failing(a, b):
            if Path(a).name == "my_own.jpg":
                raise PermissionError(13, "locked")
            return real(a, b)
        face_sync.os.replace = failing
        try:
            face_sync.forget_person(lib, 1)
        finally:
            face_sync.os.replace = real
        try:
            left = [q for q in env.people.rglob("my_own.jpg")]
            assert left, "user file was deleted"
            assert all(q.parts[-3].startswith("_") or q.parts[-2].startswith("_") for q in left), left
            assert "p0001" in lib.store.forgotten_codes()
            p = face_sync.create_person(lib, "Back", code="p0001")
            assert p["id"] != 1 and p["code"] != "p0001", p
        finally:
            lib.close()
        assert '"forgotten": "p0001"' in lib.store.feedback_path.read_text(encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        failed = run_all(globals())
    finally:
        face_models.models = _REAL_MODELS
    sys.exit(1 if failed else 0)
