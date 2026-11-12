"""
Smoke test of the REAL face models (YuNet + the configured embedder). Runs only when FACES_SAMPLE points at a
photo with one face (the models are downloaded once if missing):

    set FACES_SAMPLE=E:\\Photos\\me.jpg
    .venv\\Scripts\\python.exe tests\\test_face_models.py
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core import config  # noqa: E402
from photos import face_models  # noqa: E402
from test_music import run_all  # noqa: E402

CFG = config.load()
SAMPLE = os.environ.get("FACES_SAMPLE")


def test_models_and_alignment():
    det, emb = face_models.models(CFG)
    for entry in [CFG["faces"]["models"]["detector"], CFG["faces"]["models"]["embedders"][emb.name]]:
        p = face_models.model_dir(CFG["faces"]) / entry["file"]
        assert p.is_file() and face_models._sha256(p) == entry["sha256"]
    bgr, info = face_models.load_image(SAMPLE, CFG["faces"])
    faces = det.detect(bgr)
    assert faces, "no face found in FACES_SAMPLE"
    d = max(faces, key=lambda x: x["box"][2] * x["box"][3])
    a = emb.aligned(bgr, d)
    assert a is not None and a.shape[:2] == (112, 112)
    again = det.detect(face_models.cv2_pad(a))
    assert again, "the aligned face is not found again"
    (rx, ry), (lx, ly) = again[0]["points"][:2]
    assert abs((ry - 56) - 51) <= 6 and abs((ly - 56) - 51) <= 6, (ry - 56, ly - 56)   # upright, eyes level
    v = emb.embed([a])[0]
    assert abs(float(np.linalg.norm(v)) - 1) < 1e-4
    # the photo vs its mirror image: the same person
    flip = np.ascontiguousarray(bgr[:, ::-1])
    f2 = max(det.detect(flip), key=lambda x: x["box"][2] * x["box"][3])
    v2 = emb.embed([emb.aligned(flip, f2)])[0]
    assert float(v @ v2) > 0.8, float(v @ v2)


def test_bad_checksum_is_caught():
    import tempfile
    fc = dict(CFG["faces"])
    d = Path(tempfile.mkdtemp())
    fc["models"] = dict(fc["models"], dir=str(d))
    entry = dict(CFG["faces"]["models"]["detector"], sha256="0" * 64)
    (d / entry["file"]).write_bytes(b"not a model")
    try:
        face_models.ensure_model(entry, fc, None)
        raise AssertionError("bad checksum accepted")
    except face_models.ModelError as e:
        assert "checksum" in str(e) and not (d / entry["file"]).exists()


if __name__ == "__main__":
    if not SAMPLE:
        print("skipped: set FACES_SAMPLE=<photo with one face> to run the real-model test")
        sys.exit(0)
    sys.exit(1 if run_all(globals()) else 0)
