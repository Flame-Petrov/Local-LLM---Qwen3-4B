"""
Fake models + a synthetic living-room scene for the /layers tests (no downloads, no ONNX).

Scene 400x300: wall, floor in the bottom 25 %, a sofa, a person overlapping the sofa's front,
a lamp further back on the left, a plant in front on the right.
"""

import copy
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ADE = ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet",
       "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water",
       "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
       "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
       "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand",
       "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
       "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove",
       "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
       "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
       "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle",
       "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
       "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven",
       "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
       "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
       "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
       "clock", "flag"]

W, H = 400, 300
# name: (x0, y0, x1, y1, colour, depth, saliency) - painted in this order (later = on top)
BOXES = {
    "lamp": (20, 60, 60, 200, (230, 200, 40), 0.4, 0.05),
    "sofa": (100, 150, 300, 250, (60, 90, 170), 0.72, 0.7),
    "person": (170, 90, 230, 230, (200, 60, 60), 0.8, 0.95),
    "plant": (330, 150, 380, 290, (40, 150, 60), 0.95, 0.05),
}


def scene(second_person: bool = False, saliency: bool = True):
    """-> (rgb uint8, classes uint8, depth float32, saliency float32) at 400x300."""
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = (205, 200, 190)
    C = np.full((H, W), ADE.index("wall"), np.uint8)
    D = np.full((H, W), 0.1, np.float32)
    S = np.full((H, W), 0.05, np.float32)
    fy = int(H * 0.75)
    rgb[fy:] = (120, 95, 70)
    C[fy:] = ADE.index("floor")
    D[fy:] = np.linspace(0.1, 0.9, H - fy, dtype=np.float32)[:, None]
    boxes = dict(BOXES)
    if second_person:
        boxes = {"lamp": BOXES["lamp"], "sofa": BOXES["sofa"], "person": BOXES["person"],
                 "person2": (245, 95, 290, 230, (190, 80, 120), 0.8, 0.95), "plant": BOXES["plant"]}
    for name, (x0, y0, x1, y1, col, d, s) in boxes.items():
        rgb[y0:y1, x0:x1] = col
        C[y0:y1, x0:x1] = ADE.index("person" if name.startswith("person") else name)
        D[y0:y1, x0:x1] = d
        S[y0:y1, x0:x1] = s
    if not saliency:
        S[:] = 0.0
    return rgb, C, D, S


class _Fake:
    def __init__(self, arr, interp=cv2.INTER_NEAREST):
        self.arr, self.interp, self.calls, self.delay = arr, interp, 0, 0.0

    def _fit(self, rgb, arr=None):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        arr = self.arr if arr is None else arr
        return cv2.resize(arr, (rgb.shape[1], rgb.shape[0]), interpolation=self.interp)


class FakeMatte(_Fake):
    def predict(self, rgb):
        return self._fit(rgb)


class FakeDepth(_Fake):
    def predict(self, rgb):
        return self._fit(rgb)


class FakeSegmenter(_Fake):
    labels = ADE

    def predict(self, rgb):
        c = self._fit(rgb)
        return c, np.ones(c.shape, np.float32)


class FakeInpainter:
    calls = 0

    def fill(self, rgb, hole):
        self.calls += 1
        out = rgb.copy()
        out[hole] = (10, 200, 200)
        return out


def fakes(**kw) -> dict:
    """{"matte", "depth", "segmenter", "inpaint"} for layer_models.install_fakes()."""
    _, C, D, S = scene(**kw)
    return {"matte": FakeMatte(S), "depth": FakeDepth(D), "segmenter": FakeSegmenter(C), "inpaint": FakeInpainter()}


def temp_config(base: dict, out_dir=None, **layer_overrides) -> dict:
    """A copy of the config that writes into a temp folder and never calls the vision model."""
    cfg = copy.deepcopy(base)
    L = cfg["layers"]
    L["output_dir"] = str(out_dir or tempfile.mkdtemp(prefix="layers_test_"))
    L["describe"]["chat"] = False
    L["describe"]["mcp"] = False
    L["keep_models_loaded"] = False
    cfg["debug"]["enabled"] = False
    for k, v in layer_overrides.items():
        node = L
        *path, last = k.split("__")
        for p in path:
            node = node[p]
        node[last] = v
    return cfg


def write_scene_png(folder, name="room.png", scale: int = 1, **kw) -> Path:
    from PIL import Image
    rgb = scene(**kw)[0]
    if scale != 1:
        rgb = cv2.resize(rgb, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    p = Path(folder) / name
    Image.fromarray(rgb).save(p)
    return p


def write_config(cfg: dict, folder) -> Path:
    p = Path(folder) / "config.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p
