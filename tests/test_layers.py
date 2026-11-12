"""
Offline tests for /cutout and /layers (layers.py, layer_ops.py) with fake models - plain asserts:

    .venv\\Scripts\\python.exe tests\\test_layers.py
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config  # noqa: E402
from photos import layer_models  # noqa: E402
from photos import layer_ops as ops  # noqa: E402
from photos import layers  # noqa: E402
import layers_fakes as lf  # noqa: E402
from test_music import run_all  # noqa: E402

BASE = config.load()
L = BASE["layers"]
TMP = Path(tempfile.mkdtemp(prefix="layers_tests_"))


def _stats(focus=None, max_layers=5, **kw):
    """The scene through regions -> focus -> groups (work size 400x300)."""
    rgb, C, D, S = lf.scene(**kw)
    labels, _ = ops.regions(C, int(L["grouping"]["min_region_pct"] / 100 * C.size))
    stats = ops.region_stats(labels, C, S, D, lf.ADE)
    f = ops.pick_focus(stats, S, L, focus)
    alpha, _ = ops.focus_alpha(rgb, S, labels, f["ids"], L, stats)
    fmask = alpha >= 0.5
    if not fmask.any():
        fmask = np.isin(labels, f["ids"])
    groups = ops.group_layers(stats, labels, fmask, D, L, max_layers, f["ids"])
    by_id = {s["id"]: s for s in stats}
    return stats, f, [(g["kind"], sorted({by_id[i]["class"] for i in g["ids"]})) for g in groups]


def _run(mode="layers", cfg=None, src=None, again=False, fakes=None, **opts):
    cfg = cfg or lf.temp_config(BASE, TMP / "out")
    fk = fakes or lf.fakes()
    layer_models.install_fakes(fk)
    src = src or lf.write_scene_png(TMP)
    req = layers.make_request(mode, str(src), opts)
    req["again"] = again
    return layers.run(req, cfg), cfg, fk


def _png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0


# ---------------- 1-9: layer_ops ----------------

def test_01_guided_filter_constant_and_edge():
    guide = np.zeros((60, 80, 3), np.float32)
    guide[:, 40:] = 1.0
    const = ops.guided_filter(guide, np.full((60, 80), 0.37, np.float32), 8, 1e-3)
    assert np.abs(const - 0.37).max() < 1e-3
    step = np.zeros((60, 80), np.float32)
    step[:, 40:] = 1.0
    out = ops.guided_filter(guide, step, 8, 1e-3)
    assert out[:, 42:].min() >= 0.9 and out[:, :38].max() <= 0.1, (out[:, 42:].min(), out[:, :38].max())


def test_02_guided_filter_is_linear():
    rng = np.random.default_rng(3)
    guide = rng.random((50, 70, 3)).astype(np.float32)
    a = (rng.random((50, 70)) > 0.5).astype(np.float32)
    b = 1.0 - a
    ga, gb = ops.guided_filter(guide, a, 6, 1e-3), ops.guided_filter(guide, b, 6, 1e-3)
    gab = ops.guided_filter(guide, a + b, 6, 1e-3)
    assert np.abs(ga + gb - gab).max() < 1e-4
    assert np.abs(ga + gb - 1.0).max() < 1e-3          # a partition stays a partition


def test_03_alphas_reproduce_the_colour():
    rng = np.random.default_rng(5)
    w = rng.random((4, 30, 40)).astype(np.float32)
    w[1, :10] = 0
    w[3, 20:] = 0
    w /= w.sum(axis=0, keepdims=True)
    a = ops.alphas_from_weights(w)
    colour = rng.random((30, 40, 3)).astype(np.float32)
    stack = [np.dstack([colour, a[j]]) for j in range(3, -1, -1)]     # back to front
    out = ops.compose(stack)
    assert np.abs(out[:, :, :3] - colour).max() < 1e-5 and np.abs(out[:, :, 3] - 1).max() < 1e-5
    assert np.all(a[3][w[3] > 0] == 1.0)
    assert np.all(a[2][(w[3] == 0) & (w[2] > 0)] == 1.0)


def test_04_no_seams_between_stacked_layers():
    w = np.zeros((3, 4, 4), np.float32)
    w[0], w[1] = 0.5, 0.5                    # a soft boundary between layer 1 and layer 2, no background there
    a = ops.alphas_from_weights(w)
    c = np.full((4, 4, 3), 0.5, np.float32)
    out = ops.compose([np.dstack([c, a[1]]), np.dstack([c, a[0]])])
    assert np.allclose(out[:, :, 3], 1.0) and np.allclose(a[0], 0.5) and np.allclose(a[1], 1.0)


def test_05_blur_fusion_keeps_the_subject_colour():
    h, w = 80, 200
    alpha = np.clip((np.arange(w, dtype=np.float32) - 90) / 20.0, 0, 1)[None].repeat(h, 0)   # ramp 90..110
    red, blue = np.array([1, 0, 0], np.float32), np.array([0, 0, 1], np.float32)
    img = alpha[:, :, None] * red + (1 - alpha[:, :, None]) * blue
    F = ops.estimate_foreground(img, alpha)
    assert np.abs(F[:, 150:] - red).max() < 1e-6 and np.abs(F[:, :50] - img[:, :50]).max() < 1e-6
    edge = (alpha > 0.3) & (alpha < 1)
    assert np.abs(F[edge] - red).max() < 0.1, np.abs(F[edge] - red).max()
    assert np.abs(img[edge] - red).max() > 0.5          # the plain photo colour would be far off


def test_06_tiny_island_is_dissolved():
    C = np.full((40, 40), 3, np.uint8)
    C[10, 10:13] = 7
    labels, n = ops.regions(C, 20)
    assert n == 1 and len(np.unique(labels)) == 1
    stats = ops.region_stats(labels, C, np.zeros((40, 40), np.float32), np.zeros((40, 40), np.float32), lf.ADE)
    assert stats[0]["class"] == "floor" and stats[0]["px"] == 1600


def test_07_pick_focus():
    stats, f, groups = _stats()
    assert f["label"] == "person" and f["count"] == 1 and not f["warnings"]      # wins over the sofa
    stats, f, groups = _stats(focus="couch")
    assert f["label"] == "sofa" and f["user"]
    assert groups[0][0] == "front" and "person" in groups[0][1], groups           # the person is now in front
    stats, f, _ = _stats(focus="person on the left", second_person=True)
    assert f["count"] == 1
    chosen = next(s for s in stats if s["id"] == f["ids"][0])
    assert chosen["class"] == "person" and chosen["centroid"][0] < 240
    stats, f, _ = _stats(focus="the unicorn")
    assert f["label"] == "person" and "unicorn" in f["warnings"][0]
    stats, f, _ = _stats(saliency=False)
    assert f["confidence"] == "low" and f["label"] not in L["grouping"]["background_classes"]
    assert f["warnings"] and f["warnings"][0].startswith("No clear subject")
    for word, cls in (("диван", "sofa"), ("the dog", "animal"), ("tv", "television receiver")):
        assert ops.parse_focus(word, L)["classes"] == [cls], word
    assert ops.parse_focus("човека отляво", L) == {"classes": ["person"], "positions": ["left"]}


def test_08_two_people():
    stats, f, _ = _stats(second_person=True)
    assert f["label"] == "person" and f["count"] == 2
    assert ops.focus_name("person", 2, L["grouping"]["class_names"]) == "2 people"
    assert ops.focus_name("windowpane", 1, L["grouping"]["class_names"]) == "window"


def test_09_group_layers_table():
    assert _stats(max_layers=5)[2] == [("front", ["plant"]), ("focus", ["person"]), ("touching", ["sofa"]),
                                       ("depth", ["lamp"]), ("background", ["floor", "wall"])]
    assert _stats(max_layers=3)[2] == [("focus", ["person"]), ("touching", ["plant", "sofa"]),
                                       ("background", ["floor", "lamp", "wall"])]
    assert _stats(max_layers=2)[2] == [("focus", ["person"]),
                                       ("background", ["floor", "lamp", "plant", "sofa", "wall"])]


# ---------------- 10-16: run() with fake models ----------------

def test_10_run_writes_the_files():
    res, cfg, _ = _run()
    folder = Path(res["folder"])
    files = sorted(str(p.relative_to(folder)).replace("\\", "/") for p in folder.rglob("*") if p.is_file())
    want = ["cutout.png", "mask.png", "layers.json", "layers.psd", "preview.jpg",
            "layers/layer_0_plant.png", "layers/layer_1_person.png", "layers/layer_2_sofa.png",
            "layers/layer_3_lamp.png", "layers/layer_4_background.png",
            "stacks/stack_1-2.png", "stacks/stack_1-3.png"]
    assert files == sorted(want), files
    man = json.loads((folder / "layers.json").read_text(encoding="utf-8"))
    assert [l["file"] for l in man["layers"]] == want[5:10]
    assert [l["name"] for l in man["layers"]] == ["plant", "person", "sofa", "lamp", "background: wall, floor"]
    assert [s["file"] for s in man["stacks"]] == want[10:] and man["stacks"][0]["layers"] == [1, 2]
    assert man["files"] == {"cutout": "cutout.png", "mask": "mask.png", "psd": "layers.psd",
                            "preview": "preview.jpg", "without_subject": None}
    assert man["focus"]["label"] == "person" and man["focus"]["where"] == "center" and man["size"] == [400, 300]
    assert not man["focus"]["nearest"]                      # the plant is nearer
    assert abs(sum(l["area_pct"] for l in man["layers"]) - 100) < 1
    assert man["layers"][2]["touches_focus"] and man["layers"][0]["in_front"]
    from PIL import Image
    im = Image.open(folder / "cutout.png")
    assert im.mode == "RGBA" and im.size == (400, 300) and not im.getexif()
    assert Image.open(folder / "mask.png").mode == "L"
    from psd_tools import PSDImage
    psd = PSDImage.open(folder / "layers.psd")
    assert [l.name for l in psd] == ["4 background: wall, floor", "3 lamp", "2 sofa", "1 person", "0 plant"]
    assert psd.size == (400, 300)
    # cutout mode: only the cutout files
    res, _, _ = _run("cutout")
    folder = Path(res["folder"])
    assert sorted(p.name for p in folder.rglob("*") if p.is_file()) == ["cutout.png", "layers.json", "mask.png",
                                                                        "preview.jpg"]
    # crop
    res, _, _ = _run("cutout", crop=True)
    w, h = Image.open(Path(res["folder"]) / "cutout.png").size
    assert w < 120 and h < 180, (w, h)


def _reconstruct(scale):
    cfg = lf.temp_config(BASE, TMP / "out", edges__decontaminate=False, edges__decontaminate_layers=False,
                         work_max_side=400)
    src = lf.write_scene_png(TMP, f"room{scale}.png", scale=scale)
    res, _, _ = _run(cfg=cfg, src=src)
    folder = Path(res["folder"])
    imgs = [_png(folder / l["file"]) for l in res["layers"]]
    assert imgs[0].shape[:2] == (300 * scale, 400 * scale)
    out = ops.compose(imgs[::-1])
    photo = _png(src)
    assert np.abs(out[:, :, 3] - 1).max() < 1e-6
    assert np.abs(out[:, :, :3] - photo[:, :, :3]).max() <= 1 / 255 + 1e-6
    top = np.max([im[:, :, 3] for im in imgs], axis=0)
    assert top.min() == 1.0                                  # 12: some layer is opaque at every pixel
    return res, folder, imgs


def test_11_12_reconstruction():
    _reconstruct(1)
    res, folder, imgs = _reconstruct(2)                      # full size 2x the work size: guided upsampling
    person = imgs[1][:, :, 3]
    assert person[320, 400] > 0.9 and person[20, 20] == 0 and person[320, 700] == 0
    bg = imgs[-1][:, :, 3]
    assert bg[20, 20] == 1.0 and bg[420, 300] == 0           # hidden under the sofa: transparent


def test_13_stacks_have_no_seams():
    res, _, _ = _run()
    folder = Path(res["folder"])
    st = _png(folder / "stacks/stack_1-2.png")[:, :, 3]
    assert st[195:205, 160:240].min() == 1.0, st[195:205, 160:240].min()    # across the person / sofa boundary
    assert st[160:240, 110:160].min() == 1.0                 # the sofa itself
    assert st[10, 10] == 0 and st[100, 40] == 0              # only wall / the lamp: transparent
    st3 = _png(folder / "stacks/stack_1-3.png")[:, :, 3]
    assert st3[100, 40] == 1.0 and st3[10, 10] == 0


def test_14_input_transparency_is_kept():
    from PIL import Image
    rgb = lf.scene()[0]
    a = np.full(rgb.shape[:2], 255, np.uint8)
    a[:40, :40] = 0
    src = TMP / "transparent.png"
    Image.fromarray(np.dstack([rgb, a]), "RGBA").save(src)
    res, _, _ = _run(src=src)
    folder = Path(res["folder"])
    for rel in ["cutout.png"] + [l["file"] for l in res["layers"]] + [s["file"] for s in res["stacks"]]:
        assert _png(folder / rel)[:40, :40, 3].max() == 0, rel
    assert _png(folder / res["layers"][-1]["file"])[60, 300, 3] == 1.0


def test_15_again_reuses_the_analysis():
    layers._CACHE.clear()
    res, cfg, fk = _run()
    calls = (fk["matte"].calls, fk["depth"].calls, fk["segmenter"].calls)
    assert calls == (1, 1, 1)
    req = layers.parse_request("/layers again layers=3", cfg["layers"])
    assert req["again"] and req["opts"] == {"layers": 3}
    res2 = layers.run(req, cfg)
    assert (fk["matte"].calls, fk["depth"].calls, fk["segmenter"].calls) == calls
    assert [l["name"] for l in res2["layers"]] == ["person", "sofa, plant", "background: wall, floor, lamp"]
    assert res2["folder"] != res["folder"] and [s["file"] for s in res2["stacks"]] == ["stacks/stack_1-2.png"]
    res3 = layers.run(layers.parse_request("/layers again layers=2", cfg["layers"]), cfg)
    assert [l["n"] for l in res3["layers"]] == [1, 2] and not res3["stacks"]
    # the same photo again as a new request: still cached
    layers.run(layers.make_request("cutout", res["source"]), cfg)
    assert (fk["matte"].calls, fk["depth"].calls, fk["segmenter"].calls) == calls


def test_16_report_and_history():
    res, cfg, _ = _run()
    lc = cfg["layers"]
    rep = layers.report(res, lc, color=False)
    lines = rep.split("\n")
    assert "\x1b" not in rep
    assert lines[0] == "[layers] Object of focus: person"
    assert lines[1].startswith("         center, ") and "% of the photo" in lines[1]
    assert f"Background removed -> {Path(res['folder']) / 'cutout.png'}" in rep
    assert "Layers (front to back):" in lines
    i = lines.index("Layers (front to back):")
    assert re.match(r"  0  plant\s+\d+%   in front of the subject$", lines[i + 1]), lines[i + 1]
    assert re.match(r"  1  person\s+\d+%$", lines[i + 2])
    assert re.match(r"  2  sofa\s+\d+%   touches the subject$", lines[i + 3])
    assert re.match(r"  4  background: wall, floor\s+\d+%   hidden parts transparent", lines[i + 5])
    assert str(Path(res["folder"]) / "layers.psd") in rep and str(Path(res["folder"]) / "stacks") in rep
    assert lines[-1].startswith("Took ")
    hist = layers.history(res, lc)
    assert hist.startswith(f"Photo: {res['source']} (400x300)\nObject of focus: person (center, ")
    assert "Layers, front to back: 0 plant (" in hist and "; 4 background: wall, floor (" in hist
    assert f"Folder: {res['folder']} (layers.psd, stacks, preview.jpg)" in hist
    s = layers.summary(res)
    json.dumps(s)
    assert Path(s["cutout"]).is_file() and all(Path(l["file"]).is_file() for l in s["layers"])
    cut, _, _ = _run("cutout")
    rep = layers.report(cut, lc, color=False).split("\n")
    assert len(rep) == 6 and rep[2].startswith("Background removed -> ") and rep[3].startswith("Mask: ")
    assert rep[4].startswith("Preview: ") and rep[5].startswith("Took ")


def test_16b_description_and_fill():
    real = layer_models.describe_focus
    cfg = lf.temp_config(BASE, TMP / "out")
    try:
        layer_models.describe_focus = lambda *a, **k: {"name": "young woman", "description": "a woman on a sofa"}
        res, _, _ = _run(cfg=cfg, describe=True, src=lf.write_scene_png(TMP, "d1.png", second_person=False))
        rep = layers.report(res, cfg["layers"], color=False)
        assert rep.startswith('[layers] Object of focus: person (young woman) - "a woman on a sofa"'), rep[:90]
        assert Path(res["folder"], "layers/layer_1_person.png").is_file()     # file names keep the class label

        def boom(*a, **k):
            raise RuntimeError("no vision model")
        layer_models.describe_focus = boom
        layers._CACHE.clear()
        res, _, _ = _run(cfg=cfg, describe=True)
        assert any("could not describe" in w for w in res["warnings"]) and res["focus"]["description"] is None
    finally:
        layer_models.describe_focus = real
    assert layer_models.parse_description("Name: ginger cat Description: A cat sleeping on grass.", 5) == \
        {"name": "ginger cat", "description": "A cat sleeping on grass."}
    assert layer_models.parse_description("it is a cat, I think", 5) is None
    assert layer_models.parse_description("Name: a b c d e f g Description: x y", 5)["name"] == ""
    # fill: the background becomes opaque, without_subject.png appears
    for method in ("fast", "lama"):
        res, _, fk = _run(fill=method)
        folder = Path(res["folder"])
        assert _png(folder / res["layers"][-1]["file"])[:, :, 3].min() == 1.0
        ws = _png(folder / "without_subject.png")
        assert ws[160, 200, 0] < 0.6                         # the red person is gone
        assert "hidden parts filled" in layers.report(res, cfg["layers"], color=False)
        assert fk["inpaint"].calls == (2 if method == "lama" else 0)


def test_16c_errors_and_stop():
    cfg = lf.temp_config(BASE, TMP / "out_err")
    layer_models.install_fakes(lf.fakes())
    for src, text in ((str(TMP / "nope.jpg"), "No such image"), (str(Path(__file__)), "Can't read")):
        try:
            layers.run(layers.make_request("cutout", src), cfg)
            assert False, src
        except layers.LayersError as e:
            assert text in str(e), e
    from PIL import Image
    tiny = TMP / "tiny.png"
    Image.new("RGB", (20, 20)).save(tiny)
    try:
        layers.run(layers.make_request("cutout", str(tiny)), cfg)
        assert False
    except layers.LayersError as e:
        assert "too small" in str(e)
    flat = TMP / "flat.png"
    Image.new("RGB", (200, 200), (9, 9, 9)).save(flat)
    C = np.zeros((300, 400), np.uint8)
    layer_models.install_fakes({"matte": lf.FakeMatte(np.zeros((300, 400), np.float32)),
                                "depth": lf.FakeDepth(np.zeros((300, 400), np.float32)),
                                "segmenter": lf.FakeSegmenter(C)})
    try:
        layers.run(layers.make_request("cutout", str(flat)), cfg)
        assert False
    except layers.LayersError as e:
        assert str(e) == cfg["layers"]["texts"]["nothing"]
    layer_models.install_fakes(lf.fakes())
    layers._CACHE.clear()
    try:
        layers.run(layers.make_request("layers", str(lf.write_scene_png(TMP, "stop.png"))), cfg,
                   should_stop=lambda: True)
        assert False
    except layers.LayersError as e:
        assert "Stopped" in str(e)
    assert not (TMP / "out_err").exists() or not any((TMP / "out_err").iterdir())   # nothing was saved


# ---------------- 17-20: chat requests + config ----------------

def test_17_parse_request():
    d = Path(tempfile.mkdtemp(prefix="layers parse "))
    try:
        x = lf.write_scene_png(d, "x.jpg".replace(".jpg", ".png"))
        ab = lf.write_scene_png(d, "a b.png")
        cyr = d / "Снимки"
        cyr.mkdir()
        me = lf.write_scene_png(cyr, "аз.png")
        r = layers.parse_request(f"/cutout {x} crop focus=dog", L)
        assert r == {"mode": "cutout", "src": str(x), "opts": {"crop": True, "focus": "dog"}, "again": False}, r
        r = layers.parse_request(f'/layers "{ab}" layers=3 fill=lama', L)
        assert r["mode"] == "layers" and r["src"] == str(ab) and r["opts"] == {"layers": 3, "fill": "lama"}, r
        r = layers.parse_request(f"/layers {ab} fill nopsd nostacks nodescribe layers=99", L)
        assert r["src"] == str(ab) and r["opts"] == {"fill": "lama", "psd": False, "stacks": False,
                                                     "describe": False, "layers": 8}, r
        r = layers.parse_request(f'/cutout {x} focus="person on the left" to="{d}"', L)
        assert r["src"] == str(x) and r["opts"] == {"focus": "person on the left", "to": str(d)}, r
        r = layers.parse_request("/cutout https://example.com/a/photo.jpg?x=1 focus=dog crop", L)
        assert r["src"] == "https://example.com/a/photo.jpg?x=1" and r["opts"] == {"focus": "dog", "crop": True}
        r = layers.parse_request(f"remove the background from {x}", L)
        assert r["mode"] == "cutout" and r["src"] == str(x) and r["opts"] == {}, r
        r = layers.parse_request(f"split {x} into layers", L)
        assert r["mode"] == "layers" and r["src"] == str(x)
        r = layers.parse_request(f"махни фона на {me}", L)
        assert r["mode"] == "cutout" and r["src"] == str(me), r
        assert layers.parse_request(f"раздели {me} на слоеве", L)["mode"] == "layers"
        assert layers.parse_request(f"изрежи кучето от {me}", L)["opts"] == {"focus": "animal"}
        assert layers.parse_request(f"make {x} transparent please", L)["mode"] == "cutout"
        r = layers.parse_request("please remove the background of https://example.com/pics/cat.webp", L)
        assert r["mode"] == "cutout" and r["src"] == "https://example.com/pics/cat.webp"
        assert layers.parse_request(f"cut out the person on the left in {x}", L)["opts"] == {"focus": "person left"}
        assert layers.parse_request(f"remove the background from this picture {x}", L)["opts"] == {}
        assert layers.parse_request("/layers", L) == L["texts"]["usage_layers"]
        assert layers.parse_request("/cutout", L) == L["texts"]["usage_cutout"]
        assert layers.parse_request(r"/cutout E:\nope_does_not_exist.jpg", L) == \
            L["texts"]["not_found"].format(path=r"E:\nope_does_not_exist.jpg")
        layers._LAST.clear()
        assert layers.parse_request("/layers again layers=4", L) == L["texts"]["no_last"]
        layers._LAST.update(src=str(x), key=None, opts={})
        r = layers.parse_request("/layers again layers=4", L)
        assert r == {"mode": "layers", "src": str(x), "opts": {"layers": 4}, "again": True}
        layers._LAST.clear()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_18_not_ours():
    d = Path(tempfile.mkdtemp(prefix="layers_no_"))
    try:
        x = lf.write_scene_png(d, "x.png")
        (d / "contacts.txt").write_text("a@b.c", encoding="utf-8")
        for msg in (f"sort the photos in {d} by face", f"who is in {x}?", f"/see {x}",
                    f"extract all emails from {d / 'contacts.txt'}", "what is a background check?",
                    "remove the background from my photo", f"describe {x}", f"/faces {d}",
                    "remove the background from https://example.com/page.html",
                    f"cut out the text of {d / 'contacts.txt'}"):
            assert layers.parse_request(msg, L) is None, msg
        off = dict(L, enabled=False)
        assert layers.parse_request(f"/cutout {x}", off) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_19_config_is_complete():
    for k in ("enabled", "output_dir", "folder_name", "image_extensions", "work_max_side", "threads", "models",
              "focus", "grouping", "edges", "outputs", "fill", "describe", "intent_patterns", "layer_patterns",
              "colors_256", "texts", "again_cache", "keep_models_loaded", "max_download_mb", "timeout_s"):
        assert k in L, k
    for k in ("server_name", "instructions", "keep_models_loaded", "return_preview", "preview_max_side",
              "max_layers_limit"):
        assert k in BASE["mcp"], k
    status = BASE["ui"]["status"]
    keys = ["layers_model_download", "layers_loading", "layers_reading", "layers_subject", "layers_depth",
            "layers_segment", "layers_grouping", "layers_describe", "layers_vision", "layers_fill", "layers_saving"]
    assert all(k in status for k in keys)
    ade = set(lf.ADE)
    assert len(lf.ADE) == 150 and len(ade) == 150
    for k in L["focus"]["class_weights"]:
        assert k == "default" or k in ade, k
    for k in L["grouping"]["background_classes"] + list(L["focus"]["synonyms"]) + list(L["grouping"]["class_names"]):
        assert k in ade, k
    assert "{label}" in config.text(L["describe"]["prompt"])
    for p in L["intent_patterns"] + L["layer_patterns"]:
        re.compile(p)
    m = L["models"]
    assert m["matte"] in m["mattes"] and m["segmenter"] in m["segmenters"]
    for e in list(m["mattes"].values()) + list(m["segmenters"].values()) + [m["depth"], m["inpaint"]]:
        assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) and e["repo"] and e["file"]
    T = L["texts"]
    assert T["layer_line"].format(n=1, name="x", pct=5, note="") and T["took"].format(s=1, parts="a")


def test_20_layers_goes_before_faces():
    from photos import faces
    d = Path(tempfile.mkdtemp(prefix="layers_order_"))
    try:
        x = lf.write_scene_png(d, "tmp.png")
        msg = f"cut out the person in {x}"
        assert faces.parse_request(msg, BASE["faces"]) is not None       # why layers must be asked first
        r = layers.parse_request(msg, L)
        assert r["mode"] == "cutout" and r["opts"] == {"focus": "person"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    failed = run_all(globals())
    layer_models.install_fakes(None)
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
