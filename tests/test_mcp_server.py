"""
Tests for mcp_server.py with fake models - in-process and over real stdio:

    .venv\\Scripts\\python.exe tests\\test_mcp_server.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import anyio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config  # noqa: E402
from photos import layer_models  # noqa: E402
from photos import layers  # noqa: E402
import layers_fakes as lf  # noqa: E402
import mcp_server  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402
from test_music import run_all  # noqa: E402

BASE = config.load()
TMP = Path(tempfile.mkdtemp(prefix="mcp_tests_"))


def _setup(name):
    out = TMP / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = lf.temp_config(BASE, out)
    layer_models.install_fakes(lf.fakes())
    layers._CACHE.clear()
    return cfg, out, lf.write_scene_png(TMP, f"{name}.png")


def test_in_process():
    cfg, out, png = _setup("inproc")
    mcp_server.configure(cfg)

    async def go():
        async with Client(mcp_server.server) as c:
            tools = (await c.list_tools()).tools
            assert {t.name for t in tools} == {"remove_background", "split_layers"}
            for t in tools:
                assert t.input_schema["required"] == ["image"], t.input_schema
                assert "ctx" not in t.input_schema["properties"]
                assert t.annotations.destructive_hint is False
                assert "remove the background" in t.description and "image:" in t.description
            seen = []

            async def on_progress(progress, total, message):
                seen.append(message)

            r = await c.call_tool("split_layers", {"image": str(png), "layers": 3}, progress_callback=on_progress)
            assert not r.is_error, r.content
            assert [b.type for b in r.content] == ["text", "image"]
            assert r.content[0].text.startswith("[layers] Object of focus: person")
            assert "(confidence:" in r.content[0].text and "\x1b" not in r.content[0].text
            assert r.content[1].mime_type == "image/jpeg" and len(r.content[1].data) < 1_400_000
            s = r.structured_content
            assert len(s["layers"]) == 3 and all(Path(l["file"]).is_file() for l in s["layers"])
            assert Path(s["psd"]).is_file() and Path(s["cutout"]).is_file() and s["focus"]["label"] == "person"
            assert Path(s["folder"]).parent == out
            assert any("Saving" in m for m in seen), seen

            r = await c.call_tool("remove_background", {"image": str(png), "focus": "couch", "crop": True})
            assert not r.is_error and r.structured_content["focus"]["label"] == "sofa"
            assert r.structured_content["layers"] == [] and r.structured_content["psd"] is None

            r = await c.call_tool("remove_background", {"image": str(TMP / "missing.jpg")})
            assert r.is_error and "No such image" in r.content[0].text
            r = await c.call_tool("remove_background", {"image": "relative.jpg"})
            assert r.is_error and "full path" in r.content[0].text
            r = await c.call_tool("split_layers", {"image": str(png), "fill": "magic"})
            assert r.is_error and "fill" in r.content[0].text

    anyio.run(go)


def _params(cfg, delay=0.0):
    env = dict(os.environ, MCP_FAKE_DELAY=str(delay))
    return StdioServerParameters(command=sys.executable,
                                 args=[str(Path(__file__).with_name("run_mcp_with_fakes.py")),
                                       "--config", str(lf.write_config(cfg, cfg["layers"]["output_dir"]))],
                                 env=env)


def test_real_stdio():
    """initialize + list_tools + one call over real pipes: nothing may pollute stdout."""
    cfg, out, png = _setup("stdio")

    async def go():
        async with Client(_params(cfg)) as c:
            assert {t.name for t in (await c.list_tools()).tools} == {"remove_background", "split_layers"}
            r = await c.call_tool("split_layers", {"image": str(png)})
            assert not r.is_error, r.content
            assert [b.type for b in r.content] == ["text", "image"]
            assert len(r.structured_content["layers"]) == 5
            assert (Path(r.structured_content["folder"]) / "layers.json").is_file()

    anyio.run(go)


def test_cancellation():
    """A cancelled call stops at the next stage and writes nothing."""
    cfg, out, png = _setup("cancel")

    async def go():
        async with Client(_params(cfg, delay=1.5)) as c:
            with anyio.move_on_after(0.7) as scope:
                await c.call_tool("split_layers", {"image": str(png)})
            assert scope.cancelled_caught
            await anyio.sleep(6)                    # longer than all three fake stages together
            assert len((await c.list_tools()).tools) == 2     # the server is still alive

    anyio.run(go)
    assert not [p for p in out.rglob("layers.json")], list(out.rglob("*"))
    assert not [p for p in out.iterdir() if p.is_dir()]


if __name__ == "__main__":
    failed = run_all(globals())
    layer_models.install_fakes(None)
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
