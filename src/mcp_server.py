"""
Local photo tools for Claude Code / Claude Desktop / any MCP client (stdio):

    remove_background   cut the object of focus out of a photo (transparent PNG) and say what it is
    split_layers        also split the photo into depth-ordered layers (PNGs, stacks, layered PSD)

The engine is layers.py (the same one the chat's /cutout and /layers use). Everything runs locally on the CPU.

    .venv\\Scripts\\python.exe src\\mcp_server.py --download-models      once, before the first call
    claude mcp add local-llm-tools --scope user -- C:\\path\\to\\local-llm\\.venv\\Scripts\\python.exe C:\\path\\to\\local-llm\\src\\mcp_server.py

stdout carries only protocol messages (mcp 2.x serves from a private copy of fd 1 and points fd 1 at stderr).
Settings: config.json -> mcp, layers.
"""

import argparse
import sys
import threading
import time
from functools import partial
from pathlib import Path

import anyio
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp_types import CallToolResult, TextContent, ToolAnnotations

from core import config
from core import debuglog
from photos import layer_models
from photos import layers as engine
from core.debuglog import log

STAGES = ["layers_reading", "layers_model_download", "layers_loading", "layers_subject", "layers_depth",
          "layers_segment", "layers_grouping", "layers_describe", "layers_vision", "layers_fill", "layers_saving"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Local photo tools as an MCP server (stdio)")
    p.add_argument("--config", help="path to a config JSON file (default: config/config.json)")
    p.add_argument("--download-models", action="store_true",
                   help="download + check all models, run each once, then exit")
    return p.parse_known_args(argv)[0]


def configure(cfg: dict) -> dict:
    """Use this config (the MCP server keeps the ONNX models loaded between calls)."""
    global CFG
    cfg["layers"]["keep_models_loaded"] = bool(cfg["mcp"]["keep_models_loaded"])
    CFG = cfg
    return cfg


ARGS = parse_args()
CFG: dict = {}
configure(config.load(ARGS.config))
server = MCPServer(CFG["mcp"]["server_name"], instructions=config.text(CFG["mcp"]["instructions"]))
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)


@server.tool(annotations=WRITES)
async def remove_background(image: str, ctx: Context, focus: str | None = None, describe: bool | None = None,
                            crop: bool = False, output_dir: str | None = None) -> CallToolResult:
    """Cut the main subject (object of focus) out of a photo and remove the background.
    Runs locally on the user's PC (CPU): about 15-30 s, plus 20-60 s with describe=true.
    Says what it detected as the object of focus (label, position, size; with describe=true also one sentence
    from a local vision model) and saves a transparent full-size PNG (cutout.png) and mask.png in a new folder.
    image: absolute path of a local image (jpg png webp bmp tif gif heic) or an http(s) image URL.
    focus: what to cut out if it is not the obvious subject, in plain words ("sofa", "the dog", "person on the left").
    crop: crop the cutout to the subject instead of keeping the full canvas.
    output_dir: parent folder for the new result folder (default: the app's downloads/layers)."""
    return await _run(ctx, "remove_background", "cutout", image,
                      dict(focus=focus, describe=describe, crop=crop or None, to=output_dir))


@server.tool(annotations=WRITES)
async def split_layers(image: str, ctx: Context, layers: int = 5, focus: str | None = None,
                       describe: bool | None = None, fill: str = "none", stacks: bool = True, psd: bool = True,
                       output_dir: str | None = None) -> CallToolResult:
    """Split a photo into depth-ordered layers and remove the background. Layer 1 = the object of
    focus (e.g. a person), 2 = what it touches (the couch it sits on), then things further back (furniture),
    last = background (walls, floor, sky); layer 0 = things in front of the subject, if any. Saves every layer
    as a full-size transparent PNG, cumulative stacks (1-2 = person + couch, ...), layers.psd
    (Photoshop/GIMP/Krita/Photopea), preview.jpg and layers.json. Local CPU, about 20-40 s.
    image: absolute path of a local image (jpg png webp bmp tif gif heic) or an http(s) image URL.
    layers: total number of layers incl. subject and background, 2-8 (2 = plain background removal).
    focus: what the subject is if it is not the obvious one, in plain words ("sofa", "the dog", "person on the left").
    describe: one sentence about the subject from a local vision model (+20-60 s).
    fill: "none" (hidden parts stay transparent), "fast" or "lama" (fill the background behind the subject
    and also save without_subject.png).
    stacks / psd: set false to skip the cumulative stacks / the layered PSD."""
    return await _run(ctx, "split_layers", "layers", image,
                      dict(layers=layers, focus=focus, describe=describe, fill=fill, stacks=stacks, psd=psd,
                           to=output_dir))


def _error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _bridge(ctx):
    """progress(key, **values) for the engine's worker thread -> MCP progress notifications."""
    status = CFG["ui"]["status"]

    def progress(key, **values):
        try:
            step = STAGES.index(key) + 1 if key in STAGES else 0
            anyio.from_thread.run(ctx.report_progress, step, len(STAGES), status[key].format(**values))
        except Exception:
            pass    # no progress token, or the request is already cancelled
    return progress


async def _run(ctx, tool: str, mode: str, image: str, opts: dict) -> CallToolResult:
    lcfg, mcfg = CFG["layers"], CFG["mcp"]
    T = lcfg["texts"]
    t0 = time.monotonic()
    log("mcp_call", tool=tool, image=image, args=opts)
    image = str(image or "").strip().strip('"')
    is_url = image.lower().startswith(("http://", "https://"))
    if not image or (not is_url and not Path(image).expanduser().is_absolute()):
        return _error(T["relative_path"].format(path=image))
    if opts.get("layers") is not None:
        opts["layers"] = max(2, min(int(mcfg["max_layers_limit"]), int(opts["layers"])))
    if opts.get("fill") is not None:
        opts["fill"] = str(opts["fill"]).lower()
        if opts["fill"] not in ("none", "fast", "lama"):
            return _error('fill must be "none", "fast" or "lama"')
    if opts.get("describe") is None:
        opts["describe"] = bool(lcfg["describe"]["mcp"])
    if opts.get("to") and not Path(opts["to"]).expanduser().is_absolute():
        return _error(f"output_dir must be an absolute path: {opts['to']}")
    req = engine.make_request(mode, image, opts)
    stop = threading.Event()
    try:
        result = await anyio.to_thread.run_sync(
            partial(engine.run, req, CFG, progress=_bridge(ctx), should_stop=stop.is_set),
            abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        stop.set()                       # the worker thread stops at its next stage check
        log("mcp_result", tool=tool, cancelled=True, seconds=round(time.monotonic() - t0, 1))
        raise
    except engine.LayersError as e:
        log("mcp_result", tool=tool, is_error=True, error=str(e), seconds=round(time.monotonic() - t0, 1))
        return _error(str(e))
    except Exception as e:
        debuglog.log_exception("mcp_" + tool, e)
        log("mcp_result", tool=tool, is_error=True, error=repr(e), seconds=round(time.monotonic() - t0, 1))
        return _error(T["failed"].format(error=e))
    content = [TextContent(type="text", text=engine.report(result, lcfg, color=False))]
    if mcfg["return_preview"] and result.get("preview_small"):
        try:
            content.append(Image(path=result["preview_small"]).to_image_content())
        except Exception as e:
            log("mcp_preview_failed", error=repr(e))
    log("mcp_result", tool=tool, is_error=False, seconds=round(time.monotonic() - t0, 1), folder=result["folder"])
    return CallToolResult(content=content, structured_content=engine.summary(result))


def main():
    debuglog.setup(CFG["debug"])         # logs/ only, never stdout
    debuglog.set_session("mcp_server")
    if ARGS.download_models:
        layer_models.download_all(CFG)   # prints to stderr
        sys.exit(0)
    server.run()                         # stdio


if __name__ == "__main__":
    main()
