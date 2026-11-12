"""
Debug log: one JSON Lines file per app run (config debug.log_dir), one event per line.

    {"time": "...", "t": 12.345, "session": "chat_...", "event": "search", ...}

Read it with any text editor, or e.g.:  python -c "import json;[print(json.loads(l)['event']) for l in open('logs/run_....jsonl', encoding='utf-8')]"

Logging must never break the chat: every write is wrapped and failures are ignored.
Everything stays on this machine.
"""

import json
import time
import traceback
from datetime import datetime, timedelta

from core import config

_state = {"enabled": False, "cfg": None, "file": None, "path": None,
          "session": None, "t0": time.monotonic()}


def setup(dcfg: dict):
    _state["cfg"] = dcfg
    _state["enabled"] = bool(dcfg["enabled"])
    _cleanup_old(dcfg)


def _cleanup_old(dcfg: dict):
    """Delete logs older than debug.keep_days (0 = keep forever)."""
    days = dcfg["keep_days"]
    d = config.path(dcfg["log_dir"])
    if not days or not d.is_dir():
        return
    cutoff = (datetime.now() - timedelta(days=days)).timestamp()
    for f in d.glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def enabled() -> bool:
    return _state["enabled"]


def set_enabled(on: bool):
    _state["enabled"] = on


def path():
    """Current log file (created on first event)."""
    return _state["path"]


def set_session(name: str):
    _state["session"] = name


def clip(value, kind: str = "text"):
    """Apply the size/privacy settings to a big field.
    kind: 'prompt' (debug.include_prompts), 'page' (debug.include_page_text) or 'text'."""
    cfg = _state["cfg"]
    if value is None or cfg is None:
        return value
    s = str(value)
    if kind == "prompt" and not cfg["include_prompts"]:
        return f"<omitted: {len(s)} chars>"
    if kind == "page" and not cfg["include_page_text"]:
        return f"<omitted: {len(s)} chars>"
    limit = cfg["max_field_chars"]
    if limit and len(s) > limit:
        return s[:limit] + f"... <clipped, {len(s)} chars total>"
    return s


def _open():
    if _state["file"] is None:
        cfg = _state["cfg"]
        d = config.path(cfg["log_dir"])
        d.mkdir(parents=True, exist_ok=True)
        p = d / (datetime.now().strftime(cfg["file_name_format"]) + ".jsonl")
        _state["file"] = open(p, "a", encoding="utf-8", buffering=1)  # line-buffered: survives crashes
        _state["path"] = p
    return _state["file"]


def log(event: str, **data):
    if not _state["enabled"]:
        return
    try:
        rec = {"time": datetime.now().isoformat(timespec="milliseconds"),
               "t": round(time.monotonic() - _state["t0"], 3),
               "session": _state["session"], "event": event, **data}
        _open().write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def log_exception(where: str, exc: BaseException):
    log("error", where=where, error=repr(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
