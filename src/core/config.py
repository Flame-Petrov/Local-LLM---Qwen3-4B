"""Loads config.json — the single source of all prompts, rules and settings."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/core/config.py -> project root
DEFAULT_PATH = ROOT / "config" / "config.json"


def load(path: str | None = None) -> dict:
    p = Path(path).resolve() if path else DEFAULT_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(
            f"Config file not found: {p}\n"
            "For a new checkout, copy config/config.example.json to config/config.json."
        )
    except json.JSONDecodeError as e:
        sys.exit(f"Config file {p} is not valid JSON: {e}")


def text(value) -> str:
    """Prompts may be a string or a list of lines (easier to edit in JSON)."""
    return "\n".join(value) if isinstance(value, list) else value


def params(section: dict) -> dict:
    """A config section without its _comment keys, ready to pass as **kwargs."""
    return {k: v for k, v in section.items() if not k.startswith("_")}


def path(value: str) -> Path:
    """Config paths are relative to the app folder unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
