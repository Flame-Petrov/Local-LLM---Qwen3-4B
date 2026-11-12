"""
Offline tests for writer.py (no model: the model is a fake). Plain asserts - run:

    .venv\\Scripts\\python.exe tests\\test_writer.py

Files are written to a temp folder.
"""

import copy
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import config  # noqa: E402
from files import writer  # noqa: E402

CFG = config.load()
TMP = Path(tempfile.mkdtemp(prefix="writer_test_"))


def wcfg():
    w = copy.deepcopy(CFG["writer"])
    w["output_dir"] = str(TMP / "documents")
    return w


class FakeSpin:
    def stop(self):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


def fake_generate(llm, msgs, gen):
    return "FILENAME: notes\n# Notes\n\nSome text.", None


MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "find info"},
            {"role": "assistant", "content": "Here is a lot of research text about the topic. " * 5}]


def run_write(rest):
    cfg = copy.deepcopy(CFG)
    cfg["writer"] = wcfg()
    return writer.write(None, MESSAGES, rest, cfg, fake_generate, FakeSpin())


# ---------- H11: URLs and paths that escape documents/ ----------

def test_url_is_not_a_file_name():
    req = writer.parse_request("https://x.com/r.html summary", wcfg())
    assert req[0] is None, req


def test_write_url_does_not_crash():
    out = run_write("https://x.com/r.html short summary")
    assert isinstance(out, str)
    assert not (TMP / "documents" / "https:").exists()


def test_escaping_paths_rejected():
    w = wcfg()
    for name in ["../x.txt", "..\\..\\x.txt", "C:x.txt", "sub/../../x.txt", "a:b.txt"]:
        assert writer.target_path(name, w) is None, name


def test_safe_paths_allowed():
    w = wcfg()
    base = (TMP / "documents").resolve()
    assert writer.target_path("report.docx", w) == base / "report.docx"
    assert writer.target_path("sub/report.txt", w) == base / "sub" / "report.txt"
    absolute = str(TMP / "elsewhere" / "x.txt")
    assert writer.target_path(absolute, w) == Path(absolute)


def test_write_rejects_escape_without_writing():
    out = run_write("../../escaped.txt notes")
    assert "escaped.txt" in out and not (TMP / "escaped.txt").exists() and not (TMP.parent / "escaped.txt").exists()


def test_write_normal_file():
    out = run_write("plain.txt notes")
    assert (TMP / "documents" / "plain.txt").exists(), out


# ---------- W9: typed() keeps text-like numbers ----------

def test_typed_leading_zeros_kept():
    assert writer.typed("007") == "007"
    assert writer.typed("0888 123 456") == "0888 123 456"


def test_typed_thousands_comma():
    assert writer.typed("1,234") == 1234
    assert writer.typed("1,234,567") == 1234567


def test_typed_decimals():
    assert writer.typed("1.5") == 1.5
    assert writer.typed("1,5") == 1.5
    assert writer.typed("0.25") == 0.25
    assert writer.typed("42") == 42
    assert writer.typed("1 234") == 1234
    assert writer.typed("abc") == "abc"


# ---------- low: RTF beyond the BMP, TOML mixed lists ----------

def test_rtf_emoji_surrogate_pair():
    rtf = writer.build_rtf("Hi \U0001F600")
    assert "\\u-10179?\\u-8704?" in rtf, rtf


def test_toml_mixed_list_keeps_dicts():
    out = writer.to_toml({"items": [1, {"a": 2}]})
    assert "a = 2" in out, out


def run_all(module_globals):
    tests = [(n, fn) for n, fn in module_globals.items() if n.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    try:
        sys.exit(1 if run_all(globals()) else 0)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
