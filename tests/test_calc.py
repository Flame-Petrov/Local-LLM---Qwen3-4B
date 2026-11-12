"""
Offline tests for calc.py (pure functions, no model). Plain asserts - run:

    .venv\\Scripts\\python.exe tests\\test_calc.py
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import calc  # noqa: E402
from core import config  # noqa: E402

C = config.load()["calculator"]


# ---------- H1: the whole left-hand side is checked ----------

def test_chained_correct_not_flagged():
    for t in ["2 + 3 * 4 = 14", "10 - 3 - 2 = 5", "(2 + 3) * 4 = 20", "1 + 2 + 3 + 4 = 10",
              "The sum: 2 + 3 * 4 = 14.", "x = 10 - 3 - 2 = 5"]:
        assert calc.check_equations(t, C) == [], t


def test_chained_wrong_flagged():
    assert calc.check_equations("2 + 3 * 4 = 20", C) == [("2 + 3 * 4", "20", "14")]
    assert calc.check_equations("10 - 3 - 2 = 9", C) == [("10 - 3 - 2", "9", "5")]


def test_simple_wrong_still_flagged():
    w = calc.check_equations("268,000,000 − 668 = 267,993,332", C)
    assert len(w) == 1 and w[0][2] == "267,999,332", w


def test_rounding_and_text_ignored():
    assert calc.check_equations("1,929.51 * 2 = 3,859.02", C) == []
    assert calc.check_equations("5 apples = 5", C) == []
    assert calc.check_equations("Total is 5. 3 x 4 = 12", C) == []


# ---------- W1: huge results do not crash ----------

def test_huge_result_refused():
    big = "*".join(["999999999999**100"] * 5)
    value, err = calc.safe_eval(big, C)
    assert value is None and err, (value, err)


def test_fmt_never_raises():
    assert calc.fmt(10 ** 5000)  # beyond Python's int->str limit
    assert calc.fmt(267999332) == "267,999,332"


# ---------- W2: function arguments, complex results ----------

def test_function_args_not_merged():
    assert calc.safe_eval("min(5,100)", C) == (5, None)
    assert calc.safe_eval("max(5, 100)", C) == (100, None)


def test_thousands_separator_still_works():
    assert calc.safe_eval("1,000 + 5", C) == (1005, None)
    assert calc.safe_eval("268,000,000 - 668", C) == (267999332, None)


def test_complex_result_refused():
    value, err = calc.safe_eval("(-8)**0.5", C)
    assert value is None and err


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
    sys.exit(1 if run_all(globals()) else 0)
