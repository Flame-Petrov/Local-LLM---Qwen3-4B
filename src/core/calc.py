"""
Calculator tool for the model + a backstop check of arithmetic in answers.

The MODEL decides when to calculate: it writes a line   CALC: 268000000 - 668
The program evaluates it exactly (safe parser, no code execution) and inserts
        = 267,999,332
into the answer, then the model continues writing with the correct number.

Backstop: equations the model wrote itself ("268,000,000 − 668 = 267,993,332") are
re-checked; wrong ones get a correction line. Settings: config.json -> "calculator".
"""

import ast
import math
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "log10": math.log10, "ln": math.log}
_MAX_DIGITS = 1000  # larger results are refused (fmt / Python's int->str digit limit)
_NUM = r"\d{1,3}(?:[,  ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"


def normalize(expr: str) -> str:
    """Human / model notation -> Python arithmetic: '268,000,000 − 668', '3 × 4', '15% of 200', '2^10'."""
    e = expr.strip().strip("`*").split("=")[0].strip()  # "2350 * 18% = 2350 * 0.18" -> "2350 * 18%"
    e = e.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/").replace("^", "**")
    # thousands separators; inside a function call ("min(5,100)") a comma separates arguments
    sep = r"[  ]" if re.search(r"[A-Za-z_]\w*\s*\(", e) else r"[,  ]"
    e = re.sub(rf"(?<=\d){sep}(?=\d{{3}}\b)", "", e)
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", e, flags=re.I)  # 15% of 200
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", e)              # 15%
    e = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", e)                  # 3 x 4
    e = re.sub(r"(?<=\d)\s*[kK]\b", "*1000", e)
    e = re.sub(r"(?<=\d)\s*M\b", "*1000000", e)
    e = re.sub(r"(?<=\d)\s*(?:B|bn)\b", "*1000000000", e)
    return e


def safe_eval(expr: str, ccfg: dict):
    """(value, None) or (None, error). Only numbers, + - * / // % ** ( ) and a few math functions."""
    e = normalize(expr)
    if len(e) > ccfg["max_expression_chars"]:
        return None, "expression too long"
    try:
        tree = ast.parse(e, mode="eval")
    except SyntaxError:
        return None, f"cannot read '{expr.strip()}'"

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > ccfg["max_exponent"] or abs(a) > 1e12):
                raise ValueError("number too large")
            return _check(_OPS[type(n.op)](a, b))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _check(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise ValueError("only numbers and + - * / % ** ( ) are allowed")

    def _check(v):
        if isinstance(v, complex):
            raise ValueError("result is not a real number")
        if isinstance(v, int) and v.bit_length() > _MAX_DIGITS * 3.33:
            raise ValueError("number too large")
        return v

    try:
        value = ev(tree)
    except ZeroDivisionError:
        return None, "division by zero"
    except (ValueError, TypeError, OverflowError) as err:
        return None, str(err)
    return value, None


def fmt(value) -> str:
    """267999332 -> '267,999,332'; 3.14159265 -> '3.14159265'; 2.5 -> '2.5'."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    if isinstance(value, int):
        try:
            return f"{value:,}"
        except ValueError:  # beyond Python's int->str digit limit
            return "(number too large)"
    return f"{value:,.10g}" if abs(value) >= 1e-4 else f"{value:.6g}"


def _to_num(s: str):
    try:
        return float(re.sub(r"[,  ]", "", s))
    except ValueError:
        return None


def check_equations(text: str, ccfg: dict) -> list[tuple[str, str, str]]:
    """Wrong equations the model wrote itself: [(expression, what it said, correct result)]."""
    plain = text.replace("**", "").replace("`", "")
    wrong = []
    for m in re.finditer(rf"=\s*({_NUM})(?![\d.,]*\s*[+\-−–×*/÷^(])", plain):
        said = m.group(1)
        # the WHOLE left-hand side: back to the previous "=", the line start or a sentence break
        left = re.split(r"[=\n;:!?]|\.\s", plain[:m.start()])[-1]
        run = re.search(r"[\d\s,  .+\-−–×*/÷xX()%^]*$", left).group(0)
        expr = re.sub(r"^[^\d(]*", "", run).strip()
        if not re.search(r"\d\s*[+\-−–×*/÷xX^%]", expr):
            continue
        if re.search(r"[xX]", expr) and not re.search(r"\d\s*[xX]\s*\d", expr):
            continue
        value, err = safe_eval(expr, ccfg)
        claimed = _to_num(said)
        if err or value is None or claimed is None:
            continue
        # exact: only the rounding the answer itself shows is allowed ("1,929.51" -> +-0.005, "267,999,332" -> 0)
        decimals = len(said.split(".")[1]) if "." in said else 0
        if abs(value - claimed) > 0.5 * 10 ** -decimals + 1e-9:
            wrong.append((expr, said, fmt(value)))
    return wrong
