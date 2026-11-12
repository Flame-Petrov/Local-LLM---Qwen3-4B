"""
File reading: '/read <file> [question]', '/read again <question>', or any message that names an
EXISTING file ("extract all emails from contacts.txt"). The model reads the file and answers.

  1. load      the file -> plain text (txt/code, csv/tsv, json/yaml/toml/xml, html, docx, pptx,
               xlsx, odt, rtf, pdf). Cached per file in downloads/reads/<file_id>/.
  2. facts     the program itself counts lines/words/pages/rows and sums numeric columns, and
               regex-finds emails/links/phones/dates when the question asks for them
               (a 4B model can't count reliably).
  3. answer    small file  -> the whole file goes into the chat (path A)
               big file    -> read part by part, only the relevant notes are kept (path B, cached
                              per question so Ctrl+C + re-run resumes)
               too big     -> refused, with a suggestion to read a range (path C)
  4. verify    every number / email / link / quoted text in the answer is looked up in the file;
               what isn't there is listed under the answer.
Settings, prompts and texts: config.json -> "reader".
"""

import csv
import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from core import config
from core.config import text
from core.debuglog import clip, log

LOADER_VERSION = 1  # bump when a loader changes: old cached extractions are then ignored
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif", ".avif")
MEDIA_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp",
              ".mp3", ".m4a", ".opus", ".wav", ".flac", ".ogg", ".aac")
TEXT_EXTS = {"txt", "md", "log", "ini", "cfg", "conf", "srt", "vtt", "py", "js", "ts", "sql", "sh", "bat",
             "ps1", "c", "h", "cpp", "hpp", "java", "cs", "go", "rs", "rb", "php", "kt", "swift", "lua",
             "r", "css", "scss", "jsx", "tsx", "vue", "env", "properties", "gitignore", "tex", "rst"}
OLD_FORMATS = {"doc": "docx", "xls": "xlsx", "ppt": "pptx"}
RANGE_UNITS = {"page": "page", "pages": "page", "страница": "page", "страници": "page", "стр": "page",
               "slide": "slide", "slides": "slide", "слайд": "slide", "слайдове": "slide",
               "sheet": "sheet", "лист": "sheet", "line": "line", "lines": "line", "ред": "line", "редове": "line"}
MARKERS = {"page": "--- Page {name} ---", "slide": "--- Slide {name} ---", "sheet": "=== Sheet: {name} ==="}
LOOKALIKE = str.maketrans("авекмнорстухіјѕАВЕКМНОРСТУХІЈЅ", "abekmhopctyxijsABEKMHOPCTYXIJS")

LAST: dict | None = None  # the last file read ({"path", "range"}) for '/read again'; cleared by a new chat


class ReadError(Exception):
    """A message for the user (file not found, unsupported, too big...)."""


@dataclass
class Doc:
    kind: str                                  # "text" or "table"
    parts: list = field(default_factory=list)  # [unit, name, text]: unit = page|slide|sheet|None
    meta: dict = field(default_factory=dict)   # encoding, pages, pages_with_text, slides, paragraphs, tables

    @property
    def text(self) -> str:
        out = []
        for unit, name, body in self.parts:
            out.append((MARKERS[unit].format(name=name) + "\n" + body) if unit else body)
        return "\n\n".join(out)


def T(rcfg, key, **values) -> str:
    return rcfg["texts"][key].format(**values)


# ---------------- decoding ----------------

def decode(data: bytes, rcfg) -> tuple[str, str]:
    """Bytes -> (text, encoding). BOM -> UTF-8 -> Bulgarian cp1251 -> charset_normalizer."""
    for bom, enc in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if data.startswith(bom):
            return _nl(data.decode(enc, "replace")), enc
    try:
        return _nl(data.decode("utf-8")), "utf-8"
    except UnicodeDecodeError:
        pass
    # short cp1251 texts are often mis-detected (as big5 etc.): if a legacy encoding turns every
    # non-ASCII byte into a letter of its script, take it
    for enc in rcfg.get("legacy_encodings", []):
        try:
            s = data.decode(enc)
        except UnicodeDecodeError:
            continue
        odd = [c for c in s if ord(c) > 127]
        if odd and sum(1 for c in odd if "\u0400" <= c <= "\u04ff" or c in "«»„“”–—№…’‘•€") / len(odd) >= 0.95:
            return _nl(s), enc
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best is not None:
            return _nl(str(best)), best.encoding
    except ImportError:
        pass
    return _nl(data.decode("cp1252", "replace")), "cp1252"


def _nl(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def looks_binary(data: bytes) -> bool:
    head = data[:8192]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False  # UTF-16 text has NUL bytes but a BOM
    return b"\x00" in head


# ---------------- loaders: each returns a Doc ----------------

def load_plain(path: Path, rcfg) -> Doc:
    data = path.read_bytes()
    if looks_binary(data):
        raise ReadError(T(rcfg, "binary", name=path.name))
    s, enc = decode(data, rcfg)
    return Doc("text", [[None, None, s.strip("\n")]], {"encoding": enc})


def rows_to_csv(rows) -> str:
    """Rows -> CSV text, empty trailing rows and columns dropped."""
    rows = [["" if v is None else str(v) for v in r] for r in rows]
    while rows and not any(c.strip() for c in rows[-1]):
        rows.pop()
    width = max((max((i + 1 for i, c in enumerate(r) if c.strip()), default=0) for r in rows), default=0)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in rows:
        w.writerow((r + [""] * width)[:width])
    return buf.getvalue().rstrip("\n")


def load_csv(path: Path, rcfg) -> Doc:
    d = load_plain(path, rcfg)
    s = d.parts[0][2]
    try:
        dialect = csv.Sniffer().sniff(s[:8192], delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows = list(csv.reader(io.StringIO(s), delimiter=delim))
    return Doc("table", [[None, None, rows_to_csv(rows)]], {**d.meta, "delimiter": delim})


def _table_from(obj):
    """A list of flat dicts (JSON/YAML records) -> rows with a header, else None."""
    if not (isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj)):
        return None
    if any(isinstance(v, (dict, list)) for x in obj for v in x.values()):
        return None
    keys = []
    for x in obj:
        keys += [k for k in x if k not in keys]
    return [keys] + [[_cell(x.get(k)) for k in keys] for x in obj]


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if hasattr(v, "isoformat"):
        s = v.isoformat()
        return s[:10] if s.endswith("T00:00:00") else s
    return str(v)


def load_json(path: Path, rcfg) -> Doc:
    d = load_plain(path, rcfg)
    try:
        obj = json.loads(d.parts[0][2])
    except ValueError:
        return d  # broken JSON is still text
    rows = _table_from(obj)
    if rows:
        return Doc("table", [[None, None, rows_to_csv(rows)]], d.meta)
    return Doc("text", [[None, None, json.dumps(obj, ensure_ascii=False, indent=1)]], d.meta)


def load_yaml(path: Path, rcfg) -> Doc:
    d = load_plain(path, rcfg)
    try:
        import yaml
        rows = _table_from(yaml.safe_load(d.parts[0][2]))
    except Exception:
        rows = None
    return Doc("table", [[None, None, rows_to_csv(rows)]], d.meta) if rows else d


def load_xml(path: Path, rcfg) -> Doc:
    d = load_plain(path, rcfg)
    s = "\n".join(ln.strip() for ln in d.parts[0][2].split("\n") if ln.strip())
    return Doc("text", [[None, None, s]], d.meta)


def load_html(path: Path, rcfg) -> Doc:
    data = path.read_bytes()
    s, enc = decode(data, rcfg)
    out = ""
    try:
        import trafilatura
        out = trafilatura.extract(s, include_tables=True, include_comments=False) or ""
    except Exception:
        out = ""
    if not out.strip():
        try:
            import lxml.html
            root = lxml.html.fromstring(s)
            for bad in root.xpath("//script|//style|//noscript"):
                bad.drop_tree()
            out = root.text_content()
        except Exception:
            out = re.sub(r"<[^>]+>", " ", s)
        out = "\n".join(" ".join(ln.split()) for ln in out.split("\n"))
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return Doc("text", [[None, None, out]], {"encoding": enc})


def _need(lib: str, pip_name: str, ext: str, rcfg):
    try:
        return __import__(lib)
    except ImportError:
        raise ReadError(T(rcfg, "missing_lib", ext=ext, lib=pip_name)) from None


def load_docx(path: Path, rcfg) -> Doc:
    _need("docx", "python-docx", "docx", rcfg)
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(str(path))
    out, paras, tables = [], 0, 0
    for el in doc.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, doc)
            t = p.text.strip()
            if not t:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            level = int(m.group(1)) if m else 1 if style == "Title" else 0
            out.append(("#" * level + " " + t) if level else t)
            paras += 1
        elif tag == "tbl":
            tables += 1
            rows = []
            for r in Table(el, doc).rows:
                cells, last = [], None
                for c in r.cells:  # a merged cell repeats the same cell object: keep one (equal text stays)
                    if last is not None and c._tc is last:
                        continue
                    last = c._tc
                    cells.append(" ".join(c.text.split()))
                rows.append(" | ".join(cells))
            out.append("\n".join(rows))
    return Doc("text", [[None, None, "\n\n".join(out)]], {"paragraphs": paras, "tables": tables})


def load_pptx(path: Path, rcfg) -> Doc:
    _need("pptx", "python-pptx", "pptx", rcfg)
    from pptx import Presentation
    prs = Presentation(str(path))

    def shape_text(sh, out):
        if getattr(sh, "shape_type", None) == 6 and hasattr(sh, "shapes"):  # group
            for s in sh.shapes:
                shape_text(s, out)
            return
        if getattr(sh, "has_table", False) and sh.has_table:
            out += [" | ".join(" ".join(c.text.split()) for c in r.cells) for r in sh.table.rows]
        elif getattr(sh, "has_text_frame", False) and sh.has_text_frame:
            t = "\n".join(p.text for p in sh.text_frame.paragraphs if p.text.strip())
            if t.strip():
                out.append(t)

    parts = []
    for i, slide in enumerate(prs.slides, 1):
        out = []
        title = slide.shapes.title
        if title is not None and title.has_text_frame and title.text_frame.text.strip():
            out.append("# " + " ".join(title.text_frame.text.split()))
        for sh in slide.shapes:
            if title is not None and sh.shape_id == title.shape_id:
                continue
            shape_text(sh, out)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            if notes:
                out.append("Speaker notes: " + notes)
        parts.append(["slide", str(i), "\n".join(out)])
    return Doc("text", parts, {"slides": len(parts)})


def load_xlsx(path: Path, rcfg) -> Doc:
    _need("openpyxl", "openpyxl", "xlsx", rcfg)
    from openpyxl import load_workbook
    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    try:
        for ws in wb.worksheets:
            body = rows_to_csv([[_cell(v) for v in r] for r in ws.iter_rows(values_only=True)])
            if body.strip():
                parts.append(["sheet", ws.title, body])
    finally:
        wb.close()
    return Doc("table", parts, {"sheets": len(parts)})


def load_odt(path: Path, rcfg) -> Doc:
    import zipfile
    from lxml import etree
    with zipfile.ZipFile(path) as z:
        root = etree.fromstring(z.read("content.xml"))
    ns_text = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    ns_table = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    out, tables = [], 0

    def txt(el):
        return " ".join("".join(el.itertext()).split())

    def walk(el):
        nonlocal tables
        for ch in el:
            tag = ch.tag
            if tag == f"{{{ns_text}}}h":
                lvl = int(ch.get(f"{{{ns_text}}}outline-level", "1") or 1)
                out.append("#" * lvl + " " + txt(ch))
            elif tag == f"{{{ns_text}}}p":
                if txt(ch):
                    out.append(txt(ch))
            elif tag == f"{{{ns_table}}}table":
                tables += 1
                rows = [" | ".join(txt(c) for c in r.iter(f"{{{ns_table}}}table-cell"))
                        for r in ch.iter(f"{{{ns_table}}}table-row")]
                out.append("\n".join(rows))
            else:
                walk(ch)

    walk(root)
    return Doc("text", [[None, None, "\n\n".join(out)]], {"paragraphs": len(out), "tables": tables})


RTF_SKIP = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer", "headerl", "headerr",
            "footerl", "footerr", "listtable", "listoverridetable", "rsidtbl", "generator", "themedata",
            "colorschememapping", "latentstyles", "datastore", "xmlnstbl", "object", "fldinst"}


def rtf_to_text(s: str) -> str:
    """Small RTF control-word stripper (WordPad / writer.build_rtf level, not the full spec)."""
    m = re.search(r"\\ansicpg(\d+)", s)
    cp = f"cp{m.group(1)}" if m else "cp1252"
    out, stack, skip, uc, i, n = [], [], False, 1, 0, len(s)
    pending = bytearray()

    def flush():
        if pending:
            try:
                out.append(pending.decode(cp))
            except (UnicodeDecodeError, LookupError):
                out.append(pending.decode("cp1252", "replace"))
            pending.clear()

    while i < n:
        c = s[i]
        if c == "{":
            flush()
            stack.append((skip, uc))
            i += 1
            if s.startswith("\\*", i):
                skip = True
            continue
        if c == "}":
            flush()
            skip, uc = stack.pop() if stack else (False, 1)
            i += 1
            continue
        if c == "\\":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt in "\\{}":
                flush()
                if not skip:
                    out.append(nxt)
                i += 2
                continue
            if nxt == "'":
                if not skip:
                    try:
                        pending.append(int(s[i + 2:i + 4], 16))
                    except ValueError:
                        pass
                i += 4
                continue
            if nxt == "~":
                flush()
                out.append("" if skip else " ")
                i += 2
                continue
            m = re.match(r"\\([a-zA-Z]+)(-?\d+)? ?", s[i:])
            if not m:
                i += 2
                continue
            flush()
            word, arg = m.group(1), m.group(2)
            i += m.end()
            if word in RTF_SKIP:
                skip = True
            elif skip:
                pass
            elif word in ("par", "line", "row", "sect", "page"):
                out.append("\n")
            elif word == "tab":
                out.append("\t")
            elif word == "cell":
                out.append(" | ")
            elif word == "bullet":
                out.append("• ")
            elif word in ("emdash", "endash"):
                out.append("—" if word == "emdash" else "–")
            elif word == "uc" and arg:
                uc = int(arg)
            elif word == "u" and arg:
                v = int(arg)
                out.append(chr(v + 65536 if v < 0 else v))
                k = 0
                while k < uc and i < n:  # skip the ANSI replacement char(s); a \'xx escape is one char
                    if s.startswith("\\'", i):
                        i += 4
                    elif s[i] in "{}\\":
                        break
                    else:
                        i += 1
                    k += 1
            continue
        if c in "\r\n":
            i += 1
            continue
        flush()
        if not skip:
            out.append(c)
        i += 1
    flush()
    t = "".join(out)
    # \uN surrogate pairs (emoji) -> one character; a lone surrogate -> U+FFFD (UTF-8 can't store it)
    t = t.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    t = re.sub(r" \| \n", "\n", t)  # row end after the last cell
    t = "\n".join(ln.strip() for ln in t.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def load_rtf(path: Path, rcfg) -> Doc:
    raw = path.read_bytes().decode("latin-1")  # RTF is 7-bit; \'xx escapes are decoded with its code page
    if not raw.lstrip().startswith("{\\rtf"):
        return load_plain(path, rcfg)
    return Doc("text", [[None, None, rtf_to_text(raw)]], {})


def load_pdf(path: Path, rcfg) -> Doc:
    _need("pypdf", "pypdf", "pdf", rcfg)
    from pypdf import PdfReader
    r = PdfReader(str(path))
    if r.is_encrypted:
        try:
            ok = r.decrypt("")  # pypdf returns a status (0 = wrong password) instead of raising
        except Exception:
            ok = 0
        if not ok:
            raise ReadError(T(rcfg, "encrypted", name=path.name)) from None
    parts = []
    for i, page in enumerate(r.pages, 1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = "\n".join(ln.rstrip() for ln in _nl(t).split("\n"))
        parts.append(["page", str(i), re.sub(r"\n{3,}", "\n\n", t).strip()])
    chars = sum(len(p[2].strip()) for p in parts)
    if not parts or chars / len(parts) < rcfg["scanned_chars_per_page"]:
        raise ReadError(T(rcfg, "scanned_pdf", name=path.name))
    return Doc("text", parts, {"pages": len(parts),
                               "pages_with_text": sum(1 for p in parts if len(p[2].strip()) >= 20)})


LOADERS = {"csv": load_csv, "tsv": load_csv, "json": load_json, "yaml": load_yaml, "yml": load_yaml,
           "xml": load_xml, "html": load_html, "htm": load_html, "docx": load_docx, "pptx": load_pptx,
           "xlsx": load_xlsx, "xlsm": load_xlsx, "odt": load_odt, "rtf": load_rtf, "pdf": load_pdf}


def file_id(path: Path) -> str:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime}|v{LOADER_VERSION}"
    return "file_" + hashlib.sha1(key.encode()).hexdigest()[:12]


def cache_folder(path: Path, rcfg) -> Path:
    return config.path(rcfg["cache_dir"]) / file_id(path)


def load_text(path, rcfg, use_cache: bool = True) -> Doc:
    """Any supported file -> Doc. Raises ReadError with a message for the user."""
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext in OLD_FORMATS:
        raise ReadError(T(rcfg, "old_format", ext=ext, new=OLD_FORMATS[ext]))
    size = path.stat().st_size
    if size > rcfg["max_file_mb"] * 1024 * 1024:
        raise ReadError(T(rcfg, "file_too_big", name=path.name, mb=round(size / 1048576, 1), max=rcfg["max_file_mb"]))
    folder = cache_folder(path, rcfg) if use_cache else None
    cached = folder / "doc.json" if folder else None
    if cached and cached.is_file():
        try:
            d = json.loads(cached.read_text(encoding="utf-8"))
            return Doc(d["kind"], d["parts"], d["meta"])
        except (OSError, ValueError, KeyError):
            pass
    loader = LOADERS.get(ext)
    if loader is None and ext not in TEXT_EXTS and ext not in rcfg["extensions"]:
        # unknown extension: text if it has no NUL bytes in the first 8 KB, else binary
        try:
            with open(path, "rb") as fh:
                head = fh.read(8192)  # only the first 8 KB, not the whole file
        except OSError as e:
            raise ReadError(T(rcfg, "load_failed", name=path.name, error=e)) from None
        if looks_binary(head):
            raise ReadError(T(rcfg, "binary", name=path.name))
    try:
        doc = (loader or load_plain)(path, rcfg)
    except ReadError:
        raise
    except Exception as e:
        log("read_load_failed", path=str(path), error=repr(e))
        raise ReadError(T(rcfg, "load_failed", name=path.name, error=e)) from None
    if folder:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"kind": doc.kind, "parts": doc.parts, "meta": doc.meta,
                                          "source": str(path.resolve())}, ensure_ascii=False), encoding="utf-8")
            (folder / "text.txt").write_text(doc.text, encoding="utf-8")
        except (OSError, ValueError):  # ValueError: text the encoder can't write must not crash the app
            pass
    return doc


# ---------------- request ----------------

def search_dirs(rcfg) -> list[Path]:
    return [config.path(d) for d in rcfg["search_dirs"]] + [Path.cwd()]


def find_file(name: str, rcfg):
    """The file as given, else looked up in the search folders (documents, app folder, current dir)."""
    name = name.strip().strip("\"'")
    if not name:
        return None
    p = Path(name).expanduser()
    if p.is_file():
        return p.resolve()
    if not p.is_absolute():
        for d in search_dirs(rcfg):
            q = d / name
            if q.is_file():
                return q.resolve()
    return None


def _is_media(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS + MEDIA_EXTS)


def parse_range(s: str):
    """'pages 3-7 rest...' -> ((unit, spec), rest) or (None, s)."""
    m = re.match(r"\s*(" + "|".join(sorted(RANGE_UNITS, key=len, reverse=True)) + r")\.?\s+"
                 r"(\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*|\"[^\"]+\"|\S+)(?=\s|$)(.*)", s, re.I | re.S)
    if not m:
        return None, s
    unit = RANGE_UNITS[m.group(1).lower()]
    spec = m.group(2).strip('"')
    if unit != "sheet" and not re.fullmatch(r"[\d\s,-]+", spec):
        return None, s
    return (unit, spec), m.group(3).strip()


def _numbers_in(spec: str) -> set[int]:
    out = set()
    for piece in spec.split(","):
        a, _, b = piece.strip().partition("-")
        if a.strip().isdigit():
            lo = int(a)
            hi = int(b) if b.strip().isdigit() else lo
            out.update(range(min(lo, hi), max(lo, hi) + 1))
    return out


def parse_request(message: str, rcfg):
    """{'path', 'question', 'range'} for a read request, a str (message for the user) when
    '/read' can't be done, or None when the message is not about a file."""
    global LAST
    msg = message.strip()
    is_cmd = msg.lower() == "/read" or msg.lower().startswith("/read ")
    if is_cmd:
        body = msg[5:].strip()
        if not body:
            return T(rcfg, "usage")
        if re.match(r"(again|пак|отново)(?=\s|$)", body, re.I):  # not "again.txt"
            if not LAST:
                return T(rcfg, "nothing_yet") + "\n" + T(rcfg, "usage")
            q = re.sub(r"^(again|пак|отново)\s*", "", body, flags=re.I)
            rng, q = parse_range(q)
            if not Path(LAST["path"]).is_file():
                return T(rcfg, "not_found", path=LAST["path"], dirs=", ".join(rcfg["search_dirs"]))
            return {"path": Path(LAST["path"]), "question": q.strip(), "range": rng or LAST.get("range"), "again": True}
        path, rest = None, ""
        m = re.match(r'"([^"]+)"\s*(.*)', body, re.S)
        if m:
            path, rest = find_file(m.group(1), rcfg), m.group(2)
            shown = m.group(1)
        else:
            words = body.split(" ")
            shown = words[0]
            for i in range(len(words), 0, -1):  # unquoted paths with spaces: longest prefix that exists
                p = find_file(" ".join(words[:i]), rcfg)
                if p:
                    path, rest = p, " ".join(words[i:])
                    break
        if not path:
            return T(rcfg, "not_found", path=shown, dirs=", ".join(rcfg["search_dirs"]))
        if _is_media(path.name):
            return T(rcfg, "media", name=path.name)
        rng, q = parse_range(rest)
        return {"path": path, "question": q.strip(), "range": rng, "again": False}

    if msg.startswith("/"):
        return None
    # natural message: a token with a supported extension that EXISTS as a file
    exts = tuple("." + e.lower() for e in rcfg["extensions"])
    cands = [(q, q) for q in re.findall(r'"([^"]+)"', msg)]
    words = msg.split()
    for j, w in enumerate(words):
        if not w.strip("\"'.,;:!?()[]").lower().endswith(exts):
            continue
        for i in range(0, j + 1):  # longest span ending at this word first (unquoted paths with spaces)
            span = " ".join(words[i:j + 1])
            cands.append((span.strip("\"'([").rstrip("\"'.,;:!?)]"), span))
    for name, raw in cands:
        if not name.lower().endswith(exts) or _is_media(name):
            continue
        p = find_file(name, rcfg)
        if p:
            q = msg.replace(f'"{raw}"', " ") if f'"{raw}"' in msg else msg.replace(raw, " ")
            q = re.sub(r"\s+", " ", q).strip()
            rng, q2 = parse_range(q)
            return {"path": p, "question": q2 if rng else q, "range": rng, "again": False}
    return None


def apply_range(doc: Doc, rng, rcfg, name: str) -> Doc:
    if not rng:
        return doc
    unit, spec = rng
    if unit == "line":
        nums = _numbers_in(spec)
        lines = doc.text.split("\n")
        keep = [ln for i, ln in enumerate(lines, 1) if i in nums]
        if not keep:
            raise ReadError(T(rcfg, "range_empty", name=name, range=f"{unit} {spec}", have=f"{len(lines)} lines"))
        return Doc(doc.kind if doc.kind == "text" else "text", [[None, None, "\n".join(keep)]],
                   {**doc.meta, "range": f"lines {spec}"})
    have = [p for p in doc.parts if p[0] == unit]
    if unit == "sheet":
        sel = [p for p in have if p[1].lower() == spec.lower()]
    else:
        nums = _numbers_in(spec)
        sel = [p for p in have if p[1].isdigit() and int(p[1]) in nums]
    if not sel:
        avail = ", ".join(p[1] for p in have[:20]) if unit == "sheet" else f"{len(have)} {unit}s"
        raise ReadError(T(rcfg, "range_empty", name=name, range=f"{unit} {spec}", have=avail or f"no {unit}s"))
    return Doc(doc.kind, sel, {**doc.meta, "range": f"{unit}s {spec}" if unit != "sheet" else f"sheet {spec}"})


# ---------------- facts the program computes ----------------

NUM_RX = re.compile(r"[-+−]?\d{1,3}(?:[ \u00a0\u202f.,'’]\d{3})+(?:[.,]\d+)?|[-+−]?\d+(?:[.,]\d+)?")


def number_values(tok: str) -> set[float]:
    """'1 234,50' / '1,234.50' / '1234.5' -> {1234.5}; ambiguous '1,234' -> {1.234, 1234}."""
    t = tok.replace("−", "-").replace("\u00a0", " ").replace("\u202f", " ").replace("’", "'")
    t = re.sub(r"[ ']", "", t)
    out = set()
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        th = "." if dec == "," else ","
        cands = [t.replace(th, "").replace(dec, ".")]
    elif "," in t or "." in t:
        sep = "," if "," in t else "."
        pieces = t.split(sep)
        cands = []
        if len(pieces) == 2:
            cands.append(t.replace(sep, "."))  # decimal
        if all(len(p) == 3 for p in pieces[1:]):
            cands.append(t.replace(sep, ""))   # thousands
    else:
        cands = [t]
    for c in cands:
        try:
            out.add(round(float(c), 9))
        except ValueError:
            pass
    return out


def parse_number(s: str):
    s = s.strip()
    m = re.fullmatch(r"[^\d+\-−]{0,4}(" + NUM_RX.pattern + r")\s*[^\d]{0,5}", s)
    if not m:
        return None
    vals = number_values(m.group(1))
    if not vals:
        return None
    # ambiguous '1,234': a table cell is more often 1234 than 1.234
    return max(vals) if len(vals) > 1 else next(iter(vals))


def fmt_num(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.10f}".rstrip("0").rstrip(".")


def table_facts(label: str, body: str) -> list[str]:
    rows = list(csv.reader(io.StringIO(body)))
    if not rows:
        return []
    head, data = rows[0], rows[1:]
    where = f"{label}: " if label else ""
    out = [f"- {where}{len(data)} data rows + 1 header row, {len(head)} columns: " + " | ".join(h or "(empty)" for h in head)]
    for ci, h in enumerate(head):
        cells = [r[ci].strip() for r in data if ci < len(r) and r[ci].strip()]
        nums = [v for v in (parse_number(c) for c in cells) if v is not None]
        if nums and len(nums) >= 0.6 * len(cells):
            out.append(f"  - column \"{h or ci + 1}\": {len(nums)} numbers, min {fmt_num(min(nums))}, "
                       f"max {fmt_num(max(nums))}, sum {fmt_num(round(sum(nums), 9))}")
    return out


def size_text(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB" if n < 1048576 else f"{n / 1048576:.1f} MB"


def facts(doc: Doc, name: str, size_bytes: int) -> tuple[str, list[str]]:
    """(FILE header line, FACTS lines) - all counted by the program."""
    s = doc.text
    lines = s.count("\n") + 1 if s else 0
    words = len(s.split())
    bits = [doc.kind, size_text(size_bytes), f"{lines} lines", f"{words} words", f"{len(s)} characters"]
    if doc.meta.get("encoding"):
        bits.append(f"encoding {doc.meta['encoding']}")
    if doc.meta.get("range"):
        bits.append(f"only {doc.meta['range']}")
    head = f"FILE: {name}  (" + ", ".join(bits) + ")"
    out = []
    m = doc.meta
    if "pages" in m:
        out.append(f"- pages: {m['pages']} ({m.get('pages_with_text', m['pages'])} with text)"
                   + (f"; read: {len(doc.parts)}" if m.get("range") else ""))
    if "slides" in m:
        out.append(f"- slides: {m['slides']}" + (f"; read: {len(doc.parts)}" if m.get("range") else ""))
    if "paragraphs" in m:
        out.append(f"- paragraphs: {m['paragraphs']}, tables: {m.get('tables', 0)}")
    if doc.kind == "table":
        if len(doc.parts) > 1 or doc.parts and doc.parts[0][0] == "sheet":
            out.append(f"- sheets: {len(doc.parts)}: " + ", ".join(p[1] for p in doc.parts))
        for unit, pname, body in doc.parts:
            out += table_facts(f"sheet \"{pname}\"" if unit == "sheet" else "", body)
    return head, out


def found_items(question: str, full_text: str, rcfg) -> list[str]:
    """'extract all emails ...' -> the complete regex-found list from the WHOLE file."""
    out = []
    for name, p in rcfg["patterns"].items():
        if name.startswith("_") or not any(re.search(r"(?<!\w)" + re.escape(w), question, re.I) for w in p["words"]):
            continue
        items = []
        for m in re.finditer(p["regex"], full_text):
            v = m.group(0).strip().rstrip(".,;:")
            if p.get("exclude") and re.fullmatch(p["exclude"], v):
                continue
            if p.get("min_digits") and len(re.sub(r"\D", "", v)) < p["min_digits"]:
                continue
            if v not in items:
                items.append(v)
        cap = rcfg["max_found_items"]
        more = f" (first {cap} shown)" if len(items) > cap else ""
        out.append(f"PROGRAM-FOUND ITEMS ({name}, complete list from the whole file, {len(items)} found{more}): "
                   + ("; ".join(items[:cap]) if items else "none"))
    return out


# ---------------- splitting a big file ----------------

def split(doc: Doc, count, chunk_tokens: int, overlap_lines: int) -> list[dict]:
    """Doc -> [{'text', 'label'}], each <= chunk_tokens. Breaks at page/slide/sheet, then blank lines,
    then lines (a line longer than a whole chunk is the only thing ever cut). Text chunks repeat the last
    `overlap_lines` lines of the previous chunk of the same part; table chunks repeat the header row."""
    cost = lambda s: count(s) + 1
    recs = []  # (text, part_index, line_no, role)
    for pi, (unit, name, body) in enumerate(doc.parts):
        if unit:
            recs.append((MARKERS[unit].format(name=name), pi, 0, "marker"))
        for li, ln in enumerate(body.split("\n"), 1):
            if doc.kind == "table":
                recs.append((ln if li == 1 else f"[r{li}] {ln}", pi, li, "header" if li == 1 else "body"))
            else:
                recs.append((ln, pi, li, "body"))

    def marker(pi):
        u, n, _ = doc.parts[pi]
        return MARKERS[u].format(name=n) if u else None

    def header(pi):
        return doc.parts[pi][2].split("\n", 1)[0] if doc.kind == "table" else None

    # blocks: a marker starts one, a blank line ends one, a table row is one
    blocks, cur = [], []
    for r in recs:
        if cur and (r[3] == "marker" or r[1] != cur[-1][1] or doc.kind == "table"):
            blocks.append(cur)
            cur = []
        cur.append(r)
        if doc.kind != "table" and r[3] == "body" and not r[0].strip():
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    def pieces(block, pi):
        room = chunk_tokens - cost(marker(pi) or "") - cost(header(pi) or "") - 2
        if sum(cost(r[0]) for r in block) <= room:
            return [block]
        out = []
        for r in block:
            if cost(r[0]) <= room:
                out.append([r])
                continue
            s = r[0]  # one giant line (minified JSON...): cut it by size
            while s:
                take = max(1, int(len(s) * room / max(1, cost(s)) * 0.9))
                while take > 1 and cost(s[:take]) > room:
                    take = int(take * 0.8)
                out.append([(s[:take], r[1], r[2], r[3])])
                s = s[take:]
        return out

    chunks, cur, n = [], [], 0

    def flush():
        if any(r[3] in ("body", "header") and r[0].strip() for r in cur):
            chunks.append(list(cur))

    for block in blocks:
        for piece in pieces(block, block[0][1]):
            pn = sum(cost(r[0]) for r in piece)
            if cur and n + pn > chunk_tokens:
                flush()
                pi = piece[0][1]
                prefix = []
                if piece[0][3] != "marker" and marker(pi):
                    prefix.append((marker(pi)[:-4] + " (continued) ---" if not marker(pi).startswith("===")
                                   else marker(pi) + " (continued)", pi, 0, "marker"))
                if header(pi) is not None and piece[0][3] != "header":
                    prefix.append((header(pi), pi, 1, "header"))
                overlap = []
                if doc.kind != "table" and overlap_lines:
                    overlap = [r for r in cur if r[1] == pi and r[3] == "body"][-overlap_lines:]
                while overlap and sum(cost(r[0]) for r in prefix + overlap) + pn > chunk_tokens:
                    overlap.pop(0)
                cur = prefix + overlap
                n = sum(cost(r[0]) for r in cur)
            cur += piece
            n += pn
    flush()

    out = []
    for ch in chunks:
        out.append({"text": "\n".join(r[0] for r in ch).strip("\n"), "label": _label(doc, ch)})
    return out


def _label(doc, ch) -> str:
    # the repeated header row is left out of the range, else every chunk label starts at row 1
    body = ([r for r in ch if r[3] == "body" and r[0].strip()]
            or [r for r in ch if r[3] == "header" and r[0].strip()] or ch)
    pis = sorted({r[1] for r in body})
    unit = doc.parts[pis[0]][0]
    if unit in ("page", "slide"):
        names = [doc.parts[p][1] for p in pis]
        return f"{unit} {names[0]}" if len(names) == 1 else f"{unit}s {names[0]}-{names[-1]}"
    lines = [r[2] for r in body if r[1] == pis[0]]
    lo, hi = min(lines), max(lines)
    if unit == "sheet":
        where = doc.parts[pis[0]][1]
        return f"{where} rows {lo}-{hi}" if len(pis) == 1 else f"sheets {where}-{doc.parts[pis[-1]][1]}"
    if doc.kind == "table":
        return f"rows {lo}-{hi}"
    return f"lines {lo}-{hi}"


# ---------------- map / reduce ----------------

def is_none(note: str) -> bool:
    t = note.strip().strip(".*`").strip().upper()
    return not t or t == "NONE" or (t.startswith("NONE") and len(t) < 12)


def notes_file(folder: Path, question: str, rng, rcfg) -> Path:
    key = f"{question.strip().lower()}|{rng}|{rcfg['chunk_tokens']}|{rcfg['overlap_lines']}"
    return folder / f"notes_{hashlib.sha1(key.encode()).hexdigest()[:10]}.json"


def map_parts(chunks, question, name, llm, rcfg, cache: Path | None, progress, should_stop=lambda: False):
    """Ask the model for the relevant items of every part. Every finished part is cached at once, so
    Ctrl+C (or 'stop') ends the loop and a re-run continues where it stopped.
    Returns (notes {part number: text}, stopped early?)."""
    notes = {}
    if cache and cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("parts") == len(chunks):
                notes = {int(k): v for k, v in data["notes"].items()}
        except (OSError, ValueError, KeyError):
            notes = {}
    if notes:
        log("read_resume", cached_parts=sorted(notes), parts=len(chunks))
    stopped = False
    for i, ch in enumerate(chunks, 1):
        if i in notes:
            continue
        if should_stop():
            stopped = True
            break
        progress(T(rcfg, "progress", part=i, parts=len(chunks), location=ch["label"]))
        prompt = text(rcfg["prompts"]["extract_chunk"]).format(
            file=name, part=i, parts=len(chunks), location=ch["label"], question=question, text=ch["text"])
        t0 = time.monotonic()
        try:
            out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                             max_tokens=rcfg["notes_max_tokens"],
                                             temperature=rcfg["extract_temperature"])
        except KeyboardInterrupt:
            stopped = True
            break
        notes[i] = out["choices"][0]["message"]["content"].strip()
        log("read_part", part=i, parts=len(chunks), location=ch["label"], seconds=round(time.monotonic() - t0, 1),
            usage=out.get("usage"), notes=clip(notes[i]))
        if cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"question": question, "parts": len(chunks),
                                             "labels": [c["label"] for c in chunks],
                                             "notes": {str(k): v for k, v in sorted(notes.items())}},
                                            ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError:
                pass
    return notes, stopped


def join_notes(notes: dict, chunks) -> str:
    return "\n\n".join(f"Part {i} ({chunks[i - 1]['label']}):\n{v}" for i, v in sorted(notes.items()) if not is_none(v))


def reduce_notes(joined: str, question, name, llm, rcfg, count, progress) -> str:
    """Notes too long for the chat: merge them in groups (at most 2 levels), then cut what is left."""
    level = 0
    while count(joined) > rcfg["inline_tokens"] and level < 2:
        level += 1
        groups = split(Doc("text", [[None, None, joined]]), count, rcfg["chunk_tokens"], 0)
        merged = []
        try:
            for gi, g in enumerate(groups, 1):
                progress(T(rcfg, "merging", part=gi, parts=len(groups)))
                prompt = text(rcfg["prompts"]["merge_notes"]).format(file=name, question=question, text=g["text"])
                out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                                 max_tokens=rcfg["notes_max_tokens"],
                                                 temperature=rcfg["extract_temperature"])
                merged.append(out["choices"][0]["message"]["content"].strip())
        except KeyboardInterrupt:  # Ctrl+C while merging: keep the merged groups + the raw rest, cut below
            log("read_reduce_stopped", level=level, merged_groups=len(merged), groups=len(groups))
            merged += [g["text"] for g in groups[len(merged):]]
            joined = "\n\n".join(m for m in merged if not is_none(m))
            break
        joined = "\n\n".join(m for m in merged if not is_none(m))
        log("read_reduce_level", level=level, groups=len(groups), tokens=count(joined))
    if count(joined) > rcfg["inline_tokens"]:
        keep = int(len(joined) * rcfg["inline_tokens"] / count(joined) * 0.95)
        joined = joined[:keep].rsplit("\n", 1)[0] + "\n[... more notes cut to fit the context]"
    return joined


# ---------------- main entry ----------------

def prepare(req: dict, llm, cfg: dict, progress, confirm, should_stop=lambda: False, count=None) -> dict:
    """Read the file of a parse_request() result. Returns
    {'block': user message for the history, 'rule': extra system prompt for this answer,
     'evidence': text the answer is checked against, 'mode': inline|notes, ...}.
    Raises ReadError with a message for the user."""
    global LAST
    rcfg = cfg["reader"]
    path, rng = Path(req["path"]), req.get("range")
    question = (req.get("question") or "").strip() or rcfg["prompts"]["default_question"]
    count = count or (lambda s: len(llm.tokenize(s.encode("utf-8"), add_bos=False)))
    progress(T(rcfg, "loading", name=path.name))
    t0 = time.monotonic()
    full = load_text(path, rcfg)
    doc = apply_range(full, rng, rcfg, path.name)
    body = doc.text
    if not body.strip():
        raise ReadError(T(rcfg, "empty", name=path.name))
    LAST = {"path": str(path), "range": rng}
    size = path.stat().st_size
    head, fact_lines = facts(doc, path.name, size)
    found = found_items(question, body, rcfg)
    facts_block = "\n".join([head] + (["FACTS (counted by the program, exact):"] + fact_lines if fact_lines else []) + found)
    evidence = "\n".join([body, facts_block, question])
    tokens = count(body)
    log("read_loaded", path=str(path), kind=doc.kind, range=rng, chars=len(body), tokens=tokens,
        seconds=round(time.monotonic() - t0, 2), facts=facts_block)
    progress(None)

    answer_rule = text(rcfg["prompts"]["answer_rule"])
    if tokens + count(facts_block) <= rcfg["inline_tokens"]:  # ---- A. the whole file into the chat
        block = (f"{rcfg['texts']['history_prefix']}\n{facts_block}\n--- BEGIN FILE ---\n{body}\n--- END FILE ---\n\n"
                 f"{rcfg['texts']['question_prefix']} {question}")
        log("read_inline", path=str(path), tokens=tokens)
        return {"block": block, "rule": answer_rule, "evidence": evidence, "mode": "inline",
                "name": path.name, "tokens": tokens, "question": question}

    if len(body) > rcfg["max_chars"]:  # ---- C. too big
        parts = math.ceil(tokens / rcfg["chunk_tokens"])
        raise ReadError(T(rcfg, "too_big", name=path.name, tokens=tokens, parts=parts,
                          minutes=round(parts * rcfg["seconds_per_chunk"] / 60), max_chars=rcfg["max_chars"]))

    # ---- B. part by part
    chunks = split(doc, count, rcfg["chunk_tokens"], rcfg["overlap_lines"])
    cache = notes_file(cache_folder(path, rcfg), question, rng, rcfg)
    done_before = 0
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            done_before = len(data["notes"]) if data.get("parts") == len(chunks) else 0
        except (OSError, ValueError, KeyError):
            pass
    todo = len(chunks) - done_before
    minutes = math.ceil(todo * rcfg["seconds_per_chunk"] / 60)
    log("read_chunked", path=str(path), tokens=tokens, parts=len(chunks), cached_parts=done_before, minutes=minutes)
    if minutes > rcfg["confirm_above_minutes"]:
        ask = T(rcfg, "confirm", name=path.name, tokens=tokens, parts=len(chunks), minutes=minutes)
        if done_before:
            ask = T(rcfg, "resume", done=done_before, parts=len(chunks)) + " " + ask
        if not confirm(ask):
            raise ReadError(T(rcfg, "declined", name=path.name))
    t1 = time.monotonic()
    notes, stopped = map_parts(chunks, question, path.name, llm, rcfg, cache, progress, should_stop)
    if notes:
        n_new = len(notes) - done_before
        if n_new > 0:
            log("read_speed", seconds_per_part=round((time.monotonic() - t1) / n_new, 1), parts=n_new)
    if not notes:
        raise ReadError(T(rcfg, "stopped_nothing", name=path.name))
    joined = join_notes(notes, chunks) or rcfg["texts"]["no_notes"]
    joined = reduce_notes(joined, question, path.name, llm, rcfg, count, progress)
    progress(None)
    done = len(notes)
    partial = T(rcfg, "partial", done=done, parts=len(chunks)) if done < len(chunks) else ""
    block = (f"{rcfg['texts']['notes_prefix']}\n{facts_block}\n"
             f"NOTES (the file was read in {len(chunks)} parts; only items relevant to the question below, with locations):\n"
             f"--- BEGIN NOTES ---\n{joined}\n--- END NOTES ---\n" + (partial + "\n" if partial else "")
             + f"\n{rcfg['texts']['question_prefix']} {question}")
    rule = answer_rule + text(rcfg["prompts"]["big_file_rule"]) + (
        "\n" + text(rcfg["prompts"]["partial_rule"]).format(partial=partial) if partial else "")
    return {"block": block, "rule": rule, "evidence": evidence, "mode": "notes", "name": path.name,
            "tokens": tokens, "parts": len(chunks), "done": done, "stopped": stopped, "partial": partial,
            "question": question}


# ---------------- verbatim check ----------------

def _norm(s: str) -> str:
    return " ".join(s.translate(LOOKALIKE).lower().split())


def unverified(answer: str, evidence: str, rcfg) -> list[str]:
    """Numbers (3+ digits), emails, links and quoted texts in the answer that are nowhere in the file."""
    calc_cfg_skip = rcfg.get("skip_line_patterns", [])
    lines = [ln for ln in answer.split("\n") if not any(re.search(p, ln) for p in calc_cfg_skip)]
    ans = re.sub(r"\[[^\]\n]{1,40}\]", " ", "\n".join(lines))  # [p.3] / [lines 120-160] location tags
    ev_norm = _norm(evidence)
    ev_vals = set()
    for m in NUM_RX.finditer(evidence):
        ev_vals |= number_values(m.group(0))
    ev_digits = re.sub(r"(?<=\d)[ \u00a0\u202f.,'’/()-]+(?=\d)", "", evidence)
    bad = []

    def add(x):
        if x not in bad:
            bad.append(x)

    for m in re.finditer(r"https?://[^\s<>\"')\]]+", ans):
        u = m.group(0).rstrip(".,;:!?")
        if _norm(u).rstrip("/") not in ev_norm:
            add(u)
    ans_no_urls = re.sub(r"https?://\S+", " ", ans)
    for m in re.finditer(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", ans_no_urls):
        if _norm(m.group(0)) not in ev_norm:
            add(m.group(0))
    rest = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", ans_no_urls)
    for m in NUM_RX.finditer(rest):
        tok = m.group(0).strip()
        digits = re.sub(r"\D", "", tok)
        if len(digits) < 3:
            continue
        if number_values(tok) & ev_vals:
            continue
        if re.search(r"(?<!\d)" + digits + r"(?!\d)", ev_digits):
            continue
        add(tok.lstrip("+"))
    for m in re.finditer(r"\"([^\"\n]{4,80})\"|“([^”\n]{4,80})”|„([^“”\n]{4,80})[“”]|«([^»\n]{4,80})»", rest):
        q = next(g for g in m.groups() if g)
        if _norm(q) not in ev_norm:
            add(f"\"{q}\"")
    return bad


def verify(reply: str, evidence: str, rcfg) -> tuple[str, str | None]:
    """(reply + warning line, warning) when the answer has values that are not in the file."""
    if not rcfg["verify_answer"] or not reply:
        return reply, None
    bad = unverified(reply, evidence, rcfg)
    log("read_unverified", items=bad, ok=not bad)
    if not bad:
        return reply, None
    cap = rcfg["max_unverified_shown"]
    items = ", ".join(bad[:cap]) + (f" (+{len(bad) - cap} more)" if len(bad) > cap else "")
    warning = T(rcfg, "unverified", items=items)
    return f"{reply}\n\n{warning}", warning
