"""
/write <file.ext> [what to put in it] - turn what the AI researched in this chat into a file.

The model writes in an intermediate format that fits the file's family, the code builds
the real file and structures it for that type:
  document  (md txt html docx pdf rtf odt pptx)  model writes Markdown -> headings, lists, tables
  table     (csv tsv xlsx)                       model writes CSV      -> header row, typed cells
  data      (json yaml yml xml toml)             model writes JSON     -> validated, converted
  anything else (py js sql sh ini ...)           model writes the file's raw text
Prompts and settings: config.json -> "writer".
"""

import csv
import io
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path

from core import config
from core.config import text
from core.debuglog import clip, log

FAMILIES = {
    "document": {"md", "markdown", "txt", "html", "htm", "docx", "pdf", "rtf", "odt", "pptx"},
    "table": {"csv", "tsv", "xlsx", "xlsm"},
    "data": {"json", "yaml", "yml", "xml", "toml"},
}


def family(ext: str) -> str:
    for name, exts in FAMILIES.items():
        if ext in exts:
            return name
    return "raw"


ALL_EXTS = set().union(*FAMILIES.values())


def parse_request(rest: str, wcfg):
    """Returns (file name or None, ext, request).
    '/write report.docx short summary'          -> ('report.docx', 'docx', 'short summary')
    '/write all their information in txt file'  -> (None, 'txt', 'all their information ...')
    No name given -> the model names the file from its content."""
    rest = rest.strip()
    if not rest:
        return None
    first, _, what = rest.partition(" ")
    if re.fullmatch(r"[^\s]+\.[A-Za-z0-9]{1,5}", first) and "://" not in first:  # explicit file name (not a URL)
        return first, Path(first).suffix.lower().lstrip("."), what.strip()
    low = rest.lower()
    ext = None
    for phrase, e in wcfg["format_words"].items():                 # "excel", "word document", "slides"...
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", low):
            ext = e
            break
    if not ext:                                                     # "txt file", "as .pdf", "in json"
        known = ALL_EXTS | set(wcfg["format_hints"]) - {"_default"}
        m = re.search(r"(?<![\w.])\.?(" + "|".join(sorted(map(re.escape, known), key=len, reverse=True)) + r")(?!\w)", low)
        ext = m.group(1) if m else wcfg["default_ext"]
    return None, ext, rest


def target_path(name: str, wcfg):
    """Where to save `name`: inside output_dir, or an absolute path given on purpose.
    None for names that would escape output_dir ('../x.txt', 'C:x.txt', URLs)."""
    p = Path(name)
    if p.is_absolute():
        return p
    if ":" in name or ".." in p.parts or p.drive or p.root:
        return None
    base = config.path(wcfg["output_dir"]).resolve()
    target = (base / p).resolve()
    return target if target == base or base in target.parents else None


def slug(s: str, wcfg) -> str:
    """'Цени на RTX 4060 Ti (ноември)' -> 'цени_на_rtx_4060_ti_ноември' (safe on Windows, any language)."""
    s = re.sub(r"[*`#\"'“”„]", "", s).strip().lower()
    s = re.sub(r"\.[a-z0-9]{1,5}$", "", s)                         # model added an extension
    s = re.sub(r"[^\w]+", "_", s, flags=re.U).strip("_")
    return s[: wcfg["name_max_chars"]].rstrip("_")


def take_filename(raw: str, wcfg):
    """The model's first line is 'FILENAME: ...' -> (name, content without that line)."""
    lines = raw.lstrip().splitlines()
    for i, line in enumerate(lines[:3]):
        m = re.match(rf"^\W*{re.escape(wcfg['filename_label'])}\W*[:\-]\s*(.+)$", line.strip(), re.I)
        if m:
            return slug(m.group(1), wcfg), "\n".join(lines[:i] + lines[i + 1:]).strip()
    return "", raw


def fallback_name(content: str, fam: str, request: str, messages, wcfg) -> str:
    """Model gave no name: subject (document title / JSON root key / first question of the chat)
    + the words of the request, e.g. 'planche_in_calisthenics' + 'key_properties'."""
    stop = set(wcfg["name_stopwords"]) | {w for phrase in wcfg["format_words"] for w in phrase.split()}

    def words(s, n):
        return [w for w in re.findall(r"\w+", s.lower(), re.U) if w not in stop and w not in ALL_EXTS][:n]
    subject = ""
    if fam == "document":
        subject = plain(next((b[2] for b in md_blocks(strip_fences(content)) if b[0] == "h"), ""))
    elif fam == "data":
        try:
            data = parse_json(content)
            if isinstance(data, dict) and len(data) == 1:
                subject = next(iter(data))
        except ValueError:
            pass
    if not subject:
        subject = next((m["content"] for m in messages[1:] if m.get("role") == "user"), "")
    subj, req = words(subject.replace("_", " "), 4), words(request, 4)
    parts = subj + [w for w in req if w not in subj]
    return slug(" ".join(parts), wcfg) or "document"


def research_context(messages, wcfg, budget_tokens=None, count=None) -> str:
    """Everything said in this chat except the system prompt. If it does not fit the model's
    context, the newest messages are kept (and the oldest kept one is cut to the remaining room)."""
    parts = [f"[{m['role']}]\n{m['content']}" for m in messages[1:] if m.get("role") in ("user", "assistant")]
    count = count or (lambda s: len(s) // 3)
    budget = budget_tokens if budget_tokens is not None else wcfg["max_context_chars"] // 3
    out, used = [], 0
    for p in reversed(parts):
        n = max(1, count(p))
        if used + n > budget:
            room = budget - used
            if room > 200:  # keep the end of the oldest message that partly fits
                out.append("[...]\n" + p[-int(len(p) * room / n):])
            break
        out.append(p)
        used += n
    return "\n\n".join(reversed(out))


def strip_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```[\w+-]*\s*\n(.*?)\n?```\s*$", s, re.S)
    return m.group(1) if m else s


# ---------------- Markdown -> blocks (shared by all document builders) ----------------

def md_blocks(md: str):
    """Small Markdown parser: [('h', level, text) | ('p', text) | ('ul'|'ol', [items]) | ('table', rows) | ('code', text)]."""
    blocks, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            j, code = i + 1, []
            while j < len(lines) and not lines[j].startswith("```"):
                code.append(lines[j]); j += 1
            blocks.append(("code", "\n".join(code))); i = j + 1
        elif m := re.match(r"^(#{1,6})\s+(.*)", line):
            blocks.append(("h", len(m.group(1)), m.group(2).strip())); i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", line):
            kind = "ol" if re.match(r"^\s*\d", line) else "ul"
            items = []
            while i < len(lines) and (m := re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)", lines[i])):
                items.append(m.group(1).strip()); i += 1
            blocks.append((kind, items))
        else:
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|\s*([-*+]|\d+[.)])\s)", lines[i]):
                para.append(lines[i].strip()); i += 1
            blocks.append(("p", " ".join(para)))
    return blocks


def runs(s: str):
    """'a **b** *c*' -> [(text, bold, italic)]."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", s):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append((part[2:-2], True, False))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append((part[1:-1], False, True))
        elif part.startswith("`") and part.endswith("`"):
            out.append((part[1:-1], False, False))
        else:
            out.append((part, False, False))
    return out


def plain(s: str) -> str:
    return "".join(t for t, _, _ in runs(s))


def title_of(blocks, fallback):
    return next((b[2] for b in blocks if b[0] == "h"), fallback)


# ---------------- builders ----------------

def build_html(md, title):
    def inline(s):
        return "".join(f"<strong>{escape(t)}</strong>" if b else f"<em>{escape(t)}</em>" if it else escape(t)
                       for t, b, it in runs(s))
    body = []
    for b in md_blocks(md):
        if b[0] == "h":
            body.append(f"<h{b[1]}>{inline(b[2])}</h{b[1]}>")
        elif b[0] == "p":
            body.append(f"<p>{inline(b[1])}</p>")
        elif b[0] in ("ul", "ol"):
            body.append(f"<{b[0]}>" + "".join(f"<li>{inline(x)}</li>" for x in b[1]) + f"</{b[0]}>")
        elif b[0] == "code":
            body.append(f"<pre><code>{escape(b[1])}</code></pre>")
        elif b[0] == "table" and b[1]:
            head = "".join(f"<th>{inline(c)}</th>" for c in b[1][0])
            rows = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in b[1][1:])
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
    css = ("body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.55}"
           "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}"
           "th{background:#f3f3f3}pre{background:#f6f6f6;padding:.8rem;overflow:auto}")
    return (f'<!doctype html>\n<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{css}</style></head>\n<body>\n" + "\n".join(body) + "\n</body></html>\n")


def build_txt(md):
    out = []
    for b in md_blocks(md):
        if b[0] == "h":
            t = plain(b[2])
            out.append(f"{t}\n{('=' if b[1] == 1 else '-') * len(t)}" if b[1] <= 2 else t.upper())
        elif b[0] == "p":
            out.append(plain(b[1]))
        elif b[0] in ("ul", "ol"):
            out.append("\n".join(f"{'  -' if b[0] == 'ul' else f'  {n}.'} {plain(x)}" for n, x in enumerate(b[1], 1)))
        elif b[0] == "code":
            out.append(b[1])
        elif b[0] == "table" and b[1]:
            rows = [[plain(c) for c in r] for r in b[1]]
            w = [max(len(r[k]) if k < len(r) else 0 for r in rows) for k in range(max(map(len, rows)))]
            fmt = lambda r: "  ".join((r[k] if k < len(r) else "").ljust(w[k]) for k in range(len(w)))
            out.append("\n".join([fmt(rows[0]), "  ".join("-" * x for x in w)] + [fmt(r) for r in rows[1:]]))
    return "\n\n".join(out) + "\n"


def build_docx(md, path, title):
    from docx import Document
    doc = Document()
    doc.core_properties.title = title

    def add_runs(par, s):
        for t, b, it in runs(s):
            r = par.add_run(t); r.bold, r.italic = b, it
    first_h = True
    for b in md_blocks(md):
        if b[0] == "h":
            # the first # heading is the document title, later ones are normal headings
            add_runs(doc.add_heading(level=0 if first_h and b[1] == 1 else min(b[1], 4)), b[2])
            first_h = False
        elif b[0] == "p":
            add_runs(doc.add_paragraph(), b[1])
        elif b[0] in ("ul", "ol"):
            for x in b[1]:
                add_runs(doc.add_paragraph(style="List Bullet" if b[0] == "ul" else "List Number"), x)
        elif b[0] == "code":
            p = doc.add_paragraph(); r = p.add_run(b[1]); r.font.name = "Consolas"
        elif b[0] == "table" and b[1]:
            cols = max(map(len, b[1]))
            t = doc.add_table(rows=len(b[1]), cols=cols); t.style = "Table Grid"
            for ri, row in enumerate(b[1]):
                for ci, c in enumerate(row):
                    cell = t.cell(ri, ci); cell.text = ""
                    add_runs(cell.paragraphs[0], f"**{c}**" if ri == 0 and c else c)
    doc.save(path)


def build_pdf(md, path, title, wcfg):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_title(title)
    fonts = wcfg["pdf_fonts"]
    if Path(fonts["regular"]).exists():
        pdf.add_font("U", "", fonts["regular"]); pdf.add_font("U", "B", fonts["bold"])
        pdf.add_font("U", "I", fonts["italic"]); fam = "U"
    else:
        fam = "Helvetica"
    pdf.set_auto_page_break(True, 15); pdf.add_page()
    W = pdf.epw

    def write_md(s, size=11):
        pdf.set_font(fam, size=size)
        pdf.multi_cell(W, size * 0.5, s, markdown=True, new_x="LMARGIN", new_y="NEXT")
    for b in md_blocks(md):
        if b[0] == "h":
            pdf.ln(2); pdf.set_font(fam, "B", {1: 20, 2: 16, 3: 13}.get(b[1], 12))
            pdf.multi_cell(W, 9, plain(b[2]), new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
        elif b[0] == "p":
            write_md(b[1]); pdf.ln(2)
        elif b[0] in ("ul", "ol"):
            for n, x in enumerate(b[1], 1):
                write_md(("• " if b[0] == "ul" else f"{n}. ") + x)
            pdf.ln(2)
        elif b[0] == "code":
            pdf.set_font("Courier", size=9); pdf.multi_cell(W, 4.5, b[1], new_x="LMARGIN", new_y="NEXT"); pdf.ln(2)
        elif b[0] == "table" and b[1]:
            pdf.set_font(fam, size=9)
            cols = max(map(len, b[1]))
            with pdf.table(first_row_as_headings=True, line_height=5) as t:
                for row in b[1]:
                    r = t.row()
                    for c in row + [""] * (cols - len(row)):
                        r.cell(plain(c))
            pdf.ln(3)
    pdf.output(str(path))


def build_pptx(md, path, title):
    """One slide per ## section (# = title slide); lists become bullets, tables become slide tables."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    slides, cur = [], None
    for b in md_blocks(md):
        if b[0] == "h" and b[1] <= 2:
            cur = {"title": plain(b[2]), "level": b[1], "body": []}; slides.append(cur)
        else:
            if cur is None:
                cur = {"title": title, "level": 2, "body": []}; slides.append(cur)
            cur["body"].append(b)
    # a table gets its own slide, so the bullets next to it are not lost
    split = []
    for s in slides:
        tables = [x for x in s["body"] if x[0] == "table" and x[1]]
        rest = [x for x in s["body"] if not (x[0] == "table" and x[1])]
        if tables and rest:
            split.append({**s, "body": rest})
            split += [{**s, "level": 2, "body": [t]} for t in tables]
        else:
            split.append(s)
    slides = split
    for s in slides:
        if s["level"] == 1 and not any(x[0] != "p" for x in s["body"]):
            sl = prs.slides.add_slide(prs.slide_layouts[0])
            sl.shapes.title.text = s["title"]
            sl.placeholders[1].text = " ".join(plain(x[1]) for x in s["body"])[:300]
            continue
        tables = [x for x in s["body"] if x[0] == "table" and x[1]]
        sl = prs.slides.add_slide(prs.slide_layouts[5 if tables else 1])
        sl.shapes.title.text = s["title"]
        if tables:
            rows = tables[0][1]; cols = max(map(len, rows))
            shape = sl.shapes.add_table(len(rows), cols, Inches(0.5), Inches(1.6), Inches(9), Inches(0.4) * len(rows))
            for ri, row in enumerate(rows):
                for ci in range(cols):
                    cell = shape.table.cell(ri, ci); cell.text = plain(row[ci]) if ci < len(row) else ""
                    cell.text_frame.paragraphs[0].font.size = Pt(12)
            continue
        tf = sl.placeholders[1].text_frame; first = True
        for x in s["body"]:
            items = x[1] if x[0] in ("ul", "ol") else [x[1]] if x[0] in ("p", "code") else [plain(x[2])] if x[0] == "h" else []
            for it in items:
                p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
                p.text = plain(it); p.level = 1 if x[0] == "h" else 0
    prs.save(path)


def _rtf_u(c):
    """RTF unicode escape; characters beyond the BMP (emoji) become a UTF-16 surrogate pair."""
    b = c.encode("utf-16-le")
    units = [int.from_bytes(b[i:i + 2], "little") for i in range(0, len(b), 2)]
    return "".join(f"\\u{u if u < 32768 else u - 65536}?" for u in units)


def build_rtf(md):
    def esc(s):
        return "".join(c if ord(c) < 128 and c not in "\\{}" else f"\\{c}" if c in "\\{}" else _rtf_u(c) for c in s)

    def inline(s):
        return "".join("{\\b " + esc(t) + "}" if b else "{\\i " + esc(t) + "}" if it else esc(t) for t, b, it in runs(s))
    out = ["{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Arial;}{\\f1 Consolas;}}\\fs22"]
    for b in md_blocks(md):
        if b[0] == "h":
            out.append(f"\\pard\\sb200\\sa100{{\\b\\fs{ {1: 36, 2: 30, 3: 26}.get(b[1], 24) } {inline(b[2])}}}\\par")
        elif b[0] == "p":
            out.append(f"\\pard\\sa120 {inline(b[1])}\\par")
        elif b[0] in ("ul", "ol"):
            out += ["\\pard\\li360\\sa60 " + ("\\bullet" if b[0] == "ul" else f"{n}.") + f" {inline(x)}\\par"
                    for n, x in enumerate(b[1], 1)]
        elif b[0] == "code":
            out.append("\\pard{\\f1\\fs18 " + esc(b[1]).replace("\n", "\\line ") + "}\\par")
        elif b[0] == "table":
            for ri, row in enumerate(b[1]):
                cells = "".join(f"\\clbrdrt\\brdrs\\clbrdrb\\brdrs\\clbrdrl\\brdrs\\clbrdrr\\brdrs\\cellx{(k + 1) * 2000}" for k in range(len(row)))
                out.append("\\trowd" + cells + "".join("\\pard\\intbl " + ("{\\b " if ri == 0 else "{") + inline(c) + "}\\cell" for c in row) + "\\row")
    return "\n".join(out) + "\n}"


def build_odt(md, path, title):
    """Minimal valid OpenDocument text, no extra library."""
    import zipfile
    from xml.sax.saxutils import escape as x
    body = []
    for b in md_blocks(md):
        if b[0] == "h":
            body.append(f'<text:h text:outline-level="{b[1]}">{x(plain(b[2]))}</text:h>')
        elif b[0] in ("p", "code"):
            body.append(f"<text:p>{x(plain(b[1]) if b[0] == 'p' else b[1])}</text:p>")
        elif b[0] in ("ul", "ol"):
            body.append("<text:list>" + "".join(f"<text:list-item><text:p>{x(plain(i))}</text:p></text:list-item>" for i in b[1]) + "</text:list>")
        elif b[0] == "table" and b[1]:
            cols = max(map(len, b[1]))
            body.append(f'<table:table><table:table-column table:number-columns-repeated="{cols}"/>' + "".join(
                "<table:table-row>" + "".join(f"<table:table-cell><text:p>{x(plain(c))}</text:p></table:table-cell>" for c in r)
                + "</table:table-row>" for r in b[1]) + "</table:table>")
    content = ('<?xml version="1.0" encoding="UTF-8"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
               'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
               'office:version="1.2"><office:body><office:text>' + "".join(body) + "</office:text></office:body></office:document-content>")
    manifest = ('<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">'
                '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>'
                '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/></manifest:manifest>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/vnd.oasis.opendocument.text")
        z.writestr("content.xml", content, zipfile.ZIP_DEFLATED)
        z.writestr("META-INF/manifest.xml", manifest, zipfile.ZIP_DEFLATED)


def parse_table(raw: str):
    """CSV from the model (or a Markdown table if it ignored the rule) -> rows."""
    raw = strip_fences(raw)
    md_tables = [b[1] for b in md_blocks(raw) if b[0] == "table" and b[1]]
    if md_tables:
        return [[plain(c) for c in r] for r in md_tables[0]]
    lines = [l for l in raw.splitlines() if l.strip()]
    delim = "\t" if lines and lines[0].count("\t") > lines[0].count(",") else ";" if lines and lines[0].count(";") > lines[0].count(",") else ","
    return [r for r in csv.reader(lines, delimiter=delim) if any(c.strip() for c in r)]


def typed(v: str):
    s = v.strip()
    n = re.sub(r"(?<=\d)[  ](?=\d{3}\b)", "", s)
    if re.fullmatch(r"-?0\d+", n):                  # "007", "0888 123": keep leading zeros
        return s
    if re.fullmatch(r"-?\d+", n):
        return int(n)
    if re.fullmatch(r"-?\d+\.\d+", n) or re.fullmatch(r"-?\d+,\d{1,2}", n):  # "1,234" is a thousands separator
        return float(n.replace(",", "."))
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", n):   # "1,234" / "1,234,567"
        return int(n.replace(",", ""))
    return s


def build_xlsx(rows, path, title):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook(); ws = wb.active; ws.title = re.sub(r"[\\/*?:\[\]]", "", title)[:31] or "Sheet1"
    for ri, row in enumerate(rows):
        ws.append([c if ri == 0 else typed(c) for c in row])
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDDDDD")
    ws.freeze_panes = "A2"
    if rows:
        ws.auto_filter.ref = ws.dimensions
    for k in range(1, ws.max_column + 1):
        width = max(len(str(c.value or "")) for c in ws[get_column_letter(k)])
        ws.column_dimensions[get_column_letter(k)].width = min(max(10, width + 2), 60)
    wb.save(path)


def to_xml(obj, tag="root", indent=0):
    from xml.sax.saxutils import escape as x
    tag = re.sub(r"[^\w.-]", "_", str(tag)) or "item"
    if tag[0].isdigit() or tag[0] in ".-":
        tag = "_" + tag
    pad = "  " * indent
    if isinstance(obj, dict):
        return f"{pad}<{tag}>\n" + "".join(to_xml(v, k, indent + 1) for k, v in obj.items()) + f"{pad}</{tag}>\n"
    if isinstance(obj, list):
        item = (tag[:-3] + "y" if tag.endswith("ies") else tag[:-2] if tag.endswith(("sses", "xes", "ches", "shes"))
                else tag[:-1] if tag.endswith("s") and not tag.endswith("ss") and len(tag) > 2 else "item")
        return f"{pad}<{tag}>\n" + "".join(to_xml(v, item, indent + 1) for v in obj) + f"{pad}</{tag}>\n"
    return f"{pad}<{tag}>{x('' if obj is None else str(obj).lower() if isinstance(obj, bool) else str(obj))}</{tag}>\n"


def to_toml(obj, prefix=""):
    def val(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, (int, float)): return str(v)
        if isinstance(v, list): return "[" + ", ".join(val(i) for i in v) + "]"
        if isinstance(v, dict): return "{" + ", ".join(f"{key(str(k))} = {val(x)}" for k, x in v.items()) + "}"
        return json.dumps("" if v is None else str(v), ensure_ascii=False)
    key = lambda k: k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k, ensure_ascii=False)
    lines, tables = [], []
    for k, v in obj.items():
        if isinstance(v, dict):
            tables.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            for i in v:
                tables.append((k, i, True))
        else:
            lines.append(f"{key(k)} = {val(v)}")
    for t in tables:
        name = f"{prefix}{key(t[0])}"
        lines.append(f"\n[[{name}]]" if len(t) == 3 else f"\n[{name}]")
        lines.append(to_toml(t[1], name + ".").strip())
    return "\n".join(lines) + "\n"


def parse_json(raw: str):
    """JSON from the model (fences / text around it tolerated) -> value, or None if invalid."""
    s = strip_fences(raw)
    for candidate in (s, (re.search(r"[\[{].*[\]}]", s, re.S) or [None])[0]):
        if candidate:
            try:
                return json.loads(candidate)
            except ValueError:
                pass
    return None


def usable(raw: str, fam: str) -> bool:
    """Is the model's output something we can build the file from?"""
    if not raw or len(raw.strip()) < 2:
        return False
    if fam == "data":
        return parse_json(raw) is not None
    if fam == "table":
        rows = parse_table(raw)
        return len(rows) >= 2 or (len(rows) == 1 and len(rows[0]) > 1)
    return True


# ---------------- main entry ----------------

def write(llm, messages, rest, cfg, generate, spin):
    wcfg = cfg["writer"]
    req = parse_request(rest, wcfg)
    if not req:
        return wcfg["texts"]["usage"]
    name, ext, what = req
    if name and target_path(name, wcfg) is None:
        return wcfg["texts"]["failed"].format(error=f"unsafe file name '{name}'")
    fam = family(ext)
    count = (lambda s: len(llm.tokenize(s.encode("utf-8"), add_bos=False))) if llm else None
    budget = (llm.n_ctx() - wcfg["answer_room_tokens"] - wcfg["prompt_overhead_tokens"]) if llm else None
    research = research_context(messages, wcfg, budget, count)
    if not research.strip():
        return wcfg["texts"]["nothing"]

    hint = wcfg["format_hints"].get(ext, wcfg["format_hints"]["_default"]).format(ext=ext)
    prompt = text(wcfg["prompts"][fam]).format(
        ext=ext, hint=hint, request=what or wcfg["texts"]["default_request"], research=research)
    if not name:  # let the model name its own work (first line of its answer); last = followed best
        prompt += "\n\n" + text(wcfg["prompts"]["naming_rule"]).format(label=wcfg["filename_label"])
    msgs = [{"role": "system", "content": text(wcfg["prompts"]["system"])}, {"role": "user", "content": prompt}]
    log("write_request", file=name, ext=ext, family=fam, request=what, context_chars=len(research))
    print(wcfg["texts"]["writing"].format(file=name or f"(.{ext})"))
    gen = {**config.params(cfg["generation"]), "temperature": wcfg["temperature"]}
    raw, _ = generate(llm, msgs, gen)
    spin.stop()
    given, raw = take_filename(raw or "", wcfg)
    if not usable(raw, fam):
        # small models sometimes stop right after the FILENAME line, or break the JSON/CSV:
        # ask once more, without the naming rule (the name is already known)
        log("write_retry", family=fam, output=clip(raw))
        print(wcfg["texts"]["retry"])
        retry = text(wcfg["prompts"][fam]).format(
            ext=ext, hint=hint, request=what or wcfg["texts"]["default_request"], research=research)
        msgs[1] = {"role": "user", "content": retry + "\n\n" + text(wcfg["prompts"]["retry_rule"])}
        raw2, _ = generate(llm, msgs, gen)
        spin.stop()
        g2, raw2 = take_filename(raw2 or "", wcfg)
        given = given or g2
        if usable(raw2, fam):
            raw = raw2
    if not raw.strip():
        return wcfg["texts"]["empty"]

    if not name:
        stem = given if given and given not in wcfg["name_stopwords"] else fallback_name(raw, fam, what, messages, wcfg)
        name = f"{stem}.{ext}"
        log("write_named", name=name, by_model=bool(given))
    target = target_path(name, wcfg)
    if target is None:
        log("write_bad_name", name=name)
        return wcfg["texts"]["failed"].format(error=f"unsafe file name '{name}'")

    title = Path(name).stem.replace("_", " ").replace("-", " ").strip().capitalize()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{target.stem}_{datetime.now():%Y%m%d_%H%M%S}{target.suffix}")
        if fam == "document":
            md = strip_fences(raw) if ext not in ("md", "markdown") or raw.strip().startswith("```") else raw.strip()
            title = plain(title_of(md_blocks(md), title))
            if ext in ("md", "markdown"):
                target.write_text(md + "\n", encoding="utf-8")
            elif ext == "txt":
                target.write_text(build_txt(md), encoding="utf-8")
            elif ext in ("html", "htm"):
                target.write_text(build_html(md, title), encoding="utf-8")
            elif ext == "docx":
                build_docx(md, target, title)
            elif ext == "pdf":
                build_pdf(md, target, title, wcfg)
            elif ext == "pptx":
                build_pptx(md, target, title)
            elif ext == "rtf":
                target.write_text(build_rtf(md), encoding="ascii")
            elif ext == "odt":
                build_odt(md, target, title)
        elif fam == "table":
            rows = parse_table(raw)
            if not rows:
                return wcfg["texts"]["bad_output"].format(kind="table")
            if ext in ("xlsx", "xlsm"):
                build_xlsx(rows, target, title)
            else:
                buf = io.StringIO()
                csv.writer(buf, delimiter="\t" if ext == "tsv" else ",", lineterminator="\n").writerows(rows)
                # utf-8-sig: Excel opens Cyrillic etc. correctly
                target.write_text(buf.getvalue(), encoding="utf-8-sig" if ext == "csv" else "utf-8")
        elif fam == "data":
            data = parse_json(raw)
            if data is None:
                return wcfg["texts"]["bad_output"].format(kind="JSON")
            if ext == "json":
                out = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            elif ext in ("yaml", "yml"):
                import yaml
                out = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
            elif ext == "xml":
                root = next(iter(data)) if isinstance(data, dict) and len(data) == 1 else "root"
                out = '<?xml version="1.0" encoding="UTF-8"?>\n' + to_xml(data[root] if root != "root" else data, root)
            else:
                out = to_toml(data if isinstance(data, dict) else {"items": data})
            target.write_text(out, encoding="utf-8")
        else:
            target.write_text(strip_fences(raw) + "\n", encoding="utf-8")
    except ImportError as e:
        return wcfg["texts"]["missing_lib"].format(lib=e.name, ext=ext)
    except Exception as e:  # never crash the chat because a file could not be built
        log("write_failed", file=str(target), error=repr(e), output=clip(raw))
        return wcfg["texts"]["failed"].format(error=e)

    log("write_done", file=str(target), bytes=target.stat().st_size)
    return wcfg["texts"]["saved"].format(file=target, kb=round(target.stat().st_size / 1024, 1))
