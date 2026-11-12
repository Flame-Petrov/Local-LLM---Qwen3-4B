"""
Offline tests for reader.py (no model: token count = len // 3, the model is a fake). Plain asserts - run:

    .venv\\Scripts\\python.exe tests\\test_reader.py

Test files are generated in a temp folder.
"""

import copy
import csv
import io
import json
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import config  # noqa: E402
from files import reader  # noqa: E402
from files import writer  # noqa: E402

CFG = config.load()
COUNT = lambda s: len(s) // 3  # noqa: E731
TMP = Path(tempfile.mkdtemp(prefix="reader_test_"))


def rcfg(**over):
    r = copy.deepcopy(CFG["reader"])
    r["cache_dir"] = str(TMP / "cache")
    r["search_dirs"] = [str(TMP / "docs")]
    r.update(over)
    return r


R = rcfg()
(TMP / "docs").mkdir()


def f(name, data):
    p = TMP / name
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.write_bytes if isinstance(data, bytes) else lambda d: p.write_text(d, encoding="utf-8"))(data)
    return p


def load(p, r=None):
    return reader.load_text(p, r or R, use_cache=False)


def raises(fn, key, **values):
    try:
        fn()
    except reader.ReadError as e:
        want = R["texts"][key].split("{")[0]
        assert want in str(e), (str(e), want)
        return str(e)
    raise AssertionError(f"expected ReadError {key}")


# ---------------- 1. loaders ----------------

def test_txt_encodings():
    bg = "Планш: задържане 30 секунди.\nВтори ред."
    assert load(f("u8.txt", bg.encode("utf-8"))).text == bg
    d = load(f("bom.txt", b"\xef\xbb\xbf" + bg.encode("utf-8")))
    assert d.text == bg and d.meta["encoding"] == "utf-8-sig"
    d = load(f("w.txt", bg.replace("\n", "\r\n").encode("cp1251")))
    assert d.text == bg, d.text
    assert d.meta["encoding"] == "cp1251"
    d = load(f("short.txt", "Цена: 12,50 лв.".encode("cp1251")))  # charset_normalizer alone says big5
    assert d.text == "Цена: 12,50 лв.", d.text


def test_csv_semicolon_and_facts():
    p = f("data.csv", "name;city;amount\nIvan;Sofia;1 234,50\nMaria;Plovdiv;100\nPetar;Sofia;15,5\n")
    d = load(p)
    assert d.kind == "table" and d.meta["delimiter"] == ";"
    rows = list(csv.reader(io.StringIO(d.text)))
    assert rows[0] == ["name", "city", "amount"] and rows[1] == ["Ivan", "Sofia", "1 234,50"]
    head, facts = reader.facts(d, p.name, p.stat().st_size)
    joined = "\n".join(facts)
    assert "3 data rows + 1 header row, 3 columns: name | city | amount" in joined, joined
    assert 'column "amount": 3 numbers, min 15.5, max 1234.5, sum 1350' in joined, joined
    assert head.startswith("FILE: data.csv  (table,")


def test_json_list_to_table_and_nested():
    d = load(f("p.json", json.dumps([{"a": 1, "b": "x"}, {"a": 2.0, "c": True}])))
    assert d.kind == "table"
    assert d.text.split("\n") == ["a,b,c", "1,x,", "2,,true"], d.text
    d = load(f("n.json", '{"x":{"y":[1,2]}}'))
    assert d.kind == "text" and '"y": [' in d.text
    d = load(f("bad.json", "{not json"))
    assert d.text == "{not json"


def test_yaml_xml():
    d = load(f("t.yaml", "- name: A\n  n: 1\n- name: B\n  n: 2\n"))
    assert d.kind == "table" and d.text.startswith("name,n")
    d = load(f("c.yml", "server:\n  port: 80\n"))
    assert d.kind == "text" and "port: 80" in d.text
    d = load(f("x.xml", "<a>\n   <b>hi</b>\n\n   <c>there</c>\n</a>"))
    assert d.text == "<a>\n<b>hi</b>\n<c>there</c>\n</a>"


def test_docx_with_table():
    md = "# Title\n\nFirst paragraph with 4750.00 EUR.\n\n| Name | Total |\n|---|---|\n| Ivan | 120 |\n"
    p = TMP / "d.docx"
    writer.build_docx(md, p, "Title")
    d = load(p)
    assert "First paragraph with 4750.00 EUR." in d.text, d.text
    assert "Name | Total" in d.text and "Ivan | 120" in d.text, d.text
    assert d.meta["tables"] == 1


def test_pptx_with_notes():
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Planche"
    s.placeholders[1].text = "Hold 30 s"
    s.notes_slide.notes_text_frame.text = "Say it slowly"
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Table"
    t = s2.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(1)).table
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "a", "b", "1", "2"
    p = TMP / "s.pptx"
    prs.save(p)
    d = load(p)
    assert d.meta["slides"] == 2
    assert d.text.startswith("--- Slide 1 ---\n# Planche\nHold 30 s\nSpeaker notes: Say it slowly"), d.text
    assert "--- Slide 2 ---\n# Table\na | b\n1 | 2" in d.text, d.text


def test_xlsx_two_sheets_trailing_empty():
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    for r in (["Item", "Qty", "Price"], ["Bar", 3, 10.5], ["Rings", 1, 40]):
        ws.append(r)
    ws["E9"] = " "   # whitespace far away: an empty trailing row + columns
    ws2 = wb.create_sheet("Other")
    ws2.append(["x"])
    ws2.append([1])
    wb.create_sheet("Empty")
    p = TMP / "w.xlsx"
    wb.save(p)
    d = load(p)
    assert [x[1] for x in d.parts] == ["Sales", "Other"], d.parts
    assert d.parts[0][2] == "Item,Qty,Price\nBar,3,10.5\nRings,1,40", repr(d.parts[0][2])
    assert d.text.startswith("=== Sheet: Sales ===\nItem,Qty,Price")
    _, facts = reader.facts(d, p.name, p.stat().st_size)
    joined = "\n".join(facts)
    assert "sheets: 2: Sales, Other" in joined
    assert 'sheet "Sales": 2 data rows + 1 header row, 3 columns' in joined
    assert 'column "Price": 2 numbers, min 10.5, max 40, sum 50.5' in joined, joined


def test_odt_rtf_roundtrip():
    md = "# Заглавие\n\nCena 1 234,50 lv.\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    p = TMP / "r.odt"
    writer.build_odt(md, p, "t")
    d = load(p)
    assert "# Заглавие" in d.text and "Cena 1 234,50 lv." in d.text and "one" in d.text, d.text
    assert "1 | 2" in d.text
    p = f("r.rtf", writer.build_rtf(md))
    d = load(p)
    assert "Заглавие" in d.text and "Cena 1 234,50 lv." in d.text and "• one" in d.text, d.text
    assert "1 | 2" in d.text, d.text
    # WordPad style: code page + \'xx bytes
    wp = r"{\rtf1\ansi\ansicpg1251\deff0{\fonttbl{\f0 Arial;}}{\*\generator Riched20;}\pard \'cf\'eb\'e0\'ed\'f8\par second\par}"
    assert load(f("wp.rtf", wp)).text == "Планш\nsecond"


def test_html():
    html = "<html><head><style>x{}</style><script>var a=1</script></head><body><p>Hello <b>world</b></p><p>Total: 99 лв.</p></body></html>"
    d = load(f("p.html", html))
    assert "Hello world" in d.text and "Total: 99 лв." in d.text and "var a" not in d.text, d.text


# ---------------- 2./3. pdf, binary, limits ----------------

def test_pdf_text_scanned_missing_lib():
    from fpdf import FPDF
    from pypdf import PdfWriter
    pdf = FPDF()
    for t in ("Invoice 12345 total 1234.50 EUR - first page text", "Second page: invoice 777 total 99.90"):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(text=t)
    p = TMP / "t.pdf"
    pdf.output(str(p))
    d = load(p)
    assert d.meta["pages"] == 2 and d.meta["pages_with_text"] == 2
    assert d.text.startswith("--- Page 1 ---\nInvoice 12345") and "--- Page 2 ---\nSecond page" in d.text, d.text
    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.add_blank_page(200, 200)
    with open(TMP / "scan.pdf", "wb") as fh:
        w.write(fh)
    raises(lambda: load(TMP / "scan.pdf"), "scanned_pdf")
    saved = sys.modules.get("pypdf")
    sys.modules["pypdf"] = None  # import pypdf -> ImportError
    try:
        msg = raises(lambda: load(p), "missing_lib")
        assert "pip install pypdf" in msg
    finally:
        sys.modules["pypdf"] = saved


def test_binary_old_formats_size():
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(100)
    raises(lambda: load(f("img.txt", png)), "binary")
    raises(lambda: load(f("blob.xyz", png)), "binary")
    assert load(f("notes.xyz", "plain words")).text == "plain words"  # unknown extension, but text
    raises(lambda: load(f("old.doc", b"x")), "old_format")
    raises(lambda: load(f("big.txt", "x" * 2048), rcfg(max_file_mb=0.001)), "file_too_big")


def test_cache():
    p = f("cached.txt", "first version")
    assert reader.load_text(p, R).text == "first version"
    assert (reader.cache_folder(p, R) / "text.txt").read_text(encoding="utf-8") == "first version"
    import os
    import time
    p.write_text("second version, longer", encoding="utf-8")
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert reader.load_text(p, R).text == "second version, longer"  # new size/mtime -> new id


# ---------------- 4. parse_request ----------------

def test_parse_request():
    sp = f("my notes.txt", "hello")
    f("docs/found_here.txt", "in docs")
    q = reader.parse_request(f'/read "{sp}" list all dates', R)
    assert q["path"] == sp.resolve() and q["question"] == "list all dates", q
    q = reader.parse_request(f"/read {sp} what is it", R)  # unquoted, with a space
    assert q["path"] == sp.resolve() and q["question"] == "what is it", q
    q = reader.parse_request("/read found_here.txt", R)  # bare name -> search_dirs
    assert q["path"] == (TMP / "docs" / "found_here.txt").resolve() and q["question"] == ""
    q = reader.parse_request("summarise found_here.txt please", R)  # natural message
    assert q and q["path"].name == "found_here.txt" and q["question"] == "summarise please", q
    q = reader.parse_request(f"extract all emails from {sp}", R)  # natural, unquoted with space
    assert q and q["path"] == sp.resolve() and q["question"] == "extract all emails from", q
    assert reader.parse_request("what is a .txt file?", R) is None
    assert reader.parse_request("write missing_file.txt with a poem", R) is None
    assert reader.parse_request("/write found_here.txt", R) is None
    f("clip.mp4", b"\x00\x00")
    f("pic.jpg", b"\xff\xd8")
    assert reader.parse_request(f"look at {TMP / 'clip.mp4'}", R) is None
    assert reader.parse_request(f"look at {TMP / 'pic.jpg'}", R) is None
    assert "image/video" in reader.parse_request(f"/read {TMP / 'pic.jpg'}", R)
    assert "not found" in reader.parse_request("/read nope.txt x", R).lower()
    assert reader.parse_request("/read", R).startswith("usage")
    # ranges
    pdf = f("rep.pdf", b"%PDF")
    q = reader.parse_request(f"/read {pdf} pages 3-7 what are the totals", R)
    assert q["range"] == ("page", "3-7") and q["question"] == "what are the totals", q
    q = reader.parse_request(f"/read {pdf} sheet Sales sum", R)
    assert q["range"] == ("sheet", "Sales") and q["question"] == "sum"
    q = reader.parse_request(f"/read {pdf} редове 10-20", R)
    assert q["range"] == ("line", "10-20") and q["question"] == ""
    q = reader.parse_request(f"/read {pdf} pages are what", R)  # not a range
    assert q["range"] is None and q["question"] == "pages are what"


def test_read_again():
    reader.LAST = None
    assert "Nothing has been read" in reader.parse_request("/read again x", R)
    reader.LAST = {"path": str(f("again.txt", "abc")), "range": ("line", "1-2")}
    q = reader.parse_request("/read again new question", R)
    assert q["again"] and q["question"] == "new question" and q["range"] == ("line", "1-2")
    reader.LAST = None


def test_apply_range():
    doc = reader.Doc("text", [["page", str(i), f"text of page {i}"] for i in range(1, 11)], {"pages": 10})
    d = reader.apply_range(doc, ("page", "3-4,9"), R, "x.pdf")
    assert [p[1] for p in d.parts] == ["3", "4", "9"]
    raises(lambda: reader.apply_range(doc, ("page", "50"), R, "x.pdf"), "range_empty")
    t = reader.Doc("text", [[None, None, "\n".join(f"line {i}" for i in range(1, 101))]])
    assert reader.apply_range(t, ("line", "10-12"), R, "x").text == "line 10\nline 11\nline 12"
    s = reader.Doc("table", [["sheet", "Sales", "a\n1"], ["sheet", "Other", "b\n2"]])
    assert reader.apply_range(s, ("sheet", "sales"), R, "x").parts == [["sheet", "Sales", "a\n1"]]


# ---------------- 5. splitter ----------------

def test_split_text_pages():
    pages = []
    for p in range(1, 6):
        body = "\n\n".join("\n".join(f"p{p} para {k} line {j} " + "word " * 8 for j in range(4)) for k in range(6))
        pages.append(["page", str(p), body])
    doc = reader.Doc("text", pages, {"pages": 5})
    orig_lines = set(doc.text.split("\n"))
    chunks = reader.split(doc, COUNT, 300, 3)
    assert len(chunks) > 5
    for c in chunks:
        assert COUNT(c["text"]) <= 300, (COUNT(c["text"]), c["label"])
        for ln in c["text"].split("\n"):
            assert ln in orig_lines or ln.endswith("(continued) ---"), ln  # never a cut line
        assert c["text"].startswith("--- Page "), c["text"][:40]  # every part knows its page
        assert c["label"].startswith("page"), c["label"]
    # overlap: a chunk that continues a page starts with the last lines of the previous chunk
    for a, b in zip(chunks, chunks[1:]):
        if "(continued)" in b["text"].split("\n")[0]:
            prev_tail = [ln for ln in a["text"].split("\n") if ln.strip()][-1]
            assert prev_tail in b["text"].split("\n")[1:5], (prev_tail, b["text"][:200])
    # every line of the file is in some chunk
    seen = set(ln for c in chunks for ln in c["text"].split("\n"))
    assert orig_lines <= seen | {""}


def test_split_table_repeats_header():
    rows = ["id,name,amount"] + [f"{i},person number {i},{i * 10}" for i in range(1, 200)]
    doc = reader.Doc("table", [["sheet", "Data", "\n".join(rows)]])
    chunks = reader.split(doc, COUNT, 250, 3)
    assert len(chunks) > 3
    for c in chunks:
        lines = c["text"].split("\n")
        assert lines[0].startswith("=== Sheet: Data ===") and lines[1] == "id,name,amount", lines[:2]
        assert COUNT(c["text"]) <= 250
        assert c["label"].startswith("Data rows "), c["label"]
    assert "[r2] 1,person number 1,10" in chunks[0]["text"]
    assert chunks[-1]["text"].endswith("[r200] 199,person number 199,1990")


def test_split_giant_line():
    doc = reader.Doc("text", [[None, None, "x" * 5000]])
    chunks = reader.split(doc, COUNT, 300, 3)
    assert all(COUNT(c["text"]) <= 300 for c in chunks)
    assert "".join(c["text"] for c in chunks) == "x" * 5000


# ---------------- 6./7. facts and patterns ----------------

def test_number_parsing():
    assert reader.parse_number("1 234,50") == 1234.5
    assert reader.parse_number("1,234.50") == 1234.5
    assert reader.parse_number("12.5 лв.") == 12.5
    assert reader.parse_number("€99") == 99
    assert reader.parse_number("1,234") == 1234
    assert reader.parse_number("abc") is None
    assert reader.parse_number("2026-11-05") is None


def test_patterns():
    txt = ("Contact reader.one@example.com or reader.two@example.com. Site: https://example.com/a?b=1, "
           "phone +1 202 555 0100, other 202-555-0199. Date 05.11.2026 and 2026-11-05.")
    out = "\n".join(reader.found_items("extract all emails and links", txt, R))
    assert "emails, complete list from the whole file, 2 found): reader.one@example.com; reader.two@example.com" in out, out
    assert "urls" in out and "https://example.com/a?b=1" in out and "phones" not in out
    out = "\n".join(reader.found_items("дай ми всички имейли и телефони", txt, R))
    assert "emails" in out and "phones" in out, out
    assert "+1 202 555 0100" in out and "202-555-0199" in out
    assert "05.11.2026" not in out.split("phones")[1], out
    out = "\n".join(reader.found_items("which dates?", txt, R))
    assert "05.11.2026" in out and "2026-11-05" in out
    assert reader.found_items("summarise it", txt, R) == []
    assert reader.found_items("update the text", txt, R) == []  # 'date' inside 'update' is no trigger


# ---------------- 8. verbatim check ----------------

def test_verify():
    ev = "Invoice 12345 total 1 234,50 EUR, contact sender@example.com, see https://x.com/inv.\nFACTS: sum 98765"
    ans = 'Invoice 12345: total 1234.50 EUR (sum 98765). Contact sender@example.com. Link: https://x.com/inv [p.3] [lines 120-160]'
    assert reader.unverified(ans, ev, R) == []
    bad = reader.unverified('Total 4 750.00, email other@example.com, "a quote not there", https://y.com', ev, R)
    assert "4 750.00" in bad and "other@example.com" in bad and '"a quote not there"' in bad and "https://y.com" in bad, bad
    assert reader.unverified("Phone 0888123456", "tel: 0888 123 456", R) == []
    assert reader.unverified("Планш", "Планш", R) == []
    assert reader.unverified('He said "Планш е труден"', "планш е трудeн", R) == []  # Latin e look-alike
    assert reader.unverified("CALC: 12345 * 2\n= 24690", ev, R) == []  # calculator lines are skipped
    reply, warn = reader.verify("The total is 4750.", ev, R)
    assert warn and "4750" in warn and reply.endswith(warn)
    assert reader.verify("All fine: 12345.", ev, R) == ("All fine: 12345.", None)


# ---------------- prepare: paths A / B / C + notes cache ----------------

class FakeLLM:
    def __init__(self, stop_at=None):
        self.calls, self.merges, self.stop_at = 0, 0, stop_at  # calls = parts read, merges = reduce calls

    def create_chat_completion(self, messages, max_tokens, temperature):
        prompt = messages[0]["content"]
        if not prompt.startswith("You are reading part"):
            self.merges += 1
            return {"choices": [{"message": {"content": "[lines] merged notes"}}]}
        self.calls += 1
        if self.stop_at and self.calls == self.stop_at:
            raise KeyboardInterrupt
        part = prompt.split("part ", 1)[1].split(" ", 1)[0] if "part " in prompt else "?"
        return {"choices": [{"message": {"content": "NONE" if part == "2" else f"[lines] item from part {part}"}}]}


def test_prepare_inline():
    p = f("small.csv", "a,b\n1,2\n3,4\n")
    res = reader.prepare({"path": p, "question": "sum of b?", "range": None}, None, {"reader": R},
                         progress=lambda *a: None, confirm=lambda q: True, count=COUNT)
    assert res["mode"] == "inline"
    assert res["block"].startswith(R["texts"]["history_prefix"] + "\nFILE: small.csv")
    assert "--- BEGIN FILE ---\na,b\n1,2\n3,4\n--- END FILE ---" in res["block"]
    assert 'column "b": 2 numbers, min 2, max 4, sum 6' in res["block"]
    assert res["block"].endswith(R["texts"]["question_prefix"] + " sum of b?")
    assert reader.LAST["path"] == str(p)
    res = reader.prepare({"path": p, "question": "", "range": None}, None, {"reader": R},
                         progress=lambda *a: None, confirm=lambda q: True, count=COUNT)
    assert res["question"] == R["prompts"]["default_question"]


def test_prepare_chunked_cache_resume():
    r = rcfg(inline_tokens=200, chunk_tokens=200, confirm_above_minutes=0, seconds_per_chunk=60)
    big = "\n\n".join(f"paragraph {i} " + "text " * 40 for i in range(30))
    p = f("big.txt", big)
    req = {"path": p, "question": "find things", "range": None}
    asked = []
    llm = FakeLLM(stop_at=3)  # Ctrl+C while part 3 is read
    res = reader.prepare(req, llm, {"reader": r}, progress=lambda *a: None,
                         confirm=lambda q: asked.append(q) or True, count=COUNT)
    parts = res["parts"]
    assert parts > 4 and res["done"] == 2 and res["stopped"]
    assert asked and f"{parts} parts" in asked[0]
    assert res["block"].startswith(r["texts"]["notes_prefix"])
    assert "item from part 1" in res["block"] and "item from part 2" not in res["block"]  # NONE dropped
    assert f"(based on parts 1-2 of {parts}; stopped early)" in res["block"]
    assert "stopped early" in res["rule"]
    # re-run: parts 1-2 come from the cache, only the rest is read
    llm2, asked2 = FakeLLM(), []
    res2 = reader.prepare(req, llm2, {"reader": r}, progress=lambda *a: None,
                          confirm=lambda q: asked2.append(q) or True, count=COUNT)
    assert llm2.calls == parts - 2, (llm2.calls, parts)
    assert res2["done"] == parts and not res2["partial"]
    assert "2 of" in asked2[0] and "already read" in asked2[0]
    # 28 notes don't fit inline_tokens=200: they were merged (reduce step), the result is what's attached
    assert llm2.merges > 0 and "merged notes" in res2["block"] and "item from part" not in res2["block"]
    # a different question = new notes
    llm3 = FakeLLM()
    reader.prepare({**req, "question": "other"}, llm3, {"reader": r}, progress=lambda *a: None,
                   confirm=lambda q: True, count=COUNT)
    assert llm3.calls == parts
    # declined
    raises(lambda: reader.prepare({**req, "question": "third"}, FakeLLM(), {"reader": r},
                                  progress=lambda *a: None, confirm=lambda q: False, count=COUNT), "declined")


def test_prepare_too_big():
    r = rcfg(inline_tokens=50, max_chars=1000)
    p = f("huge.txt", "word " * 1000)
    raises(lambda: reader.prepare({"path": p, "question": "", "range": None}, None, {"reader": r},
                                  progress=lambda *a: None, confirm=lambda q: True, count=COUNT), "too_big")


# ---------------- 10. history pinning ----------------

def test_oldest_droppable_pins_file_block():
    import chat
    chat.CFG = CFG
    fp = CFG["reader"]["texts"]["history_prefix"]
    np_ = CFG["reader"]["texts"]["notes_prefix"]
    vp = CFG["video"]["texts"]["history_prefix"]
    sys_ = {"role": "system", "content": "s"}
    u = lambda c: {"role": "user", "content": c}  # noqa: E731
    a = {"role": "assistant", "content": "a"}
    assert chat.oldest_droppable([sys_, u(fp + "\nFILE"), a, u("q"), a]) == 2
    assert chat.oldest_droppable([sys_, u(np_ + "\nNOTES"), a, u("q"), a]) == 2
    assert chat.oldest_droppable([sys_, u(vp + "\nVIDEO"), a, u("q"), a]) == 2
    assert chat.oldest_droppable([sys_, u("old"), a, u(fp + "\nFILE"), a]) == 1
    # only the NEWEST data block is pinned: an older video can go
    assert chat.oldest_droppable([sys_, u(vp + "\nVIDEO"), a, u(fp + "\nFILE"), a]) == 1


def test_help_lists_read():
    import chat
    cmds = [ln.split()[0] for ln in chat.HELP.splitlines() if ln.startswith("  /")]
    assert "/read" in cmds and cmds == sorted(cmds)


def test_rtf_unicode_word_and_emoji():
    r"""H10: Word writes \uN\'3f - the \'3f is the replacement char; emoji come as surrogate pairs."""
    word = r"{\rtf1\ansi\uc1 \u1087\'3f\u1088\'3f\u1080\'3f\par}"
    assert reader.rtf_to_text(word) == "при", reader.rtf_to_text(word)
    emoji = r"{\rtf1\ansi\uc1 hi \u-10179\'3f\u-8704\'3f!\par}"
    assert reader.rtf_to_text(emoji) == "hi \U0001F600!", ascii(reader.rtf_to_text(emoji))
    lone = r"{\rtf1\ansi\uc1 a\u-10179\'3f b\par}"
    t = reader.rtf_to_text(lone)
    t.encode("utf-8")  # no lone surrogate left
    p = f("emoji.rtf", emoji)
    assert "\U0001F600" in load(p).text  # the cache write must not crash


def test_read_again_txt_is_a_file():
    assert not re.match(r"(again|пак|отново)(?=\s|$)", "again.txt", re.I)


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
