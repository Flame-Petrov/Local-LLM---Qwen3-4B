"""
Follow-ups in normal chat about the songs a music job skipped:

    "download them anyway" · "download 15 and 17" · "get the Avicii one" · "all except Moby"
    "skip 9 but download the rest" · "no, skip them" · "свали ги все пак" · "where is the Daft Punk song?"
    "from now on don't skip similar songs"

After a music job, whatever was skipped, failed, was too long or (check mode) is not downloaded yet
stays OPEN (PENDING). Every normal message then goes through interpret():

  0. pre-checks   /commands and messages with a link are not follow-ups
  1. gate         instant rules: no number from the list, no song name, no category word, and (outside
                  the first weak_reference_turns messages) no "yes" / "them" -> not about it (normal chat)
  2. rules        clauses, numbers / ranges, ordinals, titles, categories, "all except", negation
  3. chat model   only when the rules are unsure: labeled-line answer, validated against the list
  4. ask          ambiguous names -> "Which one? 1) ... 2) ..."; big downloads -> "Download 34 songs? yes / no"

The model never acts by itself: it only fills a small answer that the code checks. execute() runs the
decision through music.py (download / copy / replace / skip). While a download runs, typed instructions
('get #15 too', "don't download #12", 'stop') are applied between songs (live_adjust, rules only).
All words, texts and prompts: config.json -> music.followup / music.texts / music.prompts.
"""

import json
import re
import time

from core import live
from media import music
from core import ui
from core.config import params, text
from core.debuglog import clip, log

PENDING: dict | None = None   # the open decision (one at a time; a new music job replaces it)
RUNNING = None                # music.Job while a download runs (for live instructions)

DOWNLOADABLE = ("same_video", "same_song", "too_long", "failed", "new")


# ---------------- the open decision ----------------

def open_decision(result: dict, cfg: dict) -> str:
    """After a music job: open the list of skipped / failed / new songs. Returns the 👉 hint ('' = nothing open)."""
    global PENDING
    mcfg, T = cfg["music"], cfg["music"]["texts"]
    items = [] if result.get("error") else (result.get("pending") or [])
    actionable = [it for it in items if it["actionable"]]
    if not actionable or not mcfg["followup"]["enabled"]:
        close_decision("new_job")
        return ""
    PENDING = {"source_url": result["url"], "playlist_id": result.get("playlist_id", ""),
               "playlist_title": result["playlist_title"], "folder": result["folder"], "root": result["root"],
               "format": result["format"], "single": result["single"], "size": result["size"],
               "offset": result.get("offset", 0), "report_path": result.get("report_path"), "check_only": result["check_only"], "turns_seen": 0,
               "items": items, "clarify": None, "confirm": None,
               "done": {s["index"]: {"title": s["title"], "path": s["path"], "id": s["id"]} for s in result["saved"]}}
    log("music_pending", state="opened", items=len(items), actionable=len(actionable))
    if result["check_only"]:
        return T["followup_hint_check"]
    if result["single"]:
        return T["followup_hint_single"]
    return T["followup_hint"].format(example=_pad(actionable[0]["n"], result["size"]))


def close_decision(reason: str) -> None:
    global PENDING
    if PENDING:
        log("music_pending", state="closed", reason=reason,
            items=sum(1 for it in PENDING["items"] if it["actionable"]))
    PENDING = None


def expire_weak() -> None:
    """Another feature just reported: 'yes' / 'them' no longer mean the open songs (numbers and names still do)."""
    if PENDING:
        PENDING["turns_seen"] = 10 ** 6


def restore(messages: list, cfg: dict):
    """/load of a chat whose last music job still had open songs: open them again (weak references
    like 'yes' don't count - numbers and song names do). Returns the note to print, or None."""
    global PENDING
    T = cfg["music"]["texts"]
    head = T["history_report"].split("{file}")[0]
    for m in reversed(messages):
        content = m.get("content", "") if m.get("role") == "user" else ""
        if T["history_prefix"] not in content:
            continue
        found = re.search(re.escape(head) + r"(.+\.json)\s*$", content, re.M)
        if not found:
            continue
        try:
            with open(found.group(1).strip(), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not all(k in data for k in ("source_url", "folder", "root")):
            return None  # an old/broken report: a missing key must not end the app (C2)
        items = data.get("pending") or []
        if not any(it.get("actionable") for it in items):
            return None
        done = {r["index"]: {"title": r.get("title", ""), "path": r["path"], "id": r.get("id")}
                for r in data.get("entries", []) if r.get("status") in ("downloaded", "copied", "replaced")
                and r.get("path")}
        PENDING = {"source_url": data["source_url"], "playlist_id": data.get("playlist_id", ""),
                   "playlist_title": data.get("playlist_title", ""), "folder": data["folder"], "root": data["root"],
                   "format": data.get("format", cfg["music"]["audio_format"]), "single": data.get("single", False),
                   "size": data.get("size") or len(data.get("entries", [])),
                   "offset": data.get("number_offset", 0), "report_path": found.group(1).strip(),
                   "check_only": False, "turns_seen": 10 ** 6, "items": items, "clarify": None, "confirm": None,
                   "done": done}
        n = sum(1 for it in items if it["actionable"])
        log("music_pending", state="restored", items=n, report=found.group(1).strip())
        return T["followup_restored"].format(n=n, title=PENDING["playlist_title"])
    return None


def prompt_extra(cfg: dict, question: bool = False) -> str:
    """Added to the system prompt of a normal answer while songs are open (never saved in the history)."""
    if not PENDING:
        return ""
    mcfg = cfg["music"]
    extra = text(mcfg["prompts"]["answer_rule"])
    if question:
        cap = mcfg["followup"]["max_list_in_prompt"]
        lst = "\n".join(music.history_line(it, mcfg) for it in PENDING["items"][:cap])
        extra += text(mcfg["prompts"]["pending_context"]).format(pending_list=lst)
    return extra


# ---------------- understanding a message (rules) ----------------

def _rx(words):
    return music._words_rx(tuple(words))


def _pad(n, size) -> str:
    return str(n).zfill(max(2, len(str(size or 1))))


def _context(items: list, done: dict, cfg: dict, weak: bool, options=None, confirm=None, then="download") -> dict:
    dcfg = cfg["music"]["duplicates"]
    songs = {}
    for it in items:
        if it.get("artist") and it.get("track"):
            songs[it["n"]] = music.song_info(it["track"], it["artist"], it.get("channel"), dcfg)
        else:
            songs[it["n"]] = music.song_info(it.get("title") or "", None, it.get("channel"), dcfg)
    return {"items": items, "by_n": {it["n"]: it for it in items}, "done": done or {}, "songs": songs,
            "weak": weak, "options": options, "confirm": confirm, "then": then}


def _pending_context(P: dict, cfg: dict, clarify=None, confirm=None) -> dict:
    weak = P["turns_seen"] <= cfg["music"]["followup"]["weak_reference_turns"]
    return _context(P["items"], P["done"], cfg, weak, (clarify or {}).get("options"), confirm,
                    (clarify or {}).get("action", "download"))


def _job_context(job, cfg: dict) -> dict:
    return _context(job.items(), {}, cfg, weak=False)


def _title_hits(tokens: set, items: list, ctx: dict, f: dict, stop: set):
    """Items whose artist or title words are (mostly) in the message: [(score, item)], best first."""
    out = []
    for it in items:
        s = ctx["songs"].get(it["n"])
        if not s:
            continue
        best, letters = 0.0, 0
        for part in (s["artist"], s["title"], s["key"]):
            words = {w for w in part.split() if w not in stop and len(w) > 1}
            if not words:
                continue
            hit = words & tokens
            score = len(hit) / len(words)
            if score > best:
                best, letters = score, sum(len(w) for w in hit)
        # "where is the Andrew Willis song?" - 2+ words of the message, all from this song's name
        hit = {w for w in s["key"].split() if w not in stop} & tokens
        if len(hit) >= 2 and len(hit) / len(tokens) > best:
            best, letters = len(hit) / len(tokens), sum(len(w) for w in hit)
        if best >= f["title_match_min"] and letters >= 3:
            out.append((best, it))
    out.sort(key=lambda x: -x[0])
    return out


def _clauses(t: str, R: dict, f: dict) -> list:
    parts = [p for p in _rx(f["clause_split"]).split(t) if p and p.strip()]
    act = "|".join(R[k].pattern for k in ("download_words", "skip_words", "redownload_words", "no_words"))
    and_rx = re.compile(r"(?<!\w)(?:and|и)\s+(?=(?:also\s+|then\s+|още\s+)?(?:" + act + "))", re.I)
    out = []
    for p in parts:
        out += [q for q in and_rx.split(p) if q and q.strip()]
    return out or [t]


def analyze(message: str, ctx: dict, cfg: dict) -> dict:
    """Rule reading of a message against the open songs.
    Returns {gate, decision (or None when unsure), question}."""
    f = cfg["music"]["followup"]
    names = ("download_words", "again_words", "redownload_words", "skip_words", "yes_words", "no_words",
             "all_words", "except_words", "preference_words", "question_words", "strong_words")
    R = {k: _rx(f[k]) for k in names}
    cats = {c: _rx(ws) for c, ws in f["category_words"].items()}
    unit = re.compile(r"\s*(?:" + _rx(f["unit_words"]).pattern + ")", re.I)
    count_units = _rx(f["count_units"])
    stop = set(music._norm(" ".join(f["stopwords"])).split())
    t = re.sub(r"\s+", " ", music._norm(message)).strip()
    items, by_n, done, options = ctx["items"], ctx["by_n"], ctx["done"], ctx["options"]
    actionable = [it for it in items if it["actionable"]]
    res = {"gate": False, "decision": None, "question": False}

    def resolve_num(n: int, hashed: bool):
        if options:
            if n in options:
                return n
            if not hashed and 1 <= n <= len(options):
                return options[n - 1]
        return n if (n in by_n or n in done) else None

    def targets(c: str) -> dict:
        """Numbers, ordinals, categories, titles and 'all' in one piece of text."""
        r = {"nums": [], "unknown": [], "all": False, "cats": set(), "titles": [], "ambiguous": [], "strong": False,
             "leftover": set()}
        rest = c
        for m in re.finditer(r"(?<![\w.])#?(\d{1,4})\s*(?:-|–|—|to|до|through)\s*#?(\d{1,4})(?!\w|\.\d)", c):
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            r["nums"] += [n for n in range(a, b + 1) if resolve_num(n, True) is not None]
            rest = rest.replace(m.group(0), " ")
        for m in re.finditer(r"(?<![\w.])(#)?(\d{1,4})(?!\w|\.\d)", rest):
            u = unit.match(rest, m.end())
            n = int(m.group(2))
            if u:
                if count_units.match(u.group(0).strip()) and n == len(actionable):
                    r["all"] = True  # "the 2 songs"
                continue
            got = resolve_num(n, bool(m.group(1)))
            (r["nums"] if got is not None else r["unknown"]).append(got if got is not None else n)
        rest = re.sub(r"#?\d+", " ", rest)
        for word, val in f["ordinals"].items():
            w = music._norm(word)
            pat = (rf"(?<!\w){re.escape(w)}(?!\w)" if (options or not w.isascii())
                   else rf"(?<!\w)(?:the\s+{re.escape(w)}(?!\w)|{re.escape(w)}\s+(?:one|song|track)s?(?!\w))")
            if re.search(pat, rest):
                pool = options or [it["n"] for it in actionable]
                idx = val - 1 if val > 0 else len(pool) + val
                if 0 <= idx < len(pool):
                    r["nums"].append(pool[idx])
                rest = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", " ", rest)
        for cat, rx in cats.items():
            if rx.search(rest):
                r["cats"].add(cat)
                rest = rx.sub(" ", rest)
        if R["all_words"].search(rest):
            r["all"] = True
        if R["strong_words"].search(rest):
            r["strong"] = True
        for k in names:
            rest = R[k].sub(" ", rest)
        tokens = {w for w in re.findall(r"\w+", rest) if w not in stop and len(w) > 1}
        pool = [by_n[n] for n in options if n in by_n] if options else items
        hits = _title_hits(tokens, pool, ctx, f, stop) if tokens else []
        if hits:
            top = hits[0][0]
            win = [it for sc, it in hits if top - sc < f["title_match_gap"]]
            if len(win) == 1:
                r["titles"] = [win[0]["n"]]
            else:
                r["ambiguous"] = [it["n"] for it in win]
        else:
            r["leftover"] = tokens  # other words: maybe not about the songs at all ("I want to go to the cinema")
        return r

    def expand(r: dict, named_elsewhere: set) -> list:
        out = list(r["nums"]) + list(r["titles"])
        for c in r["cats"]:
            want = ("same_video", "same_video_here") if c == "same_video" else (c,)
            out += [it["n"] for it in actionable if it["category"] in want]
        if r["all"]:
            out += [it["n"] for it in actionable if it["n"] not in named_elsewhere]
        return list(dict.fromkeys(out))

    # ---- clauses ----
    question = bool(R["question_words"].match(t))
    clauses = []
    for c in _clauses(t, R, f):
        main, excl = c, ""
        ex_m = R["except_words"].search(c)
        if ex_m:  # "all except moby" / "всички без 17"
            main, excl = c[: ex_m.start()], c[ex_m.end():]
        masked = R["skip_words"].sub(" ", main)
        sk = R["skip_words"].search(main)
        dl = R["download_words"].search(masked)
        again = R["again_words"].search(main)
        red = R["redownload_words"].search(main)
        yes = R["yes_words"].search(masked)
        no = R["no_words"].search(masked)
        word_pos = min([m.start() for m in (sk, dl, again) if m] or [10 ** 6])
        negated = bool(no and no.start() < word_pos)
        if red:
            act = "skip" if negated else "redownload"
        elif sk and dl:
            act = "conflict"
        elif sk:
            act = "download" if negated else "skip"
        elif dl or again:
            act = "skip" if negated else "download"
        elif no:
            act = "no"
        elif yes:
            act = "yes"
        else:
            act = None
        tg, ex = targets(main), targets(excl) if excl.strip() else None
        clauses.append({"text": c, "act": act, "t": tg, "ex": ex, "again": bool(again or red),
                        "pref": bool(R["preference_words"].search(main)), "yes": bool(yes), "no": bool(no)})

    has_target = [bool(cl["t"]["nums"] or cl["t"]["titles"] or cl["t"]["cats"] or cl["t"]["ambiguous"])
                  for cl in clauses]
    strong = any(has_target) or any(cl["t"]["strong"] for cl in clauses) or any(
        cl["ex"] and (cl["ex"]["nums"] or cl["ex"]["titles"]) for cl in clauses)
    signal = any(cl["act"] or cl["t"]["all"] or cl["pref"] for cl in clauses)
    pending_question = bool(options or ctx["confirm"])
    res["gate"] = strong or (ctx["weak"] and signal) or (pending_question and signal)
    if not res["gate"]:
        return res

    def decision(action, items_=(), skip_items=(), **kw):
        return {"action": action, "items": list(dict.fromkeys(items_)), "skip_items": list(dict.fromkeys(skip_items)),
                "source": "rules", "again": any(cl["again"] for cl in clauses), **kw}

    # preference: "from now on don't skip similar songs"
    pref = next((cl for cl in clauses if cl["pref"] and cl["act"] in ("download", "skip")), None)
    if pref:
        keys = [k for k in ("same_song", "same_video") if k in pref["t"]["cats"]] or ["same_song", "same_video"]
        res["decision"] = decision("preference", pref={k: pref["act"] for k in keys})
        return res

    acting = [cl for cl in clauses if cl["act"] in ("download", "skip", "redownload")]
    if question or (t.endswith("?") and not acting):
        res["question"] = True
        if strong:
            res["decision"] = decision("question")
        return res

    leftover = any(cl["t"]["leftover"] for cl in clauses)
    # pure "yes" / "no" (answer to the 👉 hint, a confirmation or a question)
    if not any(has_target) and not any(cl["t"]["all"] for cl in clauses) and not acting:
        if leftover:
            return res  # "yes, I love that movie" -> unsure
        acts = {cl["act"] for cl in clauses if cl["act"]}
        if acts == {"yes"}:
            if ctx["confirm"]:
                res["decision"] = {**ctx["confirm"], "source": "confirm"}
            elif options:
                res["decision"] = decision("clarify", options=options, then=ctx["then"])
            elif ctx["weak"]:
                res["decision"] = decision("download", [it["n"] for it in actionable])
        elif acts == {"no"}:
            if ctx["confirm"] or options:
                res["decision"] = decision("cancel")
            elif ctx["weak"]:
                res["decision"] = decision("skip", [it["n"] for it in actionable])
        return res

    # actions flow into following clauses without one ("15, 17" / "download 15, not 17")
    prev = None
    for i, cl in enumerate(clauses):
        if cl["act"] in ("download", "skip", "redownload"):
            prev = cl["act"]
        elif cl["act"] == "no" and has_target[i]:
            # "download the similar ones, not the duplicates" / "no, not 15"
            cl["act"] = "exclude" if prev == "download" else "skip"
        elif cl["act"] in (None, "yes", "no") and (has_target[i] or cl["t"]["all"]):
            cl["act"] = prev or next((c["act"] for c in clauses[i + 1:] if c["act"] in ("download", "skip", "redownload")),
                                     None) or ctx["then"] or "download"
            prev = cl["act"] if cl["act"] in ("download", "skip", "redownload") else prev
    if any(cl["act"] == "conflict" for cl in clauses):
        return res
    for cl in clauses:
        if cl["t"]["ambiguous"]:
            res["decision"] = decision("clarify", options=cl["t"]["ambiguous"],
                                       then=cl["act"] if cl["act"] in ("download", "skip", "redownload") else "download")
            return res

    named = {cl_i: set(cl["t"]["nums"]) | set(cl["t"]["titles"]) for cl_i, cl in enumerate(clauses)}
    sets = {"download": [], "skip": [], "redownload": [], "exclude": []}
    unknown = []
    for i, cl in enumerate(clauses):
        unknown += cl["t"]["unknown"]
        if cl["act"] not in sets:
            continue
        elsewhere = set().union(*(v for k, v in named.items() if k != i)) if len(named) > 1 else set()
        got = expand(cl["t"], elsewhere)
        if cl["ex"]:
            drop = set(expand(cl["ex"], set()))
            got = [n for n in got if n not in drop]
        if not has_target[i] and cl["t"]["leftover"]:
            return res  # "I want to watch it tonight" - other words and no song named: unsure
        if not got and not has_target[i] and not cl["t"]["all"] and not cl["t"]["unknown"] \
                and cl["act"] in ("download", "skip", "redownload"):
            # "download" / "skip" with no target: everything open - only right after the report, or with
            # "anyway" / "the skipped songs"
            if ctx["weak"] or cl["again"] or cl["t"]["strong"]:
                got = [it["n"] for it in actionable if it["n"] not in elsewhere]
        if not got:
            if unknown and not any(has_target):
                res["decision"] = decision("unknown", unknown=unknown)
            return res  # unsure -> stage 3
        sets[cl["act"]] += got
    dl = [n for n in sets["download"] if n not in sets["exclude"]]
    sk, red = sets["skip"], sets["redownload"]
    if set(dl) & set(sk) or set(red) & set(sk):
        return res
    if red:
        res["decision"] = decision("redownload", red + dl, sk)
    elif dl:
        res["decision"] = decision("download", dl, sk)
    elif sk:
        res["decision"] = decision("skip", sk)
    return res


def is_music_instruction(text_: str, cfg: dict) -> bool:
    """Rules only (no model): does this live-typed message clearly add / remove songs of the running
    (or open) music job? Unclear ones are NOT taken - they wait for the end of the job."""
    try:
        if RUNNING is not None:
            ctx = _job_context(RUNNING, cfg)
        elif PENDING:
            ctx = {**_pending_context(PENDING, cfg), "weak": False}
        else:
            f = cfg["music"]["followup"]
            t = music._norm(text_)
            return bool((_rx(f["download_words"]).search(t) or _rx(f["skip_words"]).search(t))
                        and re.search(r"#?\d{1,4}(?!\w)", t))
        d = analyze(text_, ctx, cfg)["decision"]
        return bool(d and d["action"] in ("download", "skip") and (d["items"] or d["skip_items"]))
    except Exception as e:
        log("music_followup_live", text=text_, error=repr(e))
        return False


# ---------------- understanding a message (chat model, only when the rules are unsure) ----------------

def _llm_targets(value: str, ctx: dict, actionable: list) -> list:
    if not value:
        return []
    v = music._norm(value)
    if re.match(r"\s*(none|-|n/a)\s*$", v):
        return []
    out = []
    if re.search(r"(?<!\w)all(?!\w)", v):
        out += [it["n"] for it in actionable]
    for cat in ("same_video", "same_song", "failed", "too_long", "new"):
        if re.search(rf"(?<!\w){cat}(?!\w)", v):
            want = ("same_video", "same_video_here") if cat == "same_video" else (cat,)
            out += [it["n"] for it in actionable if it["category"] in want]
    for m in re.finditer(r"#?(\d{1,4})", v):
        n = int(m.group(1))
        if n in ctx["by_n"] or n in ctx["done"]:  # numbers that are not in the list are dropped
            out.append(n)
    return list(dict.fromkeys(out))


def _llm_resolve(message: str, ctx: dict, llm, cfg: dict, spin, history):
    mcfg = cfg["music"]
    f, T = mcfg["followup"], mcfg["texts"]
    items = ctx["items"][: f["max_list_in_prompt"]]
    pending_list = "\n".join(music.history_line(it, mcfg) for it in items)
    options = ""
    if ctx["options"]:
        options = T["llm_options"].format(options="  ".join(
            f"{i}) #{n} {ctx['by_n'][n]['title']}" for i, n in enumerate(ctx["options"], 1) if n in ctx["by_n"]))
    recent = [m for m in (history or []) if m.get("role") in ("user", "assistant")][-2:]
    recent_text = "\n".join(f"{m['role']}: {' '.join(m['content'].split())[:300]}" for m in recent) or "-"
    prompt = text(mcfg["prompts"]["followup_resolver"]).format(pending_list=pending_list, clarify_options=options,
                                                               recent=recent_text, message=message)
    spin.status("music_understanding")
    t = time.monotonic()
    try:
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                         **params(f["generation"]))["choices"][0]["message"]["content"]
    finally:
        spin.stop()
    dec = parse_llm(out, ctx, cfg)
    log("music_followup", text=message, stage="llm", decision=dec, model_output=out,
        prompt=clip(prompt, "prompt"), seconds=round(time.monotonic() - t, 2))
    return dec


def parse_llm(out: str, ctx: dict, cfg: dict):
    """The model's labeled lines -> a validated decision, or None (not about the songs / unreadable)."""
    T = cfg["music"]["texts"]

    def field(name):
        m = re.search(rf"^\W*{name}\s*[:\-]\s*(.+)$", out or "", re.I | re.M)
        return m.group(1).strip().strip("\"'`*").strip() if m else None
    about, action = field("ABOUT_SKIPPED"), field("ACTION")
    if not about or not action or not re.match(r"(yes|no)\b", about, re.I):
        return None
    if about.lower().startswith("no"):
        return None
    act = action.lower().split()[0].strip(".,")
    if act not in ("download", "redownload", "skip", "ask", "none"):
        return None
    actionable = [it for it in ctx["items"] if it["actionable"]]
    songs = _llm_targets(field("SONGS"), ctx, actionable)
    skip = _llm_targets(field("SKIP_SONGS"), ctx, actionable)
    base = {"items": [], "skip_items": [], "source": "llm", "again": act == "redownload"}
    if act == "none":
        return {**base, "action": "question"}
    if act == "ask" or (act in ("download", "redownload", "skip") and not songs):
        ask = field("QUESTION") if act == "ask" else None
        return {**base, "action": "clarify", "ask": ask or T["followup_ask_default"], "options": ctx["options"] or [],
                "then": "download" if act == "ask" else act}
    if act == "skip":
        return {**base, "action": "skip", "items": songs}
    return {**base, "action": act, "items": [n for n in songs if n not in skip], "skip_items": skip}


def interpret(message: str, llm, cfg: dict, spin, history=None):
    """Decision for a chat message while songs are open, or None (not about them -> normal chat)."""
    P = PENDING
    f = cfg["music"]["followup"]
    if not P or not f["enabled"]:
        return None
    msg = message.strip()
    if not msg or msg.startswith("/"):
        return None
    P["turns_seen"] += 1
    clarify, confirm = P.get("clarify"), P.get("confirm")
    P["clarify"] = P["confirm"] = None  # a question counts only for the very next message
    t = time.monotonic()
    if re.search(cfg["music"]["url_pattern"], msg):
        log("music_followup", text=msg, stage="pre", decision=None)
        return None  # a new link: the music / video routing handles it
    ctx = _pending_context(P, cfg, clarify, confirm)
    a = analyze(msg, ctx, cfg)
    if not a["gate"]:
        log("music_followup", text=msg, stage="gate", decision=None, seconds=round(time.monotonic() - t, 3))
        return None
    if a["decision"]:
        log("music_followup", text=msg, stage="rules", decision=a["decision"], seconds=round(time.monotonic() - t, 3))
        return a["decision"]
    if a["question"] or not f["use_llm"] or llm is None:
        log("music_followup", text=msg, stage="rules", decision=None, unsure=True)
        return None
    return _llm_resolve(msg, ctx, llm, cfg, spin, history)


# ---------------- doing it ----------------

def execute(dec: dict, llm, cfg: dict, spin) -> str:
    """Run a decision. Prints its own lines; returns the plain report for the chat history."""
    global PENDING
    mcfg = cfg["music"]
    T, f, colors = mcfg["texts"], mcfg["followup"], mcfg["colors_256"]
    lines = []

    def say(s, color=None):
        print(ui.paint(s, colors.get(color)) if color else s)
        lines.append(s)

    def say_path(p):
        shown = ui.file_link(p) if mcfg["clickable_paths"] else p
        print(T["dup_path"].format(path=ui.paint(shown, colors["path"])))
        lines.append(T["dup_path"].format(path=p))

    P = PENDING
    if P is None:
        say(T["followup_nothing_open"])
        return "\n".join(lines)
    pad = lambda n: _pad(n, P["size"])  # noqa: E731
    act = dec["action"]
    if act == "clarify":
        opts = [n for n in dec.get("options") or [] if any(it["n"] == n for it in P["items"])]
        if opts and not dec.get("ask"):
            say(T["followup_ask_which"].format(options="   ".join(
                f"{i}) #{pad(n)} {next(it['title'] for it in P['items'] if it['n'] == n)}" for i, n in enumerate(opts, 1))))
        else:
            say(dec.get("ask") or T["followup_ask_default"])
        P["clarify"] = {"options": opts, "action": dec.get("then") or "download"}
        return "\n".join(lines)
    if act == "unknown":
        open_ = ", ".join(f"#{pad(it['n'])}" for it in P["items"] if it["actionable"])
        say(T["followup_unknown_number"].format(numbers=", #".join(map(str, dec.get("unknown") or [])), open=open_))
        return "\n".join(lines)
    if act == "cancel":
        say(T["followup_confirm_cancelled"])
        return "\n".join(lines)
    if act == "preference":
        for key, a in dec["pref"].items():
            mcfg["duplicates"][key] = a
            log("setting_changed", key=f"music.duplicates.{key}", value=a, by="music_followup")
            say(T["followup_pref_set"].format(what=T["followup_pref_what"][key], action=T["followup_pref_action"][a],
                                              key=key, value=a))
        return "\n".join(lines)
    if act == "question":
        return ""

    by_n = {it["n"]: it for it in P["items"]}
    handled, changes, jobs = set(), {}, []
    if act in ("download", "redownload"):
        for n in dec["items"]:
            it, done = by_n.get(n), P["done"].get(n)
            if it is None and done:
                if act == "redownload" or dec.get("again"):
                    jobs.append({"entry": _entry({"n": n, "id": done.get("id"), "title": done["title"]}, P),
                                 "mode": "replace", "path": done["path"], "force": True})
                else:
                    say(T["followup_done"].format(index=pad(n), title=done["title"]))
                    say_path(done["path"])
                continue
            if it is None:
                continue
            if not it["actionable"]:
                say(T["followup_not_actionable"].format(index=pad(n), title=it["title"],
                                                        reason=music.reason_text(it, mcfg)), "error")
                continue
            cat = it["category"]
            if cat == "same_video_here" and act != "redownload":
                say(T["followup_here"].format(index=pad(n), title=it["title"]))
                say_path(it["existing_path"])
                handled.add(n)
                continue
            job = {"entry": _entry(it, P), "mode": "download", "force": True}
            if act == "redownload" and it.get("existing_path") and cat in ("same_video_here", "same_video"):
                job.update(mode="replace", path=it["existing_path"])
            elif cat == "same_video" and f["copy_instead_of_download"] and it.get("existing_path") \
                    and it["existing_path"].lower().endswith("." + P["format"]):
                job.update(mode="copy", path=it["existing_path"])
            jobs.append(job)
        net = [j for j in jobs if j["mode"] != "copy"]
        if len(net) > f["confirm_over_songs"] and dec.get("source") != "confirm":
            kbps = mcfg["mp3_quality_kbps"] if P["format"] == "mp3" else mcfg["estimate_kbps"]
            mb = round(sum((j["entry"].get("duration") or 240) for j in net) * kbps / 8 / 1000)
            P["confirm"] = {**dec, "source": "confirm"}
            say(T["followup_confirm_many"].format(n=len(net), mb=mb))
            return "\n".join(lines)
        if jobs:
            res = music.run_jobs(P, jobs, cfg, spin)
            for s in res["saved"]:
                handled.add(s["index"])
                P["done"][s["index"]] = {"title": s["title"], "path": s["path"], "id": s["id"]}
                changes[s["index"]] = {"status": s["how"], "path": s["path"]}
                lines.append(T["history_saved_line"].format(index=s["index"], title=s["title"], path=s["path"]))
            for d in res["failed"]:
                n = d["entry"]["index"]
                if n in by_n:
                    by_n[n].update(category="unavailable" if d["kind"] == "unavailable" else "failed",
                                   reason=d["reason"], actionable=d["kind"] != "unavailable")
                changes[n] = {"status": "failed", "reason": d["reason"]}
                lines.append(T["history_failed_line"].format(index=n, title=d["entry"]["title"], reason=d["reason"]))
            if res["disk"]:
                lines.append(res["disk"])
            if res["rate_limited"]:
                say(T["rate_limited"].format(saved=len(res["saved"])), "warn")
            elif res["stopped"]:
                say(T["followup_stopped"].format(saved=len(res["saved"])), "warn")
    skip_targets = dec["items"] if act == "skip" else dec.get("skip_items") or []
    skipped = [n for n in skip_targets if n in by_n and by_n[n]["actionable"] and n not in handled]
    if skipped:
        handled.update(skipped)
        for n in skipped:
            changes[n] = {"status": "skipped_by_user"}
        say(T["followup_skipped"].format(numbers=", ".join(f"#{pad(n)}" for n in skipped)))
    elif act == "skip" and not handled:
        say(T["followup_nothing_to_skip"])

    P["items"] = [it for it in P["items"] if it["n"] not in handled]
    P["turns_seen"] = 0  # "yes" / "them" refer to what is left, right after this
    if P.get("report_path"):
        music.update_report(P["report_path"], changes, P["items"])
    left = [it["n"] for it in P["items"] if it["actionable"]]
    if left:
        shown = ", ".join(f"#{pad(n)}" for n in left[:15]) + (" ..." if len(left) > 15 else "")
        say(T["followup_still_open"].format(numbers=shown))
    else:
        close_decision("all_handled")
    if P.get("report_path"):
        lines.append(T["history_report"].format(file=P["report_path"]))
    print()
    return "\n".join(lines)


def _entry(it: dict, P: dict) -> dict:
    vid = it.get("id") or ""
    return {"index": it["n"], "id": vid, "url": it.get("url") or f"https://www.youtube.com/watch?v={vid}",
            "title": it.get("title") or vid, "channel": it.get("channel") or "", "duration": it.get("duration"),
            "artist": it.get("artist"), "track": it.get("track"), "available": True, "reason": None}


# ---------------- while a download runs ----------------

def live_adjust(job, cfg: dict, spin) -> None:
    """Between two songs: take the typed instructions that clearly concern this job (rules only)."""
    texts = live.take_if(lambda s: live.is_stop([s]) or is_music_instruction(s, cfg))
    if not texts:
        return
    T, o = cfg["music"]["texts"], job.ctx["out"]
    pad = lambda n: _pad(n, job.ctx["size"])  # noqa: E731
    if live.is_stop(texts):
        job.stop = True
        o.line(T["followup_live_stop"], "warn")
        log("music_followup_live", text=texts, effect="stop")
        live.mark_used(texts)
        return
    for s in texts:
        d = analyze(s, _job_context(job, cfg), cfg)["decision"] or {}
        dl = d.get("items", []) if d.get("action") == "download" else []
        sk = (d.get("items", []) if d.get("action") == "skip" else []) + d.get("skip_items", [])
        added = [n for n in dl if job.add(n)]
        removed = [n for n in sk if job.remove(n)]
        if added:
            o.line(T["followup_live_added"].format(numbers=", ".join(f"#{pad(n)}" for n in added)), "ok")
        if removed:
            o.line(T["followup_live_removed"].format(numbers=", ".join(f"#{pad(n)}" for n in removed)), "ok")
        if not added and not removed:
            o.line(T["followup_live_nothing"].format(text=s), "dim")
        log("music_followup_live", text=s, effect="queued" if added else "removed" if removed else "nothing",
            added=added, removed=removed)
    live.mark_used(texts)


# ---------------- /music pending | get | skip | rescan ----------------

def command(sub: str, args: str, llm, cfg: dict, spin) -> dict:
    """Exact commands. Returns {"show": text to print, "history": plain text for the chat or None}."""
    mcfg = cfg["music"]
    T = mcfg["texts"]
    sub = sub.lower().strip()
    if sub == "rescan":
        spin.status("music_library")
        try:
            return {"show": music.rescan(cfg), "history": None}
        finally:
            spin.stop()
    if not PENDING:
        return {"show": T["followup_nothing_open"], "history": None}
    P = PENDING
    if sub == "pending":
        rows = [T["pending_head"].format(title=P["playlist_title"] or P["source_url"],
                                         n=sum(1 for it in P["items"] if it["actionable"]))]
        for it in P["items"]:
            rows.append(T["pending_line"].format(index=_pad(it["n"], P["size"]), title=it["title"],
                                                 reason=music.reason_text(it, mcfg)))
            if it.get("existing_path"):
                rows.append(T["dup_path"].format(path=it["existing_path"]))
        return {"show": "\n".join(rows), "history": None}
    args = (args or "").strip()
    if sub not in ("get", "skip") or (sub == "get" and not args):
        return {"show": T["usage"], "history": None}
    word = mcfg["followup"]["download_words"][0] if sub == "get" else mcfg["followup"]["skip_words"][0]
    ctx = {**_pending_context(P, cfg), "weak": True, "options": None, "confirm": None}
    dec = analyze(f"{word} {args or 'all'}", ctx, cfg)["decision"]
    if not dec or dec["action"] not in ("download", "skip", "clarify", "unknown"):
        return {"show": T["usage"], "history": None}
    dec["source"] = "command"
    log("music_followup", text=f"/music {sub} {args}", stage="command", decision=dec)
    return {"show": "", "history": execute(dec, llm, cfg, spin)}
