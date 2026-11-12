"""
Follow-ups in normal chat about the faces library (same design as music_followup):

    "12 is Maria" · "човек 12 е Мария" · "rename Maria to Maria Petrova" · "12 and 3 are the same person"
    "1 and 3 yes, 2 no" · "2 is Ivan" · "IMG_0450 is not Maria" · "in IMG_0450 Maria is the left one"
    "ignore 12" · "forget Maria" · "ok, sort them" (after a check) · "be stricter with Maria"
    "who is in IMG_0450?" (answered by code from the DB) · "where are Maria's photos?" (normal answer + PEOPLE)

Every normal message goes through interpret():
  0. pre-checks   /commands, links and new faces requests are not follow-ups
  1. gate         no strong reference (person 12, #12, a known name + an action word, a known photo name) and,
                  outside the first weak_reference_turns messages after a report, no weak one ('yes', bare
                  numbers, 'all') -> normal chat
  2. rules        clauses, references, action words, name extraction
  3. chat model   only when the rules are unsure: labeled-line form, validated against the real lists
  4. ask          ambiguous names, person-or-photo numbers, big merges, forget -> one question, valid for the
                  very next message

The model never acts by itself. execute() does the work through face_sync / faces.
All words, texts and prompts: config.json -> faces.followup / faces.texts / faces.prompts.
"""

import re
import time
from pathlib import Path

from photos import face_store
from photos import face_sync
from photos import faces
from media import music
from core.config import params, text
from core.debuglog import clip, log

CONTEXT: dict | None = None    # the last faces report: new people, the numbered 'not sure' list, check run...
PENDING_Q: dict | None = None  # a question we asked (valid for the very next message only)


# ---------------- context ----------------

def open_context(result: dict, cfg: dict) -> str:
    """After a faces report. Returns the 👉 hint ('' = none)."""
    global CONTEXT, PENDING_Q
    T = cfg["faces"]["texts"]
    PENDING_Q = None
    if result.get("error") or not cfg["faces"]["followup"]["enabled"]:
        CONTEXT = None
        return ""
    if result.get("mode") == "who":
        CONTEXT = {"turns_seen": 0, "new_people": [], "maybe": [], "check": False, "source": result["source"],
                   "opts": {}, "who": result, "photos": {result["photo"].lower()}}
        log("faces_followup", stage="context", state="opened", mode="who")
        return ""
    CONTEXT = {"turns_seen": 0, "new_people": [p["number"] for p in result["new_people"] if p.get("number")],
               "maybe": faces.OPEN_MAYBE if not result["check"] else [], "check": result["check"],
               "source": result["source"], "opts": dict(result.get("opts") or {}), "who": None,
               "photos": {m["photo"].lower() for m in result["maybe"]}}
    log("faces_followup", stage="context", state="opened", new_people=len(CONTEXT["new_people"]),
        maybe=len(CONTEXT["maybe"]), check=result["check"])
    if result["check"]:
        return T["hint_check"]
    if result["new_people"] or result["maybe"]:
        return T["hint_scan"]
    return ""


def open_maybe_context(items: list, cfg: dict) -> None:
    """/faces maybe listed the open 'not sure' faces: their numbers now count (and 'yes' for a moment)."""
    global CONTEXT
    if CONTEXT is None:
        CONTEXT = {"turns_seen": 0, "new_people": [], "maybe": items, "check": False, "source": None, "opts": {},
                   "who": None, "photos": set()}
    else:
        CONTEXT.update(maybe=items, turns_seen=0)


def ask_confirm(decision: dict, question: str) -> None:
    global PENDING_Q
    PENDING_Q = {"kind": "confirm", "decision": decision, "question": question}


def close(reason: str) -> None:
    global CONTEXT, PENDING_Q
    if CONTEXT or PENDING_Q:
        log("faces_followup", stage="context", state="closed", reason=reason)
    CONTEXT = None
    PENDING_Q = None


def expire_weak() -> None:
    """Another feature just reported: 'yes' / 'them' no longer mean the faces list (numbers and names still do)."""
    if CONTEXT:
        CONTEXT["turns_seen"] = 10 ** 6


# ---------------- reading the library (fast, read-only) ----------------

def _current_maybe() -> list:
    """The numbered 'not sure' list answers refer to. An open context wins even when its list is empty
    (a check / who report) - never fall back to an older scan's list (review H7)."""
    if CONTEXT is not None:
        return CONTEXT.get("maybe") or []
    return faces.OPEN_MAYBE


def _db_exists(fcfg: dict) -> bool:
    return (face_sync.library_dir(fcfg) / fcfg["db_file"]).is_file()


def _snapshot(fcfg: dict):
    """(persons {id: row}, store) - store must be closed by the caller. None when there is no library yet."""
    if not _db_exists(fcfg):
        return None
    try:
        st = face_store.Store.open(face_sync.library_dir(fcfg), fcfg)
    except Exception as e:
        log("faces_followup", stage="pre", error=repr(e))
        return None
    return {p["id"]: p for p in st.persons()}, st


class _LazyStore:
    """Opens the face DB only when the rules really need it (a photo name lookup)."""

    def __init__(self, fcfg: dict):
        self._fcfg, self._st = fcfg, None

    def __getattr__(self, attr):
        if self._st is None:
            self._st = face_store.Store.open(face_sync.library_dir(self._fcfg), self._fcfg)
        return getattr(self._st, attr)

    def close(self) -> None:
        if self._st is not None:
            self._st.close()
            self._st = None


_PERSONS_CACHE = {"key": None, "persons": None}


def _db_key(fcfg: dict):
    """Change marker of the DB files (main + WAL); None when there is no library."""
    db = face_sync.library_dir(fcfg) / fcfg["db_file"]
    key = []
    for q in (db, db.with_name(db.name + "-wal")):
        try:
            s = q.stat()
            key.append((s.st_mtime_ns, s.st_size))
        except OSError:
            if q == db:
                return None
            key.append(None)
    return (str(db), tuple(key))


def _cached_snapshot(fcfg: dict):
    """Like _snapshot, but reuses the persons list while the DB files are unchanged, and opens the store
    lazily - so a plain chat message does not open (and write to) faces.db (review C8)."""
    key = _db_key(fcfg)
    if key is None:
        return None
    if _PERSONS_CACHE["key"] != key:
        snap = _snapshot(fcfg)
        if snap is None:
            return None
        persons, st = snap
        st.close()
        _PERSONS_CACHE.update(key=_db_key(fcfg), persons=persons)
    return dict(_PERSONS_CACHE["persons"]), _LazyStore(fcfg)


# ---------------- rules ----------------

def _rx(words):
    return music._words_rx(tuple(words))


def _clauses(t: str, f: dict) -> list:
    parts = [p for p in _rx(f["clause_split"]).split(t) if p and p.strip()]
    return parts or [t]


_PHOTO_RX = re.compile(r"(?<![\w\\/])([\w\-\(\) ]{0,40}?\.(?:jpe?g|png|webp|bmp|tiff?|gif|heic|heif))(?!\w)|"
                       r"(?<![\w\\/])([A-Za-z]{0,6}[_\-]?\d{3,}[\w\-]*)(?!\w)", re.I)


def _photo_refs(t: str, st) -> list:
    """Known photo names in the text: [(stem, [images])]."""
    out = []
    for m in _PHOTO_RX.finditer(t):
        tok = (m.group(1) or m.group(2) or "").strip()
        stem = Path(tok).stem if m.group(1) else tok
        if not stem or len(stem) < 4:
            continue
        hits = st.images_by_stem(stem) if st else []
        if hits:
            out.append((stem, hits, m.span()))
    return out


def _name_refs(t: str, persons: dict) -> list:
    """Known names / aliases in the text (whole words, longest first, no overlaps): [(pid, span)]."""
    cands = []
    for pid, p in persons.items():
        if p["status"] not in ("active", "ignored", "merged"):
            continue
        for n in [p["name"], *(p.get("aliases") or [])]:
            k = face_sync.norm_name(n)
            if not k or k.isdigit():
                continue
            for m in re.finditer(r"(?<!\w)" + re.escape(k) + r"(?!\w)", t):
                cands.append((m.start(), m.end(), pid))
    cands.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    taken, out = [], []
    for a, b, pid in cands:
        if any(not (b <= x or a >= y) for x, y in taken):
            continue
        taken.append((a, b))
        out.append((pid, (a, b)))
    out.sort(key=lambda x: x[1][0])
    return out


def _resolve(pid, persons):
    return face_sync.resolve_ref(pid, persons) if pid is not None else None


def _clean_name(raw: str, f: dict, cfg: dict, strong: bool):
    """The new name after a rename word. Returns (name or None, ok: bool). None + ok False = unsure."""
    s = raw.strip().strip(" .!,;:")
    q = re.match(r"[\"'“„«](.+?)[\"'”“»]", s)
    quoted = bool(q)
    if q:
        s = q.group(1).strip()
    if not s:
        return None, False
    if re.search(r"[\\/:*?\"<>|#\d]", s):
        return None, False
    stop = set(music._norm(" ".join(f["stopwords"])).split())
    words = s.split()
    while words and music._norm(words[0]) in stop:
        words = words[1:]
    if not words or len(words) > f["max_name_words"]:
        return None, False
    if not re.match(r"[^\W\d_]", words[0]):
        return None, False
    s = " ".join(words)
    lower = s == s.lower()
    nn = _rx(f["not_names"])
    if not quoted and lower:
        if nn.search(music._norm(s)) or any(music._norm(w) in stop for w in words):
            return None, False
        if not strong and len(words) > 2:
            return None, False
        if cfg["faces"]["matching"]["capitalize_lowercase_names"]:
            s = " ".join(w[:1].upper() + w[1:] for w in words)
    elif not quoted and nn.fullmatch(music._norm(s)):
        return None, False
    return s, True


def analyze(message: str, ctx: dict, cfg: dict) -> dict:
    """Rule reading of a message. ctx: {persons, store, weak, maybe (list), check, pending}.
    Returns {gate, decision (None = unsure), question}. A name two people share ("Maria" = #3 and #19) ->
    "Which one? 1) #3 Maria 2) #19 Maria (2)"."""
    res = _analyze(message, ctx, cfg)
    d = res.get("decision")
    if not d or d.get("source") in ("clarify", "confirm"):
        return res
    persons = ctx["persons"]
    groups = {}
    for pid, p in persons.items():
        if p["status"] not in ("active", "ignored"):
            continue
        for n in [p["name"], *(p.get("aliases") or [])]:
            k = face_sync.norm_name(n)
            if k and re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", music._norm(message)):
                groups.setdefault(k, set()).add(pid)
    for k, pids in groups.items():
        if len(pids) < 2:
            continue
        for slot in ("person", "a", "b"):
            if d.get(slot) in pids and not re.search(r"(?<![\w.])#?%d(?![\w.])" % d[slot], message):
                res["decision"] = {"action": "ask_which", "options": sorted(pids), "decision": d, "slot": slot,
                                   "source": "rules"}
                return res
    return res


def _analyze(message: str, ctx: dict, cfg: dict) -> dict:
    fcfg = cfg["faces"]
    f = fcfg["followup"]
    names = ("person_words", "rename_words", "same_words", "not_words", "ignore_words", "unignore_words",
             "forget_words", "sort_words", "yes_words", "no_words", "all_words", "except_words", "stricter_words",
             "looser_words", "suggested_words", "preference_words", "question_words")
    R = {k: _rx(f[k]) for k in names}
    persons, st = ctx["persons"], ctx["store"]
    maybe = [m for m in (ctx.get("maybe") or [])]
    maybe_n = {m["n"]: m for m in maybe}
    orig = re.sub(r"\s+", " ", message).strip()
    t = music._norm(orig)
    res = {"gate": False, "decision": None, "question": False}
    pq = ctx.get("pending")

    def D(action, **kw):
        return {"action": action, "source": "rules", **kw}

    # references in the whole message
    pw = R["person_words"].pattern
    strong_nums = []
    for m in re.finditer(r"(?:(?:" + pw + r")\s*(?:#|no\.?|№)?\s*|(?<![\w&])#)(\d{1,5})(?!\w)", t):
        strong_nums.append((int(m.group(1)), m.span(1)))
    bare = [(int(m.group(1)), m.span(1)) for m in re.finditer(r"(?<![\w#.\-])(\d{1,5})(?![\w.])", t)
            if not any(m.span(1) == s for _, s in strong_nums)]
    photos = _photo_refs(orig, st)
    photo_spans = [s for _, _, s in photos]
    bare = [(n, s) for n, s in bare if not any(a <= s[0] < b for a, b in photo_spans)]
    name_refs = _name_refs(t, persons)
    action_hit = any(R[k].search(t) for k in names if k not in ("person_words", "all_words", "question_words",
                                                                "yes_words", "no_words"))
    yes_no = bool(R["yes_words"].search(t) or R["no_words"].search(t))
    # a command word right before / after a number is unambiguous: "name 12 Maria", "ignore 12", "merge 12 into 3",
    # "12 is a stranger", "forget 12"
    verbs = "|".join(R[k].pattern for k in ("ignore_words", "unignore_words", "forget_words"))
    cmd_num = bool(re.search(r"(?:(?<!\w)(?:name|call|rename|merge|кръсти|наречи|преименувай|слей|обедини)\s+|(?:"
                             + verbs + r")\s*)#?\d{1,5}(?!\w)", t)
                   or re.search(r"(?<![\w#])\d{1,5}(?!\d)\s+(?:\w+\s+){0,3}?(?:" + verbs + ")", t))
    strong = bool(strong_nums or photos or (name_refs and action_hit) or cmd_num)
    weak_signal = bool(bare or R["all_words"].search(t) or yes_no or R["sort_words"].search(t))
    question = bool(R["question_words"].match(t)) or t.endswith("?")
    res["question"] = question
    res["gate"] = strong or (pq is not None) or (ctx["weak"] and (weak_signal or action_hit)) or bool(
        question and (name_refs or photos))
    if not res["gate"]:
        return res

    # 0. the answer to a question we asked
    if pq:
        kind = pq["kind"]
        only_yes = R["yes_words"].search(t) and not R["no_words"].search(t)
        only_no = R["no_words"].search(t) and not R["yes_words"].search(t)
        if kind == "confirm":
            if only_yes:
                res["decision"] = {**pq["decision"], "source": "confirm"}
                return res
            if only_no:
                res["decision"] = D("cancel")
                return res
        elif kind == "merge_or_rename":
            if only_yes:
                res["decision"] = D("merge", a=pq["a"], b=pq["b"], source="confirm")
                return res
            if only_no:
                res["decision"] = D("rename", person=pq["a"], name=pq["alt_name"], source="confirm")
                return res
        elif kind == "which":
            opts = pq["options"]
            m = re.match(r"\s*#?(\d+)\s*$", t)
            pick = None
            if m:
                n = int(m.group(1))
                pick = opts[n - 1] if 1 <= n <= len(opts) and n not in opts else (n if n in opts else None)
            elif name_refs:
                c = [pid for pid, _ in name_refs if pid in opts]
                pick = c[0] if len(c) == 1 else None
            if pick:
                dec = dict(pq["decision"])
                dec[pq["slot"]] = pick
                dec["source"] = "clarify"
                res["decision"] = dec
                return res
        elif kind == "person_or_item":
            if re.search(r"(?<!\w)(person|човек|лице|folder|папка)", t):
                res["decision"] = {**pq["as_person"], "source": "clarify"}
                return res
            if re.search(r"(?<!\w)(photo|picture|item|list|not sure|снимк\w*|списък\w*)", t):
                res["decision"] = {**pq["as_item"], "source": "clarify"}
                return res
        # not an answer: the question is dropped, the message is read normally

    # 1. questions: "who is in IMG_0450?" by code; others -> normal answer with the people context
    if question and photos and re.search(r"(?<!\w)(who|кой|коя|кои)(?!\w)", t):
        res["decision"] = D("who_photo", photo=photos[0][0], images=[h["image_id"] for h in photos[0][1]])
        return res
    if R["question_words"].match(t) or (t.endswith("?") and not (bare and yes_no)):
        if strong or name_refs or photos:
            res["decision"] = D("question")
        return res

    clauses_t = _clauses(t, f)
    clauses_o = _clauses(orig, f) if len(_clauses(orig, f)) == len(clauses_t) else [orig]
    if len(clauses_o) != len(clauses_t):
        clauses_t = [t]

    # pure answers to the 'not sure' list: "1 and 3 yes, 2 no" · "да, всички" · "all yes" · "yes"
    # (a verdict and its numbers may sit in neighbouring clauses)
    if maybe and not photos and not name_refs and not strong_nums and not cmd_num:
        got = _answers_only(clauses_t, R, f, maybe, ctx["weak"])
        if got == "unsure":
            return res
        if got is not None:
            res["decision"] = D("answer", ops=got) if got and got[0][0] != "unknown" else D(
                "unknown_item", numbers=got[0][1])
            return res

    ops, singles = [], []

    def person_refs(c: str):
        """Person references in a clause: [(kind 'strong'|'bare'|'name', value, span)] in order."""
        refs = []
        for m in re.finditer(r"(?:(?:" + pw + r")\s*(?:#|no\.?|№)?\s*|(?<![\w&])#)(\d{1,5})(?!\w)", c):
            refs.append(("strong", int(m.group(1)), m.span()))
        for m in re.finditer(r"(?<![\w#.\-])(\d{1,5})(?![\w.])", c):
            if not any(a <= m.start(1) < b for _, _, (a, b) in refs):
                refs.append(("bare", int(m.group(1)), m.span()))
        for pid, sp in _name_refs(c, persons):
            refs.append(("name", pid, sp))
        refs.sort(key=lambda r: r[2][0])
        return refs

    for ci, c in enumerate(clauses_t):
        co = clauses_o[ci] if ci < len(clauses_o) else c
        refs = person_refs(c)
        prefs = [r for r in refs if r[0] in ("strong", "name")]
        cphotos = _photo_refs(co, st)
        if cphotos:
            sp = [s for _, _, s in cphotos]
            refs = [r for r in refs if not any(a <= r[2][0] < b for a, b in sp)]
            prefs = [r for r in refs if r[0] in ("strong", "name")]

        def pid_of(r):
            return _resolve(r[1], persons)

        # forget / unignore / ignore
        if R["forget_words"].search(c):
            target = next((pid_of(r) for r in refs if pid_of(r)), None)
            if target:
                singles.append(D("ask_forget", person=target))
                continue
            return res
        if R["unignore_words"].search(c) and refs:
            target = next((pid_of(r) for r in refs if pid_of(r)), None)
            if target:
                singles.append(D("unignore", person=target))
                continue
        if R["ignore_words"].search(c):
            target = next((pid_of(r) for r in refs if pid_of(r)), None)
            if target:
                singles.append(D("ignore", person=target))
                continue
            return res
        # stricter / looser / suggested threshold
        if R["suggested_words"].search(c) and refs:
            target = next((pid_of(r) for r in refs if pid_of(r)), None)
            if target:
                singles.append(D("suggested", person=target))
                continue
        for word, act in (("stricter_words", "stricter"), ("looser_words", "looser")):
            if R[word].search(c):
                target = next((pid_of(r) for r in prefs if pid_of(r)), None)
                singles.append(D(act, person=target))
                break
        else:
            act = None
        if act:
            continue
        # preference: "from now on only make folders for people in at least 2 photos"
        if R["preference_words"].search(c) and refs and not prefs:
            m = re.search(r"(\d+)\s*(?:\w+\s+)?(?:photos?|pictures?|times|снимк\w*|пъти)", c)
            if m:
                singles.append(D("preference", key="min_faces_for_new_person", value=max(1, int(m.group(1)))))
                continue
        # sort them (after a check run)
        if R["sort_words"].search(c) and ctx.get("check") and ctx.get("source"):
            singles.append(D("sort"))
            continue
        # merge: "12 and 3 are the same person", "merge 12 into Maria", "12 is also Maria"
        if R["same_words"].search(c):
            ps = [pid_of(r) for r in refs if pid_of(r)]
            ps = list(dict.fromkeys(ps))
            if len(ps) == 2:
                singles.append(D("merge", a=ps[0], b=ps[1]))
                continue
            if len(ps) < 2 and refs:
                return res  # unsure
        # not: "IMG_0450 is not Maria", "remove IMG_0450 from Maria"
        if R["not_words"].search(c) and cphotos:
            target = next((pid_of(r) for r in prefs if pid_of(r)), None)
            if target:
                singles.append(D("reject_photo", person=target, photo=cphotos[0][0],
                                 images=[h["image_id"] for h in cphotos[0][1]]))
                continue
            return res
        # position: "in IMG_0450 Maria is the left one"
        pos = None
        for key, words in f["position_words"].items():
            if _rx(words).search(c):
                pos = key
        if cphotos and pos and prefs:
            target = pid_of(prefs[0])
            if target:
                singles.append(D("confirm_position", person=target, photo=cphotos[0][0], position=pos,
                                 images=[h["image_id"] for h in cphotos[0][1]]))
                continue
        # maybe answers: "1 and 3 yes", "2 no", "all yes", "да, всички", "yes"
        has_yes = bool(R["yes_words"].search(c))
        has_no = bool(R["no_words"].search(c)) and not R["not_words"].search(c)
        nums = [r[1] for r in refs if r[0] == "bare"]
        rng = [(int(a), int(b)) for a, b in re.findall(r"(?<!\w)(\d{1,4})\s*(?:-|–|to|до)\s*(\d{1,4})(?!\w)", c)]
        for a, b in rng:
            nums += list(range(min(a, b), max(a, b) + 1))
        nums = list(dict.fromkeys(nums))
        allw = bool(R["all_words"].search(c))
        for word, val in f["ordinal_words"].items():
            if _rx(val).search(c) and maybe:
                nums.append(int(word))
        rename_m = R["rename_words"].search(c)
        if (has_yes or has_no) and not rename_m and maybe and (nums or allw or ctx["weak"]):
            if has_yes and has_no and not nums:
                return res
            verdict = "yes" if has_yes and not has_no else "no"
            if nums:
                bad = [n for n in nums if n not in maybe_n]
                if bad and not any(n in maybe_n for n in nums):
                    singles.append(D("unknown_item", numbers=bad))
                    continue
                items = [n for n in nums if n in maybe_n]
            elif allw or len([m for m in maybe if not m.get("done")]) == 1:
                items = [m["n"] for m in maybe if not m.get("done")]
            elif ctx["weak"]:
                return res  # "yes" with several open items: which ones? -> unsure
            else:
                return res
            ops += [(n, verdict, None) for n in items]
            continue
        # rename / assign: "12 is Maria", "#12 = Maria", "name 12 Maria", "кръсти 12 Мария", "rename Maria to X"
        if rename_m or re.search(r"=", c):
            got = _rename(c, co, refs, persons, ctx, f, cfg, D, R)
            if got is None:
                return res
            if got["action"] == "assign_item":
                ops.append((got["item"], "assign", got["person"]))
            else:
                singles.append(got)
            continue
        if refs or cphotos:
            return res  # something about people, but no rule fits: unsure
    if ops and not singles:
        res["decision"] = D("answer", ops=ops)
    elif singles and not ops and len(singles) == 1:
        res["decision"] = singles[0]
    elif singles and not ops and all(s["action"] in ("rename",) for s in singles):
        res["decision"] = D("multi", decisions=singles)
    return res


def _answers_only(clauses: list, R: dict, f: dict, maybe: list, weak: bool):
    """[(n, 'yes'|'no', None)] when the message only answers the 'not sure' list; None = not such a message;
    'unsure' = it is, but unclear; [('unknown', [numbers])] = numbers that are not on the list."""
    by_n = {m["n"]: m for m in maybe}
    open_ = [m["n"] for m in maybe if not m.get("done")]
    other = ("rename_words", "same_words", "ignore_words", "unignore_words", "forget_words", "sort_words",
             "stricter_words", "looser_words", "suggested_words", "preference_words", "question_words")
    parsed = []
    for c in clauses:
        if any(R[k].search(c) for k in other) or re.search(r"=", c):
            return None
        yes = bool(R["yes_words"].search(c))
        no = bool(R["no_words"].search(c) or R["not_words"].search(c))
        nums = []
        for a, b in re.findall(r"(?<!\w)(\d{1,4})\s*(?:-|–|to|до)\s*(\d{1,4})(?!\w)", c):
            nums += list(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
        c2 = re.sub(r"(?<!\w)\d{1,4}\s*(?:-|–|to|до)\s*\d{1,4}(?!\w)", " ", c)
        nums += [int(x) for x in re.findall(r"(?<![\w#])(\d{1,4})(?!\w)", c2)]
        for word, val in f["ordinal_words"].items():
            if _rx(val).search(c):
                nums.append(int(word))
        allw = bool(R["all_words"].search(c))
        rest = c
        for k in ("yes_words", "no_words", "not_words", "all_words", "except_words"):
            rest = R[k].sub(" ", rest)
        rest = re.sub(r"\d+|[\s,.;:!\-–]|(?<!\w)(and|и|to|до)(?!\w)", " ", rest).strip()
        for val in f["ordinal_words"].values():
            rest = _rx(val).sub(" ", rest).strip()
        if rest:
            return None if not (yes or no or nums) else "unsure"   # other words: not just an answer
        if yes and no:
            return "unsure"
        if not (yes or no or nums or allw):
            continue
        parsed.append({"v": "yes" if yes else "no" if no else None, "nums": nums, "all": allw})
    if not parsed or not any(p["v"] for p in parsed):
        return None
    if not weak and not any(p["nums"] for p in parsed):
        return None        # "yes" / "all" long after the report: not about the list
    ops, held_v, held_t = [], None, None
    for p in parsed:
        targets = p["nums"] or (open_ if p["all"] else None)
        if p["v"] and targets:
            ops += [(n, p["v"]) for n in targets]
        elif p["v"]:
            if held_t:
                ops += [(n, p["v"]) for n in held_t]
                held_t = None
            else:
                held_v = p["v"]
        elif targets:
            if held_v:
                ops += [(n, held_v) for n in targets]
                held_v = None
            else:
                held_t = targets
    if held_t:
        return "unsure"
    if held_v:
        if len(open_) == 1:
            ops.append((open_[0], held_v))
        else:
            return "unsure"
    bad = [n for n, _ in ops if n not in by_n]
    ops = [(n, v) for n, v in ops if n in by_n]
    if not ops and bad:
        return [("unknown", bad)]
    if not ops:
        return "unsure"
    seen = {}
    for n, v in ops:
        seen[n] = v
    return [(n, v, None) for n, v in seen.items()]


def _rename(c: str, co: str, refs: list, persons: dict, ctx: dict, f: dict, cfg: dict, D, R):
    """A rename / assign clause -> decision, or None (unsure)."""
    maybe_n = {m["n"]: m for m in (ctx.get("maybe") or [])}
    T = cfg["faces"]["texts"]
    co_l = music._norm(co)
    explicit = re.search(r"(?<!\w)(rename|преименувай)(?!\w)\s+(.+?)\s+(?:to|as|на|в|→|->)\s+(.+)$", co, re.I)
    target_ref, raw_name = None, None
    if explicit:
        who_txt = music._norm(explicit.group(2))
        raw_name = explicit.group(3)
        m = re.fullmatch(r"(?:\w+\s+)?#?(\d{1,5})", who_txt)
        if m:
            target_ref = ("strong", int(m.group(1)))
        else:
            hits = _name_refs(who_txt, persons)
            if len(hits) == 1:
                target_ref = ("name", hits[0][0])
            elif len({p for p, _ in hits}) > 1:
                return None
    else:
        verb = re.search(r"(?<!\w)(name|call|кръсти|наречи)(?!\w)\s+(.+)$", co, re.I)
        if verb:
            rest = verb.group(2)
            m = re.match(r"(?:(?:" + R["person_words"].pattern + r")\s*)?#?(\d{1,5})\s+(.+)$", music._norm(rest), re.I)
            if m:
                target_ref = ("strong", int(m.group(1)))
                raw_name = rest[len(rest) - len(m.group(2)):]
        if target_ref is None:
            m = re.match(r"\s*(?:(" + R["person_words"].pattern + r")\s*(?:#|no\.?|№)?\s*|(#))?(\d{1,5})\s*"
                         r"(?:is|=|е|се казва|казва се|is called|is named)\s+(.+)$", co_l, re.I)
            if m:
                target_ref = ("strong" if (m.group(1) or m.group(2)) else "bare", int(m.group(3)))
                raw_name = co[len(co) - len(m.group(4)):]
    if target_ref is None or raw_name is None:
        return None
    strong = target_ref[0] in ("strong", "name") or ctx["weak"]
    if target_ref[0] == "bare" and not ctx["weak"]:
        return None
    # the new name: an existing person? ("2 is Ivan")
    nm_norm = music._norm(raw_name).strip(" .!,;:\"'")
    other = None
    hits = _name_refs(nm_norm, persons)
    if hits and hits[0][1][0] == 0 and hits[0][1][1] >= len(nm_norm.rstrip()) - 1:
        cands = [pid for pid, _ in hits]
        other = cands[0]
    elif nm_norm.isdigit() or re.fullmatch(r"#\d+", nm_norm):
        other = _resolve(int(nm_norm.lstrip("#")), persons)
    name, ok = _clean_name(raw_name, f, cfg, strong)
    n = target_ref[1]
    if target_ref[0] == "name":
        pid = _resolve(n, persons)
        if not pid:
            return None
        if not ok:
            return None
        if other and other != pid:
            return D("ask_merge", a=pid, b=other, alt_name=name)
        return D("rename", person=pid, name=name)
    is_person = _resolve(n, persons) is not None
    is_item = n in maybe_n and target_ref[0] == "bare"
    if is_item and other:
        if is_person and n in (ctx.get("new_people") or []):
            pid = _resolve(n, persons)
            as_person = D("ask_merge", a=pid, b=other, alt_name=persons[other]["name"] + " 2") if other != pid else None
            return D("ask_person_or_item", n=n, as_item=D("answer", ops=[(n, "assign", other)]),
                     as_person=as_person or D("cancel"))
        return {"action": "assign_item", "item": n, "person": other}
    if not is_person:
        if is_item and not other:
            return D("clarify", ask=T["ask_default"])
        return D("unknown_person", number=n)
    pid = _resolve(n, persons)
    if other and other != pid:
        return D("ask_merge", a=pid, b=other, alt_name=f"{persons[other]['name']} 2")
    if other == pid:
        return D("noop", person=pid)
    if not ok:
        return None
    # the same name as someone else (not found as a name ref because of different case / alias)
    same = face_sync.name_index(persons.values()).get(face_sync.norm_name(name)) or []
    same = [q for q in same if q != pid and persons[q]["status"] in ("active", "ignored")]
    if same:
        return D("ask_merge", a=pid, b=same[0], alt_name=f"{name} 2")
    return D("rename", person=pid, name=name)


# ---------------- the chat model (only when the rules are unsure) ----------------

def _llm_resolve(message: str, ctx: dict, llm, cfg: dict, spin, history):
    fcfg = cfg["faces"]
    f, T = fcfg["followup"], fcfg["texts"]
    persons = ctx["persons"]
    shown = [p for p in persons.values() if p["status"] in ("active", "ignored")][: f["max_people_in_prompt"]]
    people = "\n".join(f"{p['id']}: {p['name']}" + (f" [{', '.join(p['aliases'])}]" if p.get("aliases") else "")
                       for p in shown) or "-"
    maybe = "\n".join(f"{m['n']}: {m['photo']} - {faces._cands_plain(m['candidates'], T)}"
                      for m in (ctx.get("maybe") or [])[: f["max_maybe_in_prompt"]] if not m.get("done")) or "-"
    opts = ""
    if ctx.get("pending") and ctx["pending"].get("options"):
        opts = T["llm_options"].format(options="  ".join(f"{i}) #{pid} {persons[pid]['name']}"
                                                         for i, pid in enumerate(ctx["pending"]["options"], 1)
                                                         if pid in persons))
    recent = [m for m in (history or []) if m.get("role") in ("user", "assistant")][-2:]
    recent_text = "\n".join(f"{m['role']}: {' '.join(m['content'].split())[:300]}" for m in recent) or "-"
    prompt = text(fcfg["prompts"]["followup_resolver"]).format(people=people, maybe=maybe, clarify_options=opts,
                                                               recent=recent_text, message=message)
    spin.status("faces_understanding")
    t = time.monotonic()
    try:
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                         **params(f["generation"]))["choices"][0]["message"]["content"]
    finally:
        spin.stop()
    dec = parse_llm(out, ctx, cfg)
    log("faces_followup", text=message, stage="llm", decision=dec, model_output=out, prompt=clip(prompt, "prompt"),
        seconds=round(time.monotonic() - t, 2))
    return dec


def parse_llm(out: str, ctx: dict, cfg: dict):
    """The model's labeled lines -> a validated decision, or None (not about the faces / unreadable)."""
    fcfg = cfg["faces"]
    T = fcfg["texts"]
    persons = ctx["persons"]
    maybe_n = {m["n"]: m for m in (ctx.get("maybe") or [])}

    def field(name):
        m = re.search(rf"^\W*{name}\s*[:\-]\s*(.+)$", out or "", re.I | re.M)
        v = m.group(1).strip().strip("\"'`*").strip() if m else None
        return None if v is None or re.fullmatch(r"(none|-|n/a|null)", v, re.I) else v

    about, action = field("ABOUT_FACES"), field("ACTION")
    if not about or not action or not re.match(r"(yes|no)\b", about, re.I):
        return None
    act = action.lower().split()[0].strip(".,")
    if about.lower().startswith("no") or act == "none":
        return {"action": "question", "source": "llm"} if about.lower().startswith("yes") else None
    known = ("rename", "merge", "confirm", "reject", "assign", "ignore", "forget", "stricter", "looser", "ask")
    if act not in known:
        return None

    def person(v):
        if not v:
            return None
        m = re.fullmatch(r"#?(\d+)", v.strip())
        if m:
            return _resolve(int(m.group(1)), persons)           # numbers not in the list are dropped
        hits = face_sync.name_index(persons.values()).get(face_sync.norm_name(v)) or []
        return hits[0] if len(hits) == 1 else None

    P, O = person(field("PERSON")), person(field("OTHER"))
    items_v = field("ITEMS") or ""
    items = []
    if re.search(r"(?<!\w)all(?!\w)", items_v, re.I):
        items = [n for n, m in maybe_n.items() if not m.get("done")]
    else:
        items = [int(x) for x in re.findall(r"\d+", items_v) if int(x) in maybe_n]
    base = {"source": "llm"}
    ask = {"action": "clarify", "ask": field("QUESTION") or T["ask_default"], **base}
    if act == "ask":
        return ask
    if act == "rename":
        name, ok = _clean_name(field("NAME") or "", fcfg["followup"], cfg, strong=True)
        if not P or not ok:
            return ask
        return {"action": "rename", "person": P, "name": name, **base}
    if act == "merge":
        if not P or not O or P == O:
            return ask
        return {"action": "merge", "a": P, "b": O, **base}
    if act in ("confirm", "reject"):
        if not items:
            return ask
        return {"action": "answer", "ops": [(n, "yes" if act == "confirm" else "no", None) for n in items], **base}
    if act == "assign":
        if not items or not O:
            return ask
        return {"action": "answer", "ops": [(n, "assign", O) for n in items], **base}
    if act in ("ignore", "forget"):
        if not P:
            return ask
        return {"action": "ignore" if act == "ignore" else "ask_forget", "person": P, **base}
    if act in ("stricter", "looser"):
        return {"action": act, "person": P, **base}
    return None


# ---------------- interpret ----------------

def interpret(message: str, llm, cfg: dict, spin, history=None):
    """Decision for a chat message about the faces library, or None (not about it -> normal chat)."""
    global PENDING_Q
    fcfg = cfg["faces"]
    f = fcfg["followup"]
    if not f["enabled"] or not fcfg["enabled"]:
        return None
    msg = message.strip()
    if not msg or msg.startswith("/"):
        return None
    t0 = time.monotonic()
    pq, PENDING_Q = PENDING_Q, None   # a question counts only for the very next message
    if CONTEXT:
        CONTEXT["turns_seen"] += 1
    if re.search(r"https?://", msg, re.I) or faces.parse_request(msg, fcfg):
        log("faces_followup", text=msg, stage="pre", decision=None)
        return None
    snap = _cached_snapshot(fcfg)
    if snap is None:
        return None
    persons, st = snap
    try:
        if not persons and not CONTEXT and not pq:
            return None
        weak = bool(CONTEXT and CONTEXT["turns_seen"] <= f["weak_reference_turns"])
        ctx = {"persons": persons, "store": st, "weak": weak, "pending": pq,
               "maybe": _current_maybe(),
               "check": (CONTEXT or {}).get("check"), "source": (CONTEXT or {}).get("source"),
               "new_people": (CONTEXT or {}).get("new_people") or []}
        a = analyze(msg, ctx, cfg)
        if not a["gate"]:
            log("faces_followup", text=msg, stage="gate", decision=None, seconds=round(time.monotonic() - t0, 3))
            return None
        if a["decision"]:
            log("faces_followup", text=msg, stage="rules", decision=a["decision"], seconds=round(time.monotonic() - t0, 3))
            return a["decision"]
        if a["question"] or not f["use_llm"] or llm is None:
            log("faces_followup", text=msg, stage="rules", decision=None, unsure=True)
            return {"action": "question", "source": "rules"} if a["question"] else None
        return _llm_resolve(msg, ctx, llm, cfg, spin, history)
    finally:
        st.close()


# ---------------- doing it ----------------

def execute(dec: dict, cfg: dict, spin, llm=None) -> str:
    """Run a decision. Prints its own lines; returns the plain report for the chat history."""
    global PENDING_Q
    fcfg = cfg["faces"]
    T, f = fcfg["texts"], fcfg["followup"]
    act = dec["action"]
    if act == "question":
        return ""
    if act == "sort":
        src = (CONTEXT or {}).get("source")
        opts = dict((CONTEXT or {}).get("opts") or {})
        opts["check"] = False
        res = faces.run(src, opts or faces.parse_options("", fcfg), cfg, spin)
        spin.stop()
        hint = open_context(res, cfg)
        print("\n" + faces.report(res, fcfg, color=True) + ("\n" + hint if hint else "") + "\n")
        return faces.history(res, fcfg)
    if act == "preference":
        face_sync.SESSION[dec["key"]] = dec["value"]
        msg = T["preference_set"].format(what=T["preference_what"][dec["key"]].format(value=dec["value"]),
                                         key=f"matching.{dec['key']}")
        print(msg + "\n")
        log("setting_changed", key=f"faces.matching.{dec['key']}", value=dec["value"], by="faces_followup")
        return msg
    lib = face_sync.Lib(cfg, spin)
    out, st = lib.out, lib.store
    try:
        persons = {p["id"]: p for p in st.persons()}

        def name(pid):
            return persons[pid]["name"] if pid in persons else f"#{pid}"

        if act == "cancel":
            out.line(T["cancelled"])
        elif act == "clarify":
            out.line(dec.get("ask") or T["ask_default"], "maybe")
        elif act == "unknown_person":
            out.line(T["unknown_person"].format(who=f"#{dec['number']}"), "error")
        elif act == "unknown_item":
            out.line(T["unknown_item"].format(numbers=", ".join(map(str, dec["numbers"]))), "error")
        elif act == "noop":
            out.line(T["already_named"].format(number=dec["person"], name=name(dec["person"])))
        elif act == "who_photo":
            for image_id in dec["images"][:1]:
                rows = st.faces(image_id=image_id, with_embedding=False)
                res = {"mode": "who", "photo": dec["photo"], "faces": [], "error": None}
                for r in rows:
                    res["faces"].append({"status": r["status"], "person_id": r["person_id"], "score": r["similarity"],
                                         "quality": r["quality"], "name": name(r["person_id"]) if r["person_id"] else None,
                                         "candidate_names": [{"number": q, "name": name(q), "score": round(s, 2)}
                                                             for q, s in r["candidates"] if q in persons]})
                print(faces.who_report(res, fcfg, color=True))
                out.lines.append(faces.who_report(res, fcfg, color=False))
        elif act == "rename":
            p = persons[dec["person"]]
            old = p["name"]
            p2 = face_sync.rename_person(lib, p, dec["name"])
            st.commit()
            out.line(T["renamed"].format(number=p["id"], old=old, name=p2["name"]), "known")
            out.path(lib.folder(p2))
        elif act == "multi":
            for d in dec["decisions"]:
                out.lines.append(execute(d, cfg, spin))
        elif act == "ask_merge":
            a, b = dec["a"], dec["b"]
            q = T["ask_merge"].format(name=name(b), number=b, photos=st.photo_count(b), other=a)
            q = q.replace(f'"{name(b)} 2"', f'"{dec.get("alt_name") or name(b) + " 2"}"')
            PENDING_Q = {"kind": "merge_or_rename", "a": a, "b": b, "alt_name": dec.get("alt_name") or f"{name(b)} 2"}
            out.line(q, "maybe")
        elif act == "ask_which":
            opts = [pid for pid in dec["options"] if pid in persons]
            PENDING_Q = {"kind": "which", "options": opts, "decision": dec["decision"], "slot": dec["slot"]}
            out.line(T["ask_which"].format(options="  ".join(
                f"{i}) #{pid} {persons[pid]['folder'] or name(pid)}" for i, pid in enumerate(opts, 1))), "maybe")
        elif act == "ask_person_or_item":
            PENDING_Q = {"kind": "person_or_item", "as_person": dec["as_person"], "as_item": dec["as_item"]}
            out.line(T["ask_person_or_item"].format(n=dec["n"]), "maybe")
        elif act == "merge":
            a, b = dec["a"], dec["b"]
            big = f["confirm_merge_over_photos"]
            if dec.get("source") != "confirm" and st.photo_count(a) >= big and st.photo_count(b) >= big:
                q = T["ask_merge_big"].format(a=a, a_name=name(a), a_photos=st.photo_count(a), b=b, b_name=name(b),
                                              b_photos=st.photo_count(b))
                PENDING_Q = {"kind": "confirm", "decision": {**dec, "source": "confirm"}}
                out.line(q, "maybe")
            else:
                B = face_sync.merge_person(lib, a, b)
                out.line(T["merged"].format(from_number=a, from_name=name(a), number=b, name=B["name"],
                                            photos=st.photo_count(b)), "known")
                out.path(lib.folder(B))
        elif act == "ignore":
            face_sync.ignore_person(lib, dec["person"])
            out.line(T["ignored"].format(number=dec["person"], name=name(dec["person"])), "known")
        elif act == "unignore":
            p2 = face_sync.unignore_person(lib, dec["person"])
            out.line(T["unignored"].format(number=dec["person"], name=p2["name"], photos=st.photo_count(dec["person"])),
                     "known")
        elif act == "ask_forget":
            pid = dec["person"]
            q = T["ask_forget"].format(number=pid, name=name(pid), photos=st.photo_count(pid))
            PENDING_Q = {"kind": "confirm", "decision": {"action": "forget", "person": pid, "source": "confirm"}}
            out.line(q, "maybe")
        elif act == "forget":
            pid = dec["person"]
            nm = name(pid)
            face_sync.forget_person(lib, pid)
            out.line(T["forgotten"].format(number=pid, name=nm), "known")
        elif act == "answer":
            maybe = _current_maybe()
            if not maybe and CONTEXT is None:
                maybe = faces.open_maybe(lib)
            by_n = {m["n"]: m for m in maybe}
            answers = [(by_n[n]["face_id"], v, pid) for n, v, pid in dec["ops"] if n in by_n]
            if not answers:
                out.line(T["nothing_open"])
            else:
                faces.answer_maybe(lib, answers, "chat")
            left = [m["n"] for m in maybe if not m.get("done")]
            if left:
                out.line(T["still_open"].format(numbers=", ".join(map(str, left[:20]))), "maybe")
            if CONTEXT:
                CONTEXT["turns_seen"] = 0
        elif act in ("reject_photo", "confirm_position"):
            pid = dec["person"]
            done = False
            for image_id in dec["images"]:
                rows = st.faces(image_id=image_id, with_embedding=False)
                if act == "reject_photo":
                    hit = [r for r in rows if r["person_id"] == pid and r["status"] in ("auto", "confirmed", "maybe")]
                    for r in hit:
                        face_sync.reject_face(lib, r["id"], pid, "chat", dec["photo"])
                        done = True
                    if hit:
                        r2 = face_sync.reevaluate(lib, [pid], include_ids=[r["id"] for r in hit])
                        face_sync.sync_disk(lib, {pid} | r2["touched"])
                        face_sync.update_learned(lib, {pid} | r2["touched"])
                        photo = face_sync._photo_name(lib, image_id)
                        out.line(T["rejected"].format(photo=photo, name=name(pid)), "known")
                        after = face_sync.describe_changes(lib, r2["changes"])
                        if after:
                            out.line(T["after_changes"].format(changes=after), "known")
                else:
                    ok = [r for r in rows if r["quality"] != "poor"] or rows
                    pos = dec["position"]
                    idx = {"left": 0, "right": len(ok) - 1, "middle": len(ok) // 2}[pos] if ok else None
                    if ok and not (pos == "middle" and len(ok) % 2 == 0):
                        face = sorted(ok, key=lambda r: r["x"] + r["w"] / 2)[idx]
                        faces.answer_maybe(lib, [(face["id"], "assign", pid)], "chat")
                        done = True
                if done:
                    break
            if not done:
                out.line(T["photo_not_theirs"].format(photo=dec["photo"], name=name(pid)), "error")
        elif act in ("stricter", "looser", "suggested"):
            pid = dec.get("person")
            step = fcfg["matching"]["strictness_step"] * (1 if act == "stricter" else -1)
            if pid is None and act != "suggested":
                cap = fcfg["matching"]["strictness_max"]
                old = face_sync.SESSION["strictness"]
                face_sync.SESSION["strictness"] = round(max(-cap, min(cap, old + step)), 3)
                out.line(T["session_strictness"].format(value=f"{face_sync.SESSION['strictness']:+.2f}"), "known")
            else:
                p = persons[pid]
                s = dict(p["settings"] or {})
                default = fcfg["matching"][lib.embedder]["match_threshold"]
                old = s.get("match_threshold")
                if act == "suggested":
                    new = (p.get("learned") or {}).get("suggested_threshold")
                    if new is None:
                        out.line(T["no_suggestion"].format(name=p["name"], n=fcfg["matching"]["suggest_min_confirmed"]),
                                 "maybe")
                        return "\n".join(out.lines)
                else:
                    lo, hi = fcfg["matching"]["threshold_range"]
                    new = round(min(hi, max(lo, (old if old is not None else default) + step)), 3)
                s["match_threshold"] = new
                st.update_person(pid, settings=s)
                st.commit()
                lib.dirty()
                face_sync.write_person_file(lib, st.person(pid))
                out.line(T["threshold_set"].format(name=p["name"], old=old if old is not None else f"{default} (global)",
                                                   new=new), "known")
                r = face_sync.reevaluate(lib, [pid])
                face_sync.sync_disk(lib, {pid} | r["touched"])
                face_sync.update_learned(lib, {pid} | r["touched"])
                after = face_sync.describe_changes(lib, r["changes"])
                if after:
                    out.line(T["after_changes"].format(changes=after), "known")
        else:
            out.line(T["ask_default"], "maybe")
        st.commit()
        log("faces_followup", stage="execute", decision=dec, lines=out.lines)
        return "\n".join(out.lines)
    finally:
        spin.stop()
        print()
        lib.close()


# ---------------- keeping the chat model informed and honest ----------------

def prompt_extra(cfg: dict, message: str, question: bool = False) -> str:
    """Added to the system prompt of a normal answer (never saved in the history)."""
    fcfg = cfg["faces"]
    if not fcfg["enabled"]:
        return ""
    snap = _snapshot(fcfg)
    if snap is None:
        return ""
    persons, st = snap
    try:
        active = [p for p in persons.values() if p["status"] in ("active", "ignored")]
        if not active:
            return ""
        extra = text(fcfg["prompts"]["answer_rule"])
        t = music._norm(message)
        mentioned = [pid for pid, _ in _name_refs(t, persons)]
        mentioned += [face_sync.resolve_ref(int(n), persons) for n in re.findall(r"#(\d+)", t)]
        mentioned = [m for m in dict.fromkeys(mentioned) if m]
        about = bool(question or mentioned or _photo_refs(message, st)
                     or any(re.search(p, message, re.I) for p in fcfg["intent_patterns"]))
        if about:
            cap = fcfg["followup"]["max_people_in_prompt"]
            order = [persons[m] for m in mentioned if m in persons] + [p for p in active if p["id"] not in mentioned]
            lib_dir = face_sync.library_dir(fcfg) / fcfg["people_folder"]
            rows = []
            for p in order[:cap]:
                al = f" (also: {', '.join(p['aliases'])})" if p.get("aliases") else ""
                notes = f" - notes: {p['notes']}" if p.get("notes") else ""
                where = (lib_dir / (p.get("folder") or face_sync.safe_name(p["name"]))) if p["status"] == "active" else "-"
                rows.append(f"#{p['id']} {p['name']}{al}: {st.photo_count(p['id'])} photos"
                            + (" (ignored - a stranger)" if p["status"] == "ignored" else f", folder {where}") + notes)
            extra += text(fcfg["prompts"]["people_context"]).format(people_list="\n".join(rows), library=lib_dir)
        return extra
    finally:
        st.close()
