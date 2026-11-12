"""
Public profile check: "check opssdamnit ig profile", "@name on instagram", "instagram.com/name".

Reads ONLY what any logged-out visitor sees in the profile page's public summary
(og:title / og:description): display name, followers, following, posts. No login, no
posts or photos, no searching other sites. The answer is built by code from that
page - the model never writes the numbers, so it cannot invent them.
Also: if the chat recently mentioned a SIMILAR handle (a likely typo: oopsdamnit vs
opssdamnit), that one is checked too and the difference is pointed out.
All patterns / texts: config.json -> "profiles".
"""

import html as html_lib
import re
import time
from datetime import date

import requests

from core.debuglog import log


def _platform_for(word: str, pcfg: dict):
    w = word.lower()
    for name, p in pcfg["platforms"].items():
        if name.startswith("_"):
            continue
        if w in p["aliases"]:
            return name
    return None


def known_handles(history: list, pcfg: dict) -> dict:
    """Handles the program already checked in this chat: {handle_lower: (platform, handle)}."""
    out = {}
    for m in history:
        c = m.get("content", "")
        if pcfg["texts"]["history_prefix"] not in c:
            continue
        for name, p in pcfg["platforms"].items():
            if name.startswith("_"):
                continue
            for h in re.findall(rf"{re.escape(p['label'])} @([\w.]+) exists", c):
                out[h.lower()] = (name, h)
    return out


def parse_request(message: str, pcfg: dict, known: dict | None = None):
    """(platform, handle) if the message asks to check a social profile, else None.
    known = handles already checked in this chat: a follow-up about one of them
    ('tell me more about the nat.geo account') is answered by the checker, not the model."""
    text = message.strip()
    if known and not any(re.search(x, text, re.I) for x in pcfg["comparison_patterns"]):
        # (comparisons / calculations go to the model: it has the numbers + a calculator)
        for w in re.findall(r"@?[\w.]+", text):
            k = w.strip("@.").lower()
            if k in known and any(re.search(x, text, re.I) for x in
                                  pcfg["intent_patterns"] + pcfg["followup_patterns"]):
                return known[k]
    for name, p in pcfg["platforms"].items():
        if name.startswith("_"):
            continue
        m = re.search(p["url_pattern"], text, re.I)  # instagram.com/handle
        if m and m.group(1).lower() not in p["reserved_paths"]:
            return name, m.group(1)
    strong = any(re.search(x, text, re.I) for x in pcfg["intent_patterns"])
    weak = any(re.search(x, text, re.I) for x in pcfg["weak_intent_patterns"])
    if not strong and not weak:
        return None
    words = re.findall(r"@?[\w.]+", text)
    if not strong:  # only "followers"-type words: need an explicit @name or "of / does <name>"
        plat = next((_platform_for(w.strip("@."), pcfg) for w in words if _platform_for(w.strip("@."), pcfg)), None)
        if not plat:
            return None
        at = next((x[1:].strip(".") for x in words if x.startswith("@") and len(x) > 1), None)
        if at:
            return plat, at
        filler = {f.lower() for f in pcfg["filler_words"]}
        for rx in pcfg["weak_handle_patterns"]:
            m = re.search(rx, text, re.I)
            if m and m.group(1).lower() not in filler and not _platform_for(m.group(1), pcfg):
                return plat, m.group(1).strip(".")
        return None
    for i, w in enumerate(words):
        plat = _platform_for(w.strip("@."), pcfg)
        if not plat:
            continue
        # handle = an @word anywhere, else the nearest non-filler word before / after the platform word
        at = next((x[1:] for x in words if x.startswith("@") and len(x) > 1), None)
        if at:
            return plat, at.strip(".")
        filler = {f.lower() for f in pcfg["filler_words"]}
        for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(words))):
            cand = words[j].strip("@.")
            if cand and cand.lower() not in filler and not _platform_for(cand, pcfg) \
                    and re.fullmatch(pcfg["handle_pattern"], cand):
                return plat, cand
    return None


def _num(s: str):
    s = s.replace(",", "").strip().upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1:], 1)
    try:
        return int(float(s.rstrip("KMB")) * mult)
    except ValueError:
        return None


def fetch_profile(platform: str, handle: str, cfg: dict) -> dict:
    pcfg = cfg["profiles"]["platforms"][platform]
    url = pcfg["profile_url"].format(handle=handle)
    out = {"platform": platform, "handle": handle, "url": url, "exists": None, "name": None,
           "followers": None, "following": None, "posts": None, "raw": None, "error": None}
    t = time.monotonic()
    try:
        r = requests.get(url, headers={"User-Agent": cfg["search"]["user_agent"]},
                         timeout=cfg["profiles"]["timeout_s"])
    except Exception as e:
        out["error"] = repr(e)
        log("profile_check", **out)
        return out
    if r.status_code == 404:
        out["exists"] = False
    elif r.status_code >= 400:
        out["error"] = f"HTTP {r.status_code}"
    else:
        page = r.text
        desc = re.search(r'property="og:description"[^>]*content="([^"]*)"', page) or \
            re.search(r'content="([^"]*)"[^>]*property="og:description"', page)
        title = re.search(r'property="og:title"[^>]*content="([^"]*)"', page) or \
            re.search(r'content="([^"]*)"[^>]*property="og:title"', page)
        if not desc:
            out["exists"] = False  # no public summary: no such profile (or not shown logged-out)
        else:
            d = html_lib.unescape(desc.group(1))
            out["raw"] = d
            m = re.search(pcfg["counts_pattern"], d, re.I)
            if m:
                out["followers"], out["following"], out["posts"] = (_num(m.group(k)) for k in ("followers", "following", "posts"))
                out["counts_text"] = {k: m.group(k) for k in ("followers", "following", "posts")}
            tt = html_lib.unescape(title.group(1)) if title else ""
            nm = re.search(pcfg["name_pattern"], tt)
            out["name"] = (nm.group(1).strip() if nm else "") or None
            if out["name"] and out["name"].lstrip("@").lower() == handle.lower():
                out["name"] = None  # no display name set - the title just repeats the handle
            found_handle = re.search(r"@([\w.]+)", tt + " " + d)
            out["exists"] = not found_handle or found_handle.group(1).lower() == handle.lower()
    log("profile_check", seconds=round(time.monotonic() - t, 2), **{k: v for k, v in out.items() if k != "raw"})
    return out


def _distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similar_handles(handle: str, history: list, pcfg: dict) -> list[str]:
    """Handles mentioned in the recent chat that differ by a typo or two (likely what was meant)."""
    seen = []
    for m in history[-pcfg["history_messages"]:]:
        for h in re.findall(r"@([\w.]{3,30})", m.get("content", "")):
            h = h.strip(".")
            if h.lower() != handle.lower() and 0 < _distance(h, handle) <= pcfg["typo_distance"] and h not in seen:
                seen.append(h)
    return seen[: pcfg["max_similar"]]


def check(platform: str, handle: str, history: list, cfg: dict, progress=None, question: str = "") -> str:
    """Check the profile (and likely-meant similar handles) -> answer text built by code.
    question: the user's message - asked things the public page cannot show (verified badge,
    bio, posts) are answered honestly as 'not visible' instead of being left to the model."""
    P = cfg["profiles"]
    T = P["texts"]
    progress = progress or (lambda *a, **k: None)
    progress("profile", platform=P["platforms"][platform]["label"], handle=handle)
    main = fetch_profile(platform, handle, cfg)
    others = []
    for h in similar_handles(handle, history, P):
        progress("profile", platform=P["platforms"][platform]["label"], handle=h)
        others.append(fetch_profile(platform, h, cfg))

    label = P["platforms"][platform]["label"]

    def block(p):
        if p["error"]:
            return T["error"].format(platform=label, handle=p["handle"], error=p["error"])
        if not p["exists"]:
            return T["not_found"].format(platform=label, handle=p["handle"], url=p["url"])
        lines = [T["found"].format(platform=label, handle=p["handle"], url=p["url"])]
        if p["name"]:
            lines.append(T["name"].format(name=p["name"]))
        if p.get("counts_text"):
            lines.append(T["counts"].format(**p["counts_text"]))
        return "\n".join(lines)

    parts = [block(main)]
    for key, patterns in P["not_visible"].items():  # e.g. "is it verified?" -> can't be seen logged-out
        if key.startswith("_"):
            continue
        if any(re.search(p, question, re.I) for p in patterns):
            parts.append(T[f"not_visible_{key}"])
    for o in others:
        parts += ["", T["similar_intro"].format(other=o["handle"], handle=handle), block(o)]
    parts += ["", T["footer"].format(today=date.today().strftime(cfg["date_format"]))]
    return "\n".join(parts)
