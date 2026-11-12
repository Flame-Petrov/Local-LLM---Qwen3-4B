"""
Search planner: turns the user's message into short, keyword-focused search queries
(SEO-style) instead of pasting the raw message into the search engine.
Prompt, rules and filler patterns come from config.json -> "prompts.query_planner" and "planner".
"""

import re
import time
from datetime import date

from core.config import params, text
from core.debuglog import clip, log
from web.prices import detect_region


def clean_query(message: str, pcfg: dict) -> str:
    """Rule-based fallback: strip filler like 'can you do a google search for'."""
    q = message
    for pattern in pcfg["filler_patterns"]:
        q = re.sub(pattern, " ", q, flags=re.I)
    q = " ".join(q.split()).strip(" ?.!,;:")
    return q or message.strip()


def _parse(out: str, pcfg: dict) -> list[str]:
    label = re.escape(pcfg["label"])
    queries = []
    for line in out.splitlines():
        m = re.match(rf"^\W*{label}\s*\d*\s*[:\-]\s*(.+)$", line.strip(), re.I)
        if not m:
            continue
        q = m.group(1).strip().strip('"\'`*').strip()
        if q and q.lower() not in (x.lower() for x in queries):
            queries.append(q)
    return queries[: pcfg["max_queries"]]


def _parse_product(out: str, pcfg: dict):
    label = re.escape(pcfg["product_label"])
    for line in out.splitlines():
        m = re.match(rf"^\W*{label}\s*[:\-]\s*(.+)$", line.strip(), re.I)
        if m:
            product = m.group(1).strip().strip('"\'`*').strip()
            if product and product.lower() not in ("none", "n/a", "-"):
                return product
    return None


def plan_queries(llm, question: str, history: list, cfg: dict, spin):
    """Return (queries, product): 1..planner.max_queries search queries for `question`,
    and the product name the user is asking about (None if not a product question).
    `history` = earlier chat messages (used to resolve follow-ups like 'how old is he?')."""
    pcfg = cfg["planner"]
    fallback = [clean_query(question, pcfg)]
    if not pcfg["enabled"]:
        return fallback, None

    t = time.monotonic()
    recent = [m for m in history if m.get("role") in ("user", "assistant")][-pcfg["history_messages"]:]
    convo = "\n".join(f"{m['role']}: {' '.join(m['content'].split())[: pcfg['history_chars']]}"
                      for m in recent) or pcfg["no_history_text"]
    prompt = text(cfg["prompts"]["query_planner"]).format(
        today=date.today().strftime(cfg["date_format"]), year=date.today().year,
        question=question, history=convo, n=pcfg["max_queries"], label=pcfg["label"],
        product_label=pcfg["product_label"],
        region=detect_region(question, cfg["price"]) or cfg["price"]["no_region_text"])

    spin.status("planning")
    out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                     **params(pcfg["generation"]))["choices"][0]["message"]["content"]
    queries = _parse(out, pcfg)
    product = _parse_product(out, pcfg)
    log("query_plan", question=question, history_used=len(recent), prompt=clip(prompt, "prompt"),
        model_output=out, queries=queries or fallback, product=product, fallback_used=not queries,
        seconds=round(time.monotonic() - t, 1))
    return queries or fallback, product
