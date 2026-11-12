"""
One-time page audit for price reading.

    .venv\\Scripts\\python.exe scripts/audit.py

For each query in config.json -> audit.queries it searches the web, opens the shop
pages, finds every element that shows a price (HTML names + the label next to it),
and asks the local model what each one means - or to say UNKNOWN / UNSURE.
Everything is written to audit_report.json, which is used to build page_dict.json
(the dictionary the bot reads to understand price elements).
"""

import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from core import config
from web import prices
from core.config import params, text
from web.search import domain, web_search


def fetch(url, scfg, pcfg):
    """(html, problem) - problem is None when the page is usable."""
    try:
        r = requests.get(url, headers={"User-Agent": scfg["user_agent"]}, timeout=scfg["page_timeout_s"])
    except Exception as e:
        return None, f"error {e!r}"
    if r.status_code >= 400:
        return None, f"HTTP {r.status_code}"
    if prices.looks_blocked(r.text, pcfg) and not prices.extract_offers(r.text, pcfg):
        return None, "blocked (robot check)"
    return r.text, None


def parse_model(out: str, n_items: int):
    rows, unknown_words = {}, []
    for line in out.splitlines():
        line = line.strip().lstrip("-*• ")
        m = re.match(r"^\[?(\d+)\]?\s*[|.:)]\s*(.*?)\s*\|\s*(SURE|UNSURE|UNKNOWN)\b", line, re.I)
        if m and 1 <= int(m.group(1)) <= n_items:
            rows[int(m.group(1))] = {"meaning": m.group(2).strip(), "status": m.group(3).upper()}
            continue
        m = re.match(r"^UNKNOWN WORD:\s*(.*?)\s*(?:\|\s*(.*))?$", line, re.I)
        if m and m.group(1):
            unknown_words.append({"word": m.group(1).strip(), "where": (m.group(2) or "").strip()})
    return rows, unknown_words


def main():
    cfg = config.load()
    acfg, pcfg, ecfg, scfg = cfg["audit"], cfg["price"], cfg["elements"], cfg["search"]

    from llama_cpp import Llama

    m = cfg["model"]
    print("Loading model...", flush=True)
    llm = Llama(model_path=str(config.path(m["models_dir"]) / m["file"]), n_ctx=m["context_tokens"],
                n_threads=m["threads"], n_batch=m["batch_size"], verbose=False)

    # ---- 1. collect pages ----
    urls, seen = [], set()
    for q in acfg["queries"]:
        rows = web_search(q, acfg["results_per_query"], scfg)
        print(f"search {q!r}: {len(rows)} results", flush=True)
        for r in rows:
            if r["url"] not in seen and not any(re.search(p, r["url"], re.I) for p in scfg["ad_url_patterns"]):
                seen.add(r["url"])
                urls.append((q, r["url"]))
        time.sleep(1)

    pages, skipped = [], []
    for q, url in urls:
        if len(pages) >= acfg["max_pages"]:
            break
        page_html, problem = fetch(url, scfg, pcfg)
        if problem:
            skipped.append({"url": url, "query": q, "reason": problem})
            print(f"  skip {domain(url):30} {problem}", flush=True)
            continue
        items = prices.price_elements(page_html, pcfg, ecfg)[: acfg["max_items_per_page"]]
        if not items:
            skipped.append({"url": url, "query": q, "reason": "no price elements found"})
            print(f"  skip {domain(url):30} no price elements", flush=True)
            continue
        pages.append({"url": url, "query": q, "items": items,
                      "offers": prices.extract_offers(page_html, pcfg)})
        print(f"  page {domain(url):30} {len(items)} price elements", flush=True)

    # ---- 2. ask the model what it does / doesn't understand ----
    for i, page in enumerate(pages, 1):
        listing = "\n".join(
            f"{n}. price \"{it['price_text']}\" | element names: {', '.join(it['names']) or '(none)'} | label: {it['label']}"
            for n, it in enumerate(page["items"], 1))
        prompt = text(cfg["prompts"]["page_audit"]).format(url=page["url"], items=listing)
        t = time.monotonic()
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                         **params(acfg["generation"]))["choices"][0]["message"]["content"]
        rows, words = parse_model(out, len(page["items"]))
        for n, it in enumerate(page["items"], 1):
            it.update(rows.get(n, {"meaning": None, "status": "NO ANSWER"}))
        page["unknown_words"] = words
        page["model_output"] = out
        n_unk = sum(it["status"] != "SURE" for it in page["items"])
        print(f"[{i}/{len(pages)}] {domain(page['url']):30} {n_unk} unclear items, "
              f"{len(words)} unknown words  ({time.monotonic() - t:.0f}s)", flush=True)

    # ---- 3. aggregate: what the model did not understand, across all pages ----
    names, labels = defaultdict(lambda: {"count": 0, "statuses": [], "meanings": [], "examples": []}), \
        defaultdict(lambda: {"count": 0, "statuses": [], "meanings": [], "examples": []})
    for page in pages:
        for it in page["items"]:
            ex = {"site": domain(page["url"]), "price": it["price_text"], "label": it["label"]}
            for name in it["names"]:
                d = names[name.lower()]
                d["count"] += 1
                d["statuses"].append(it["status"])
                d["meanings"].append(it["meaning"])
                if len(d["examples"]) < 3:
                    d["examples"].append(ex)
            words = re.sub(r"[#\d.,:;()\[\]/%+*-]+", " ", it["label"]).split()
            for w in {w.lower() for w in words if len(w) > 1}:
                d = labels[w]
                d["count"] += 1
                d["statuses"].append(it["status"])
                if len(d["examples"]) < 2:
                    d["examples"].append(ex)

    def unclear(d):
        return sum(s != "SURE" for s in d["statuses"])

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "queries": acfg["queries"],
        "pages_read": len(pages),
        "pages_skipped": skipped,
        "unknown_words_reported_by_model": [dict(w, site=domain(p["url"])) for p in pages for w in p["unknown_words"]],
        "element_names": dict(sorted(names.items(), key=lambda kv: (-unclear(kv[1]), -kv[1]["count"]))),
        "label_words": dict(sorted(labels.items(), key=lambda kv: -kv[1]["count"])),
        "pages": pages,
    }
    path = config.path(acfg["report_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nReport: {path}  ({len(pages)} pages, {len(names)} element names, "
          f"{len(report['unknown_words_reported_by_model'])} unknown words reported)")


if __name__ == "__main__":
    sys.exit(main())
