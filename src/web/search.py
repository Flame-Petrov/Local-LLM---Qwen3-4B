"""
Web search for the local chat. Nothing here runs unless you use /web, /agent or /auto.
All settings come from config.json -> "search" and "web".

Providers (first one available wins):
  1. Brave Search API  - reliable, needs a key in the env var named by search.brave_api_key_env
  2. ddgs (free, no key) - metasearch scraper; can return nothing when rate-limited

Page text is pulled with trafilatura (strips menus/ads, keeps article text).
"""

import os
import re
import time
from itertools import zip_longest
from urllib.parse import urlparse

import requests

from web import prices
from core.debuglog import clip, log


def _brave(query: str, n: int, scfg: dict) -> list[dict]:
    r = requests.get(
        scfg["brave_endpoint"],
        params={"q": query, "count": n},
        headers={"Accept": "application/json",
                 "X-Subscription-Token": os.environ[scfg["brave_api_key_env"]]},
        timeout=scfg["search_timeout_s"],
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
        for x in r.json().get("web", {}).get("results", [])[:n]
    ]


def _ddgs(query: str, n: int, scfg: dict, region=None) -> list[dict]:
    from ddgs import DDGS

    for attempt in range(scfg["ddgs_retries"]):
        try:
            rows = DDGS().text(query, max_results=n, backend=scfg["ddgs_backend"],
                               region=region or scfg["ddgs_region"])
            if rows:
                return [{"title": x["title"], "url": x["href"], "snippet": x["body"]} for x in rows]
            log("search_attempt", provider="ddgs", query=query, attempt=attempt + 1, result="empty")
        except Exception as e:
            log("search_attempt", provider="ddgs", query=query, attempt=attempt + 1, error=repr(e))
        if attempt + 1 < scfg["ddgs_retries"]:
            time.sleep(scfg["ddgs_retry_delay_s"] * (attempt + 1))
    return []


def web_search(query: str, n: int, scfg: dict, pcfg=None) -> list[dict]:
    """Returns [{title, url, snippet}, ...] or [] if nothing could be found.
    pcfg (price mode with a region): search that country, drop excluded domains, local sites first."""
    t = time.monotonic()
    provider, results = "ddgs", None
    region = pcfg.get("region_found") if pcfg else None
    search_region = pcfg["search_regions"].get(region) if region else None
    if os.environ.get(scfg["brave_api_key_env"]):
        try:
            provider, results = "brave", _brave(query, n, scfg)
        except Exception as e:
            log("search_attempt", provider="brave", query=query, error=repr(e))
            print(f"[web] Brave failed ({e}); falling back to free search")
            provider = "ddgs"
    if results is None:
        results = _ddgs(query, n, scfg, search_region)
    if region:
        excluded = tuple(pcfg["exclude_tlds"].get(region, []))
        local = tuple(pcfg["region_tlds"].get(region, []))
        dropped = [r["url"] for r in results if domain(r["url"]).endswith(excluded)] if excluded else []
        results = [r for r in results if not (excluded and domain(r["url"]).endswith(excluded))]
        if local:  # local shops first (stable sort keeps the engine's order otherwise)
            results.sort(key=lambda r: not domain(r["url"]).endswith(local))
        if dropped:
            log("search_region_filtered", query=query, region=region, dropped=dropped)
    ads = [r for r in results if any(re.search(p, r["url"], re.I) for p in scfg["ad_url_patterns"])]
    if ads:  # sponsored/tracking links are not sources
        log("search_ads_removed", query=query, urls=[r["url"] for r in ads])
        results = [r for r in results if r not in ads]
    log("search", provider=provider, query=query, requested=n, found=len(results),
        seconds=round(time.monotonic() - t, 2), results=results)
    return results


def read_page(url: str, max_chars: int, scfg: dict, pcfg: dict | None = None) -> dict:
    """Download a page -> {text, offers, mentions, blocked}.
    With pcfg (price mode) it also reads the shop's structured price data and the
    prices written in the text, and detects robot-check pages."""
    t, status = time.monotonic(), None
    page = {"text": "", "offers": [], "mentions": [], "elements": [], "elements_raw": 0, "blocked": False,
            "status": None, "unavailable": False, "redirected_to": None, "off_topic": False, "rows": []}
    try:
        import trafilatura

        resp = requests.get(url, headers={"User-Agent": scfg["user_agent"]},
                            timeout=scfg["page_timeout_s"])
        status = page["status"] = resp.status_code
        html = resp.text
        if status >= 400:  # dead link: don't feed an error page to the model as if it were the product
            page["unavailable"] = True
            log("page_fetch", url=url, status=status, unavailable=True, seconds=round(time.monotonic() - t, 2))
            return page
        if resp.history and urlparse(resp.url).path.strip("/") == "" and urlparse(url).path.strip("/"):
            page["redirected_to"] = resp.url  # deep link sent to the homepage: product page is gone
        tables = bool(pcfg and pcfg["include_tables"])
        full = " ".join((trafilatura.extract(html, include_comments=False, include_tables=tables) or "").split())
        if pcfg is not None:
            terms = pcfg.get("terms", [])
            title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
            offers = prices.extract_offers(html, pcfg)
            page["blocked"] = prices.looks_blocked(html, pcfg) and not offers
            page["off_topic"] = not page["blocked"] and not prices.matches(
                (title.group(1) if title else "") + " " + full, pcfg)
            if not page["off_topic"]:
                # drop offers whose own product name is clearly a different product
                page["offers"] = [o for o in offers if not o.get("product") or prices.matches(o["product"], pcfg)]
                page["mentions"] = prices.price_mentions(full, pcfg)
                # price elements in the HTML, explained with the page dictionary
                raw = prices.price_elements(html, pcfg, pcfg["elements_cfg"])
                known = {prices._num(str(o["price"])) for o in page["offers"]}
                page["elements"], page["element_labels"], element_rows = prices.explain_elements(raw, terms, pcfg, known)
                page["elements_raw"] = len(raw)
                # rows for the price list: the shop's own data first, page prices not already covered
                rows = prices.offer_rows(page["offers"], pcfg)
                have = {r["amount"] for r in rows}
                page["rows"] = rows + [r for r in element_rows if r["amount"] not in have]
        page["text"] = full[:max_chars] + ("..." if len(full) > max_chars else "")
        log("page_fetch", url=url, status=status, html_bytes=len(html), extracted_chars=len(full),
            used_chars=len(page["text"]), price_mode=pcfg is not None, blocked=page["blocked"],
            redirected_to=page["redirected_to"], off_topic=page["off_topic"],
            terms=(pcfg.get("term_sets") or pcfg.get("terms")) if pcfg else None, price_rows=page["rows"],
            offers=page["offers"], price_mentions=page["mentions"],
            price_elements=page["elements"], price_elements_found=page["elements_raw"],
            seconds=round(time.monotonic() - t, 2), text=clip(page["text"], "page"))
    except Exception as e:
        log("page_fetch", url=url, status=status, error=repr(e), seconds=round(time.monotonic() - t, 2))
    return page


def result_block(n: int, r: dict, fetch: bool, wcfg: dict, scfg: dict, pcfg=None, progress=None,
                 collect=None) -> str:
    """One numbered search result as the model sees it. In price mode, the shop's
    own price data and the prices found in the page text are listed first.
    collect: a list that receives this page's price rows (for the code-built price list)."""
    progress = progress or (lambda key, **values: None)
    lines = [f"[{n}] {r['title']}", f"URL: {r['url']}"]
    body = r["snippet"]
    if fetch:
        progress("pricing" if pcfg else "reading", site=domain(r["url"]))
        page = read_page(r["url"], pcfg["page_chars"] if pcfg else wcfg["web_page_chars"], scfg, pcfg)
        if pcfg is not None:
            L = pcfg["labels"]
            if any(m in domain(r["url"]) for m in pcfg["marketplace_domains"]):
                lines.append(L["marketplace"])
            if page["unavailable"]:
                key = "gone" if page["status"] in pcfg["gone_statuses"] else "unavailable"
                lines.append(L[key].format(status=page["status"]))
            elif page["redirected_to"]:
                lines.append(L["redirected"].format(url=page["redirected_to"]))
            elif page["blocked"]:
                lines.append(L["blocked"])
            elif page["off_topic"]:
                lines.append(L["off_topic"])
            else:
                if collect is not None:
                    market = any(m in domain(r["url"]) for m in pcfg["marketplace_domains"])
                    archive = any(re.search(p, f"{r['url']} {r['title']}", re.I) for p in pcfg["archive_patterns"])
                    for row in page["rows"]:
                        collect.append({**row, "shop": domain(r["url"]), "n": n, "title": r["title"],
                                        "marketplace": market, "archive": archive})
                lines += [f"{L['offer']}: {prices.format_offer(o)}" for o in page["offers"]]
                lines += [f"{L['element']}: {e}" for e in page["elements"]]
                if not page["elements"]:  # text mentions only when the page structure gave nothing
                    lines += [f"{L['mention']}: {m}" for m in page["mentions"]]
                # glossary only for words in the PAGE's own text, not in our explanations
                page_words = "\n".join([page.get("element_labels", "")] + page["mentions"]
                                       + [prices.format_offer(o) for o in page["offers"]])
                words = prices.explain_words(page_words, pcfg)
                if words:
                    lines.append(f"{L['words']}: " + "; ".join(words))
            if page["unavailable"] or page["redirected_to"] or page["blocked"] or page["off_topic"]:
                page["text"] = ""  # keep only the search snippet for dead/blocked/unrelated pages
        body = page["text"] or body
    lines.append(body)
    return "\n".join(lines)


def domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def merge_results(lists: list[list[dict]], limit: int) -> list[dict]:
    """Interleave results of several queries (1st of each, then 2nd of each, ...),
    skipping duplicate URLs, so every query contributes its best hits."""
    merged, seen = [], set()
    for row in zip_longest(*lists):
        for r in row:
            if r and r["url"] not in seen:
                seen.add(r["url"])
                merged.append(r)
    return merged[:limit]


def build_context(queries, wcfg: dict, scfg: dict, progress=None, pcfg=None, collect=None):
    """Search one or more queries + fetch top pages. Returns (context_text, sources) or ("", []).
    pcfg = price mode settings (reads every result page + structured price data).
    `progress(status_key, **values)` is called before each slow step (keys: config ui.status)."""
    progress = progress or (lambda key, **values: None)
    queries = [queries] if isinstance(queries, str) else queries
    lists = []
    for q in queries:
        progress("searching")
        lists.append(web_search(q, wcfg["web_results"], scfg, pcfg))
    results = merge_results(lists, wcfg["web_results"])
    if not results:
        return "", []

    n_pages = pcfg["pages"] if pcfg else wcfg["web_pages"]
    blocks = [result_block(i, r, i <= n_pages, wcfg, scfg, pcfg, progress, collect)
              for i, r in enumerate(results, 1)]
    return "\n\n".join(blocks), [(r["title"], r["url"]) for r in results]
