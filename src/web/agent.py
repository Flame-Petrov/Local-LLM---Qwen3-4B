"""
Web research agent: keeps searching until the question is answered.
All prompts, rules and limits come from config.json -> "prompts" and "agent".

Each round:
  1. search the web (skipping pages it has already read)
  2. read the top pages
  3. ONE model call that extracts new facts, judges "answered yet?",
     and proposes the next search
  4. print "Research so far" so you can follow along

It never gives up on its own (unless agent.max_rounds > 0). If the search
engine returns nothing it waits, changes the query, and tries again.

Ctrl+C  -> stop researching and write the best answer from what was found.
Ctrl+C again (while that answer is being written) -> stop completely.
"""

import re
import time
from datetime import date

from core.config import params, text
from core import live
from core.debuglog import clip, log
from web.prices import check_answer, filter_kind, format_price_list, local_queries, price_config, split_rows
from web.search import result_block, web_search
from core.ui import note_cancel

CITATION = re.compile(r"\[(\d+)\]")


def _parse(out: str, acfg: dict):
    """Pull findings, the answered flag and the next search out of the model's reply."""
    L = acfg["labels"]
    F, A, Y, N = (re.escape(L[k]) for k in ("findings", "answered", "yes", "next"))
    non_fact = re.compile("|".join(f"(?:{p})" for p in acfg["non_fact_patterns"]), re.I)
    empty = {e.lower() for e in acfg["empty_findings"]}

    m = re.search(rf"{F}:\s*(.*?)(?=\n\W*{A}|\Z)", out, re.S | re.I)
    block = m.group(1) if m else re.split(A, out, flags=re.I)[0]
    findings, dropped = [], []
    for line in block.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.lower().rstrip(".") in empty or re.match(rf"\W*({F}|{N})", line, re.I):
            continue
        if non_fact.search(line):
            dropped.append({"finding": line, "reason": "not a fact (says something is missing)"})
        elif acfg["require_source_citation"] and not CITATION.search(line):
            dropped.append({"finding": line, "reason": "no source citation"})
        else:
            findings.append(line)

    answered = bool(re.search(rf"{A}:\W*{Y}", out, re.I))
    q = re.search(rf"{N}:\s*(.+)", out, re.I)
    next_q = q.group(1).strip().strip('"*` ') if q else None
    return findings, answered, next_q, dropped


def _fresh_query(proposed, tried, question, acfg, backups=(), prefer_backups=False):
    """Use the model's proposal unless it repeats an earlier search; else a planned
    backup query; else a variation of the (keyword) question.
    prefer_backups: planned searches first (price mode while shop prices are still missing)."""
    done = {t.lower().strip() for t in tried}
    year = date.today().year
    if proposed:
        proposed = CITATION.sub("", proposed).strip()  # "... exact [1]" -> "... exact"
    first = list(backups) + [proposed] if prefer_backups else [proposed] + list(backups)
    candidates = first + [v.format(question=question, year=year) for v in acfg["query_variations"]]
    for q in candidates:
        if not q:
            continue
        q = " ".join(dict.fromkeys(q.split()))  # "... 2026 2026" -> "... 2026"
        if q.lower().strip() not in done:
            return q
    return None  # out of new search ideas: the caller stops instead of inventing junk queries


def _fit(llm, build, blocks, budget, acfg):
    """Shrink page text until the prompt fits in the context window."""
    prompt = build(blocks)
    for _ in range(acfg["fit_max_attempts"]):
        if len(llm.tokenize(prompt.encode("utf-8"), add_bos=False)) <= budget:
            break
        blocks = [b[: int(len(b) * acfg["fit_shrink_factor"])] for b in blocks]
        prompt = build(blocks)
    return prompt


def price_list_answer(llm, question, rows, notes_txt, instr_txt, cfg, pcfg, generate) -> str:
    """Price answers: the LIST is built by code from the shops' price data (reliable format,
    every row = price, currency, product, shop, source, stock); the model only adds a short
    recommendation under it. Used by /agent and /web."""
    region = pcfg.get("region_found")
    today = date.today().strftime(cfg["date_format"])
    kept = filter_kind(rows, question, pcfg)  # e.g. GPUs asked -> no laptops with that GPU
    shop_rows, other = split_rows(kept, region, pcfg)
    listing = format_price_list(shop_rows, other, region, today, pcfg)
    print(listing + "\n")
    log("price_list", region=region, rows=len(rows), kind_filtered=len(rows) - len(kept),
        shop_rows=len(shop_rows), other_rows=len(other), listing=listing)
    notes_for_comment = notes_txt if pcfg["comment_uses_notes"] else "-"
    prompt = text(cfg["prompts"]["price_comment"]).format(
        today=today, question=question, rows=listing, notes=notes_for_comment or "-", user_instructions=instr_txt)
    msgs = [{"role": "system", "content": cfg["prompts"]["system"]}, {"role": "user", "content": prompt}]
    comment, _ = generate(llm, msgs, params(cfg["generation"]))
    comment, warning = check_answer(comment, listing + "\n" + notes_txt, pcfg)
    if warning:
        print(warning + "\n")
    return f"{listing}\n\n{comment}"


def _price_lines(block: str, pcfg: dict) -> list[str]:
    """The 'Price data' / 'Price element' lines of one result block."""
    L = pcfg["labels"]
    return [l for l in block.splitlines() if l.startswith((f"{L['offer']}:", f"{L['element']}:"))]


def _collect_price_lines(evidence, pcfg, acfg) -> list[str]:
    """Price mode: '[n] title' + its price lines for every block that had prices
    (the small model often forgets to copy prices into its notes)."""
    out = []
    if pcfg:
        for block in evidence:
            lines = _price_lines(block, pcfg)
            if lines:
                out += [block.splitlines()[0]] + lines
    return out[: acfg["final_price_lines_max"]]


def _found_so_far(notes, price_lines, sources) -> str:
    """Instant answer after Ctrl+C: the collected facts and shop price lines, as they are."""
    parts = ["Research stopped (Ctrl+C). What was found so far:"]
    if notes:
        parts += ["", "Facts:"] + [f"  - {n}" for n in notes]
    if price_lines:
        parts += ["", "Prices read from the shop pages:"] + [
            ("  " if not l.startswith("[") else "") + l for l in price_lines]
    nums = sorted({int(n) for n in CITATION.findall("\n".join(notes + price_lines)) if 0 < int(n) <= len(sources)})
    if nums:
        parts += ["", "Sources:"] + [f"[{i}] {sources[i - 1][0]} - {sources[i - 1][1]}" for i in nums]
    return "\n".join(parts)


def _print_notes(notes, n_sources, show):
    print(f"\nResearch so far ({len(notes)} facts from {n_sources} sources):")
    hidden = len(notes) - show
    if hidden > 0:
        print(f"  ... {hidden} earlier facts")
    for n in notes[-show:]:
        print(f"  - {n}")


def run_agent(llm, question, cfg, spin, generate, first_query=None, started_at=None, planned=None,
              product=None):
    """Research until answered (or Ctrl+C). Returns the final answer text, or None.
    `planned` = keyword queries from the search planner: round 1 uses the first,
    the rest are backups when the model's next search would repeat an earlier one.
    Ctrl+C within chat.edit_on_cancel_seconds of `started_at` returns None at once,
    so the chat can put the question back in the input line for editing."""
    started_at = started_at or time.monotonic()
    planned = list(planned or [])
    acfg, wcfg, scfg, P = cfg["agent"], cfg["web"], cfg["search"], cfg["prompts"]
    today = date.today().strftime(cfg["date_format"])
    labels = acfg["labels"]
    notes, sources, seen, tried = [], [], set(), []
    query = planned[0] if planned else (first_query or question)
    keywords = planned[0] if planned else question  # base for query variations
    backups = planned[1:]
    pcfg = price_config(question, product, cfg)
    region = pcfg["region_found"] if pcfg else None
    if pcfg:
        print(f"[agent] price mode: reading shop price data from every result (products: "
              f"{' | '.join(', '.join(t) for t in pcfg['term_sets']) or '-'}"
              + (f"; shops in {region}" if region else "") + ")")
    evidence = []    # every result block the model saw (for the price check)
    price_rows = []  # structured shop prices -> the code-built price list
    is_list = bool(pcfg) and any(re.search(p, question, re.I) for p in acfg["list_question_patterns"])
    priced_shops = set()  # shops with real prices (in the region's currency when a region is known)
    backoff, round_no, answered, empty_streak = 0, 0, False, 0
    outcome = "answered"
    log("agent_start", question=question, first_query=query, planned=planned,
        max_rounds=acfg["max_rounds"], require_source_citation=acfg["require_source_citation"])

    need_shops = (acfg["list_min_shops"] if is_list else 1) if pcfg else 0
    fast = bool(pcfg) and acfg["price_fast_mode"]
    # list questions search every planned model once ("RTX 4070 цена", "RTX 4080 цена", "RTX 4090 цена")
    local_q = local_queries(question, product, pcfg) if pcfg else []
    instructions = []   # messages the user sent while the agent works (newest last)
    to_mark_used = []   # ... read by the model in the analysis that is running now

    def absorb():
        """Take the user's new messages. Returns True if they ask to stop and answer."""
        texts = live.take()
        if not texts:
            return False
        stop = live.is_stop(texts)
        instructions.extend(texts)
        to_mark_used.extend(texts)
        print(f"\n{live.symbol('updating')} " + live.text_for(
            "agent_stop" if stop else "agent_using", text=" / ".join(texts)))
        log("live_input_folded", where="agent", texts=texts, stop=stop, round=round_no)
        if stop:
            live.mark_used(texts)
        return stop

    print("[agent] Researching until answered. Press Ctrl+C to stop and get an answer from what's found.")
    try:
        while not answered:
            round_no += 1
            if acfg["max_rounds"] and round_no > acfg["max_rounds"]:
                print(f"\n[agent] Reached agent.max_rounds={acfg['max_rounds']}.")
                outcome = "max_rounds"
                break
            if absorb():
                outcome = "user_stop"
                break
            t_round = time.monotonic()

            print(f"\n--- Round {round_no} ---")
            spin.log(f"[web] searching: {query}")
            tried.append(query)
            spin.status("searching")
            hits = [r for r in web_search(query, wcfg["web_results"], scfg, pcfg) if r["url"] not in seen]

            if not hits:  # rate-limited or nothing new: wait, change query, keep going
                empty_streak += 1
                query = _fresh_query(None, tried, keywords, acfg, backups)
                if query is None or empty_streak >= acfg["max_empty_searches"]:
                    spin.stop()
                    outcome = "out_of_queries" if query is None else "searches_empty"
                    print(f"\n[agent] Stopping: " + ("no new search ideas left." if query is None else
                          f"{empty_streak} searches in a row found nothing new (search engines may be rate-limiting)."))
                    break
                backoff = min(acfg["retry_backoff_max_s"], backoff * 2 or acfg["retry_backoff_start_s"])
                log("agent_retry", round=round_no, query=tried[-1], reason="no new results", wait_s=backoff,
                    empty_streak=empty_streak)
                spin.log(f"[web] nothing new found - trying a different search in {backoff}s")
                spin.status("waiting", seconds=backoff)
                time.sleep(backoff)
                spin.stop()
                continue
            backoff, empty_streak = 0, 0

            blocks = []
            n_pages = pcfg["pages"] if pcfg else wcfg["web_pages"]
            for i, r in enumerate(hits):
                seen.add(r["url"])
                sources.append((r["title"], r["url"]))
                blocks.append(result_block(len(sources), r, i < n_pages, wcfg, scfg, pcfg, spin.status,
                                           collect=price_rows))
            evidence.extend(blocks)
            if pcfg:  # blogs with $ prices are not shops in Bulgaria: count only the region's currency
                priced_shops = {row["shop"] for row in split_rows(price_rows, region, pcfg)[0]
                                if not row["marketplace"]}

            if fast:  # price mode: the code reads the prices - no slow model analysis between rounds
                in_region = split_rows(filter_kind(price_rows, question, pcfg), region, pcfg)[0]
                print(f"Prices so far: {len(in_region)} offers - shops: {', '.join(sorted(priced_shops)) or 'none yet'}"
                      + (f"; classified ads: {sum(r['marketplace'] for r in in_region)}" if any(r['marketplace'] for r in in_region) else ""))
                log("agent_round", round=round_no, query=query, fast=True, new_urls=[r["url"] for r in hits],
                    priced_shops=sorted(priced_shops), offers=len(in_region),
                    seconds=round(time.monotonic() - t_round, 1))
                if absorb():
                    outcome = "user_stop"
                    break
                if to_mark_used:  # no model analysis in fast mode: the final answer uses them
                    live.mark_used(to_mark_used)
                    to_mark_used.clear()
                untried_local = [q for q in local_q if q.lower() not in {t.lower() for t in tried}]
                if len(priced_shops) >= need_shops and not (is_list and untried_local):
                    print(f"\n[agent] Prices read from {len(priced_shops)} shops - enough for the answer.")
                    break
                query = _fresh_query(None, tried, keywords, acfg, backups, prefer_backups=True)
                if query is None:
                    print("\n[agent] No more shop searches to try - answering from what was found.")
                    outcome = "out_of_queries"
                    break
                print(f"Next shop search: {query}")
                continue

            if absorb():  # typed while we were searching / reading pages
                outcome = "user_stop"
                spin.stop()
                break
            stop_now = False
            while True:
                spin.status("analyzing")
                notes_txt = "\n".join(f"- {n}" for n in notes) or acfg["empty_notes_text"]
                # user instructions go at the END of the prompt: on a restart llama.cpp reuses the rest
                instr_txt = (text(P["agent_user_instructions"]).format(
                    lines="\n".join(f"- {t}" for t in instructions)) if instructions else "")
                prompt = _fit(
                    llm,
                    lambda b: text(P["agent_analyze"]).format(
                        today=today, question=question, notes=notes_txt, tried="; ".join(tried),
                        results="\n\n".join(b), findings_label=labels["findings"],
                        answered_label=labels["answered"], yes=labels["yes"], no=labels["no"],
                        next_label=labels["next"], user_instructions=instr_txt),
                    blocks,
                    llm.n_ctx() - acfg["context_reserve_tokens"],
                    acfg,
                )
                try:
                    with live.interruptible():
                        out = llm.create_chat_completion(
                            messages=[{"role": "user", "content": prompt}], **params(acfg["analyze_generation"])
                        )["choices"][0]["message"]["content"]
                    break
                except live.NewInput:  # the user sent a message mid-analysis: redo it with the message
                    if absorb():
                        stop_now = True
                        break
            spin.stop()
            if stop_now:
                outcome = "user_stop"
                break
            live.mark_used(to_mark_used)  # the model has now worked with them
            to_mark_used.clear()

            findings, answered, next_q, dropped = _parse(out, acfg)
            known = {n.lower() for n in notes}
            new = [f for f in findings if f.lower() not in known]
            dropped += [{"finding": f, "reason": "already in notes"} for f in findings if f.lower() in known]
            notes.extend(new)
            forgotten = []
            while len("\n".join(notes)) > acfg["notes_max_chars"] and len(notes) > 1:
                forgotten.append(notes.pop(0))  # forget the oldest facts if notes outgrow the context

            if not new:
                print("(nothing new in these results)")
            _print_notes(notes, len(sources), acfg["show_notes"])

            done = answered and bool(notes)
            gated = False
            if pcfg and done and len(priced_shops) < need_shops:
                # the model thinks it is answered, but a price question needs real shop prices
                print(f"\n[agent] Only {len(priced_shops)} shop(s) with prices so far "
                      f"(need {need_shops}) - searching shops.")
                done, gated = False, True
            if pcfg and not done and len(priced_shops) >= need_shops and (is_list or answered):
                print(f"\n[agent] Prices read from {len(priced_shops)} shops "
                      f"({', '.join(sorted(priced_shops))}) - enough for the answer.")
                done = True
            searched = query
            if not done:
                answered = False
                # while shop prices are missing, the planned local-language shop searches come first
                short = bool(pcfg) and len(priced_shops) < need_shops
                query = _fresh_query(next_q, tried, keywords, acfg, backups, prefer_backups=gated or short)
                if query is None:
                    print("\n[agent] Stopping: no new search ideas left - answering from what was found.")
                    outcome, done = "out_of_queries", True
            log("agent_round", round=round_no, query=searched,
                new_urls=[r["url"] for r in hits], prompt=clip(prompt, "prompt"),
                model_output=clip(out), kept=new, dropped=dropped, forgotten=forgotten,
                model_said_answered=bool(re.search(rf"{re.escape(labels['answered'])}:\W*{re.escape(labels['yes'])}", out, re.I)),
                done=done, list_question=is_list, priced_shops=sorted(priced_shops),
                proposed_next=next_q, next_query=None if done else query,
                notes_total=len(notes), seconds=round(time.monotonic() - t_round, 1))
            if done:
                if outcome == "answered":
                    print("\n[agent] Found an answer.")
                break
            print(f"Not answered yet -> next search: {query}")

    except KeyboardInterrupt:
        spin.stop()
        note_cancel()
        print("\n[agent] Stopped by you (Ctrl+C).")
        if time.monotonic() - started_at <= cfg["chat"]["edit_on_cancel_seconds"]:
            log("agent_end", outcome="cancelled_early", rounds=round_no, searches=tried, notes=notes)
            return None  # early cancel: the chat hands the question back for editing
        outcome = "stopped_by_user"
        if not notes and not priced_shops:
            log("agent_end", outcome=outcome, rounds=round_no, searches=tried, notes=notes)
            print("Nothing useful was found yet.\n")
            return None
        if not acfg["answer_after_ctrl_c"]:
            # stop NOW: show what was collected, no slow model call
            if pcfg:
                shop_rows, other = split_rows(price_rows, region, pcfg)
                summary = _found_so_far(notes, [], sources) + "\n\n" + format_price_list(
                    shop_rows, other, region, today, pcfg)
            else:
                summary = _found_so_far(notes, [], sources)
            log("agent_end", outcome=outcome, rounds=round_no, searches=tried, notes=notes,
                sources=len(sources), priced_shops=sorted(priced_shops), answer="summary_without_model")
            print(summary + "\n")
            return summary
        print("Writing an answer from what was found so far... (Ctrl+C again to skip)\n")

    log("agent_end", outcome=outcome, rounds=round_no, searches=tried, notes=notes,
        sources=len(sources), priced_shops=sorted(priced_shops))
    if not notes and not price_rows:
        msg = acfg["nothing_found_price"] if pcfg else acfg["nothing_found"]
        print(f"\n[agent] {msg.format(searches=len(tried), pages=len(sources))}\n")
        return None

    notes_txt = "\n".join(f"- {n}" for n in notes) or acfg["empty_notes_text"]
    instr_txt = (text(P["agent_user_instructions"]).format(lines="\n".join(f"- {t}" for t in instructions))
                 if instructions else "")

    if pcfg:  # ---- price mode: list built by code + a short recommendation from the model ----
        answer = price_list_answer(llm, question, price_rows, notes_txt, instr_txt, cfg, pcfg, generate)
        nums = sorted({int(n) for n in CITATION.findall(answer) if 0 < int(n) <= len(sources)})
        src = "\n".join(f"[{i}] {sources[i - 1][0]} - {sources[i - 1][1]}" for i in nums)
        print(f"Sources ({round_no} rounds, {len(tried)} searches):\n{src}\n")
        return f"{answer}\n\nSources:\n{src}"

    # ---- final answer from the notes ----
    price_lines = _collect_price_lines(evidence, pcfg, acfg)
    price_txt = text(P["agent_final_price_lines"]).format(lines="\n".join(price_lines)) if price_lines else ""
    msgs = [
        {"role": "system", "content": P["system"]},
        {"role": "user", "content": text(P["agent_final"]).format(
            today=today, notes=notes_txt, question=question, price_lines=price_txt,
            user_instructions=instr_txt)},
    ]
    answer, _ = generate(llm, msgs, params(cfg["generation"]))
    answer, warning = check_answer(answer, notes_txt + "\n" + "\n\n".join(evidence), pcfg)
    if warning:
        print(warning + "\n")

    cited = sorted({int(n) for n in CITATION.findall(notes_txt + answer) if 0 < int(n) <= len(sources)})
    used = cited or range(1, len(sources) + 1)
    src = "\n".join(f"[{i}] {sources[i - 1][0]} - {sources[i - 1][1]}" for i in used)
    print(f"Sources ({round_no} rounds, {len(tried)} searches):\n{src}\n")
    return f"{answer}\n\nSources:\n{src}"
