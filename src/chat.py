"""
Local AI chat — no Ollama, no server, no cloud.

Runs a GGUF model directly in this Python process via llama-cpp-python.
ALL prompts, rules and settings live in config.json (edit that, not the code).

    models/  -> the model files (.gguf)
    chats/   -> your conversation history (plain JSON)

Network access happens ONLY for:
  - the one-time model download (skipped if the file already exists)
  - web search, when YOU use /web, /agent or turn on /auto
Otherwise it runs fully offline.

Usage:
    python chat.py                             # uses config.json
    python chat.py --config my_config.json     # a different config file
    python chat.py --model models/other.gguf   # any other local GGUF file
    python chat.py --help                      # command-line overrides
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

from core import calc
from core import config
from core import debuglog
from photos import faces
from photos import faces_followup
from web import images
from photos import layers
from core import live
from media import music
from media import music_followup
from web import profiles
from files import reader
from media import see
from web.agent import price_list_answer, run_agent
from core.debuglog import clip, log
from web import prices
from web.planner import clean_query, plan_queries
from core.config import params, text
from web.search import build_context
from core import ui
from media import video
from files import writer
from core.ui import Spinner, echo_user_message

CFG: dict = {}     # loaded from config.json in main()
SPIN: Spinner = None
FORMATTER = None   # the model's own chat template (from the GGUF), built after loading


class ContextFull(Exception):
    """The prompt alone (system + your message) doesn't fit the context window."""

HELP = """Commands:
  /agent <question>  research agent: keeps searching until it finds the answer
                     (Ctrl+C stops it and answers from what it found so far)
  /auto on|agent|off let the model decide by itself when to search
                     on = one quick search, agent = full research agent
  /cutout <photo|url> [focus=thing] [crop]  remove the background: the object of focus is cut out
                     (transparent PNG) and the bot says what it found. Or just write:
                     "remove the background from E:\\pics\\me.jpg"
  /debug [on|off]    show or switch the debug log (logs folder, one file per run)
  /exit              quit (conversation is auto-saved)
  /faces <folder|photo> [check|link|norecurse|new=N|strict|loose|max=N]  find the faces in
                     your photos: every person gets a folder (downloads\\faces\\people) with
                     each photo they're in, and a person.json to rename / fine-tune them.
                     Or just write: "sort the photos in E:\\Photos by face".
                     Then say "12 is Maria", "1 yes, 2 no", "IMG_0450 is not Maria".
                     /faces who <photo> | people | rename | merge | maybe | yes | no |
                     ignore | unignore | forget | learn | rebuild
  /help              show this list
  /images <url>      download the images of that page (max=20 min=300 only=word types=jpg,png)
                     - or just write: "download the images from <url>"
  /layers <photo|url> [layers=N] [focus=thing] [fill|fill=lama]  split a photo into depth layers:
                     1 = the subject, 2 = what it sits/stands on, then the things behind it,
                     last = background. A PNG per layer + stacks + layers.psd (Photoshop/GIMP).
                     /layers again [options] = the last photo again (fast)
  /load              list saved chats and pick one by number
  /load <n|name>     load chat number n from that list, or by file name
  /music <link> [options]  download a YouTube / YouTube Music playlist or song as mp3
                     (or just paste a playlist link). Songs you already have are skipped.
                     Options - write them after the link, in any order:
                       check           only show what would be downloaded
                       force           download songs you already have too
                       single          only this song, not the whole playlist
                       max=20          only the first 20 songs
                       format=m4a      m4a or opus instead of mp3
                       to="D:\\Music"   save into another folder
                       folders=...     what happens when a folder with the playlist's
                                       name already exists from ANOTHER playlist
                                       (YouTube names every artist mix "Mix - DARA"):
                         (not written)   new folder "Mix - DARA (2)" - the default
                         folders=merge   same folder, numbers go on: 51, 52, 53...
                         folders=id      folder "Mix - DARA [RDAOy8CE7...]"
                       e.g. /music https://music.youtube.com/...list=RD... folders=merge
                     After a download just say "download them anyway", "download #15" or
                     "skip them". Also: /music pending | get <numbers|all> | skip | rescan
  /read <file> [question]  the model reads a local file and answers / pulls data out of it:
                     txt md csv json yaml xml html docx pptx xlsx odt rtf pdf code ...
                     (bare names are looked up in documents\\) - or just write:
                     "extract all emails from contacts.txt". Ranges: pages 3-7, sheet Sales,
                     lines 100-400. /read again <question> = the last file again
  /reload            re-read config.json (clears the conversation)
  /reset             clear the screen and start a new chat (the old one stays saved, /load to reopen it)
  /save [name]       save conversation to the chats folder
  /see <file|url> [question]  describe an image in detail (local vision model)
  /set <key> <value> change a setting for this session, e.g.
                     /set temperature 0.9   /set web_pages 0   /set agent.max_rounds 5
                     /set music.duplicates.same_song download
  /settings          show current settings
  /system <text>     replace the system prompt (clears the conversation)
  /video <link|file> [question]  watch a video: speech (captions or Whisper), key frames
                     (local vision model), info + comments - then ask anything about it.
                     Or just paste a video link (YouTube, TikTok, X, Vimeo, Instagram...)
  /web <question>    one search, then answer from the results
  /write <file.ext> [what]  write what was researched in this chat into a file
                     (documents folder) - any type: md txt html docx pdf pptx rtf odt
                     csv tsv xlsx json yaml xml toml py js sql ... e.g. /write prices.xlsx
                     no name -> it names the file itself: /write the prices in excel
Permanent changes: edit config.json.
"""



def sorted_help(text: str) -> str:
    """Help with the commands in alphabetical order (multi-line entries stay together)."""
    lines = text.rstrip("\n").split("\n")
    head, entries, tail = [lines[0]], [], []
    for ln in lines[1:]:
        if ln.startswith("  /"):
            entries.append([ln])
        elif ln.startswith("   ") and entries and not tail:
            entries[-1].append(ln)
        else:
            tail.append(ln)
    entries.sort(key=lambda e: e[0].split()[0])  # stable: "/load" stays before "/load <n|name>"
    return "\n".join(head + [l for e in entries for l in e] + tail) + "\n"


SETTABLE = ("generation", "web", "agent", "chat", "music", "faces", "layers")  # config sections /set may change


def parse_args():
    p = argparse.ArgumentParser(description="Offline local LLM chat (settings in config/config.json)")
    p.add_argument("--config", help="path to a config JSON file (default: config/config.json)")
    p.add_argument("--model", help="path to a local .gguf file (skips download)")
    p.add_argument("--system", help="override prompts.system")
    p.add_argument("--ctx", type=int, help="override model.context_tokens")
    p.add_argument("--threads", type=int, help="override model.threads")
    p.add_argument("--max-tokens", type=int, help="override generation.max_tokens")
    p.add_argument("--temperature", type=float, help="override generation.temperature")
    p.add_argument("--auto", choices=["off", "on", "agent"], help="override chat.auto_mode")
    p.add_argument("--no-spinner", action="store_true", help="override ui.spinner = false")
    return p.parse_args()


def load_config(args) -> dict:
    cfg = config.load(args.config)
    for value, section, key in (
        (args.ctx, "model", "context_tokens"),
        (args.threads, "model", "threads"),
        (args.max_tokens, "generation", "max_tokens"),
        (args.temperature, "generation", "temperature"),
        (args.auto, "chat", "auto_mode"),
        (args.system, "prompts", "system"),
    ):
        if value is not None:
            cfg[section][key] = value
    if args.no_spinner:
        cfg["ui"]["spinner"] = False
    return cfg


def today() -> str:
    return date.today().strftime(CFG["date_format"])


def resolve_model(args) -> Path:
    if args.model:
        path = Path(args.model).expanduser().resolve()
        if not path.is_file():
            sys.exit(f"Model file not found: {path}")
        return path

    m = CFG["model"]
    models_dir = config.path(m["models_dir"])
    path = models_dir / m["file"]
    if path.is_file():
        return path

    print(f"Model not found locally. Downloading once:\n  {m['repo']}/{m['file']}\n  -> {models_dir}")
    if m["disable_hf_telemetry"]:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=models_dir)
    return path


def load_model(path: Path):
    m = CFG["model"]
    print(f"Model: {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
    SPIN.status("loading")  # started before the import: llama_cpp takes ~1s to import
    t = time.monotonic()
    try:
        import llama_cpp
        from llama_cpp import Llama

        llm = Llama(
            model_path=str(path),
            n_ctx=m["context_tokens"],
            n_threads=m["threads"],
            n_batch=m["batch_size"],
            use_mmap=m["use_mmap"],
            verbose=False,  # chat template is read from the GGUF metadata
        )
        log("model_loaded", path=str(path), size_gb=round(path.stat().st_size / 1e9, 2),
            seconds=round(time.monotonic() - t, 2), n_ctx=llm.n_ctx(), threads=m["threads"],
            llama_cpp_version=llama_cpp.__version__)
        return llm
    finally:
        SPIN.stop()


def build_formatter(llm):
    """Same chat-template formatter llama-cpp-python builds internally, so we can
    also render a half-finished answer and let the model resume it mid-sentence."""
    from llama_cpp import llama_chat_format as lcf

    template = llm.metadata.get("tokenizer.chat_template") or lcf.CHATML_CHAT_TEMPLATE
    eos_id, bos_id = llm.token_eos(), llm.token_bos()

    def tok(i):
        return llm.detokenize([i], special=True).decode("utf-8", "ignore") if i != -1 else ""

    return lcf.Jinja2ChatFormatter(template=template, eos_token=tok(eos_id),
                                   bos_token=tok(bos_id), stop_token_ids=[eos_id])


def render(llm, messages, partial: str = ""):
    """Chat messages (+ the answer so far) -> prompt tokens, formatter response."""
    r = FORMATTER(messages=messages)
    tokens = llm.tokenize(r.prompt.encode("utf-8"), add_bos=not r.added_special, special=True)
    if partial:
        tokens += llm.tokenize(partial.encode("utf-8"), add_bos=False, special=False)
    return tokens, r


def fit_prompt(llm, messages, partial: str = ""):
    """Render, making room for the answer to keep going: drop the oldest history
    first; for huge answers keep only the tail of what was already written.
    Returns (tokens, formatter response, info about what had to be dropped)."""
    m = CFG["model"]
    msgs = list(messages)
    info = {"history_dropped": 0, "answer_tail_kept_tokens": None}
    while True:
        tokens, r = render(llm, msgs, partial)
        if llm.n_ctx() - len(tokens) >= m["min_continue_room_tokens"]:
            info["prompt_text"] = r.prompt + partial
            return tokens, r, info
        if len(msgs) > 2:  # keep system prompt [0] and the current message [-1]
            del msgs[oldest_droppable(msgs)]
            info["history_dropped"] += 1
            continue
        base, _ = render(llm, msgs)
        room_for_partial = llm.n_ctx() - len(base) - m["min_continue_room_tokens"]
        keep = min(m["continuation_keep_tokens"], room_for_partial)
        ptoks = llm.tokenize(partial.encode("utf-8"), add_bos=False) if partial else []
        if keep <= 0 or len(ptoks) <= keep:
            raise ContextFull()
        partial = llm.detokenize(ptoks[-keep:]).decode("utf-8", "ignore")
        info["answer_tail_kept_tokens"] = keep


def pinned_prefixes() -> tuple:
    """History messages starting with these hold data follow-up questions need (video, file)."""
    r = CFG.get("reader", {}).get("texts", {})
    return tuple(p for p in (CFG["video"]["texts"]["history_prefix"], r.get("history_prefix"),
                             r.get("notes_prefix")) if p)


def oldest_droppable(msgs) -> int:
    """Index of the history message to drop first: the oldest one (0 = system prompt stays),
    except the newest video/file data - follow-up questions about it need it."""
    prefixes = pinned_prefixes()
    pinned = next((i for i in range(len(msgs) - 1, 0, -1)
                   if msgs[i].get("role") == "user" and msgs[i]["content"].startswith(prefixes)), None)
    return 2 if pinned == 1 and len(msgs) > 3 else 1


def count_tokens(llm, messages) -> int:
    joined = "".join(m["content"] for m in messages)
    overhead = CFG["model"]["template_overhead_tokens"] * len(messages)
    return len(llm.tokenize(joined.encode("utf-8"), add_bos=False)) + overhead


def trim_history(llm, messages, budget: int):
    """A COPY of `messages` without the oldest user/assistant turns, so the prompt fits the
    context. The live history (what gets saved) is never changed."""
    messages = list(messages)
    dropped = 0
    while len(messages) > 2 and count_tokens(llm, messages) > budget:
        del messages[oldest_droppable(messages)]  # index 0 is the system prompt
        dropped += 1
    if dropped:
        log("history_trimmed", dropped_messages=dropped, budget_tokens=budget,
            remaining_messages=len(messages))
    return messages


def chats_dir() -> Path:
    return config.path(CFG["chat"]["chats_dir"])


READY_TEXT = "Ready. Offline unless you use /web, /agent or /auto. Type /help for commands.\n"


def new_session_name(avoid: str = "") -> str:
    """Chat name from the clock; gets a -2, -3... suffix if that name is taken (e.g. two
    /reset within one second) so a new chat never overwrites an old one."""
    base = datetime.now().strftime(CFG["chat"]["chat_name_format"])
    name, n = base, 1
    while name == avoid or (chats_dir() / f"{name}.json").exists():
        n += 1
        name = f"{base}-{n}"
    return name


def start_new_chat(messages, session: str, reason: str):
    """Shared by /reset, /system and /reload: save the current chat, drop all conversation
    state and (chat.reset_new_session) switch to a new session so the old file is kept.
    Returns (new messages, new session name, whether the old chat was saved)."""
    saved = len(messages) > 1
    if saved:
        save_chat(messages, session)
    music_followup.close_decision(reason)
    faces_followup.close(reason)
    reader.LAST = None  # '/read again' must not reach into the old chat
    live.LIVE.pending_texts()  # text typed while the bot was busy: belongs to the old chat
    live.LIVE.pop_consumed()
    new_session = session
    if CFG["chat"].get("reset_new_session", True):
        new_session = new_session_name(avoid=session)
        debuglog.set_session(new_session)
    log(reason, previous_session=session, new_session=new_session, saved=saved)
    return [{"role": "system", "content": CFG["prompts"]["system"]}], new_session, saved


def save_chat(messages, name: str) -> Path:
    d = chats_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / (name if name.endswith(".json") else name + ".json")
    # Readable on disk: metadata header, one numbered entry per message, multi-line text as a
    # list of lines (one JSON line each) instead of one huge string full of "\n".
    data = {
        "name": path.stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "message_count": len(messages),
        "messages": [
            {"n": i, "role": m.get("role"),
             "content": m["content"].split("\n") if "\n" in m.get("content", "") else m.get("content", ""),
             **{k: v for k, v in m.items() if k not in ("role", "content")}}
            for i, m in enumerate(messages)
        ],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the saved chat
    return path


def safe_chat_name(name: str) -> str:
    """A /save or /load name as a plain file name inside chats/ (no folders, no '..')."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip()).strip(" .")


def load_chat(path: Path) -> list:
    """Read a chat file (new structured format or the old plain list) back into messages."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["messages"] if isinstance(data, dict) else data
    msgs = []
    for m in raw:
        m = {k: v for k, v in m.items() if k != "n"}
        if isinstance(m.get("content"), list):
            m["content"] = "\n".join(m["content"])
        msgs.append(m)
    return msgs


def chat_summary(path: Path):
    """(first user message preview, number of messages) for the /load list."""
    try:
        msgs = load_chat(path)
    except (OSError, ValueError, KeyError, TypeError):
        return "(unreadable file)", 0
    first = next((m["content"] for m in msgs if m.get("role") == "user"), "(empty)")
    first = " ".join(first.replace("\ufeff", "").split())  # BOM from early piped-input chats
    limit = CFG["chat"]["load_preview_chars"]
    if len(first) > limit:
        first = first[: limit - 3] + "..."
    return first, sum(1 for m in msgs if m.get("role") in ("user", "assistant"))


def pick_chat(arg: str, current: str):
    """Resolve /load's argument to a chat file. No argument -> show a numbered list and ask."""
    c = CFG["chat"]
    d = chats_dir()
    if arg and not arg.isdigit():
        arg = safe_chat_name(arg)
        path = d / (arg if arg.endswith(".json") else arg + ".json")
        if arg and path.is_file():
            return path
        print(f"Not found: {path}")
        return None

    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []
    if not files:
        print("No saved chats yet.")
        return None
    shown = files[: c["load_list_limit"]]

    def by_name(name):
        path = d / (name if name.endswith(".json") else name + ".json")
        return path if path.is_file() else None

    if not arg:
        print("Saved chats (newest first):")
        for i, p in enumerate(shown, 1):
            first, n = chat_summary(p)
            when = datetime.fromtimestamp(p.stat().st_mtime).strftime(c["load_date_format"])
            tag = "  (current)" if p.stem == current else ""
            print(f"  {i:>2}. {when}  {n:>3} msgs  {first}{tag}")
        if len(files) > len(shown):
            print(f"  ... {len(files) - len(shown)} older chats (load them with /load <name>)")
        # ask until the answer is valid - a wrong answer must not fall through to the chat
        for _ in range(c["load_max_tries"]):
            try:
                ans = input(c["load_prompt"]).strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            ans = re.sub(c["load_answer_prefix"], "", ans, flags=re.I).strip()  # "/load 1" -> "1"
            if not ans:
                break
            if ans.isdigit() and 1 <= int(ans) <= len(shown):
                return shown[int(ans) - 1]
            if not ans.isdigit() and by_name(ans):
                return by_name(ans)
            print(c["load_retry"].format(count=len(shown)))
        print(c["load_cancelled"])
        return None

    if arg.isdigit() and 1 <= int(arg) <= len(shown):
        return shown[int(arg) - 1]
    print(c["load_retry"].format(count=len(shown)))
    return None


def generate(llm, messages, gen, watch_for_search=False):
    """
    Stream a reply to the console until the model is FINISHED. Returns (text, search_query).

    If the answer fills the context window, older history is dropped and the
    model resumes the same answer mid-sentence (repeats until it ends by itself).
    With watch_for_search, a reply starting with the search trigger (config
    chat.search_trigger) is captured silently and returned as search_query.
    """
    trigger = CFG["chat"]["search_trigger"]
    out, shown, interrupted, got_query = [], False, False, False
    t_start, t_first, n_tokens, segment, finish, context_full = time.monotonic(), None, 0, 0, None, False
    # a trimmed COPY: messages the user sends while we work are added to it; the full history stays saved
    msgs = trim_history(llm, messages, llm.n_ctx() - CFG["model"]["answer_reserve_tokens"])
    awaiting_use = []      # those messages while the model reads them (-> "used" at the first token)

    # ---- calculator tool: the model writes "CALC: <expr>", the program inserts "= <exact result>" ----
    ccfg = CFG["calculator"]
    calc_rx = re.compile(rf"^[ \t>*`-]*{re.escape(ccfg['label'])}\s*:\s*(.+?)[\s`*]*$")
    calc_count, calc_hit = 0, False
    if ccfg["enabled"] and msgs and msgs[0].get("role") == "system":
        msgs[0] = {**msgs[0], "content": msgs[0]["content"]
                   + text(CFG["prompts"]["calculator_rule"]).format(label=ccfg["label"])}

    def run_calc(final=False):
        """The model just finished a 'CALC: ...' line -> compute it exactly and insert the result.
        Returns True when a result was inserted (the answer then continues from there)."""
        nonlocal calc_count, out
        if not ccfg["enabled"] or calc_count >= ccfg["max_per_answer"]:
            return False
        so_far = "".join(out)
        lines = so_far.split("\n")
        line = lines[-1] if final else (lines[-2] if len(lines) > 1 else "")
        m = calc_rx.match(line)
        if not m:
            return False
        expr = m.group(1)
        value, err = calc.safe_eval(expr, ccfg)
        result = calc.fmt(value) if err is None else f"? ({err})"
        # keep the text up to the end of the CALC line (the model should have stopped there)
        keep = so_far + "\n" if final else "\n".join(lines[:-1]) + "\n"
        add = f"{ccfg['result_prefix']}{result}\n"
        print(("\n" if final else "") + add, end="", flush=True)
        out = [keep, add]
        calc_count += 1
        log("calc", expression=expr, result=result, error=err)
        return True

    def reveal(s):
        """First visible output: stop the thinking animation and start the answer."""
        nonlocal shown
        SPIN.stop()
        print(CFG["ui"]["chat_labels"]["ai"] + s, end="", flush=True)
        shown = True

    def fold_in(texts):
        """Add the user's new message to the END of the prompt: llama.cpp keeps everything it
        already read (same prefix), so only the new message has to be read - a fast restart."""
        msgs.append({"role": "user",
                     "content": text(CFG["prompts"]["live_instruction"]).format(text="\n".join(texts))})
        awaiting_use.extend(texts)
        log("live_input_folded", where="answer", texts=texts, restarted=bool(out))

    user_stopped = False
    pending = live.take()  # typed while we were searching / planning
    if pending and live.is_stop(pending):
        # nothing is written yet: "stop" means "stop searching, answer now" - which is what we do
        live.mark_used(pending)
        log("live_input_stop_ignored", reason="answer not started yet", texts=pending)
    elif pending:
        fold_in(pending)

    SPIN.start()  # rotating thinking words until the first real token
    try:
        while not user_stopped:
            try:
                with live.interruptible():
                    try:
                        tokens, fmt, info = fit_prompt(llm, msgs, "".join(out))
                    except ContextFull:
                        context_full = True
                        log("context_full", n_ctx=llm.n_ctx(), segment=segment)
                        if not shown:
                            reveal("")
                        print(f"\n[!] Your message is too long for the context window "
                              f"({llm.n_ctx()} tokens). Raise model.context_tokens in config.json.", end="")
                        break

                    if segment == 0:
                        log("generate_start", messages=len(msgs), prompt_tokens=len(tokens),
                            n_ctx=llm.n_ctx(), params=gen, watch_for_search=watch_for_search,
                            history_dropped=info["history_dropped"], prompt=clip(info["prompt_text"], "prompt"))
                    else:
                        log("continuation", segment=segment, prompt_tokens=len(tokens),
                            history_dropped=info["history_dropped"],
                            answer_tail_kept_tokens=info["answer_tail_kept_tokens"], answer_tokens_so_far=n_tokens)

                    finish = None
                    for chunk in llm.create_completion(prompt=tokens, stream=True, stop=fmt.stop,
                                                       stopping_criteria=fmt.stopping_criteria, **gen):
                        choice = chunk["choices"][0]
                        finish = choice.get("finish_reason") or finish
                        piece = choice["text"]
                        if not piece:
                            continue
                        n_tokens += 1
                        t_first = t_first or time.monotonic()
                        out.append(piece)
                        if awaiting_use:  # the model has read the user's new message and answers with it
                            live.mark_used(awaiting_use)
                            awaiting_use.clear()
                        if shown:
                            SPIN.stop()  # ends the inline spinner after a continuation
                            print(piece, end="", flush=True)
                            if "\n" in piece and run_calc():
                                calc_hit = True
                                break
                            continue

                        so_far = "".join(out)
                        if watch_for_search:
                            head = so_far.lstrip().upper()
                            if head.startswith(trigger.upper()):
                                if "\n" in so_far.lstrip():  # full query line received; stop generating
                                    got_query = True
                                    break
                                continue
                            if len(head) < len(trigger) and trigger.upper().startswith(head):
                                continue  # could still become the trigger, keep buffering
                        if so_far.strip():  # don't reveal on leading whitespace
                            reveal(so_far.lstrip())
                            if "\n" in piece and run_calc():
                                calc_hit = True
                                break
                    if not calc_hit and not got_query and shown and run_calc(final=True):
                        calc_hit = True  # the answer ended right after a CALC line
            except live.NewInput:
                texts = live.take()
                if not texts:
                    continue  # nothing new after all: resume where the answer stopped
                if live.is_stop(texts):
                    live.mark_used(texts)
                    if shown:  # "stop" / "enough" / "спри" while text is on screen: end it, keep what's written
                        user_stopped = True
                        break
                    continue  # nothing written yet: "stop" = "answer now" - keep going
                SPIN.stop()
                print(("\n\n" if shown else "") + f"{live.symbol('updating')} "
                      + live.text_for("updating", text=" / ".join(texts)))
                fold_in(texts)
                out, shown, got_query, segment, finish = [], False, False, 0, None
                SPIN.start()
                continue

            if calc_hit:  # exact result inserted: the model continues the SAME answer from there
                calc_hit = False
                segment += 1
                continue
            if got_query or finish != "length":
                break  # model finished by itself (or we have a search query)
            segment += 1
            if shown:
                SPIN.start_inline()  # context was full: make room and keep writing
    except KeyboardInterrupt:
        interrupted = True
        ui.note_cancel()
        SPIN.stop()  # also ends an inline continuation spinner
        if not shown:
            reveal("")
        print(" [stopped]", end="")

    if user_stopped:
        SPIN.stop()
        if not shown:
            reveal("")
        print(" " + live.text_for("answer_stopped"), end="")
    reply = "".join(out).strip()
    query = None
    if watch_for_search and not interrupted and reply.upper().startswith(trigger.upper()):
        query = reply[len(trigger):].splitlines()[0].strip().strip('"') or None

    t_end = time.monotonic()
    gen_s = (t_end - t_first) if t_first else 0
    log("generate_end",
        finish_reason="search_trigger" if query else "cancelled" if interrupted
        else "context_full" if context_full else finish,
        tokens=n_tokens, segments=segment + 1, seconds=round(t_end - t_start, 2),
        first_token_s=round(t_first - t_start, 2) if t_first else None,
        tokens_per_s=round((n_tokens - 1) / gen_s, 2) if gen_s > 0 and n_tokens > 1 else None,
        search_query=query, reply=clip(reply))

    if query:
        return reply, query  # spinner keeps running into the search phase
    if not shown:
        reveal(reply)
    print("\n")
    return reply, None


def search_plan(llm, question, history, use_key):
    """Keyword queries for `question` (planner on) or the filler-stripped question (off)."""
    p = CFG["planner"]
    product = None
    if p["enabled"] and p[use_key]:
        queries, product = plan_queries(llm, question, history, CFG, SPIN)
    else:
        queries = [clean_query(question, p)]
    if prices.is_price_question(question, CFG["price"]):
        # searches in the shop country's language first ("RTX 4060 Ti 16GB цена") -> local shops
        local = prices.local_queries(question, product, CFG["price"])
        queries = local + [q for q in queries if q not in local]
    SPIN.log(f"[web] search plan: {' | '.join(queries)}" + (f"  (product: {product})" if product else ""))
    return queries, product


def answer_from_web(llm, messages, question, queries, product=None):
    """Search (one or more queries), then answer `question` from the merged results.
    Price questions switch to price mode (shop price data + strict rules + price check)."""
    pcfg = prices.price_config(question, product, CFG)
    rows = []
    if pcfg:
        queries = queries[: CFG["price"]["max_web_queries"]]
        SPIN.log(f"[web] price mode: reading shop price data from every result (products: "
                 f"{' | '.join(', '.join(t) for t in pcfg['term_sets']) or '-'}"
                 + (f"; shops in {pcfg['region_found']}" if pcfg["region_found"] else "") + ")")
    context, sources = build_context(queries, CFG["web"], CFG["search"], progress=SPIN.status,
                                     pcfg=pcfg, collect=rows)
    if not context:
        SPIN.stop()
        log("web_answer", queries=queries, outcome="no_results")
        print("[web] no results (search engine may be rate-limiting; try again or rephrase)\n")
        return None

    log("web_answer", queries=queries, outcome="answering", price_mode=pcfg is not None,
        sources=sources, context=clip(context, "page"))
    SPIN.log(f"[web] read {len(sources)} results")
    if pcfg:  # price list built by code + a short recommendation (same as /agent)
        reply = price_list_answer(llm, question, rows, context[: CFG["price"]["comment_context_chars"]],
                                  "", CFG, pcfg, generate)
        nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply) if 0 < int(n) <= len(sources)})
        src = "\n".join(f"[{i}] {sources[i - 1][0]} - {sources[i - 1][1]}" for i in nums)
        print(f"Sources:\n{src}\n")
        return f"{reply}\n\nSources:\n{src}"
    gen = params(CFG["generation"])
    template = CFG["prompts"]["price_answer" if pcfg else "web_answer"]
    prompt = text(template).format(today=today(), context=context, question=question)
    # Results are used only for this one generation; history keeps the plain question.
    temp = trim_history(llm, messages[:-1] + [{"role": "user", "content": prompt}],
                        llm.n_ctx() - CFG["model"]["answer_reserve_tokens"])
    reply, _ = generate(llm, temp, gen)
    reply = check_prices(reply, context, pcfg)
    src = "\n".join(f"[{i}] {t} - {u}" for i, (t, u) in enumerate(sources, 1))
    print(f"Sources:\n{src}\n")
    return f"{reply}\n\nSources:\n{src}"


def check_prices(reply: str, evidence: str, pcfg) -> str:
    """Price mode: warn about any price in the answer that appears nowhere in the sources."""
    reply, warning = prices.check_answer(reply, evidence, pcfg)
    if warning:
        print(warning + "\n")
    return reply


def arithmetic_backstop(reply: str) -> str:
    """Equations the model wrote itself ('A - B = C') are re-checked; wrong ones get a correction."""
    ccfg = CFG["calculator"]
    if not (ccfg["enabled"] and ccfg["check_equations"]) or not reply:
        return reply
    wrong = calc.check_equations(reply, ccfg)
    if not wrong:
        return reply
    notes = [ccfg["equation_warning"].format(expr=e, right=r, said=s) for e, s, r in wrong]
    print("\n".join(notes) + "\n")
    log("calc_correction", wrong=wrong)
    return reply + "\n\n" + "\n".join(notes)


def _unverified_sites(reply: str, history) -> list[str]:
    """Web addresses in a no-internet answer that appear nowhere in the chat (the model made them up)."""
    seen = " ".join(m.get("content", "") for m in history).lower()
    out = []
    for m in re.finditer(CFG["chat"]["site_pattern"], reply, re.I):
        site = m.group(0).lower().rstrip(".")
        if site not in seen and site not in out and not any(site.endswith(s) for s in CFG["chat"]["known_sites"]):
            out.append(site)
    return out


def ask_yes_no(question: str) -> bool:
    """Blocking y/n question in the middle of a job (e.g. before reading a 30-min file)."""
    SPIN.stop()
    was_live = bool(live.LIVE and live.LIVE.active)
    if was_live:
        live.LIVE.stop()
    try:
        ans = input(question + " ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    finally:
        if was_live:
            live.LIVE.start()
    log("confirm", question=question, answer=ans)
    return ans in CFG["reader"]["yes_words"]


def live_stop_requested() -> bool:
    """The user typed 'stop' / 'enough' / 'спри' while a long job runs (other messages stay queued)."""
    texts = live.take_if(lambda t: live.is_stop([t]))
    if texts:
        live.mark_used(texts)
    return bool(texts)


def find_setting(key: str):
    """'section.key', 'section.sub.key' (e.g. music.duplicates.same_song) or a plain key that
    exists in one of the SETTABLE sections."""
    if "." in key:
        *parts, name = key.split(".")
        if parts[0] not in SETTABLE:
            return None, None
        node = CFG
        for part in parts:
            node = node.get(part) if isinstance(node, dict) and not part.startswith("_") else None
        if isinstance(node, dict) and name in node and not name.startswith("_"):
            return node, name
        return None, None
    for section in SETTABLE:
        if key in CFG[section] and not key.startswith("_"):
            return CFG[section], key
    return None, None


def convert(old, raw: str):
    if isinstance(old, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    return type(old)(raw)


def main():
    global CFG, SPIN
    args = parse_args()
    CFG = load_config(args)
    global FORMATTER
    debuglog.setup(CFG["debug"])
    log("app_start", argv=sys.argv[1:], python=sys.version.split()[0],
        config=str(config.path(args.config) if args.config else config.DEFAULT_PATH), settings=CFG)
    SPIN = Spinner(CFG["ui"])
    llm = load_model(resolve_model(args))
    FORMATTER = build_formatter(llm)
    live.LIVE = live.LiveInput(CFG["live_input"], SPIN.out)  # type while it works
    SPIN.live = live.LIVE
    live.LIVE.install_sigint()

    def add_live_to_history(history):
        """Messages the user sent while the bot worked belong in the chat history too."""
        for t in live.LIVE.pop_consumed():
            history.append({"role": "user", "content": t})

    messages = [{"role": "system", "content": CFG["prompts"]["system"]}]
    session = new_session_name()
    debuglog.set_session(session)
    prefill = ""   # text put back in the input line after an early Ctrl+C
    turn = None    # (time sent, raw input, history before it) of the last message

    print(READY_TEXT)

    while True:
        # ---- early Ctrl+C on the last message: undo it and let the user edit it ----
        if turn:
            t_sent, last_raw, before = turn
            turn = None
            cancelled = ui.cancel_time()
            if cancelled is not None:
                log("cancel", after_s=round(cancelled - t_sent, 2), input=last_raw,
                    handed_back_for_edit=cancelled - t_sent <= CFG["chat"]["edit_on_cancel_seconds"])
            if cancelled is not None and cancelled - t_sent <= CFG["chat"]["edit_on_cancel_seconds"]:
                messages = before
                if len(messages) > 1:
                    save_chat(messages, session)
                else:  # nothing left in this chat: don't keep an empty file around
                    (chats_dir() / f"{session}.json").unlink(missing_ok=True)
                prefill = last_raw
                print("Cancelled - edit your message and press Enter to send it again.")

        prompt_label = CFG["ui"]["chat_labels"]["you"]
        typed = False
        waiting = live.LIVE.pending_texts()  # sent while the last answer was finishing
        if waiting:
            raw = " ".join(waiting)
            print(f"{live.symbol('waiting')} " + live.text_for("sending_next", text=raw))
            echo_user_message(prompt_label, raw, CFG["ui"]["user_message"], replace_typed=False)
        else:
            try:
                raw = ui.input_prefilled(prompt_label, prefill)
                typed = True
            except (EOFError, KeyboardInterrupt):
                raw = "/exit"
        last_prefill, prefill = prefill, ""
        user = raw.replace("\ufeff", "").strip()  # drop BOM from piped input
        if not user:
            continue
        if typed:
            echo_user_message(prompt_label, raw, CFG["ui"]["user_message"])

        ui.reset_cancel()
        turn = (time.monotonic(), raw, list(messages))
        log("user_input", text=user, typed=typed,
            resend_after_cancel=bool(last_prefill), edited=bool(last_prefill) and raw != last_prefill,
            auto_mode=CFG["chat"]["auto_mode"], history_messages=len(messages))
        cmd0, _, rest0 = user.partition(" ")
        if not user.startswith("/") or (cmd0 in ("/web", "/agent", "/video", "/see", "/music", "/faces", "/read", "/cutout", "/layers") and rest0.strip()):
            live.LIVE.start()  # pinned footer: the user can type while the bot works
        try:
            # ---- image description: "/see <file|url> [question]" ----
            if cmd0 == "/see":
                req = see.parse(rest0)
                if not req:
                    print(see.USAGE)
                    continue
                try:
                    desc = see.describe(req[0], req[1], CFG, progress=SPIN.start)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    SPIN.stop()
                    log("see_failed", src=req[0], error=repr(e))
                    print(f"could not describe the image: {e}\n")
                    continue
                SPIN.stop()
                print("\n=== Image description ===\n" + desc + "\n")
                messages.append({"role": "user", "content": f"{user}\n\n[Detailed description of the image {req[0]} by the vision model]\n{desc}"})
                messages.append({"role": "assistant", "content": desc})
                save_chat(messages, session)
                continue

            # ---- background removal / layers: "/cutout <photo>", "/layers <photo> [options]", "/layers again ...",
            #      "remove the background from <photo>", "split <photo> into layers" ----
            # (before the music/faces follow-ups and the faces/reader triggers: "cut out the person in X.jpg"
            #  would otherwise start a face scan, and the reader takes any message that names a file)
            lcfg = CFG["layers"]
            lay_req = layers.parse_request(user, lcfg) if lcfg["enabled"] and (
                cmd0 in ("/cutout", "/layers") or not user.startswith("/")) else None
            if isinstance(lay_req, str):            # usage / not found / nothing to redo
                print(lay_req + "\n")
                continue
            if lay_req:
                log("layers_request", mode=lay_req["mode"], src=lay_req["src"], options=lay_req["opts"],
                    again=lay_req["again"], text=user)
                try:
                    result = layers.run(lay_req, CFG, progress=SPIN.status, should_stop=live_stop_requested)
                except KeyboardInterrupt:
                    ui.note_cancel()
                    SPIN.stop()
                    print(" [stopped]\n")
                    continue
                except Exception as e:              # LayersError = clear message; anything else like /see does
                    SPIN.stop()
                    log("layers_failed", src=lay_req["src"], error=repr(e))
                    if isinstance(e, layers.LayersError):
                        print(f"{e}\n")
                    else:
                        debuglog.log_exception("layers", e)
                        print(lcfg["texts"]["failed"].format(error=e) + "\n")
                    continue
                SPIN.stop()
                print("\n" + layers.report(result, lcfg, color=True) + "\n")
                messages.append({"role": "user", "content": f"{user}\n\n{lcfg['texts']['history_prefix']}\n"
                                                            f"{layers.history(result, lcfg)}"})
                add_live_to_history(messages)
                save_chat(messages, session)
                continue

            # ---- music follow-up: "download them anyway", "download #15", "skip them" (only while songs are open) ----
            mcfg = CFG["music"]
            music_question = False  # a question about the open songs: the normal answer gets their list
            if music_followup.PENDING and not user.startswith("/"):
                try:
                    dec = music_followup.interpret(user, llm, CFG, SPIN, history=messages)
                    SPIN.stop()
                    if dec and dec["action"] == "question":
                        music_question = True
                    elif dec:
                        out = music_followup.execute(dec, llm, CFG, SPIN)  # prints its own lines
                        SPIN.stop()
                        messages.append({"role": "user", "content": f"{user}\n\n{mcfg['texts']['history_prefix']}\n{out}"})
                        add_live_to_history(messages)
                        save_chat(messages, session)
                        continue
                except KeyboardInterrupt:
                    ui.note_cancel()
                    SPIN.stop()
                    print(" [stopped]\n")
                    continue

            # ---- faces follow-up: "12 is Maria", "1 yes, 2 no", "IMG_0450 is not Maria" ----
            fcfg = CFG["faces"]
            faces_question = False  # a question about the people: the normal answer gets the PEOPLE list
            if fcfg["enabled"] and not user.startswith("/"):
                try:
                    fdec = faces_followup.interpret(user, llm, CFG, SPIN, history=messages)
                    SPIN.stop()
                    if fdec and fdec["action"] == "question":
                        faces_question = True
                    elif fdec:
                        out = faces_followup.execute(fdec, CFG, SPIN, llm)  # prints its own lines
                        SPIN.stop()
                        messages.append({"role": "user", "content": f"{user}\n\n{fcfg['texts']['history_prefix']}\n{out}"})
                        add_live_to_history(messages)
                        save_chat(messages, session)
                        continue
                except KeyboardInterrupt:
                    ui.note_cancel()
                    SPIN.stop()
                    print(" [stopped]\n")
                    continue

            # ---- faces: "/faces <folder|photo> [options]", "/faces <sub-command>", "sort the photos in <folder> by face" ----
            if cmd0 == "/faces" and rest0.split()[:1] and rest0.split()[0].lower() in faces.SUBCOMMANDS \
                    and rest0.split()[0].lower() != "who":
                sub, _, args_ = rest0.strip().partition(" ")
                try:
                    res = faces.command(sub.lower(), args_, CFG, SPIN)
                except KeyboardInterrupt:
                    ui.note_cancel()
                    SPIN.stop()
                    print(" [stopped]\n")
                    continue
                SPIN.stop()
                if res["show"]:
                    print(res["show"] + "\n")
                if res["history"]:
                    messages.append({"role": "user", "content": f"{user}\n\n{fcfg['texts']['history_prefix']}\n{res['history']}"})
                    save_chat(messages, session)
                continue
            face_req = faces.parse_request(user, fcfg) if fcfg["enabled"] and (
                cmd0 == "/faces" or not user.startswith("/")) else None
            if cmd0 == "/faces" and not face_req:
                arg = rest0.strip()
                print((fcfg["texts"]["not_found"].format(path=arg) if arg and not faces.find_path(arg)
                       else fcfg["texts"]["usage"]) + "\n")
                continue
            if face_req:
                mode, path, opts = face_req
                log("faces_request", mode=mode, path=path, options=opts, text=user)
                try:
                    result = faces.who(path, CFG, SPIN) if mode == "who" else faces.run(path, opts, CFG, SPIN)
                except KeyboardInterrupt:  # before the first photo (model download / listing); later ones return stopped=True
                    ui.note_cancel()
                    SPIN.stop()
                    print(" [stopped]\n")
                    continue
                SPIN.stop()
                hint = faces_followup.open_context(result, CFG)
                music_followup.expire_weak()  # the newest report owns "yes"
                print("\n" + faces.report(result, fcfg, color=True) + ("\n" + hint if hint else "") + "\n")
                messages.append({"role": "user", "content": f"{user}\n\n{fcfg['texts']['history_prefix']}\n"
                                                            f"{faces.history(result, fcfg)}"})
                add_live_to_history(messages)
                save_chat(messages, session)
                continue

            # ---- image download: "/images <url>" or "download the images from <url>" ----
            icfg = CFG["images"]
            img_req = images.parse_request(user, icfg) if icfg["enabled"] and (
                cmd0 == "/images" or not user.startswith("/")) else None
            if cmd0 == "/images" and not img_req:
                print(icfg["texts"]["usage"])
                continue
            if img_req:
                url, opts = img_req
                log("images_request", url=url, options=opts, text=user)
                result = images.download_images(url, opts, CFG, progress=SPIN.status)
                SPIN.stop()
                summary = images.report(result, icfg)
                print(summary + "\n")
                messages.append({"role": "user", "content": f"{user}\n\n{icfg['texts']['history_prefix']}\n{summary}"})
                save_chat(messages, session)
                continue

            # ---- music: "/music <link> [options]", a pasted playlist link, or "download ... mp3 <link>" ----
            # (before video: a watch?v=...&list=... link with music words would otherwise be analyzed as a video)
            if cmd0 == "/music" and rest0.split()[:1] and rest0.split()[0].lower() in ("rescan", "pending", "get", "skip"):
                sub, _, args_ = rest0.strip().partition(" ")
                res = music_followup.command(sub, args_, llm, CFG, SPIN)
                SPIN.stop()
                if res["show"]:
                    print(res["show"] + "\n")
                if res["history"]:
                    messages.append({"role": "user", "content": f"{user}\n\n{mcfg['texts']['history_prefix']}\n{res['history']}"})
                    save_chat(messages, session)
                continue
            mus_req = music.parse_request(user, mcfg) if mcfg["enabled"] and (
                cmd0 == "/music" or not user.startswith("/")) else None
            if cmd0 == "/music" and not mus_req:
                print(mcfg["texts"]["usage"] + "\n")
                continue
            if mus_req:
                url, opts = mus_req
                log("music_request", url=url, options=opts, text=user)
                music_followup.close_decision("new_job")
                try:
                    result = music.run(url, opts, CFG, SPIN)  # prints the warnings (with paths) and song lines
                except KeyboardInterrupt:  # Ctrl+C before the downloads started (reading the playlist / library)
                    ui.note_cancel()
                    SPIN.stop()
                    print(" [stopped]\n")
                    continue
                SPIN.stop()
                hint = music_followup.open_decision(result, CFG)
                faces_followup.expire_weak()  # the newest report owns "yes"
                print("\n" + music.report(result, mcfg, color=True) + ("\n" + hint if hint else "") + "\n")
                messages.append({"role": "user", "content": f"{user}\n\n{mcfg['texts']['history_prefix']}\n"
                                                            f"{music.history(result, mcfg)}"})
                add_live_to_history(messages)
                save_chat(messages, session)
                continue

            # ---- video: "/video <link|file> [question]" or a message with a video link ----
            vcfg = CFG["video"]
            vid_req = video.parse_request(user, vcfg) if vcfg["enabled"] and (
                cmd0 == "/video" or not user.startswith("/")) else None
            if cmd0 == "/video" and not vid_req:
                print(vcfg["texts"]["usage"])
                continue
            if vid_req:
                src, question = vid_req
                log("video_request", src=src, question=question, text=user)
                try:
                    dossier, folder = video.analyze(src, llm, CFG, progress=SPIN.start, log_line=SPIN.log)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    SPIN.stop()
                    log("video_failed", src=src, error=repr(e))
                    print(vcfg["texts"]["failed"].format(error=e) + "\n")
                    continue
                SPIN.stop()
                print(vcfg["texts"]["saved"].format(folder=folder))
                question = question or vcfg["prompts"]["default_question"]
                # the video data is attached to the USER's message (like profile checks), so follow-up
                # questions and /write can use it
                messages.append({"role": "user", "content": f"{vcfg['texts']['history_prefix']}\n{dossier}\n\n"
                                                            f"{vcfg['texts']['question_prefix']} {question}"})
                system = messages[0]["content"]
                messages[0]["content"] = system + text(vcfg["prompts"]["answer_rule"])
                try:
                    reply, _ = generate(llm, messages, params(CFG["generation"]))
                finally:
                    messages[0]["content"] = system
                add_live_to_history(messages)
                messages.append({"role": "assistant", "content": reply})
                save_chat(messages, session)
                continue

            # ---- read a file: "/read <file> [question]", "/read again <q>", or a message naming an existing file ----
            rcfg = CFG["reader"]
            read_req = reader.parse_request(user, rcfg) if rcfg["enabled"] and (
                cmd0 == "/read" or not user.startswith("/")) else None
            if isinstance(read_req, str):  # /read that can't run: usage, not found, nothing read yet
                print(read_req + "\n")
                continue
            if read_req:
                log("read_request", path=str(read_req["path"]), question=read_req["question"],
                    range=read_req["range"], again=read_req["again"], text=user)
                try:
                    res = reader.prepare(read_req, llm, CFG, progress=SPIN.start, confirm=ask_yes_no,
                                         should_stop=live_stop_requested)
                except reader.ReadError as e:
                    SPIN.stop()
                    log("read_refused", path=str(read_req["path"]), message=str(e))
                    print(f"{e}\n")
                    continue
                SPIN.stop()
                if res["mode"] == "inline":
                    print(reader.T(rcfg, "inline", name=res["name"], tokens=res["tokens"]))
                else:
                    print(reader.T(rcfg, "reading_big", name=res["name"], parts=res["parts"], done=res["done"])
                          + (" " + res["partial"] if res["partial"] else ""))
                # the file data is attached to the USER's message (like a video dossier), so follow-up
                # questions and /write can use it
                messages.append({"role": "user", "content": res["block"]})
                system = messages[0]["content"]
                messages[0]["content"] = system + res["rule"]
                try:
                    reply, _ = generate(llm, messages, params(CFG["generation"]))
                finally:
                    messages[0]["content"] = system
                reply, warning = reader.verify(reply, res["evidence"], rcfg)
                if warning:
                    print(warning + "\n")
                reply = arithmetic_backstop(reply)
                add_live_to_history(messages)
                messages.append({"role": "assistant", "content": reply})
                save_chat(messages, session)
                continue

            # ---- commands ----
            if user.startswith("/"):
                cmd, _, rest = user.partition(" ")
                rest = rest.strip()
                if cmd == "/exit":
                    if len(messages) > 1:
                        print(f"Saved -> {save_chat(messages, session)}")
                    log("exit", history_messages=len(messages))
                    break
                elif cmd == "/help":
                    print(sorted_help(HELP))
                elif cmd == "/debug":
                    if rest.lower() in ("on", "off"):
                        debuglog.set_enabled(rest.lower() == "on")
                        log("debug_toggled", on=debuglog.enabled())  # only lands when turned on
                    state = "ON" if debuglog.enabled() else "OFF"
                    where = debuglog.path() or config.path(CFG["debug"]["log_dir"])
                    print(f"Debug log: {state} -> {where}")
                elif cmd in ("/web", "/agent"):
                    if not rest:
                        print(f"Usage: {cmd} <question>")
                        continue
                    messages.append({"role": "user", "content": rest})
                    if cmd == "/web":
                        queries, product = search_plan(llm, rest, messages[:-1], "use_for_web")
                        reply = answer_from_web(llm, messages, rest, queries, product)
                    else:
                        queries, product = search_plan(llm, rest, messages[:-1], "use_for_agent")
                        SPIN.stop()
                        reply = run_agent(llm, rest, CFG, SPIN, generate, started_at=turn[0],
                                          planned=queries, product=product)
                    if reply is None:
                        messages.pop()
                    else:
                        reply = arithmetic_backstop(reply)
                        add_live_to_history(messages)
                        messages.append({"role": "assistant", "content": reply})
                        save_chat(messages, session)
                elif cmd == "/write":
                    print(writer.write(llm, messages, rest, CFG, generate, SPIN) + "\n")
                elif cmd == "/auto":
                    mode = rest.lower()
                    CFG["chat"]["auto_mode"] = ("agent" if mode == "agent"
                                                else "on" if mode in ("on", "1", "true", "yes") else "off")
                    print({"off": "Auto web search: OFF - offline",
                           "on": "Auto web search: ON - model may do a quick search",
                           "agent": "Auto web search: AGENT - model may start full research (Ctrl+C stops it)",
                           }[CFG["chat"]["auto_mode"]])
                elif cmd == "/reset":
                    old_session = session
                    messages, session, saved = start_new_chat(messages, session, "reset")
                    prefill, turn = "", None
                    if CFG["chat"].get("reset_clears_screen", True) and ui.clear_screen():
                        print(READY_TEXT)
                    if saved and session != old_session:
                        print(f"New chat started (previous one saved as {old_session}, /load to reopen it).")
                    else:
                        print("New chat started.")
                elif cmd == "/system":
                    if rest:
                        CFG["prompts"]["system"] = rest
                    messages, session, _ = start_new_chat(messages, session, "system")
                    prefill, turn = "", None
                    log("system_prompt_set", text=CFG["prompts"]["system"])
                    print("System prompt set. New chat started (the previous one is saved).")
                elif cmd == "/reload":
                    try:
                        CFG = load_config(args)
                    except SystemExit as e:  # config.load exits on a missing/invalid file
                        print(f"{e}\nKept the current settings.")
                        continue
                    old_out = SPIN.out
                    was_on = debuglog.enabled()
                    debuglog.setup(CFG["debug"])
                    log("config_reloaded", debug_was_on=was_on, settings=CFG)
                    SPIN = Spinner(CFG["ui"])
                    SPIN.live = live.LIVE
                    if old_out is not sys.stdout and old_out is not SPIN.out:
                        try:
                            old_out.close()
                        except OSError:
                            pass
                    live.LIVE.cfg = CFG["live_input"]
                    messages, session, _ = start_new_chat(messages, session, "reload")
                    prefill, turn = "", None
                    print("config.json reloaded. New chat started (the previous one is saved). "
                          "(model.* changes need a restart)")
                elif cmd == "/save":
                    print(f"Saved -> {save_chat(messages, safe_chat_name(rest) or session)}")
                elif cmd == "/load":
                    path = pick_chat(rest, session)
                    if path:
                        try:
                            loaded = load_chat(path)
                        except (OSError, ValueError, KeyError, TypeError) as e:
                            print(f"Could not read {path.name}: {e}")
                            continue
                        messages, session = loaded, path.stem
                        debuglog.set_session(session)
                        log("chat_loaded", file=str(path), messages=len(messages))
                        first, n = chat_summary(path)
                        print(f"Loaded {n} messages from {path.name} - \"{first}\"")
                        music_followup.close_decision("load")
                        faces_followup.close("load")
                        reader.LAST = None  # '/read again' must not reach into the previous chat
                        live.LIVE.pending_texts()
                        live.LIVE.pop_consumed()
                        note = music_followup.restore(messages, CFG)  # its skipped songs are open again
                        if note:
                            print(note)
                elif cmd == "/settings":
                    for section in SETTABLE:
                        for k, v in params(CFG[section]).items():
                            if not isinstance(v, (dict, list)):
                                print(f"  {section}.{k} = {v}")
                elif cmd == "/set":
                    key, _, val = rest.partition(" ")
                    target, name = find_setting(key)
                    if target is None or isinstance(target[name], (dict, list)):
                        print("Unknown setting. See /settings for the list.")
                        continue
                    try:
                        target[name] = convert(target[name], val.strip())
                        log("setting_changed", key=key, value=target[name])
                        print(f"  {key} = {target[name]}")
                    except ValueError:
                        print(f"Invalid value for {key}: {val!r}")
                else:
                    print("Unknown command. /help for the list.")
                continue

            # ---- public profile check: "check opssdamnit ig profile" -> read the real profile page ----
            if CFG["profiles"]["enabled"]:
                prof = profiles.parse_request(user, CFG["profiles"],
                                              known=profiles.known_handles(messages, CFG["profiles"]))
                if prof:
                    platform, handle = prof
                    log("profile_request", platform=platform, handle=handle, text=user)
                    answer = profiles.check(platform, handle, messages, CFG, progress=SPIN.status, question=user)
                    SPIN.stop()
                    print(answer + "\n")
                    # The result goes into the history as information attached to the USER's message, and
                    # there is NO assistant turn for it: a small model copies the format of its own
                    # earlier answers - a check result there makes it fake checks, a placeholder makes it
                    # repeat the placeholder. Its own history must contain only real answers.
                    messages.append({"role": "user",
                                     "content": f"{user}\n\n{CFG['profiles']['texts']['history_prefix']}\n{answer}"})
                    save_chat(messages, session)
                    continue

            # ---- normal message ----
            gen = params(CFG["generation"])
            messages.append({"role": "user", "content": user})

            auto = CFG["chat"]["auto_mode"]
            used_web = False
            # songs the music downloader skipped are still open: the model must not claim downloads
            music_rule = music_followup.prompt_extra(CFG, question=music_question)
            # the face tool's people (names, notes, folders) - and the rule that the model can't see photos
            music_rule += faces_followup.prompt_extra(CFG, user, question=faces_question)
            if auto != "off":
                system = messages[0]["content"]
                messages[0]["content"] = system + text(CFG["prompts"]["auto_search_rule"]).format(
                    today=today(), trigger=CFG["chat"]["search_trigger"]) + music_rule
                try:
                    reply, query = generate(llm, messages, gen, watch_for_search=True)
                finally:
                    messages[0]["content"] = system  # keep saved history clean
                if query:
                    log("auto_search_triggered", mode=auto, query=query)
                    product = None
                    if CFG["planner"]["use_for_auto"]:
                        queries, product = search_plan(llm, user, messages[:-1], "use_for_auto")
                    else:  # the model already wrote its own keyword query
                        queries = [query]
                        SPIN.log(f"[web] searching: {query}")
                    if auto == "agent":
                        SPIN.stop()
                        result = run_agent(llm, user, CFG, SPIN, generate, first_query=query,
                                           started_at=turn[0], planned=queries, product=product)
                    else:
                        result = answer_from_web(llm, messages, user, queries, product)
                    reply = result if result is not None else "(web search found nothing)"
                    used_web = True
            else:
                # no internet in plain chat: the model must not pretend it checked anything
                system = messages[0]["content"]
                messages[0]["content"] = system + text(CFG["prompts"]["no_internet_rule"]) + music_rule
                try:
                    reply, _ = generate(llm, messages, gen)
                finally:
                    messages[0]["content"] = system  # keep saved history clean
            if not used_web and any(re.search(p, reply, re.I | re.M) for p in CFG["chat"]["tool_signatures"]):
                # the model imitated a tool result (fake "✓ Instagram @..." / "Checked live") -> never show it
                log("tool_imitation_blocked", reply=clip(reply))
                print("\n" + CFG["chat"]["imitation_blocked"] + "\n")
                reply = CFG["chat"]["imitation_blocked"]
            elif not used_web and (sites := _unverified_sites(reply, messages)):
                # plain chat named web addresses nobody gave it -> they may not exist (e.g. "Amazon.bg")
                warning = CFG["chat"]["unverified_sites_warning"].format(sites=", ".join(sites))
                print(warning + "\n")
                log("unverified_sites", sites=sites)
                reply = f"{reply}\n\n{warning}"
            elif not used_web and any(re.search(p, reply, re.I) for p in CFG["chat"]["claim_patterns"]):
                # safety net: it still claims to have looked something up -> warn the user
                warning = CFG["chat"]["claim_warning"]
                print(warning + "\n")
                log("fake_lookup_claim", reply=clip(reply))
                reply = f"{reply}\n\n{warning}"

            reply = arithmetic_backstop(reply)
            add_live_to_history(messages)
            messages.append({"role": "assistant", "content": reply})
            save_chat(messages, session)  # auto-save every turn
        except KeyboardInterrupt:  # Ctrl+C outside the answer stream (e.g. while searching)
            ui.note_cancel()
            SPIN.stop()
            print(" [stopped]\n")
            # don't leave a question without an answer - but only one THIS turn added
            if turn and len(messages) > len(turn[2]) and messages[-1]["role"] == "user":
                messages.pop()
        except Exception as e:  # one failing feature must not end the whole session
            SPIN.stop()
            debuglog.log_exception("turn", e)
            print(f"\n[error] {type(e).__name__}: {e}\n(The chat continues; details are in the debug log.)\n")
            if turn and len(messages) > len(turn[2]) and messages[-1]["role"] == "user":
                messages.pop()
        finally:
            SPIN.stop()
            draft = live.LIVE.stop()  # half-typed text goes back into the normal input line
            live.LIVE.pop_consumed()
            if draft and not prefill:
                prefill = draft


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        debuglog.log_exception("main", e)
        sys.exit(f"KeyError {e}: a setting may be missing from config.json, or this is a bug "
                 "(details in the debug log)")
    except Exception as e:  # unexpected crash: record it in the debug log, then show it
        debuglog.log_exception("main", e)
        raise
