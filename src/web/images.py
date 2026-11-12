"""
Image downloader: save the images of a web page the user names.

    /images https://example.com/page              all images (largest version of each)
    /images https://example.com/page max=10 min=300 only=pi
    "download the images from https://..."  /  "свали снимките от https://..."

How it finds images (shops and modern sites hide them in many places):
  <img src / data-src / data-lazy-src / data-original ...>, srcset / data-srcset (largest
  width wins), <picture><source srcset>, <a href="....jpg"> (full-size links), and the
  page's og:image / twitter:image. Relative links are resolved against the page URL.

Filters: image file types only, response must really be an image (Content-Type and file
header), smaller than min_side px or min_bytes = icon/spacer -> skipped, duplicates
(same bytes) skipped, max images / max MB per image. Everything comes from config.json
-> "images". Files go to downloads/<site>_<time>/ with index.json (source URL, alt text,
size) next to them. The bot identifies itself honestly (search.user_agent).
"""

import hashlib
import json
import re
import struct
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

from core import config
from core.debuglog import log


# ---------------- request understanding ----------------
def parse_request(message: str, icfg: dict):
    """(url, options) if the message asks to download images from a page, else None."""
    m = re.search(icfg["url_pattern"], message)
    if not m:
        return None
    url = m.group(0).rstrip(".,;:!?)\"'")
    rest = message[: m.start()] + " " + message[m.end():]
    if not message.lstrip().lower().startswith("/images") and \
            not any(re.search(p, rest, re.I) for p in icfg["intent_patterns"]):
        return None
    opts = {"max": icfg["max_images"], "min": icfg["min_side_px"], "only": None, "types": None}
    for key, val in re.findall(r"\b(max|min|only|types?)=(\S+)", rest, re.I):
        key = key.lower().rstrip("s") if key.lower().startswith("type") else key.lower()
        if key in ("max", "min") and val.isdigit():
            opts[key] = int(val)
        elif key == "only":
            opts["only"] = val.lower()
        elif key == "type":
            opts["types"] = [t.strip(".").lower() for t in val.split(",")]
    # plain language: "first 10 images", "10 снимки", "only jpg", "само png"
    if not re.search(r"\bmax=", rest, re.I):
        n = re.search(icfg["count_pattern"], rest, re.I)
        if n:
            opts["max"] = int(n.group(1))
    if not opts["types"]:
        t = re.findall(r"\b(jpe?g|png|webp|gif|svg|avif)\b", rest, re.I)
        if t:
            opts["types"] = sorted({x.lower().replace("jpeg", "jpg") for x in t})
    return url, opts


# ---------------- finding images in the page ----------------
def _largest_from_srcset(srcset: str):
    best, best_w = None, -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        w = 0
        if len(bits) > 1:
            d = re.match(r"(\d+(?:\.\d+)?)([wx])", bits[1])
            if d:
                w = float(d.group(1)) * (1 if d.group(2) == "w" else 1000)
        if w > best_w:
            best, best_w = bits[0], w
    return best


def find_images(page_html: str, page_url: str, icfg: dict) -> list[dict]:
    """[{url, alt, how}] in page order, duplicates removed, largest srcset candidate per image."""
    from lxml import html as lxml_html

    doc = lxml_html.fromstring(page_html)
    found, seen = [], {}  # seen: url -> index in found
    exts = tuple(f".{e}" for e in icfg["extensions"])
    skip_rx = re.compile("|".join(icfg["skip_patterns"]), re.I) if icfg["skip_patterns"] else None
    rewrites = [(re.compile(r["pattern"]), r["replace"]) for r in icfg["full_size_rewrites"] if "pattern" in r]

    def add(src, alt, how, hint="", declared=None, shown=None):
        """shown = the image as displayed on the page: fallback if the bigger version fails."""
        if not src:
            return
        src = src.strip()
        if src.startswith("data:") or src.startswith("javascript:"):
            return
        full = urljoin(page_url, src).split("#")[0]
        as_on_page = urljoin(page_url, shown).split("#")[0] if shown else full
        for rx, rep in rewrites:  # thumbnail URL -> full-size original where the site's pattern is known
            full = rx.sub(rep, full)
        fallback = as_on_page if full != as_on_page else None
        decoration = bool(skip_rx and skip_rx.search(f"{unquote(urlparse(full).path)} {alt} {hint}"))
        entry = {"url": full, "alt": (alt or "").strip()[:200], "how": how, "declared": declared,
                 "fallback": fallback}
        if decoration:
            entry["skip"] = "decoration"
        if full in seen:  # seen before: a normal occurrence beats an earlier 'decoration' guess
            old = found[seen[full]]
            if old.get("skip") and not decoration:
                found[seen[full]] = entry
            return
        seen[full] = len(found)
        found.append(entry)

    def declared_size(el):
        try:
            w, h = int(el.get("width") or 0), int(el.get("height") or 0)
            return (w, h) if w and h else None
        except ValueError:
            return None

    not_file = re.compile("|".join(icfg["not_image_link_patterns"]), re.I)

    def is_image_link(href):
        path = unquote(urlparse(href or "").path)
        return path.lower().endswith(exts) and not not_file.search(path)

    def link_target(el):
        """<a href="big.jpg"><img src="tiny.jpg"></a> -> the big one (shops do this for zoom).
        Not '/wiki/File:x.jpg' - that is a description PAGE, not an image file."""
        node = el.getparent()
        for _ in range(icfg["link_levels"]):
            if node is None:
                return None
            if node.tag == "a" and is_image_link(node.get("href")):
                return node.get("href")
            node = node.getparent()
        return None

    def own_src(el):
        srcset = next((el.get(a) for a in icfg["srcset_attributes"] if el.get(a)), None)
        best = _largest_from_srcset(srcset) if srcset else None
        return best or next((el.get(a) for a in icfg["src_attributes"]
                             if el.get(a) and not el.get(a).startswith("data:")), None)

    for el in doc.iter("img", "source"):
        alt = el.get("alt") or el.get("title") or ""
        hint = " ".join(filter(None, [el.get("class"), el.get("id"),
                                      el.getparent().get("class") if el.getparent() is not None else None]))
        big = link_target(el)
        if big:
            add(big, alt, "link", hint, shown=own_src(el))
            continue
        srcset = next((el.get(a) for a in icfg["srcset_attributes"] if el.get(a)), None)
        best = _largest_from_srcset(srcset) if srcset else None
        if best:
            add(best, alt, "srcset", hint)
            continue
        src = next((el.get(a) for a in icfg["src_attributes"] if el.get(a) and not el.get(a).startswith("data:")), None)
        add(src, alt, "img", hint, declared_size(el))
    for a in doc.iter("a"):  # links straight to (full-size) image files
        href = a.get("href") or ""
        if is_image_link(href):
            img = a.find(".//img")
            add(href, (img.get("alt") if img is not None else "") or a.text_content(), "link")
    for meta in doc.iter("meta"):
        if (meta.get("property") or meta.get("name") or "").lower() in icfg["meta_image_names"]:
            add(meta.get("content"), "page preview image", "meta")
    return found


# ---------------- checking what was downloaded ----------------
def image_size(data: bytes):
    """(width, height) from the file header for PNG/GIF/JPEG/WebP, else None. No Pillow needed."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", data[16:24])
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return struct.unpack("<HH", data[6:10])
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            chunk = data[12:16]
            if chunk == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
            if chunk == b"VP8L":
                b = data[21:25]
                return 1 + (((b[1] & 0x3F) << 8) | b[0]), 1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
            if chunk == b"VP8X":
                return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if data[:2] == b"\xff\xd8":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    except (struct.error, IndexError):
        pass
    return None


def kind_of(data: bytes, content_type: str):
    """Real image type from the file header (sites lie in URLs and sometimes in headers)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:12] in (b"ftypavif", b"ftypavis"):
        return "avif"
    if b"<svg" in data[:2000].lower() and "svg" in content_type:
        return "svg"
    return None


def _safe_name(url: str, alt: str, kind: str, used: set, icfg: dict) -> str:
    stem = Path(unquote(urlparse(url).path)).stem or "image"
    if len(stem) < 4 and alt:
        stem = alt
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.U).strip("_")[: icfg["name_max_chars"]] or "image"
    name, i = f"{stem}.{kind}", 2
    while name.lower() in used:
        name, i = f"{stem}_{i}.{kind}", i + 1
    used.add(name.lower())
    return name


# ---------------- the download ----------------
def download_images(url: str, opts: dict, cfg: dict, progress=None) -> dict:
    icfg, scfg = cfg["images"], cfg["search"]
    progress = progress or (lambda *a, **k: None)
    headers = {"User-Agent": scfg["user_agent"]}
    t0 = time.monotonic()
    result = {"url": url, "folder": None, "saved": [], "skipped": {}, "error": None, "found": 0}

    def skip(reason):
        result["skipped"][reason] = result["skipped"].get(reason, 0) + 1

    progress("images_page", site=urlparse(url).netloc)
    try:
        resp = requests.get(url, headers=headers, timeout=icfg["timeout_s"])
    except Exception as e:
        result["error"] = icfg["texts"]["page_error"].format(error=e)
        log("images_page", url=url, error=repr(e))
        return result
    if resp.status_code >= 400:
        result["error"] = icfg["texts"]["page_http"].format(status=resp.status_code)
        log("images_page", url=url, status=resp.status_code)
        return result
    ctype = resp.headers.get("Content-Type", "")
    if ctype.startswith("image/"):  # the "page" is itself an image
        candidates = [{"url": resp.url, "alt": "", "how": "direct"}]
    else:
        from web.prices import extract_offers, looks_blocked
        if looks_blocked(resp.text, cfg["price"]) and not extract_offers(resp.text, cfg["price"]):
            result["error"] = icfg["texts"]["page_blocked"]
            log("images_page", url=url, status=resp.status_code, blocked=True)
            return result
        candidates = find_images(resp.text, resp.url, icfg)
    result["found"] = len(candidates)
    log("images_page", url=url, status=resp.status_code, found=len(candidates),
        how={h: sum(c["how"] == h for c in candidates) for h in {c["how"] for c in candidates}})

    allowed = set(opts["types"] or icfg["extensions"])
    if not icfg["include_svg"] and not opts["types"]:
        allowed.discard("svg")
    if opts["only"]:
        candidates = [c for c in candidates if opts["only"] in (c["alt"] + " " + c["url"]).lower()] or []
        if not candidates:
            skip("no_match_only")

    site = re.sub(r"[^\w.-]", "_", urlparse(resp.url).netloc.removeprefix("www."))
    folder = config.path(icfg["download_dir"]) / f"{site}_{datetime.now().strftime(icfg['folder_time_format'])}"
    used, hashes, index = set(), set(), []
    max_bytes = icfg["max_mb_per_image"] * 1024 * 1024
    try:
        _download_all(candidates, opts, allowed, icfg, headers, resp.url, max_bytes, folder,
                      used, hashes, index, result, skip, progress)
    except KeyboardInterrupt:  # Ctrl+C: keep what is already saved
        result["stopped"] = True

    if index:
        (folder / icfg["index_file"]).write_text(json.dumps(
            {"page": resp.url, "downloaded": datetime.now().isoformat(timespec="seconds"), "images": index},
            ensure_ascii=False, indent=2), encoding="utf-8")
        result["folder"] = str(folder)
    log("images_done", url=url, saved=len(index), skipped=result["skipped"], folder=result["folder"],
        stopped=result.get("stopped", False), seconds=round(time.monotonic() - t0, 1), options=opts)
    return result


def _stem_key(url: str, icfg: dict):
    """Same picture in another size/format ('/tiny/x-452615.jpg', '/original/x-452615.jpg',
    'a5835955.png' / 'a5835955.webp') -> same key. Only for distinctive names."""
    stem = Path(unquote(urlparse(url).path)).stem.lower()
    stem = re.sub(icfg["size_suffix_pattern"], "", stem)
    return stem if len(stem) >= icfg["stem_min_chars"] and re.search(r"\d", stem) else None


def _download_all(candidates, opts, allowed, icfg, headers, referer, max_bytes, folder,
                  used, hashes, index, result, skip, progress):
    stems = set()
    for i, c in enumerate(candidates, 1):
        if len(result["saved"]) >= opts["max"]:
            skip("over_max")
            continue
        if c.get("skip"):
            skip(c["skip"])
            continue
        if c.get("declared") and min(c["declared"]) < opts["min"]:
            skip("too_small")  # the page says it is small: don't even download it
            continue
        key = _stem_key(c["url"], icfg)
        if key and key in stems:
            skip("duplicate")
            continue
        path_ext = Path(urlparse(c["url"]).path).suffix.lower().lstrip(".").replace("jpeg", "jpg")
        if path_ext and path_ext not in allowed and path_ext in icfg["extensions"] + ["svg"]:
            skip("type_not_wanted")
            continue
        progress("images_download", n=len(result["saved"]) + 1, total=min(opts["max"], len(candidates)))

        def fetch(u):
            with requests.get(u, headers={**headers, "Referer": referer}, timeout=icfg["timeout_s"],
                              stream=True) as r:  # "with": the streamed connection is always released
                if r.status_code >= 400:
                    return r.status_code, b"", ""
                data = bytearray()
                for chunk in r.iter_content(65536):
                    data += chunk
                    if len(data) > max_bytes:
                        break
                return r.status_code, bytes(data), r.headers.get("Content-Type", "")

        try:
            status, data, ctype = fetch(c["url"])
            if (status >= 400 or not kind_of(data, ctype)) and c.get("fallback"):
                # the bigger version was refused or is not an image: take the one shown on the page
                c["url"] = c["fallback"]
                status, data, ctype = fetch(c["url"])
        except Exception:
            skip("download_error")
            continue
        if status >= 400:
            skip(f"http_{status}")
            continue
        if len(data) > max_bytes:
            skip("too_big")
            continue
        kind = kind_of(data, ctype)
        if not kind:
            skip("not_an_image")
            continue
        if kind not in allowed:
            skip("type_not_wanted")
            continue
        if len(data) < icfg["min_bytes"]:
            skip("too_small")
            continue
        dims = image_size(data)
        if dims and min(dims) < opts["min"]:
            skip("too_small")
            continue
        h = hashlib.sha1(data).hexdigest()
        if h in hashes:
            skip("duplicate")
            continue
        hashes.add(h)
        if key:
            stems.add(key)
        folder.mkdir(parents=True, exist_ok=True)
        name = _safe_name(c["url"], c["alt"], kind, used, icfg)
        (folder / name).write_bytes(data)
        entry = {"file": name, "source": c["url"], "alt": c["alt"], "found_in": c["how"],
                 "bytes": len(data), "width": dims[0] if dims else None, "height": dims[1] if dims else None}
        index.append(entry)
        result["saved"].append(entry)
        time.sleep(icfg["delay_s"])  # polite: one image at a time, small pause


def report(result: dict, icfg: dict) -> str:
    T = icfg["texts"]
    if result["error"]:
        return T["failed"].format(error=result["error"])
    lines = []
    if result["saved"]:
        mb = sum(e["bytes"] for e in result["saved"]) / 1024 / 1024
        lines.append(T["saved"].format(n=len(result["saved"]), mb=f"{mb:.1f}", folder=result["folder"],
                                       found=result["found"]))
        for e in result["saved"][: icfg["report_list_max"]]:
            size = f"{e['width']}x{e['height']}" if e["width"] else "?"
            lines.append(f"  - {e['file']}  ({size}, {e['bytes'] // 1024} KB)" + (f"  \"{e['alt'][:60]}\"" if e["alt"] else ""))
        if len(result["saved"]) > icfg["report_list_max"]:
            lines.append(T["more"].format(n=len(result["saved"]) - icfg["report_list_max"]))
    else:
        lines.append(T["none"].format(found=result["found"]))
    if result.get("stopped"):
        lines.append(T["stopped"])
    if result["skipped"]:
        names = icfg["skip_reasons"]
        lines.append(T["skipped"].format(details=", ".join(
            f"{n} {names.get(k.split('_')[0] if k.startswith('http_') else k, k)}" + (f" ({k[5:]})" if k.startswith("http_") else "")
            for k, n in result["skipped"].items())))
    return "\n".join(lines)
