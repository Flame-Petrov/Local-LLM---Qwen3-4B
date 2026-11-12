"""
Price mode: reliable price gathering for shopping/price questions.
All patterns, limits and wording come from config.json -> "price" and "prompts.price_answer".

- is_price_question(): does the question ask about prices? (config price.intent_patterns)
- extract_offers():    the shop's own structured product data (schema.org JSON-LD,
                       microdata, meta tags) - this is where shops put the real price,
                       often NOT in the visible article text
- price_mentions():    prices written in the page text, with a little context
- looks_blocked():     robot-check / captcha pages (e.g. Amazon) -> skipped
- unverified_prices(): prices in the answer that appear nowhere in the sources
"""

import html as html_lib
import json
import re


def _rx(pcfg):
    cur, num = pcfg["currency_pattern"], pcfg["number_pattern"]
    return re.compile(rf"(?:{cur})\s?(?:{num})|(?:{num})\s?(?:{cur})", re.I)


def is_price_question(question: str, pcfg: dict) -> bool:
    return pcfg["enabled"] and any(re.search(p, question, re.I) for p in pcfg["intent_patterns"])


def looks_blocked(page_html: str, pcfg: dict) -> bool:
    """Robot-check pages are SMALL. A big real page may mention 'captcha' in its scripts
    (Wikipedia does), so markers only count on smallish pages."""
    size = len(page_html.encode("utf-8", "ignore"))
    if size < pcfg["blocked_max_html_bytes"]:
        return True
    low = page_html[: pcfg["blocked_scan_chars"]].lower()
    return size < pcfg["blocked_marker_max_bytes"] and any(m in low for m in pcfg["blocked_markers"])


# ---------------- is this page / offer about the product asked for? ----------------
def product_terms(question: str, pcfg: dict) -> list[str]:
    """Key product words from the question (model names, sizes, brands), e.g.
    'Raspberry Pi 5 8GB price' -> ['raspberry', '8gb']."""
    stop = {w.lower() for w in pcfg["stopwords"] + pcfg["region_words"]}
    stop |= set(re.findall(r"\w+", (pcfg["region"] or "").lower()))  # the configured region is not a product word
    toks = [t for t in re.findall(r"\w+", question.lower())
            if t not in stop and not re.fullmatch(pcfg["ignore_term_pattern"], t)]
    # short parts of a model name are glued to the word before them:
    # "raspberry pi 5" -> "raspberrypi5", "iphone 17" -> "iphone17", "wroom-32" -> "wroom32"
    glued, pending = [], ""
    for t in toks:
        if len(t) < pcfg["term_min_len"]:
            if glued:
                glued[-1] += t
            else:
                pending += t
        else:
            glued.append(pending + t)
            pending = ""
    if pending:
        glued.append(pending)
    terms = []
    for t in glued:
        if len(t) >= pcfg["term_min_len"] and t not in terms:
            terms.append(t)
    return terms


def price_config(question: str, product, cfg: dict):
    """Price-mode settings for this question, or None if it is not about prices.
    product = the planner's PRODUCT line (may list several models)."""
    pc = cfg["price"]
    if not is_price_question(question, pc):
        return None
    sets = term_sets(product, pc) if product else []
    if not sets:
        sets = [product_terms(question, pc)]
    return {**pc, "terms": sets[0], "term_sets": sets, "elements_cfg": cfg["elements"],
            "region_found": detect_region(question, pc), "products": product}


def local_queries(question: str, product, pcfg: dict) -> list[str]:
    """Searches in the shop country's language: 'RTX 4060 Ti 16GB цена' (Bulgaria)."""
    region = detect_region(question, pcfg)
    template = pcfg["local_query_templates"].get(region) if region else None
    if not template or not product:
        return []
    names = [p.strip() for p in re.split(r"[,;]", product) if p.strip()]
    out = []
    for name in names[: pcfg["max_local_queries"]]:
        # no memory sizes in the search ("RTX 4070 16GB" may not exist; shops list all sizes anyway)
        name = " ".join(re.sub(rf"(?<!\w){pcfg['soft_term_pattern']}(?!\w)", " ", name, flags=re.I).split())
        q = template.format(product=name)
        if name and q not in out:
            out.append(q)
    return out


def detect_region(question: str, pcfg: dict):
    """The configured region, or a country named in the question ('bg shops' -> Bulgaria)."""
    if pcfg["region"]:
        return pcfg["region"]
    low = question.lower()
    for region, aliases in pcfg["region_aliases"].items():
        if region.startswith("_"):
            continue
        if any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", low) for a in aliases):
            return region
    return None


def term_sets(products: str, pcfg: dict) -> list[list[str]]:
    """'RTX 4060 Ti 16GB, RTX 3090' -> [['rtx', '4060ti', '16gb'], ['rtx', '3090']] (one set per product)."""
    sets = []
    for part in re.split(r"[,;]|\s+(?:and|или|и|or)\s+", products):
        terms = product_terms(part, pcfg)
        if terms and terms not in sets:
            sets.append(terms)
    return sets


def matches(text: str, pcfg: dict) -> bool:
    """Is the text about ANY of the products asked for? (pcfg['term_sets'])"""
    sets = pcfg.get("term_sets") or ([pcfg["terms"]] if pcfg.get("terms") else [])
    if not sets:
        return True
    optional = set(pcfg["category_words"])
    return any(is_relevant(text, terms, optional, pcfg["soft_term_pattern"]) for terms in sets)


def currency_code(text: str, pcfg: dict):
    m = re.search(pcfg["currency_pattern"], text, re.I)
    if not m:
        return None
    return pcfg["currency_codes"].get(m.group(0).lower().strip(), m.group(0).upper())


def is_relevant(text: str, terms: list[str], optional=(), soft=None) -> bool:
    """Text is about the product asked for:
    - EVERY model-number term (has a digit: 'raspberrypi5', '8gb', '5090') must appear;
    - EVERY brand/model word in Latin letters ('lenovo', 'ideapad') must appear
      (one may be missing when there are 4 or more);
    - descriptive local words ('лаптоп', 'платка') are optional unless they are all there is
      (then at least half must appear)."""
    if not terms:
        return True
    low = text.lower()
    squashed = re.sub(r"[\s\-_./]", "", low)  # "8 GB" / "WROOM-32" match "8gb" / "wroom32"

    def has(t):
        return t in low or t in squashed

    core = [t for t in terms if t not in optional]
    if not core:  # only category words ("nvidia gpus"): any of them is enough
        return any(has(t) for t in terms)
    terms = core
    coded = [t for t in terms if re.search(r"\d", t)]
    if soft:  # memory sizes are soft when a real model number is also asked for ("4070" + "16gb")
        hard = [t for t in coded if not re.fullmatch(soft, t)]
        if hard:
            terms = [t for t in terms if t in hard or not re.search(r"\d", t)]
            coded = hard
    if not all(has(t) for t in coded):
        return False
    latin = [t for t in terms if re.fullmatch(r"[a-z0-9]+", t) and t not in coded]
    if latin:
        missing = sum(not has(t) for t in latin)
        return missing == 0 or (len(latin) + len(coded) >= 4 and missing <= 1)
    if coded:
        return True
    return sum(has(t) for t in terms) * 2 >= len(terms)


# ---------------- structured data (schema.org) ----------------
def _as_list(x):
    return x if isinstance(x, list) else [x] if x is not None else []


def _types(node) -> set:
    return {str(t).split("/")[-1] for t in _as_list(node.get("@type"))}


def _walk(node):
    if isinstance(node, list):
        for x in node:
            yield from _walk(x)
    elif isinstance(node, dict):
        yield node
        for key in ("@graph", "mainEntity", "itemListElement", "item", "hasVariant"):
            if key in node:
                yield from _walk(node[key])


def _jsonld_nodes(page_html: str):
    for block in re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                            page_html, re.S | re.I):
        for candidate in (block.strip(), html_lib.unescape(block.strip())):
            try:
                yield from _walk(json.loads(candidate))
                break
            except ValueError:
                continue


def _offer_row(offer: dict, name):
    spec = offer.get("priceSpecification")
    spec = spec[0] if isinstance(spec, list) and spec else spec if isinstance(spec, dict) else {}
    price = offer.get("price", spec.get("price"))
    low, high = offer.get("lowPrice"), offer.get("highPrice")
    if price is None and low is None:
        return None
    seller = offer.get("seller")
    return {
        "product": name,
        "price": price if price is not None else low,
        "high_price": high,
        "currency": offer.get("priceCurrency") or spec.get("priceCurrency"),
        "availability": str(offer.get("availability") or "").split("/")[-1] or None,
        "condition": str(offer.get("itemCondition") or "").split("/")[-1].replace("Condition", "") or None,
        "seller": seller.get("name") if isinstance(seller, dict) else seller,
    }


def extract_offers(page_html: str, pcfg: dict) -> list[dict]:
    """Prices the shop itself declares for its product(s)."""
    rows, used = [], set()
    nodes = list(_jsonld_nodes(page_html))
    for node in nodes:
        if "Product" in _types(node) or "ProductGroup" in _types(node):
            for offer in _as_list(node.get("offers")):
                for o in ([offer] if "offers" not in offer else _as_list(offer["offers"])):
                    if isinstance(o, dict):
                        rows.append(_offer_row(o, node.get("name")))
                        used.add(id(o))
    for node in nodes:  # offers not attached to a Product
        if _types(node) & {"Offer", "AggregateOffer"} and id(node) not in used:
            rows.append(_offer_row(node, None))

    if not any(rows):  # microdata / meta-tag fallback
        price = re.search(r'itemprop=["\']price["\'][^>]*content=["\']([^"\']+)', page_html, re.I) or \
            re.search(r'(?:og|product):price:amount["\'][^>]*content=["\']([^"\']+)', page_html, re.I)
        cur = re.search(r'itemprop=["\']priceCurrency["\'][^>]*content=["\']([^"\']+)', page_html, re.I) or \
            re.search(r'(?:og|product):price:currency["\'][^>]*content=["\']([^"\']+)', page_html, re.I)
        if price:
            rows.append({"product": None, "price": price.group(1), "high_price": None,
                         "currency": cur.group(1) if cur else None, "availability": None,
                         "condition": None, "seller": None})

    out, seen = [], set()
    for r in rows:
        if not r:
            continue
        key = (r["product"], str(r["price"]), r["currency"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out[: pcfg["max_offers_per_page"]]


def format_offer(o: dict) -> str:
    amount = f"{o['price']}" + (f" - {o['high_price']}" if o.get("high_price") else "")
    parts = [f"{amount} {o['currency'] or '(currency not stated)'}"]
    for key in ("availability", "condition", "seller"):
        if o.get(key):
            parts.append(f"{key}: {o[key]}")
    product = f'"{o["product"]}": ' if o.get("product") else ""
    return product + ", ".join(parts)


# ---------------- price elements in the page HTML ----------------
def price_elements(page_html: str, pcfg: dict, ecfg: dict) -> list[dict]:
    """Every HTML element whose own text is a price, with the names (class / id /
    itemprop) of the element and its parents and the label text around it.
    e.g. {"price_text": "219,00 €", "names": ["price-old", "product-price"], "label": "Стара цена: # €"}"""
    from lxml import html as lxml_html

    try:
        doc = lxml_html.fromstring(page_html)
    except Exception:
        return []
    rx = _rx(pcfg)
    layout = re.compile(ecfg["layout_name_pattern"], re.I)
    noise = re.compile(ecfg["noise_name_pattern"])
    junk = re.compile(ecfg["label_junk_pattern"], re.I)
    skip = set(ecfg["skip_tags"])

    def full_text(node):
        return " ".join(node.text_content().split())

    def is_price(node):
        t = full_text(node)
        return bool(t) and len(t) <= ecfg["max_price_text_chars"] and bool(rx.search(t))

    def value(node):
        m = rx.search(full_text(node))
        nums = re.findall(pcfg["number_pattern"], m.group(0)) if m else []
        return _num(nums[0]) if nums else None

    items, seen = [], set()
    for el in doc.iter():
        if not isinstance(el.tag, str) or el.tag in skip or not is_price(el):
            continue
        # smallest element holding the WHOLE price: "199.20" + "€" can sit in separate tags,
        # and "238." + "99 €" too - so step down only to a child with the same amount,
        # and skip a fragment whose parent shows a different (complete) amount
        v = value(el)
        if any(isinstance(c.tag, str) and is_price(c) and value(c) == v for c in el):
            continue
        parent = el.getparent()
        if parent is not None and isinstance(parent.tag, str) and is_price(parent) and value(parent) != v:
            continue
        own = full_text(el)
        names, node = [], el
        for _ in range(ecfg["ancestor_levels"] + 1):
            if node is None or not isinstance(node.tag, str):
                break
            for attr in ecfg["name_attributes"]:
                for name in (node.get(attr) or "").split():
                    name = re.sub(ecfg["name_cleanup_pattern"], "", name)
                    if name and not layout.match(name) and not noise.match(name) and name not in names:
                        names.append(name)
            node = node.getparent()
        # label = nearest surrounding element that has words besides the price itself
        label, node = own, el.getparent()
        for _ in range(ecfg["label_levels"]):
            if node is None or not isinstance(node.tag, str):
                break
            t = full_text(node)
            if junk.search(t):
                break  # CSS / script text, not a label
            if re.sub(r"[\W\d_]+", "", rx.sub("", t)):
                label = t
                break
            node = node.getparent()
        # card = the biggest surrounding block that still holds only this product
        # (on list pages it contains the product name -> used to check relevance)
        card, node = label, el.getparent()
        for _ in range(ecfg["card_levels"]):
            if node is None or not isinstance(node.tag, str):
                break
            t = full_text(node)
            if len(t) > ecfg["card_max_chars"] or len(rx.findall(t)) > ecfg["card_max_prices"] or junk.search(t):
                break
            card = t
            node = node.getparent()
        label = rx.sub("#", label)[: ecfg["label_chars"]]
        key = (own, tuple(names), label)
        if key not in seen:
            seen.add(key)
            items.append({"price_text": own, "tag": el.tag, "names": names, "label": label,
                          "card": rx.sub("#", card)[: ecfg["card_max_chars"]]})
        if len(items) >= ecfg["max_elements_per_page"]:
            break
    return items


# ---------------- page dictionary (page_dict.json) ----------------
_DICTS = {}


def _main_price_pattern(d):
    """The 'main_price' element pattern; KeyError (caught by load_dict) instead of StopIteration."""
    for r in d["element_names"]:
        if r.get("role") == "main_price" and "pattern" in r:
            return r["pattern"]
    raise KeyError("element_names has no main_price pattern")


def load_dict(pcfg: dict):
    """page_dict.json, compiled once. None if the file is missing or broken."""
    from core import config

    path = config.path(pcfg["dictionary_file"])
    if str(path) in _DICTS:
        return _DICTS[str(path)]
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        flags = [{"name": k, "meaning": v["meaning"], "rxs": [re.compile(p, re.I) for p in v["patterns"]]}
                 for k, v in d["flags"].items() if not k.startswith("_")]
        words = {k: v for k, v in d["words"].items() if not k.startswith("_")}
        compiled = {
            "roles": {k: v for k, v in d["roles"].items() if not k.startswith("_")},
            "modifier_roles": set(d["modifier_roles"]),
            "generic_roles": set(d["generic_roles"]),
            "element_rules": [{"rx": re.compile(r["pattern"], re.I), "role": r["role"]}
                              for r in d["element_names"] if "pattern" in r],
            "label_rules": [{"rx": re.compile(r["pattern"], re.I), "role": r["role"]}
                            for r in d["label_terms"] if "pattern" in r],
            "main_area": re.compile(_main_price_pattern(d), re.I),
            "old_tags": set(d["old_price_tags"]),
            "flags": flags,
            "conflicts": {k: set(v) for k, v in d["flag_conflicts"].items() if not k.startswith("_")},
            # longest first, so "с вкл. ДДС" is explained before "ДДС"
            "words": [(re.compile(rf"(?<!\w){re.escape(k)}(?!\w)", re.I), k, v)
                      for k, v in sorted(words.items(), key=lambda kv: -len(kv[0]))],
        }
    except (OSError, ValueError, KeyError, re.error) as e:
        from core.debuglog import log
        log("page_dict_error", path=str(path), error=repr(e))
        return None  # not cached: a fixed page_dict.json is picked up without a restart
    _DICTS[str(path)] = compiled
    return compiled


def classify(item: dict, d: dict):
    """(role, [flag meanings], in_main_area) for one price element, using the page dictionary."""
    names = " ".join(item["names"])
    text_all = f"{item['label']} {item['price_text']}"
    role = "old_price" if item["tag"] in d["old_tags"] else None
    if role is None:
        role = next((r["role"] for r in d["element_rules"] if r["rx"].search(names)), None)
    label_role = next((r["role"] for r in d["label_rules"] if r["rx"].search(text_all)), None)
    extra = []
    if label_role:
        if label_role in d["modifier_roles"] and role:
            extra.append(d["roles"][label_role]["meaning"])   # e.g. main price + "per piece"
        elif role is None or role in d["generic_roles"]:
            if not (label_role == "current_price" and role):
                role = label_role
    # the product card often says 'out of stock' - but only trust it for actual product prices,
    # not for shipping / add-on lines next to a store-availability list
    card = item.get("card", "") if role is None or role in d["generic_roles"] else ""
    flag_text = f"{text_all} {card} {names}"
    hit = [f for f in d["flags"] if any(rx.search(flag_text) for rx in f["rxs"])]
    removed = set().union(*(d["conflicts"].get(f["name"], set()) for f in hit)) if hit else set()
    flags = [f["meaning"] for f in hit if f["name"] not in removed]
    return role, extra + flags, bool(d["main_area"].search(names))


def explain_elements(items: list[dict], terms, pcfg: dict, known_values=()):
    """Turn raw price elements into short lines the model can trust, e.g.
    '10,90 € = price of the main product on this page; OUT OF STOCK; incl. VAT'.
    Cart totals / filters are dropped; other products' prices only if they name the product."""
    d = load_dict(pcfg)
    if not d:
        return [], "", []
    order = list(d["roles"])
    rows, price_rows = [], []
    for it in items:
        role, notes, main_area = classify(it, d)
        about_product = bool(terms or pcfg.get("term_sets")) and matches(f"{it['label']} {it.get('card', '')}", pcfg)
        if role is None:
            if not about_product:
                continue  # unknown element that doesn't name the product: can't interpret it safely
            meaning = pcfg["unknown_role_text"]
        else:
            spec = d["roles"][role]
            if spec["drop"]:
                continue
            if spec["needs_product_match"] and not main_area and not about_product:
                continue
            meaning = spec["meaning"]
        label = it["label"].strip()
        line = f"{it['price_text']} = {'; '.join([meaning] + notes)}"
        if not main_area and about_product and it.get("card"):
            # list / sidebar items: say WHICH product this price belongs to
            name = " ".join(it["card"].replace("#", " ").split())[: pcfg["elements_cfg"]["card_name_chars"]]
            line += f' (product: "{name}")'
        elif label not in ("", "#") and len(label) > 2:
            line += f' (label: "{label[: pcfg["element_label_chars"]]}")'
        nums = re.findall(pcfg["number_pattern"], it["price_text"])
        amount = _num(nums[0]) if nums else it["price_text"]
        if role in pcfg["list_roles"] or role is None:  # a real price of a product -> the price list
            name = (" ".join(it["card"].replace("#", " ").split())[: pcfg["elements_cfg"]["card_name_chars"]]
                    if not main_area and about_product and it.get("card") else None)
            price_rows.append({"product": name, "amount": amount, "price_text": it["price_text"],
                               "currency": currency_code(it["price_text"], pcfg),
                               "out_of_stock": any("OUT OF STOCK" in n for n in notes), "source": "page"})
        if role == "main_price" and not notes and amount in known_values:
            continue  # same as the shop's own "Price data" line and adds nothing
        rows.append((order.index(role) if role in order else len(order), (amount, meaning), role, line, label))
    rows.sort(key=lambda r: r[0])
    out, seen, labels, per_role = [], set(), [], {}
    for _, key, role, line, label in rows:
        if key in seen:  # same amount with the same meaning = one line ("286 €" vs "286 € / 559.37 лв.")
            continue
        if per_role.get(role, 0) >= pcfg["max_elements_per_role"]:
            continue  # e.g. five warranty add-ons -> show two
        seen.add(key)
        per_role[role] = per_role.get(role, 0) + 1
        out.append(line)
        labels.append(label)
    n = pcfg["max_elements_shown"]
    return out[:n], "\n".join(labels[:n]), price_rows


# ---------------- the price list shown to the user (built by code, not by the model) ----------------
def offer_rows(offers, pcfg) -> list[dict]:
    rows = []
    for o in offers:
        amount = _num(str(o["price"]))
        if amount is None:
            continue
        cur = pcfg["currency_codes"].get(str(o.get("currency") or "").lower(), o.get("currency"))
        rows.append({"product": o.get("product"), "amount": amount, "price_text": str(o["price"]),
                     "currency": cur, "out_of_stock": str(o.get("availability") or "").lower() in pcfg["out_of_stock_values"],
                     "source": "shop data"})
    return rows


def split_rows(rows, region, pcfg):
    """(shop rows in the region's currency, other rows) - $ prices from blogs are 'other' for Bulgaria."""
    wanted = set(pcfg["region_currencies"].get(region, [])) if region else set()
    if not wanted:
        return rows, []
    return [r for r in rows if r["currency"] in wanted], [r for r in rows if r["currency"] not in wanted]


def filter_kind(rows, question: str, pcfg: dict):
    """Asked for graphics cards -> drop laptops / PCs that merely contain one (config price.kind_filters)."""
    for kf in pcfg["kind_filters"]:
        if any(re.search(p, question, re.I) for p in kf["when"]):
            rows = [r for r in rows
                    if not any(re.search(p, r["product"] or r["title"] or "", re.I) for p in kf["drop"])]
    return rows


def format_price_list(rows, others, region, today, pcfg) -> str:
    """Shops first (cheapest first, a few rows per shop), then classified ads, then prices
    from articles / other countries - each row: price, currency, product, shop, source, stock."""
    L = pcfg["list"]

    def sort_key(r):
        return (r["currency"] or "", r["amount"] if isinstance(r["amount"], float) else 0)

    seen, uniq = set(), []
    for r in sorted(rows, key=sort_key):
        key = (r["shop"], r["amount"], (r["product"] or "")[:30].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    shops, per_shop = [], {}
    for r in (r for r in uniq if not r["marketplace"]):
        if per_shop.get(r["shop"], 0) < L["max_rows_per_shop"]:
            per_shop[r["shop"]] = per_shop.get(r["shop"], 0) + 1
            shops.append(r)
    shops = shops[: L["max_rows"]]
    ads = [r for r in uniq if r["marketplace"]][: L["max_ads"]]

    def clean(name):
        for p in L["name_strip_patterns"]:
            name = re.sub(p, "", name, flags=re.I)
        return " ".join(name.split())

    def row_text(r, in_ads=False):
        notes = (L["note_out_of_stock"] if r["out_of_stock"] else "") + \
                (L["note_archive"] if r.get("archive") else "") + \
                (L["note_marketplace"] if r["marketplace"] and not in_ads else "")
        amount = f"{r['amount']:.2f}" if isinstance(r["amount"], float) else r["price_text"]
        return L["row"].format(price=amount, currency=r["currency"] or "", product=clean(r["product"] or r["title"]),
                               shop=r["shop"], num=r["n"], notes=notes)

    out = []
    if shops:
        out = [L["header"].format(n=len(shops), shops=len({r["shop"] for r in shops}))] + [row_text(r) for r in shops]
        cur = shops[0]["currency"]
        same = [r for r in shops if r["currency"] == cur and isinstance(r["amount"], float)]
        if len(same) > 1:
            lo, hi = same[0], same[-1]
            out.append(L["summary"].format(lo=f"{lo['amount']:.2f} {cur}", lo_shop=lo["shop"],
                                           hi=f"{hi['amount']:.2f} {cur}", hi_shop=hi["shop"]))
    else:
        out = [L["none"]]
    if ads:
        out += ["", L["ads_header"]] + [row_text(r, in_ads=True) for r in ads]
    if others:
        out += ["", L["other_header"].format(region=region)] + [row_text(r) for r in sorted(others, key=sort_key)[: L["max_ads"]]]
    out.append(L["checked"].format(today=today))
    return "\n".join(out)


def explain_words(text_block: str, pcfg: dict) -> list[str]:
    """Glossary entries for dictionary words that appear in what the model will read."""
    d = load_dict(pcfg)
    if not d:
        return []
    found, covered = [], []
    for rx, word, meaning in d["words"]:
        if len(word) < pcfg["min_glossary_word_len"]:
            continue
        m = rx.search(text_block)
        if m and not any(word.lower() in c.lower() for c in covered):
            found.append(f"{word} = {meaning}")
            covered.append(word)
        if len(found) >= pcfg["max_glossary_words"]:
            break
    return found


# ---------------- prices written in text ----------------
def price_mentions(text: str, pcfg: dict) -> list[str]:
    ctx = pcfg["mention_context_chars"]
    out = []
    for m in _rx(pcfg).finditer(text):
        a, b = max(0, m.start() - ctx), min(len(text), m.end() + ctx)
        snippet = " ".join(text[a:b].split())
        if not any(m.group(0) in s for s in out):
            out.append(f"...{snippet}...")
        if len(out) >= pcfg["max_mentions_per_page"]:
            break
    return out


# ---------------- answer verification ----------------
def _num(s: str):
    s = s.strip().rstrip(".,").replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _numbers(text: str, pcfg: dict) -> set:
    return {v for v in (_num(m) for m in re.findall(pcfg["number_pattern"], text)) if v is not None}


def check_answer(answer: str, evidence: str, pcfg) -> tuple[str, str | None]:
    """Price mode: (answer + warning line, warning) for prices found nowhere in the sources."""
    from core.debuglog import log

    if not pcfg or not pcfg["verify_answer"]:
        return answer, None
    bad = unverified_prices(answer, evidence, pcfg)
    log("price_check", unverified=bad, ok=not bad)
    if not bad:
        return answer, None
    warning = pcfg["unverified_warning"].format(prices=", ".join(bad))
    return f"{answer}\n\n{warning}", warning


def unverified_prices(answer: str, evidence: str, pcfg: dict) -> list[str]:
    """Prices (number next to a currency) in the answer whose amount appears nowhere in the sources."""
    known = _numbers(evidence, pcfg)
    bad = []
    for m in _rx(pcfg).finditer(answer):
        nums = [_num(x) for x in re.findall(pcfg["number_pattern"], m.group(0))]
        if any(v is not None and v not in known for v in nums) and m.group(0) not in bad:
            bad.append(m.group(0).strip())
    return bad
