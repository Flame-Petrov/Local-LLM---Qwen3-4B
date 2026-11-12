"""
Image description: '/see <file|url> [question]' -> the local vision model (Qwen2.5-VL, same as /video)
looks at the image and writes a long, detailed description. The description is printed at the end and
added to the chat, so follow-up questions and /write can use it.
"""

import io
import re
import tempfile
from pathlib import Path

import requests

from core import config
from media import video
from core.debuglog import log

USAGE = "usage: /see <image file or URL> [question]   e.g. /see C:\\pics\\cat.jpg  or  /see https://.../photo.png what brand is the car?"

DETAIL_PROMPT = (
    "Describe this image in as much detail as possible. Cover, in order:\n"
    "1. Overall: what kind of image it is (photo, screenshot, drawing, chart...) and the main subject.\n"
    "2. Every visible object and person: position (left/right/center, foreground/background), size, "
    "colours, materials, clothing, expressions, poses, actions.\n"
    "3. The setting/background, lighting, time of day, weather, mood.\n"
    "4. ALL readable text, quoted exactly (signs, labels, captions, UI, numbers).\n"
    "5. Composition, camera angle, style, image quality.\n"
    "6. Anything unusual or notable, and what is likely happening.\n"
    "Only describe what you can actually see; say 'unclear' instead of guessing."
)

MAX_SIDE = 1024  # bigger images cost many more vision tokens and slow the 3B model down a lot


def parse(rest: str):
    rest = rest.strip()
    if not rest:
        return None
    m = re.match(r'"([^"]+)"\s*(.*)', rest) or re.match(r"(\S+)\s*(.*)", rest)
    src, q = m.group(1), m.group(2).strip()
    if not src.lower().startswith(("http://", "https://")):
        # allow unquoted paths with spaces: take the longest prefix that is an existing file
        words = rest.split(" ")
        for i in range(len(words), 0, -1):
            cand = " ".join(words[:i]).strip('"')
            if Path(cand).is_file():
                return cand, " ".join(words[i:]).strip()
    return src, q


def load_image(src: str, ua: str) -> bytes:
    if src.lower().startswith(("http://", "https://")):
        r = requests.get(src, timeout=30, headers={"User-Agent": ua})
        r.raise_for_status()
        return r.content
    p = Path(src).expanduser()
    if not p.is_absolute():
        p = config.path(src) if not p.exists() else p
    return p.read_bytes()


def prepare(data: bytes) -> Path:
    """Convert to RGB JPEG, shrink to MAX_SIDE, write to a temp file for the vision model."""
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    f = Path(tempfile.gettempdir()) / "local_llm_see.jpg"
    img.save(f, "JPEG", quality=92)
    return f


def describe(src: str, question: str, cfg: dict, progress) -> str:
    vcfg = cfg["video"]
    ua = cfg.get("search", {}).get("user_agent", "local-llm")
    progress("loading image")
    img = prepare(load_image(src, ua))
    # the video settings are tuned for short frame captions - give the model room for a long answer
    vc = dict(vcfg, vision=dict(vcfg["vision"], max_tokens=1500, context_tokens=max(vcfg["vision"]["context_tokens"], 6144)))
    vis = video.Vision(vc, progress)
    try:
        prompt = DETAIL_PROMPT + (f"\n\nAfterwards, answer this question about the image: {question}" if question else "")
        progress("looking at the image (this can take a minute)")
        with video.quiet():
            out = vis.llm.create_chat_completion(
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + __import__("base64").b64encode(img.read_bytes()).decode()}},
                    {"type": "text", "text": prompt}]}],
                max_tokens=1500, temperature=0.2)
        desc = out["choices"][0]["message"]["content"].strip()
    finally:
        vis.close()
        img.unlink(missing_ok=True)
    log("see_image", src=src, question=question, description=desc)
    return desc
