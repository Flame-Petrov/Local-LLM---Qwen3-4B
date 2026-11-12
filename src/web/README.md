# web/

Everything that goes online to answer a question: searching, reading shop pages, checking public
profiles and saving a page's images. Nothing here runs unless the user asks for web use (`/web`,
`/agent`, `/auto`, a price question, an image download or a profile check).

| File | What it does |
| --- | --- |
| `search.py` | Web search and page fetching; builds the context the model answers from. |
| `planner.py` | Turns the user's message into short, keyword-focused search queries. |
| `agent.py` | Research agent: keeps searching until the question is answered. |
| `prices.py` | Price mode: finds and checks prices on shop pages (offers, blocked pages, region). |
| `images.py` | Saves the images of a web page the user names. |
| `profiles.py` | Public profile check (e.g. Instagram), using only what a logged-out visitor sees. |

**Why they're together:** they form one pipeline and share one dependency, `prices.py`, which
knows how to read shop-page HTML:

```
planner ──► search ──► agent
   │          │          │
   └──────────┴────┬─────┘
                   ▼
                prices ◄── images (offer / blocked-page detection)
```

`profiles.py` doesn't use `prices`. It lives here because it's the same kind of work: fetch a public
page and build the answer in code from what's on it.

Outside users: `chat.py` (all of them) and `scripts/audit.py` (`search`, `prices`).
