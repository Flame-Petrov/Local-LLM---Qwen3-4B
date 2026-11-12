# files/

Reading files in and writing files out. These are the two chat commands that work with documents on
disk.

| File | What it does |
| --- | --- |
| `reader.py` | `/read <file> [question]` and `/read again`: loads a document (text/code, CSV, JSON/YAML/XML, HTML, Word, PowerPoint, PDF, …) and answers questions about it. |
| `writer.py` | `/write <file.ext> [what to put in it]`: turns what the chat researched into a file (saved to `data/documents/`). |

**Why they're together:** they're the two sides of the same job, moving content between the chat and
local files. They also share a test file (`tests/test_reader.py` covers both). Neither one depends on
the web, media or photo packages; they use only `core`.
