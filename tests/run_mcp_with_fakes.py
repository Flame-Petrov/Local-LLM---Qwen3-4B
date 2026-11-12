"""Starts mcp_server.py over real stdio with the fake models installed (used by test_mcp_server.py):

    python run_mcp_with_fakes.py --config <temp config.json>

MCP_FAKE_DELAY=<seconds> makes every fake model sleep (for the cancellation test).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from photos import layer_models  # noqa: E402
import layers_fakes  # noqa: E402

fakes = layers_fakes.fakes()
for name in ("matte", "depth", "segmenter"):
    fakes[name].delay = float(os.environ.get("MCP_FAKE_DELAY", "0"))
layer_models.install_fakes(fakes)

import mcp_server  # noqa: E402

if __name__ == "__main__":
    mcp_server.main()
