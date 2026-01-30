"""Thin wrapper for computing and plotting LCOE."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rdm_kenya.postprocessing import lcoe_main


if __name__ == "__main__":
    lcoe_main()
