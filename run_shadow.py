"""Entry point for the Phase 5 shadow trading engine.

Run from the repo root:
    python run_shadow.py
    python run_shadow.py --instrument BANKNIFTY
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# --- path setup (must happen before any local imports) ---
ROOT = Path(__file__).resolve().parent
TRADING_AGENT_PKG = ROOT / "trading_agent"

for p in (str(ROOT), str(TRADING_AGENT_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)
# ---------------------------------------------------------

from model_improver.engine import Engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Nifty shadow trading engine")
    parser.add_argument("--instrument", default="NIFTY", help="NIFTY | BANKNIFTY | SENSEX")
    parser.add_argument(
        "--artifacts-dir",
        default=str(ROOT / "trading_agent" / "core" / "model" / "artifacts"),
        help="Path to model artifacts directory",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    engine = Engine(instrument=args.instrument, artifacts_dir=args.artifacts_dir)
    engine.run()


if __name__ == "__main__":
    main()
