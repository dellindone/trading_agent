# python run_shadow_btc.py --capital 20000 --model-version v1.0

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from btc_agent.btc_engine import BtcEngine


def _configure_console_logging() -> None:
    # Keep the terminal clean for the in-place BTC ticker. Only warnings/errors
    # should break onto a new line during normal operation.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for logger_name in [
        "httpx",
        "httpcore",
        "websocket",
        "btc_agent.delta_client",
        "btc_agent.btc_reporter",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=20000.0)
    parser.add_argument("--model-version", type=str, default="v1.0")
    args = parser.parse_args()

    _configure_console_logging()
    engine = BtcEngine(capital_inr=args.capital, model_version=args.model_version)
    engine.run()
