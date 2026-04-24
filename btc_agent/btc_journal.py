"""BTC journal: DB-primary (shared paper_trade) with parquet backup."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_improver.db import ensure_table_exists, get_engine, paper_trade, upsert_trade

logger = logging.getLogger(__name__)

INR_TO_USD = 0.012
ROUND_TRIP_FEE = 0.0010


@dataclass
class BtcTradeRecord:
    trade_id: str
    symbol: str
    timestamp_entry: datetime
    timestamp_exit: datetime | None
    direction: int
    entry_price: float
    exit_price: float | None
    sl_price: float
    target_price: float
    contracts: float
    confidence: float
    direction_prob: float
    atr_at_entry: float
    exit_reason: str | None
    pnl_usd: float | None
    pnl_inr: float | None
    charges_usd: float | None
    model_version: str


TRADE_COLUMNS = [
    "trade_id",
    "symbol",
    "timestamp_entry",
    "timestamp_exit",
    "direction",
    "entry_price",
    "exit_price",
    "sl_price",
    "target_price",
    "contracts",
    "confidence",
    "direction_prob",
    "atr_at_entry",
    "exit_reason",
    "pnl_usd",
    "pnl_inr",
    "charges_usd",
    "model_version",
]

DATETIME_COLUMNS = [
    "timestamp_entry",
    "timestamp_exit",
]


def _normalize_timestamp(value):
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_row(row: dict) -> dict:
    normalized = dict(row)
    for column in DATETIME_COLUMNS:
        if column in normalized:
            normalized[column] = _normalize_timestamp(normalized[column])
    return normalized


def _to_db_row(record: BtcTradeRecord) -> dict:
    charges_inr = (record.charges_usd or 0.0) / INR_TO_USD
    return {
        "trade_id": record.trade_id,
        "instrument": record.symbol,
        "timestamp_entry": record.timestamp_entry,
        "timestamp_exit": record.timestamp_exit,
        "direction": record.direction,
        "strike": None,
        "expiry_date": None,
        "option_type": "LONG" if record.direction == 1 else "SHORT",
        "entry_premium": None,
        "exit_premium": None,
        "lot_size": None,
        "trail_bin": None,
        "trail_tf": None,
        "vix_at_entry": None,
        "entry_price": record.entry_price,
        "exit_price": record.exit_price,
        "contracts": record.contracts,
        "pnl_usd": record.pnl_usd,
        "charges_usd": record.charges_usd,
        "sl_price": record.sl_price,
        "target_price": record.target_price,
        "confidence": record.confidence,
        "direction_prob": record.direction_prob,
        "exit_reason": record.exit_reason,
        "pnl_gross": record.pnl_usd,
        "pnl_net": record.pnl_inr,
        "charges": charges_inr,
        "atr_at_entry": record.atr_at_entry,
        "model_version": record.model_version,
    }


class BtcJournal:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir is not None else (base_dir / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trades_path = self.data_dir / "btc_trades.parquet"

        if not os.getenv("DATABASE_URL", "").strip():
            logger.warning("DATABASE_URL missing; BtcJournal will use parquet backup only.")
            self._engine = None
        else:
            self._engine = get_engine()
            ensure_table_exists(self._engine)

    def _ensure_shape(self, df: pd.DataFrame) -> pd.DataFrame:
        shaped = df.copy()
        for col in TRADE_COLUMNS:
            if col not in shaped.columns:
                shaped[col] = pd.NaT if col in DATETIME_COLUMNS else None
        for col in DATETIME_COLUMNS:
            shaped[col] = pd.to_datetime(shaped[col], errors="coerce", utc=True)
        return shaped[TRADE_COLUMNS]

    def _read_parquet(self) -> pd.DataFrame:
        if not self.trades_path.exists():
            return pd.DataFrame(columns=TRADE_COLUMNS)
        try:
            df = pd.read_parquet(self.trades_path, engine="pyarrow")
        except Exception:
            df = pd.read_parquet(self.trades_path)
        df = self._ensure_shape(df)
        return df[df["symbol"].astype(str).str.upper() == "BTCUSDT"].copy()

    def _write_parquet(self, df: pd.DataFrame) -> None:
        frame = self._ensure_shape(df)
        try:
            frame.to_parquet(self.trades_path, index=False, engine="pyarrow")
        except Exception:
            frame.to_parquet(self.trades_path, index=False)

    def _load_all_from_db(self) -> pd.DataFrame | None:
        if self._engine is None:
            return None
        try:
            stmt = (
                select(paper_trade)
                .where(paper_trade.c.instrument == "BTCUSDT")
                .order_by(paper_trade.c.timestamp_entry)
            )
            with self._engine.connect() as conn:
                rows = [dict(r._mapping) for r in conn.execute(stmt)]
            if not rows:
                return pd.DataFrame(columns=TRADE_COLUMNS)

            mapped = []
            for r in rows:
                mapped.append(
                    {
                        "trade_id": r.get("trade_id"),
                        "symbol": r.get("instrument"),
                        "timestamp_entry": r.get("timestamp_entry"),
                        "timestamp_exit": r.get("timestamp_exit"),
                        "direction": r.get("direction"),
                        "entry_price": r.get("entry_price"),
                        "exit_price": r.get("exit_price"),
                        "sl_price": r.get("sl_price"),
                        "target_price": r.get("target_price"),
                        "contracts": r.get("contracts"),
                        "confidence": r.get("confidence"),
                        "direction_prob": r.get("direction_prob"),
                        "atr_at_entry": r.get("atr_at_entry"),
                        "exit_reason": r.get("exit_reason"),
                        "pnl_usd": r.get("pnl_usd"),
                        "pnl_inr": r.get("pnl_net"),
                        "charges_usd": r.get("charges_usd"),
                        "model_version": r.get("model_version"),
                    }
                )
            return self._ensure_shape(pd.DataFrame(mapped))
        except Exception:
            return None

    def _load_open_from_db(self) -> pd.DataFrame | None:
        if self._engine is None:
            return None
        try:
            stmt = (
                select(paper_trade)
                .where(paper_trade.c.instrument == "BTCUSDT")
                .where(paper_trade.c.timestamp_exit.is_(None))
                .order_by(paper_trade.c.timestamp_entry)
            )
            with self._engine.connect() as conn:
                rows = [dict(r._mapping) for r in conn.execute(stmt)]
            if not rows:
                return pd.DataFrame(columns=TRADE_COLUMNS)

            mapped = []
            for r in rows:
                mapped.append(
                    {
                        "trade_id": r.get("trade_id"),
                        "symbol": r.get("instrument"),
                        "timestamp_entry": r.get("timestamp_entry"),
                        "timestamp_exit": r.get("timestamp_exit"),
                        "direction": r.get("direction"),
                        "entry_price": r.get("entry_price"),
                        "exit_price": r.get("exit_price"),
                        "sl_price": r.get("sl_price"),
                        "target_price": r.get("target_price"),
                        "contracts": r.get("contracts"),
                        "confidence": r.get("confidence"),
                        "direction_prob": r.get("direction_prob"),
                        "atr_at_entry": r.get("atr_at_entry"),
                        "exit_reason": r.get("exit_reason"),
                        "pnl_usd": r.get("pnl_usd"),
                        "pnl_inr": r.get("pnl_net"),
                        "charges_usd": r.get("charges_usd"),
                        "model_version": r.get("model_version"),
                    }
                )
            return self._ensure_shape(pd.DataFrame(mapped))
        except Exception:
            return None

    def log_entry(self, record: BtcTradeRecord):
        row = _normalize_row(asdict(record))
        if not row.get("trade_id"):
            row["trade_id"] = str(uuid4())
            record.trade_id = row["trade_id"]

        all_trades = self._read_parquet()
        if all_trades.empty:
            updated = self._ensure_shape(pd.DataFrame([row]))
        else:
            updated = pd.concat([all_trades, pd.DataFrame([row])], ignore_index=True)
            updated = self._ensure_shape(updated)
        self._write_parquet(updated)

        upsert_trade(self._engine, _to_db_row(record))

    def log_exit(self, trade_id, exit_price, exit_reason, timestamp_exit):
        all_trades = self._read_parquet()
        if all_trades.empty:
            return

        mask = all_trades["trade_id"].astype(str) == str(trade_id)
        if not mask.any():
            return

        row = all_trades.loc[mask].iloc[-1]
        direction = int(row["direction"])
        entry_price = float(row["entry_price"])
        contracts = float(row["contracts"])
        exit_price = float(exit_price)

        pnl_usd = (exit_price - entry_price) / entry_price * contracts * direction
        charges_usd = contracts * ROUND_TRIP_FEE
        pnl_net_usd = pnl_usd - charges_usd
        pnl_inr = pnl_net_usd / INR_TO_USD

        timestamp_exit = _normalize_timestamp(timestamp_exit)
        all_trades.loc[mask, "timestamp_exit"] = timestamp_exit
        all_trades.loc[mask, "exit_price"] = exit_price
        all_trades.loc[mask, "exit_reason"] = str(exit_reason)
        all_trades.loc[mask, "pnl_usd"] = float(pnl_usd)
        all_trades.loc[mask, "charges_usd"] = float(charges_usd)
        all_trades.loc[mask, "pnl_inr"] = float(pnl_inr)
        self._write_parquet(all_trades)

        updated_row = all_trades.loc[mask].iloc[-1].to_dict()
        record = BtcTradeRecord(
            trade_id=str(updated_row["trade_id"]),
            symbol=str(updated_row["symbol"]),
            timestamp_entry=pd.to_datetime(updated_row["timestamp_entry"]).to_pydatetime(),
            timestamp_exit=pd.to_datetime(updated_row["timestamp_exit"]).to_pydatetime(),
            direction=int(updated_row["direction"]),
            entry_price=float(updated_row["entry_price"]),
            exit_price=float(updated_row["exit_price"]),
            sl_price=float(updated_row["sl_price"]),
            target_price=float(updated_row["target_price"]),
            contracts=float(updated_row["contracts"]),
            confidence=float(updated_row["confidence"]),
            direction_prob=float(updated_row["direction_prob"]),
            atr_at_entry=float(updated_row["atr_at_entry"]),
            exit_reason=str(updated_row["exit_reason"]),
            pnl_usd=float(updated_row["pnl_usd"]),
            pnl_inr=float(updated_row["pnl_inr"]),
            charges_usd=float(updated_row["charges_usd"]),
            model_version=str(updated_row["model_version"]),
        )
        upsert_trade(self._engine, _to_db_row(record))

    def load_all(self) -> pd.DataFrame:
        db_df = self._load_all_from_db()
        if db_df is not None:
            return db_df
        return self._read_parquet()

    def load_open_trades(self) -> pd.DataFrame:
        db_df = self._load_open_from_db()
        if db_df is not None:
            return db_df
        all_trades = self._read_parquet()
        if all_trades.empty:
            return all_trades
        return all_trades[all_trades["timestamp_exit"].isna()].copy()

    def open_trades(self) -> pd.DataFrame:
        return self.load_open_trades()

    # Internal helper used by shadow mode to persist trailing SL updates.
    def update_trade(self, trade_id: str, updates: dict) -> None:
        all_trades = self._read_parquet()
        if all_trades.empty:
            return

        mask = all_trades["trade_id"].astype(str) == str(trade_id)
        if not mask.any():
            return

        for key, value in updates.items():
            if key in all_trades.columns:
                if key in DATETIME_COLUMNS:
                    value = _normalize_timestamp(value)
                all_trades.loc[mask, key] = value

        self._write_parquet(all_trades)

        row = all_trades.loc[mask].iloc[-1]
        record = BtcTradeRecord(
            trade_id=str(row["trade_id"]),
            symbol=str(row["symbol"]),
            timestamp_entry=pd.to_datetime(row["timestamp_entry"]).to_pydatetime(),
            timestamp_exit=pd.to_datetime(row["timestamp_exit"]).to_pydatetime() if pd.notna(row["timestamp_exit"]) else None,
            direction=int(row["direction"]),
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]) if pd.notna(row["exit_price"]) else None,
            sl_price=float(row["sl_price"]),
            target_price=float(row["target_price"]),
            contracts=float(row["contracts"]),
            confidence=float(row["confidence"]),
            direction_prob=float(row["direction_prob"]),
            atr_at_entry=float(row["atr_at_entry"]),
            exit_reason=str(row["exit_reason"]) if pd.notna(row["exit_reason"]) else None,
            pnl_usd=float(row["pnl_usd"]) if pd.notna(row["pnl_usd"]) else None,
            pnl_inr=float(row["pnl_inr"]) if pd.notna(row["pnl_inr"]) else None,
            charges_usd=float(row["charges_usd"]) if pd.notna(row["charges_usd"]) else None,
            model_version=str(row["model_version"]),
        )
        upsert_trade(self._engine, _to_db_row(record))
