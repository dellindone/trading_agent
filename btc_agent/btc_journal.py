"""BTC journal: DB-primary (shared paper_trade) with parquet backup."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_improver.db import ensure_table_exists, get_engine, paper_trade, upsert_trade

logger = logging.getLogger(__name__)

USD_TO_INR = float(os.getenv("BTC_USD_INR", "83.0"))
INR_TO_USD = 1.0 / USD_TO_INR if USD_TO_INR > 0 else 0.012
FUTURES_TAKER_FEE_RATE = 0.0005
FUTURES_MAKER_FEE_RATE = 0.0002
GST_RATE = 0.18
ENTRY_FEE_MODE = str(os.getenv("BTC_ENTRY_FEE_MODE", "taker")).strip().lower()
EXIT_FEE_MODE = str(os.getenv("BTC_EXIT_FEE_MODE", "taker")).strip().lower()


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
    override: bool = False
    charges_inr: float | None = None
    initial_sl_price: float | None = None


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
    "charges_inr",
    "initial_sl_price",
    "override",
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
    charges_inr = float(record.charges_inr) if record.charges_inr is not None else float((record.charges_usd or 0.0) * USD_TO_INR)
    pnl_net_inr = float(record.pnl_inr) if record.pnl_inr is not None else None
    if pnl_net_inr is not None:
        pnl_gross_inr = pnl_net_inr + charges_inr
        pnl_gross_usd = pnl_gross_inr * INR_TO_USD
    elif record.pnl_usd is not None:
        pnl_gross_usd = float(record.pnl_usd)
    else:
        pnl_gross_usd = None
    row = {
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
        "pnl_gross": pnl_gross_usd,
        "pnl_net": pnl_net_inr,
        "charges": charges_inr,
        "atr_at_entry": record.atr_at_entry,
        "initial_sl_price": record.initial_sl_price,
        "override": bool(record.override),
        "model_version": record.model_version,
    }
    return {key: value for key, value in row.items() if key in paper_trade.c}


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
                if col in DATETIME_COLUMNS:
                    shaped[col] = pd.NaT
                elif col == "override":
                    shaped[col] = False
                else:
                    shaped[col] = None
        for col in DATETIME_COLUMNS:
            shaped[col] = pd.to_datetime(shaped[col], errors="coerce", utc=True)
        shaped["override"] = shaped["override"].fillna(False).astype(bool)
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

    def _overlay_override_from_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        pq = self._read_parquet()
        if pq.empty or "override" not in pq.columns:
            return df
        lookup = (
            pq.sort_values("timestamp_entry")
            .drop_duplicates(subset=["trade_id"], keep="last")
            .set_index("trade_id")
        )
        merged = df.copy()
        merged["override"] = (
            merged["trade_id"].astype(str).map(lookup["override"]).fillna(False).astype(bool)
        )
        if "initial_sl_price" in lookup.columns:
            merged["initial_sl_price"] = merged["trade_id"].astype(str).map(lookup["initial_sl_price"])
            merged["initial_sl_price"] = merged["initial_sl_price"].where(
                merged["initial_sl_price"].notna(),
                merged.get("sl_price"),
            )
        return self._ensure_shape(merged)

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
                        "charges_inr": r.get("charges"),
                        "initial_sl_price": r.get("initial_sl_price"),
                        "override": bool(r.get("override", False) or False),
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
                        "charges_inr": r.get("charges"),
                        "initial_sl_price": r.get("initial_sl_price"),
                        "override": bool(r.get("override", False) or False),
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

        gross_usd = (exit_price - entry_price) * contracts * direction
        entry_notional_usd = abs(entry_price * contracts)
        exit_notional_usd = abs(exit_price * contracts)
        entry_fee_rate = FUTURES_MAKER_FEE_RATE if ENTRY_FEE_MODE == "maker" else FUTURES_TAKER_FEE_RATE
        exit_fee_rate = FUTURES_MAKER_FEE_RATE if EXIT_FEE_MODE == "maker" else FUTURES_TAKER_FEE_RATE
        charges_usd = (entry_notional_usd * entry_fee_rate + exit_notional_usd * exit_fee_rate) * (1.0 + GST_RATE)
        pnl_net_usd = gross_usd - charges_usd
        pnl_inr = pnl_net_usd * USD_TO_INR
        charges_inr = charges_usd * USD_TO_INR

        timestamp_exit = _normalize_timestamp(timestamp_exit)
        all_trades.loc[mask, "timestamp_exit"] = timestamp_exit
        all_trades.loc[mask, "exit_price"] = exit_price
        all_trades.loc[mask, "exit_reason"] = str(exit_reason)
        all_trades.loc[mask, "pnl_usd"] = float(pnl_net_usd)
        all_trades.loc[mask, "charges_usd"] = float(charges_usd)
        all_trades.loc[mask, "charges_inr"] = float(charges_inr)
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
            override=bool(updated_row.get("override", False) or False),
            model_version=str(updated_row["model_version"]),
            charges_inr=float(updated_row["charges_inr"]) if pd.notna(updated_row.get("charges_inr")) else None,
            initial_sl_price=(
                float(updated_row["initial_sl_price"])
                if pd.notna(updated_row.get("initial_sl_price"))
                else float(updated_row["sl_price"])
            ),
        )
        upsert_trade(self._engine, _to_db_row(record))

    def load_all(self) -> pd.DataFrame:
        db_df = self._load_all_from_db()
        if db_df is not None:
            return self._overlay_override_from_parquet(db_df)
        return self._read_parquet()

    def load_open_trades(self) -> pd.DataFrame:
        db_df = self._load_open_from_db()
        if db_df is not None:
            return self._overlay_override_from_parquet(db_df)
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
            override=bool(row.get("override", False) or False),
            model_version=str(row["model_version"]),
            charges_inr=float(row["charges_inr"]) if ("charges_inr" in row and pd.notna(row["charges_inr"])) else None,
            initial_sl_price=(
                float(row["initial_sl_price"])
                if ("initial_sl_price" in row and pd.notna(row["initial_sl_price"]))
                else float(row["sl_price"])
            ),
        )
        upsert_trade(self._engine, _to_db_row(record))

    @staticmethod
    def _compute_realized_r(row: pd.Series) -> float | None:
        pnl_usd = pd.to_numeric(row.get("pnl_usd"), errors="coerce")
        entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
        initial_sl_price = pd.to_numeric(row.get("initial_sl_price"), errors="coerce")
        sl_price = initial_sl_price if pd.notna(initial_sl_price) else pd.to_numeric(row.get("sl_price"), errors="coerce")
        contracts = pd.to_numeric(row.get("contracts"), errors="coerce")
        if pd.isna(pnl_usd) or pd.isna(entry_price) or pd.isna(sl_price) or pd.isna(contracts):
            return None
        risk_usd = abs(float(entry_price) - float(sl_price)) * float(contracts)
        if risk_usd <= 0:
            return None
        return float(float(pnl_usd) / risk_usd)

    def load_closed_override_events(self, lookback_days: int = 7) -> list[tuple[datetime, float]]:
        df = self._read_parquet()
        if df.empty or "override" not in df.columns:
            return []

        now = datetime.now(UTC)
        since = now - timedelta(days=max(1, int(lookback_days)))
        ts = pd.to_datetime(df["timestamp_exit"], errors="coerce", utc=True)
        override_mask = df["override"].fillna(False).astype(bool)
        closed_mask = ts.notna()
        recent_mask = ts >= pd.Timestamp(since)
        filtered = df.loc[override_mask & closed_mask & recent_mask].copy()
        if filtered.empty:
            return []
        filtered["timestamp_exit"] = pd.to_datetime(filtered["timestamp_exit"], errors="coerce", utc=True)
        filtered = filtered.sort_values("timestamp_exit")

        events: list[tuple[datetime, float]] = []
        for _, row in filtered.iterrows():
            realized_r = self._compute_realized_r(row)
            if realized_r is None:
                continue
            exit_ts = pd.to_datetime(row.get("timestamp_exit"), errors="coerce", utc=True)
            if pd.isna(exit_ts):
                continue
            events.append((exit_ts.to_pydatetime(), float(realized_r)))
        return events

    def get_closed_override_event(self, trade_id: str) -> tuple[datetime, float] | None:
        df = self._read_parquet()
        if df.empty or "override" not in df.columns:
            return None

        row_df = df[df["trade_id"].astype(str) == str(trade_id)]
        if row_df.empty:
            return None
        row = row_df.iloc[-1]
        if not bool(row.get("override", False) or False):
            return None
        exit_ts = pd.to_datetime(row.get("timestamp_exit"), errors="coerce", utc=True)
        if pd.isna(exit_ts):
            return None
        realized_r = self._compute_realized_r(row)
        if realized_r is None:
            return None
        return (exit_ts.to_pydatetime(), float(realized_r))
