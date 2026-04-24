"""Database helpers for shadow paper-trade persistence."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime

from sqlalchemy import Column, Date, DateTime, Float, Integer, MetaData, Table, Text, create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

metadata = MetaData()

paper_trade = Table(
    "paper_trade",
    metadata,
    Column("trade_id", Text, primary_key=True),
    Column("instrument", Text, nullable=False),
    Column("timestamp_entry", DateTime, nullable=False),
    Column("timestamp_exit", DateTime, nullable=True),
    Column("direction", Integer, nullable=False),      # Nifty: 1=CE/0=PE | BTC: 1=long/-1=short

    # Nifty options columns (nullable — BTC leaves these None)
    Column("strike", Integer, nullable=True),
    Column("expiry_date", Date, nullable=True),
    Column("option_type", Text, nullable=True),        # Nifty: CE/PE | BTC: LONG/SHORT
    Column("entry_premium", Float, nullable=True),
    Column("exit_premium", Float, nullable=True),
    Column("lot_size", Integer, nullable=True),
    Column("trail_bin", Text, nullable=True),
    Column("trail_tf", Text, nullable=True),
    Column("vix_at_entry", Float, nullable=True),

    # BTC futures columns (nullable — Nifty leaves these None)
    Column("entry_price", Float, nullable=True),
    Column("exit_price", Float, nullable=True),
    Column("contracts", Float, nullable=True),
    Column("pnl_usd", Float, nullable=True),
    Column("charges_usd", Float, nullable=True),

    # Shared columns
    Column("sl_price", Float, nullable=False),
    Column("target_price", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("direction_prob", Float, nullable=False),
    Column("exit_reason", Text, nullable=True),
    Column("pnl_gross", Float, nullable=True),
    Column("pnl_net", Float, nullable=True),           # always in INR
    Column("charges", Float, nullable=True),           # always in INR
    Column("atr_at_entry", Float, nullable=False),
    Column("model_version", Text, nullable=False),
)


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def get_engine() -> Engine | None:
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if not raw_url:
        logger.warning("DATABASE_URL not set; DB persistence disabled.")
        return None
    db_url = _normalize_database_url(raw_url)
    try:
        return create_engine(db_url, future=True, pool_pre_ping=True)
    except Exception as exc:
        logger.warning("Failed to create DB engine: %s", exc)
        return None


def ensure_table_exists(engine: Engine | None) -> None:
    if engine is None:
        return
    try:
        metadata.create_all(engine, tables=[paper_trade], checkfirst=True)
    except Exception as exc:
        logger.warning("Failed to ensure paper_trade table exists: %s", exc)


def _normalize_record(record: dict) -> dict:
    normalized = dict(record)

    ts_entry = normalized.get("timestamp_entry")
    if ts_entry is not None and not isinstance(ts_entry, datetime):
        try:
            normalized["timestamp_entry"] = datetime.fromisoformat(str(ts_entry))
        except Exception:
            normalized["timestamp_entry"] = ts_entry
    ts_exit = normalized.get("timestamp_exit")
    if ts_exit is not None and not isinstance(ts_exit, datetime):
        try:
            normalized["timestamp_exit"] = datetime.fromisoformat(str(ts_exit))
        except Exception:
            normalized["timestamp_exit"] = ts_exit
    expiry = normalized.get("expiry_date")
    if expiry is not None and not isinstance(expiry, date):
        try:
            # str(pd.Timestamp) → "2026-04-29 00:00:00"; [:10] gives "2026-04-29"
            if hasattr(expiry, "date"):
                normalized["expiry_date"] = expiry.date()
            else:
                normalized["expiry_date"] = date.fromisoformat(str(expiry)[:10])
        except Exception:
            normalized["expiry_date"] = None

    # Coerce pd.NaT / NaN timestamps to None so psycopg2 doesn't choke
    for ts_field in ("timestamp_entry", "timestamp_exit"):
        val = normalized.get(ts_field)
        if val is not None and not isinstance(val, datetime):
            try:
                normalized[ts_field] = datetime.fromisoformat(str(val)[:19])
            except Exception:
                normalized[ts_field] = None

    return normalized


def upsert_trade(engine: Engine | None, record: dict) -> None:
    if engine is None:
        return
    try:
        normalized = _normalize_record(record)
        stmt = pg_insert(paper_trade).values(**normalized)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in paper_trade.columns
            if c.name != "trade_id"
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[paper_trade.c.trade_id],
            set_=update_cols,
        )
        with engine.begin() as conn:
            conn.execute(stmt)
    except Exception as exc:
        logger.warning("Failed to upsert trade into DB: %s", exc)
