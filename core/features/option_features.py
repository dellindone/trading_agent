import pandas as pd

from core.utils.market_calendar import market_calendar


def _index_to_local_dates(index: pd.Index, tz: str = "Asia/Kolkata") -> pd.Index:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(tz)
    else:
        dt_index = dt_index.tz_convert(tz)
    return pd.Index(dt_index.date)


def compute_option_features(df: pd.DataFrame, instrument: str, tf: str = "5m") -> pd.DataFrame:
    featured = df.copy()

    pcr_premium = featured["pe_premium"] / featured["ce_premium"]
    pcr_mean_20 = pcr_premium.rolling(20, min_periods=1).mean()
    pcr_std_20 = pcr_premium.rolling(20, min_periods=1).std()

    featured["pcr_premium"] = pcr_premium
    featured["pcr_zscore_20"] = (pcr_premium - pcr_mean_20) / pcr_std_20.replace(0.0, pd.NA)

    featured["atm_premium_total"] = featured["ce_premium"] + featured["pe_premium"]
    featured["atm_premium_ma_10"] = featured["atm_premium_total"].rolling(10, min_periods=1).mean()
    featured["premium_expansion"] = (
        featured["atm_premium_total"] > (featured["atm_premium_ma_10"] * 1.1)
    ).astype(int)
    featured["premium_contraction"] = (
        featured["atm_premium_total"] < (featured["atm_premium_ma_10"] * 0.9)
    ).astype(int)

    featured["ce_pe_skew"] = (
        (featured["ce_premium"] - featured["pe_premium"]) / featured["atm_premium_total"]
    )
    featured["iv_proxy"] = featured["vix"]
    local_dates = _index_to_local_dates(featured.index, tz="Asia/Kolkata")
    unique_dates = local_dates.unique()
    normalized_instrument = instrument.upper()
    expiry_map = {
        date_value: market_calendar.days_to_next_expiry(normalized_instrument, date_value)
        for date_value in unique_dates
    }
    featured["dte_proxy"] = pd.Series(local_dates, index=featured.index).map(expiry_map).astype("Int64")
    featured["theta_decay_flag"] = (featured["dte_proxy"] <= 1).astype(int)

    return featured
