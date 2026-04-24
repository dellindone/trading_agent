import pandas as pd


def compute_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    vix = featured["vix"]

    vix_mean_20 = vix.rolling(20, min_periods=1).mean()
    vix_std_20 = vix.rolling(20, min_periods=1).std()
    vix_mean_60 = vix.rolling(60, min_periods=1).mean()
    vix_std_60 = vix.rolling(60, min_periods=1).std()

    featured["vix_zscore_20"] = (vix - vix_mean_20) / vix_std_20.replace(0.0, pd.NA)
    featured["vix_zscore_60"] = (vix - vix_mean_60) / vix_std_60.replace(0.0, pd.NA)
    featured["vix_pct_change_1"] = vix.pct_change()
    featured["vix_ma_ratio_5"] = vix / vix.rolling(5, min_periods=1).mean()
    featured["vix_ma_ratio_20"] = vix / vix.rolling(20, min_periods=1).mean()

    featured["vix_high_flag"] = (vix > 25).astype(int)
    featured["vix_extreme_flag"] = (vix > 30).astype(int)
    featured["vix_low_flag"] = (vix < 12).astype(int)

    featured["vix_regime"] = 1
    featured.loc[vix < 12, "vix_regime"] = 0
    featured.loc[(vix >= 18) & (vix <= 25), "vix_regime"] = 2
    featured.loc[(vix > 25) & (vix <= 30), "vix_regime"] = 3
    featured.loc[vix > 30, "vix_regime"] = 4

    featured["vix_intraday_spike"] = (vix > (vix.shift(1) * 1.05)).astype(int)

    return featured
