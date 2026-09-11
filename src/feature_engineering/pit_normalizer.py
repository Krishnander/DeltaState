import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger("feature_engineering.pit_normalizer")

class PITExpandingNormalizer:
    """
    Computes Point-in-Time (PIT) expanding window quantiles and z-scores
    strictly using historical data up to time t to eliminate look-ahead bias.
    """

    def __init__(self, min_periods: int = 20):
        self.min_periods = min_periods

    def normalize_features(self, df_features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Transforms raw feature columns into expanding window PIT percentiles (0.0 to 1.0) and z-scores.
        Requires 'trade_date' column.
        """
        if df_features is None or df_features.empty:
            logger.warning("Empty dataframe passed to PIT normalizer.")
            return df_features

        df = df_features.sort_values("trade_date").copy().reset_index(drop=True)

        for col in feature_cols:
            if col not in df.columns:
                continue

            series = df[col].astype(float)

            # Expanding window mean and std for z-score calculation
            exp_mean = series.expanding(min_periods=self.min_periods).mean()
            exp_std = series.expanding(min_periods=self.min_periods).std().replace(0, np.nan)
            df[f"{col}_pit_zscore"] = (series - exp_mean) / exp_std

            # Expanding window percentiles (PIT)
            pit_percentiles = []
            for i in range(len(series)):
                current_val = series.iloc[i]
                if pd.isna(current_val) or i < self.min_periods - 1:
                    pit_percentiles.append(np.nan)
                else:
                    hist_window = series.iloc[:i+1].dropna()
                    if len(hist_window) == 0:
                        pit_percentiles.append(np.nan)
                    else:
                        pct = (hist_window < current_val).mean()
                        pit_percentiles.append(pct)

            df[f"{col}_pit_quantile"] = pit_percentiles

        return df
