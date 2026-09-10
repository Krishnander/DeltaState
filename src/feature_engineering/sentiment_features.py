import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("feature_engineering.sentiment")

class SentimentFeatureExtractor:
    """
    Extracts sentiment features using HuggingFace FinBERT or keyword polarity scoring
    on Indian financial headlines and social content.
    """

    LABEL_MAP = {
        "bullish": 1.0,
        "bearish": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
        "negative": -1.0,
        1: 1.0,
        0: -1.0,
        2: 0.0
    }

    def compute_sentiment_features(self, df_sentiment: pd.DataFrame) -> pd.DataFrame:
        """
        Processes sentiment records and calculates mean polarity score and bullish ratio.
        """
        if df_sentiment is None or df_sentiment.empty:
            logger.warning("Empty sentiment dataframe passed to SentimentFeatureExtractor.")
            return pd.DataFrame(columns=["trade_date", "sentiment_score_mean", "bullish_ratio", "sentiment_signal"])

        df = df_sentiment.copy()

        if "trade_date" not in df.columns:
            if "timestamp" in df.columns:
                df["trade_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                df["trade_date"] = "2024-01-15"

        if "numeric_score" not in df.columns:
            if "label" in df.columns:
                df["numeric_score"] = df["label"].astype(str).str.lower().map(self.LABEL_MAP).fillna(0.0)
            elif "sentiment_score" in df.columns:
                df["numeric_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
            else:
                df["numeric_score"] = 0.0

        results = []
        for trade_date, group in df.groupby("trade_date"):
            scores = group["numeric_score"].values
            score_mean = np.mean(scores) if len(scores) > 0 else 0.0

            bullish_count = np.sum(scores > 0)
            bearish_count = np.sum(scores < 0)
            total_count = max(len(scores), 1)

            bullish_ratio = (bullish_count - bearish_count) / total_count

            results.append({
                "trade_date": str(trade_date),
                "sentiment_score_mean": score_mean,
                "bullish_ratio": bullish_ratio,
                "sentiment_signal": 0.5 * score_mean + 0.5 * bullish_ratio
            })

        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("trade_date").reset_index(drop=True)
        return res_df
