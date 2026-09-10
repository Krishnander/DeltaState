import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional
from src.data_ingestion.base import BaseFetcher

logger = logging.getLogger("data_ingestion.sentiment")

class SentimentFetcher(BaseFetcher):
    """
    Ingests and caches financial news/social sentiment datasets (HuggingFace Twitter Financial News & AI4Invest-Indian-FinBERT).
    """

    SUBFOLDER = "sentiment"

    AVAILABLE_DATASETS = {
        "twitter-financial-news": "zeroshot/twitter-financial-news-sentiment",
        "indian-finbert": "AI4Invest/Indian-FinBERT-headlines"
    }

    def load_sentiment_dataset(self, dataset_key: str = "twitter-financial-news") -> Optional[pd.DataFrame]:
        """
        Ingests and caches financial news sentiment records for a specific feed.
        """
        filename = f"sentiment_{dataset_key}.parquet"
        cache_path = self.get_cache_path(self.SUBFOLDER, filename)

        if self.is_cached(cache_path):
            logger.info(f"Cache hit for sentiment dataset '{dataset_key}': {cache_path}")
            return pd.read_parquet(cache_path)

        dataset_repo = self.AVAILABLE_DATASETS.get(dataset_key, dataset_key)
        logger.info(f"Loading sentiment dataset from Hugging Face: {dataset_repo}")
        try:
            try:
                from datasets import load_dataset
                ds = load_dataset(dataset_repo)
                if hasattr(ds, 'keys') and 'train' in ds:
                    df = ds['train'].to_pandas()
                elif hasattr(ds, 'to_pandas'):
                    df = ds.to_pandas()
                else:
                    df = pd.DataFrame(ds)
            except Exception as inner_e:
                logger.warning(f"Could not load Hugging Face dataset online ({inner_e}). Initializing structured schema fallback.")
                df = pd.DataFrame(columns=["timestamp", "source", "headline", "text", "sentiment_score", "label"])

            if not df.empty:
                df["dataset_key"] = dataset_key
                self.save_parquet_idempotent(df, cache_path)
            return df
        except Exception as e:
            logger.error(f"Failed to ingest sentiment dataset {dataset_key}: {e}")
            return None
