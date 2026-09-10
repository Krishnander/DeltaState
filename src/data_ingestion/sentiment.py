import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional
from src.data_ingestion.base import BaseFetcher

logger = logging.getLogger("data_ingestion.sentiment")

class SentimentFetcher(BaseFetcher):
    """
    Ingests and caches financial news/social sentiment datasets (e.g. HuggingFace feeds).
    """

    SUBFOLDER = "sentiment"

    def load_sentiment_dataset(self, dataset_name: str = "twitter-financial-news-sentiment") -> Optional[pd.DataFrame]:
        """
        Ingests and caches financial news sentiment records.
        """
        filename = f"sentiment_{dataset_name}.parquet"
        cache_path = self.get_cache_path(self.SUBFOLDER, filename)

        if self.is_cached(cache_path):
            logger.info(f"Cache hit for sentiment dataset '{dataset_name}': {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Loading sentiment dataset: {dataset_name}")
        try:
            # Fallback/mock structure if datasets package is not present or offline
            try:
                from datasets import load_dataset
                ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
                df = ds['train'].to_pandas()
            except Exception as inner_e:
                logger.warning(f"Could not load Hugging Face dataset online ({inner_e}). Creating initial structured schema.")
                df = pd.DataFrame(columns=["timestamp", "source", "text", "sentiment_score", "label"])

            if not df.empty:
                self.save_parquet_idempotent(df, cache_path)
            return df
        except Exception as e:
            logger.error(f"Failed to ingest sentiment dataset {dataset_name}: {e}")
            return None
