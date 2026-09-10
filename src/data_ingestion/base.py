import os
import logging
from typing import Callable, Any, Optional
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("data_ingestion.base")

class BaseFetcher:
    """Base class for data fetchers with retries, caching, and idempotent saving."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def get_cache_path(self, subfolder: str, filename: str) -> str:
        folder = os.path.join(self.raw_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    def is_cached(self, cache_path: str) -> bool:
        return os.path.exists(cache_path) and os.path.getsize(cache_path) > 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=5, max=45),
        reraise=True
    )
    def _fetch_with_retry(self, fetch_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Fetch attempt failed for {fetch_func.__name__} with error: {e}. Retrying...")
            raise e

    def save_parquet_idempotent(self, df: pd.DataFrame, file_path: str) -> bool:
        """Saves dataframe to parquet atomically and idempotently."""
        if df is None or df.empty:
            logger.warning(f"Attempted to save empty DataFrame to {file_path}. Skipping.")
            return False

        temp_path = f"{file_path}.tmp"
        try:
            df.to_parquet(temp_path, index=False)
            os.replace(temp_path, file_path)
            logger.info(f"Successfully saved {len(df)} rows to {file_path}")
            return True
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Failed to save parquet to {file_path}: {e}")
            raise e
