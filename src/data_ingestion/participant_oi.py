import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional
from nselib import derivatives
from src.data_ingestion.base import BaseFetcher

logger = logging.getLogger("data_ingestion.participant_oi")

class ParticipantOIFetcher(BaseFetcher):
    """
    Fetches participant-wise Open Interest (OI) reports from NSE via nselib.
    Output data includes OI position breakdown by Client, Pro, FII, DII categories.
    """

    SUBFOLDER = "participant_oi"

    def fetch_for_date(self, trade_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches participant-wise OI for a given trade date (format: 'DD-MM-YYYY' or 'YYYY-MM-DD').
        """
        # Standardize date string format for nselib ('DD-MM-YYYY')
        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d")
            formatted_date = dt.strftime("%d-%m-%Y")
            file_date_str = dt.strftime("%Y%m%d")
        except ValueError:
            dt = datetime.strptime(trade_date, "%d-%m-%Y")
            formatted_date = dt.strftime("%d-%m-%Y")
            file_date_str = dt.strftime("%Y%m%d")

        filename = f"participant_oi_{file_date_str}.parquet"
        cache_path = self.get_cache_path(self.SUBFOLDER, filename)

        if self.is_cached(cache_path):
            logger.info(f"Cache hit for participant OI on {formatted_date}: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching participant OI from NSE for date: {formatted_date}")
        try:
            df = self._fetch_with_retry(derivatives.participant_wise_open_interest, trade_date=formatted_date)
            if df is not None and not df.empty:
                # Add metadata columns
                df["trade_date"] = dt.strftime("%Y-%m-%d")
                self.save_parquet_idempotent(df, cache_path)
                return df
            else:
                logger.warning(f"No participant OI data returned for {formatted_date} (likely market holiday).")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch participant OI for {formatted_date}: {e}")
            return None
