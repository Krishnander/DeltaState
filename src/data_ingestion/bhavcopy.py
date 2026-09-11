import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, List
from nselib import derivatives, capital_market
from src.data_ingestion.base import BaseFetcher

logger = logging.getLogger("data_ingestion.bhavcopy")

class BhavcopyFetcher(BaseFetcher):
    """
    Fetches F&O and Equity Cash Market Bhavcopies from NSE via nselib.
    Calculates prices, volume, and OI data for target symbols (e.g. NIFTY).
    """

    SUBFOLDER = "bhavcopy"

    def _filter_symbol(self, df: pd.DataFrame, target_symbol: Optional[str]) -> pd.DataFrame:
        if df is None or df.empty or not target_symbol:
            return df

        sym_col = None
        for col in ["SYMBOL", "TckrSymb", "symbol"]:
            if col in df.columns:
                sym_col = col
                break

        if sym_col:
            return df[df[sym_col] == target_symbol].copy()
        return df

    def fetch_fno_bhavcopy(self, trade_date: str, target_symbol: Optional[str] = "NIFTY") -> Optional[pd.DataFrame]:
        """
        Fetches F&O Bhavcopy for a given trade date ('YYYY-MM-DD' or 'DD-MM-YYYY').
        Optionally filters for target symbol (e.g. NIFTY).
        """
        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d")
            formatted_date = dt.strftime("%d-%m-%Y")
            file_date_str = dt.strftime("%Y%m%d")
        except ValueError:
            dt = datetime.strptime(trade_date, "%d-%m-%Y")
            formatted_date = dt.strftime("%d-%m-%Y")
            file_date_str = dt.strftime("%Y%m%d")

        filename = f"fno_bhavcopy_{file_date_str}.parquet"
        cache_path = self.get_cache_path(self.SUBFOLDER, filename)

        if self.is_cached(cache_path):
            logger.info(f"Cache hit for F&O Bhavcopy on {formatted_date}: {cache_path}")
            df = pd.read_parquet(cache_path)
            return self._filter_symbol(df, target_symbol)

        logger.info(f"Fetching F&O Bhavcopy from NSE for date: {formatted_date}")
        try:
            df = self._fetch_with_retry(derivatives.fno_bhav_copy, trade_date=formatted_date)
            if df is not None and not df.empty:
                df["trade_date"] = dt.strftime("%Y-%m-%d")
                self.save_parquet_idempotent(df, cache_path)
                return self._filter_symbol(df, target_symbol)
            else:
                logger.warning(f"No F&O Bhavcopy data returned for {formatted_date} (likely market holiday).")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch F&O Bhavcopy for {formatted_date}: {e}")
            return None

    def fetch_equity_bhavcopy(self, trade_date: str, target_symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetches Equity Cash Market Bhavcopy (with delivery volume) for a given trade date.
        Essential for stock-level cash volume required by Expiry Pressure metrics.
        """
        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d")
            formatted_date = dt.strftime("%d-%m-%Y")
            file_date_str = dt.strftime("%Y%m%d")
        except ValueError:
            dt = datetime.strptime(trade_date, "%d-%m-%Y")
            formatted_date = dt.strftime("%d-%m-%Y")
            file_date_str = dt.strftime("%Y%m%d")

        filename = f"equity_bhavcopy_{file_date_str}.parquet"
        cache_path = self.get_cache_path(self.SUBFOLDER, filename)

        if self.is_cached(cache_path):
            logger.info(f"Cache hit for Equity Bhavcopy on {formatted_date}: {cache_path}")
            df = pd.read_parquet(cache_path)
            return self._filter_symbol(df, target_symbol)

        logger.info(f"Fetching Equity Cash Bhavcopy from NSE for date: {formatted_date}")
        try:
            df = self._fetch_with_retry(capital_market.bhav_copy_equities, trade_date=formatted_date)
            if df is not None and not df.empty:
                df["trade_date"] = dt.strftime("%Y-%m-%d")
                self.save_parquet_idempotent(df, cache_path)
                return self._filter_symbol(df, target_symbol)
            else:
                logger.warning(f"No Equity Bhavcopy data returned for {formatted_date} (likely market holiday).")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch Equity Bhavcopy for {formatted_date}: {e}")
            return None

    def fetch_for_date(self, trade_date: str, target_symbol: Optional[str] = "NIFTY") -> Optional[pd.DataFrame]:
        """Convenience wrapper for F&O Bhavcopy."""
        return self.fetch_fno_bhavcopy(trade_date, target_symbol)
