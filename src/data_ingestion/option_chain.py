import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional
from nselib import derivatives
from src.data_ingestion.base import BaseFetcher

logger = logging.getLogger("data_ingestion.option_chain")

class OptionChainFetcher(BaseFetcher):
    """
    Fetches live / snapshot option chain data for indices/stocks (e.g. NIFTY) from NSE.
    """

    SUBFOLDER = "option_chain"

    def fetch_snapshot(self, symbol: str = "NIFTY") -> Optional[pd.DataFrame]:
        """
        Fetches current option chain snapshot for symbol.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"option_chain_{symbol}_{timestamp_str}.parquet"
        cache_path = self.get_cache_path(self.SUBFOLDER, filename)

        logger.info(f"Fetching live option chain snapshot for {symbol}")
        try:
            df = self._fetch_with_retry(derivatives.nse_live_option_chain, symbol=symbol)
            if df is not None and not df.empty:
                df["fetch_timestamp"] = datetime.now().isoformat()
                df["symbol"] = symbol
                self.save_parquet_idempotent(df, cache_path)
                return df
            else:
                logger.warning(f"No option chain data returned for {symbol}.")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {symbol}: {e}")
            return None
