import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional
import indiaopt
from nselib import derivatives
from src.data_ingestion.base import BaseFetcher

logger = logging.getLogger("data_ingestion.option_chain")

class OptionChainFetcher(BaseFetcher):
    """
    Fetches live / snapshot option chain data for indices/stocks (e.g. NIFTY) from NSE.
    Uses indiaopt.fetch_option_chain as primary engine, fallback to nselib.
    """

    SUBFOLDER = "option_chain"

    def fetch_snapshot(self, symbol: str = "NIFTY", is_index: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetches current option chain snapshot for symbol using indiaopt with nselib fallback.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"option_chain_{symbol}_{timestamp_str}.parquet"
        cache_path = self.get_cache_path(self.SUBFOLDER, filename)

        logger.info(f"Fetching live option chain snapshot for {symbol} using indiaopt")
        try:
            # Primary: indiaopt
            result = self._fetch_with_retry(indiaopt.fetch_option_chain, symbol=symbol, is_index=is_index)
            if result and hasattr(result, "rows") and result.rows:
                records = []
                for row in result.rows:
                    rec = {
                        "strike_price": row.strike_price,
                        "underlying_price": result.underlying_price,
                        "expiry_date": str(row.expiry_date) if hasattr(row, "expiry_date") else None
                    }
                    if hasattr(row, "call") and row.call:
                        rec["ce_oi"] = getattr(row.call, "open_interest", 0)
                        rec["ce_volume"] = getattr(row.call, "volume", 0)
                        rec["ce_iv"] = getattr(row.call, "implied_volatility", 0.0)
                        rec["ce_ltp"] = getattr(row.call, "last_price", 0.0)
                    if hasattr(row, "put") and row.put:
                        rec["pe_oi"] = getattr(row.put, "open_interest", 0)
                        rec["pe_volume"] = getattr(row.put, "volume", 0)
                        rec["pe_iv"] = getattr(row.put, "implied_volatility", 0.0)
                        rec["pe_ltp"] = getattr(row.put, "last_price", 0.0)
                    records.append(rec)
                df = pd.DataFrame(records)
                df["fetch_timestamp"] = datetime.now().isoformat()
                df["symbol"] = symbol
                self.save_parquet_idempotent(df, cache_path)
                return df
        except Exception as e:
            logger.warning(f"indiaopt fetch failed for {symbol}: {e}. Trying nselib fallback...")

        try:
            # Fallback: nselib
            df = self._fetch_with_retry(derivatives.nse_live_option_chain, symbol=symbol)
            if df is not None and not df.empty:
                df["fetch_timestamp"] = datetime.now().isoformat()
                df["symbol"] = symbol
                self.save_parquet_idempotent(df, cache_path)
                return df
        except Exception as fallback_e:
            logger.error(f"Failed to fetch option chain snapshot for {symbol}: {fallback_e}")

        return None
