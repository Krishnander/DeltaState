import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("feature_engineering.rpi")

class RetailPositioningIndex:
    """
    Computes Retail Positioning Index (RPI) from Participant-wise Open Interest (OI) data.
    RPI = (Client Net Future Index OI) / (Total Future Index Market OI)
    Excludes the 'TOTAL' row to avoid double-counting aggregate open interest.
    """

    def compute(self, df_participant_oi: pd.DataFrame) -> pd.DataFrame:
        """
        Computes RPI signal for a dataframe of participant OI records.
        """
        if df_participant_oi is None or df_participant_oi.empty:
            logger.warning("Empty participant OI dataframe passed to RPI compute.")
            return pd.DataFrame(columns=["trade_date", "client_net_fut_oi", "total_market_fut_oi", "rpi_raw"])

        df = df_participant_oi.copy()

        # Clean numeric columns
        numeric_cols = [
            "Future Index Long", "Future Index Short",
            "Option Index Call Long", "Option Index Put Long",
            "Option Index Call Short", "Option Index Put Short"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)

        results = []
        for trade_date, group in df.groupby("trade_date"):
            td_str = str(trade_date.iloc[0]) if isinstance(trade_date, (pd.Series, pd.Index)) else str(trade_date)

            # Filter out summary 'TOTAL' / 'Total' rows to prevent double-counting
            participants_only = group[~group["Client Type"].astype(str).str.lower().str.contains("total")]

            if participants_only.empty:
                participants_only = group

            client_row = participants_only[participants_only["Client Type"].astype(str).str.lower().str.contains("client")]

            # Total market OI across actual participant categories (Client, FII, DII, Pro)
            total_long = participants_only["Future Index Long"].sum()
            total_short = participants_only["Future Index Short"].sum()
            total_market_oi = max(total_long, total_short, 1.0)

            if not client_row.empty:
                client_long = client_row["Future Index Long"].values[0]
                client_short = client_row["Future Index Short"].values[0]
                client_net_oi = client_long - client_short
            else:
                client_net_oi = 0.0

            rpi_raw = client_net_oi / total_market_oi

            results.append({
                "trade_date": td_str,
                "client_net_fut_oi": client_net_oi,
                "total_market_fut_oi": total_market_oi,
                "rpi_raw": rpi_raw
            })

        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("trade_date").reset_index(drop=True)
        return res_df
