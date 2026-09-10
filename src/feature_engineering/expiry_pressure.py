import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("feature_engineering.expiry_pressure")

class ExpiryPressureMetrics:
    """
    Computes physical settlement / rollover expiry pressure metrics.
    Expiry Pressure = (Open Interest in expiring contract) / (Average Daily Cash Volume)
    """

    def compute(self, df_fno_bhavcopy: pd.DataFrame, df_equity_bhavcopy: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Computes Expiry Pressure per symbol and trade date.
        """
        if df_fno_bhavcopy is None or df_fno_bhavcopy.empty:
            logger.warning("Empty F&O bhavcopy passed to Expiry Pressure calculation.")
            return pd.DataFrame(columns=["trade_date", "symbol", "expiring_oi", "avg_cash_volume", "expiry_pressure_ratio"])

        df_fno = df_fno_bhavcopy.loc[:, ~df_fno_bhavcopy.columns.duplicated()].copy().reset_index(drop=True)

        # Rename standard columns
        col_map = {
            "TradDt": "trade_date",
            "SYMBOL": "symbol",
            "TckrSymb": "symbol",
            "OPEN_INT": "open_interest",
            "OpnIntrst": "open_interest",
            "EXPIRY_DT": "expiry_date",
            "XpryDt": "expiry_date"
        }
        df_fno = df_fno.rename(columns={k: v for k, v in col_map.items() if k in df_fno.columns})
        df_fno = df_fno.loc[:, ~df_fno.columns.duplicated()].copy().reset_index(drop=True)

        df_eq = None
        if df_equity_bhavcopy is not None and not df_equity_bhavcopy.empty:
            df_eq = df_equity_bhavcopy.loc[:, ~df_equity_bhavcopy.columns.duplicated()].copy().reset_index(drop=True)
            eq_col_map = {"TradDt": "trade_date", "SYMBOL": "symbol", "TckrSymb": "symbol"}
            df_eq = df_eq.rename(columns={k: v for k, v in eq_col_map.items() if k in df_eq.columns})
            df_eq = df_eq.loc[:, ~df_eq.columns.duplicated()].copy().reset_index(drop=True)

        for col in ["open_interest"]:
            if col in df_fno.columns:
                df_fno[col] = pd.to_numeric(df_fno[col].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)

        results = []
        for (trade_date, symbol), group in df_fno.groupby(["trade_date", "symbol"]):
            td_str = str(trade_date.iloc[0]) if isinstance(trade_date, (pd.Series, pd.Index)) else str(trade_date)
            sym_str = str(symbol.iloc[0]) if isinstance(symbol, (pd.Series, pd.Index)) else str(symbol)

            if "expiry_date" in group.columns:
                group["dt_exp"] = pd.to_datetime(group["expiry_date"], errors="coerce")
                group["dt_trade"] = pd.to_datetime(td_str, errors="coerce")
                group["dte"] = (group["dt_exp"] - group["dt_trade"]).dt.days
                expiring_contracts = group[group["dte"] >= 0].sort_values("dte")
                if not expiring_contracts.empty:
                    near_exp = expiring_contracts["expiry_date"].iloc[0]
                    expiring_oi = expiring_contracts[expiring_contracts["expiry_date"] == near_exp]["open_interest"].sum()
                else:
                    expiring_oi = group["open_interest"].sum()
            else:
                expiring_oi = group["open_interest"].sum()

            cash_vol = None
            if df_eq is not None and "symbol" in df_eq.columns and "trade_date" in df_eq.columns:
                td_series = df_eq["trade_date"].astype(str)
                sym_series = df_eq["symbol"].astype(str)
                eq_match = df_eq[(td_series == td_str) & (sym_series == sym_str)]
                if not eq_match.empty and "TTL_TRD_QNTY" in eq_match.columns:
                    cash_vol = pd.to_numeric(eq_match["TTL_TRD_QNTY"].iloc[0], errors="coerce")

            if cash_vol is None or pd.isna(cash_vol) or cash_vol <= 0:
                cash_vol = max(expiring_oi * 0.1, 10000.0)

            ratio = expiring_oi / cash_vol if cash_vol > 0 else 0.0

            results.append({
                "trade_date": td_str,
                "symbol": sym_str,
                "expiring_oi": expiring_oi,
                "avg_cash_volume": cash_vol,
                "expiry_pressure_ratio": ratio
            })

        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("trade_date").reset_index(drop=True)
        return res_df
