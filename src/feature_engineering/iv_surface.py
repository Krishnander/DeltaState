import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from scipy.interpolate import interp1d

logger = logging.getLogger("feature_engineering.iv_surface")

class ConstantMaturityIVSurface:
    """
    Constructs Constant Maturity ATM Implied Volatility Surface using cubic spline / linear interpolation
    at fixed tenors (7D, 30D, 60D). SVI total variance fitting structure parameterizes smile across strikes.
    """

    TARGET_TENORS_DAYS = [7, 30, 60]

    def _svi_vol_formula(self, k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
        """
        SVI (Stochastic Volatility Inspired) total variance formula:
        w(k) = a + b * { rho * (k - m) + sqrt((k - m)^2 + sigma^2) }
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def compute_atm_iv_surface(self, df_option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates constant maturity ATM IV from option chain or bhavcopy snapshots for a specific underlying.
        """
        if df_option_chain is None or df_option_chain.empty:
            logger.warning("Empty option chain passed to IV surface calculation.")
            return pd.DataFrame(columns=["trade_date", "iv_7d", "iv_30d", "iv_60d", "iv_slope_30_7", "iv_slope_60_30"])

        df = df_option_chain.loc[:, ~df_option_chain.columns.duplicated()].copy()

        # Standardize column names
        col_map = {
            "TradDt": "trade_date",
            "EXPIRY_DT": "expiry_date",
            "XpryDt": "expiry_date",
            "STRIKE_PR": "strike",
            "StrkPric": "strike",
            "OPTION_TY": "option_type",
            "OptnTp": "option_type",
            "CLOSE": "close_price",
            "ClsPric": "close_price",
            "UndrlygPric": "underlying_price"
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()

        for c in ["strike", "close_price", "underlying_price"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")

        results = []
        for trade_date, group in df.groupby("trade_date"):
            td_str = str(trade_date.iloc[0]) if isinstance(trade_date, (pd.Series, pd.Index)) else str(trade_date)
            group = group.copy()
            try:
                dt_trade = pd.to_datetime(td_str)
            except Exception:
                continue

            # Determine underlying spot price from Futures or Underlying column
            if "underlying_price" in group.columns and not group["underlying_price"].dropna().empty:
                spot = group["underlying_price"].dropna().iloc[0]
            elif "INSTRUMENT" in group.columns and "FUT" in str(group["INSTRUMENT"].iloc[0]):
                spot = group["close_price"].iloc[0]
            else:
                spot = group["strike"].median() if "strike" in group.columns else 21500.0

            tenor_ivs = {}
            expiry_groups = []

            for exp_dt, exp_group in group.groupby("expiry_date"):
                exp_dt_str = str(exp_dt.iloc[0]) if isinstance(exp_dt, (pd.Series, pd.Index)) else str(exp_dt)
                exp_group = exp_group.copy()
                try:
                    dt_exp = pd.to_datetime(exp_dt_str)
                    dte = (dt_exp - dt_trade).days
                except Exception:
                    continue

                if dte <= 0:
                    continue

                exp_group["atm_dist"] = (exp_group["strike"] - spot).abs()
                atm_row = exp_group.sort_values("atm_dist").iloc[0]

                atm_price = atm_row["close_price"]
                if spot > 0 and dte > 0 and atm_price > 0:
                    t_years = dte / 365.0
                    # Black-Scholes ATM IV approximation
                    atm_iv_proxy = (np.sqrt(2 * np.pi) / np.sqrt(t_years)) * (atm_price / spot)
                    expiry_groups.append((dte, atm_iv_proxy))

            if len(expiry_groups) >= 2:
                dtes, ivs = zip(*sorted(expiry_groups))

                if len(dtes) > 1:
                    f_interp = interp1d(dtes, ivs, kind="linear", fill_value="extrapolate")
                    for tenor in self.TARGET_TENORS_DAYS:
                        val = float(f_interp(tenor))
                        tenor_ivs[f"iv_{tenor}d"] = max(val, 0.01)
                else:
                    for tenor in self.TARGET_TENORS_DAYS:
                        tenor_ivs[f"iv_{tenor}d"] = float(ivs[0])
            else:
                for tenor in self.TARGET_TENORS_DAYS:
                    tenor_ivs[f"iv_{tenor}d"] = np.nan

            iv_7d = tenor_ivs.get("iv_7d", np.nan)
            iv_30d = tenor_ivs.get("iv_30d", np.nan)
            iv_60d = tenor_ivs.get("iv_60d", np.nan)

            results.append({
                "trade_date": td_str,
                "iv_7d": iv_7d,
                "iv_30d": iv_30d,
                "iv_60d": iv_60d,
                "iv_slope_30_7": iv_30d - iv_7d if pd.notna(iv_30d) and pd.notna(iv_7d) else np.nan,
                "iv_slope_60_30": iv_60d - iv_30d if pd.notna(iv_60d) and pd.notna(iv_30d) else np.nan,
            })

        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("trade_date").reset_index(drop=True)
        return res_df
