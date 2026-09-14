import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger("backtesting.significance")

class StatisticalSignificanceTesting:
    """
    Evaluates statistical significance of state-conditional forward returns:
    - Newey-West adjusted t-statistics (HAC standard errors for autocorrelated returns)
    - Benjamini-Hochberg False Discovery Rate (FDR) control for multiple hypothesis testing
    """

    def __init__(self, max_lags: int = 5, fdr_alpha: float = 0.05):
        self.max_lags = max_lags
        self.fdr_alpha = fdr_alpha

    def compute_newey_west_tstat(self, returns_series: pd.Series, lags: Optional[int] = None) -> Tuple[float, float]:
        """
        Computes Newey-West HAC adjusted t-statistic and p-value for mean return != 0.
        """
        clean_ret = returns_series.dropna()
        if len(clean_ret) < 5:
            return 0.0, 1.0

        if lags is None:
            lags = self.max_lags

        X = np.ones(len(clean_ret))
        y = clean_ret.values

        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            t_stat = float(model.tvalues[0])
            p_val = float(model.pvalues[0])
            return t_stat, p_val
        except Exception as e:
            logger.warning(f"Newey-West OLS computation failed: {e}. Defaulting to t=0, p=1.")
            return 0.0, 1.0

    def test_state_returns_significance(self, df_states_returns: pd.DataFrame, state_col: str = "composite_state_id", horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Runs Newey-West HAC t-tests and Benjamini-Hochberg FDR control across all states and horizons.
        """
        if df_states_returns is None or df_states_returns.empty:
            logger.warning("Empty dataframe passed to significance testing.")
            return pd.DataFrame()

        df = df_states_returns.copy()
        if state_col not in df.columns:
            if "regime_state_id" in df.columns:
                state_col = "regime_state_id"
            else:
                df[state_col] = 0

        raw_results = []
        for state_id, group in df.groupby(state_col):
            for h in horizons:
                ret_col = f"fwd_ret_{h}d"
                if ret_col in group.columns:
                    returns = group[ret_col].dropna()
                    t_stat, p_val = self.compute_newey_west_tstat(returns, lags=h)
                    mean_ret = returns.mean() if len(returns) > 0 else 0.0
                    sample_size = len(returns)
                else:
                    t_stat, p_val, mean_ret, sample_size = 0.0, 1.0, 0.0, 0

                raw_results.append({
                    "state_id": state_id,
                    "horizon_days": h,
                    "mean_return": mean_ret,
                    "sample_size": sample_size,
                    "newey_west_tstat": t_stat,
                    "p_value_raw": p_val
                })

        res_df = pd.DataFrame(raw_results)

        if not res_df.empty and "p_value_raw" in res_df.columns:
            p_vals = res_df["p_value_raw"].values
            rejected, p_adjusted, _, _ = multipletests(p_vals, alpha=self.fdr_alpha, method="fdr_bh")
            res_df["p_value_fdr_bh"] = p_adjusted
            res_df["is_significant_fdr"] = rejected

        return res_df
