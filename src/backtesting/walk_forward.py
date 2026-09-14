import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple
from src.state_machine.state_machine import PointInTimeStateMachine

logger = logging.getLogger("backtesting.walk_forward")

class WalkForwardValidator:
    """
    Implements Walk-Forward Validation for Out-Of-Sample (OOS) regime state generation.
    Splits history into rolling / expanding training windows and tests on unseen future folds to prevent look-ahead bias.
    """

    def __init__(self, train_window_days: int = 252, test_window_days: int = 63):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days

    def generate_walk_forward_folds(self, df_features: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates list of (train_df, test_df) tuples over expanding / rolling windows.
        """
        df = df_features.sort_values("trade_date").copy().reset_index(drop=True)
        n = len(df)
        folds = []

        if n <= self.train_window_days:
            logger.warning("Dataset smaller than train window. Returning single fold with all data.")
            return [(df, df)]

        start_idx = 0
        while start_idx + self.train_window_days < n:
            train_end = start_idx + self.train_window_days
            test_end = min(train_end + self.test_window_days, n)

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            folds.append((train_df, test_df))
            start_idx += self.test_window_days

        return folds

    def run_walk_forward_state_generation(self, df_features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Fits regime state models on historical fold t and classifies out-of-sample state on test fold t+1.
        """
        folds = self.generate_walk_forward_folds(df_features)
        oos_results = []

        for fold_idx, (train_df, test_df) in enumerate(folds):
            sm = PointInTimeStateMachine(min_periods=1, n_states_per_feature=2)
            train_states, _ = sm.process_state_classification(train_df, feature_cols)
            test_states, _ = sm.process_state_classification(test_df, feature_cols)
            test_states["fold_id"] = fold_idx
            oos_results.append(test_states)

        if oos_results:
            df_oos = pd.concat(oos_results, ignore_index=True)
            df_oos = df_oos.drop_duplicates(subset=["trade_date"]).reset_index(drop=True)
            return df_oos
        return df_features
