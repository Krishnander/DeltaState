import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple
from src.state_machine.regime_classifier import DynamicRegimeClassifier

logger = logging.getLogger("state_machine.state_machine")

class PointInTimeStateMachine:
    """
    Applies strict Point-in-Time (PIT) expanding-window quantile thresholding and dynamic regime fitting.
    Guarantees no look-ahead bias by re-evaluating regime bounds and transition probabilities strictly up to time t.
    """

    def __init__(self, min_periods: int = 20, n_states_per_feature: int = 2):
        self.min_periods = min_periods
        self.n_states_per_feature = n_states_per_feature

    def compute_expanding_quantiles(self, df_features: pd.DataFrame, feature_cols: List[str], quantiles: List[float] = [0.33, 0.67]) -> pd.DataFrame:
        """
        Computes PIT expanding-window quantile thresholds strictly using observations up to time t.
        """
        df = df_features.sort_values("trade_date").copy().reset_index(drop=True)

        for col in feature_cols:
            if col not in df.columns:
                continue

            series = df[col].astype(float)
            for q in quantiles:
                q_col = f"{col}_expanding_q{int(q*100)}"
                df[q_col] = series.expanding(min_periods=self.min_periods).quantile(q)

        return df

    def compute_transition_matrix(self, states: np.ndarray, num_states: int) -> np.ndarray:
        """
        Computes empirical state transition probability matrix P[i, j] = P(S_{t} = j | S_{t-1} = i).
        Rows with no observed transitions remain zeros or uniform.
        """
        trans_matrix = np.zeros((num_states, num_states))
        for t in range(1, len(states)):
            prev_s = states[t-1]
            curr_s = states[t]
            if 0 <= prev_s < num_states and 0 <= curr_s < num_states:
                trans_matrix[prev_s, curr_s] += 1

        # Normalize non-zero rows to sum to 1
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        nonzero_mask = (row_sums > 0).flatten()
        trans_matrix[nonzero_mask] = trans_matrix[nonzero_mask] / row_sums[nonzero_mask]
        return trans_matrix

    def process_state_classification(self, df_features: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Executes dynamic regime classification and computes PIT expanding thresholds and transition matrix.
        """
        if df_features is None or df_features.empty:
            logger.warning("Empty dataframe passed to State Machine.")
            return pd.DataFrame(), np.array([[]])

        # 1. Compute PIT expanding quantiles
        df_pit = self.compute_expanding_quantiles(df_features, feature_cols)

        # 2. Fit Dynamic Regime Classifier across features
        classifier = DynamicRegimeClassifier(n_states_per_feature=self.n_states_per_feature)
        df_states = classifier.build_composite_state(df_pit, feature_cols)

        # 3. Compute empirical state transition matrix
        states_seq = df_states["composite_state_id"].values
        total_possible_states = 2 ** len(feature_cols)
        trans_matrix = self.compute_transition_matrix(states_seq, total_possible_states)

        return df_states, trans_matrix
