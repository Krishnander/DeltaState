import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

logger = logging.getLogger("state_machine.regime_classifier")

class DynamicRegimeClassifier:
    """
    Implements Dynamic Regime Clustering using Gaussian Hidden Markov Models (HMM)
    or Gaussian Mixture Models (GMM) per feature.
    Collapses multi-feature state space into a tractable set of 8-12 states (e.g. 2 states per feature for 3 features = 2^3 = 8 states).
    """

    def __init__(self, n_states_per_feature: int = 2, model_type: str = "hmm", random_state: int = 42):
        self.n_states_per_feature = n_states_per_feature
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.fitted_models: Dict[str, Union[GaussianHMM, GaussianMixture]] = {}

    def fit_predict_feature(self, feature_series: pd.Series, feature_name: str) -> np.ndarray:
        """
        Fits a 2-state HMM or GMM on a single feature series and returns predicted state labels (0 or 1).
        Ensures consistent labeling (0 = Low/Bearish, 1 = High/Bullish).
        """
        vals = feature_series.dropna().values.reshape(-1, 1)
        if len(vals) < self.n_states_per_feature * 2:
            logger.warning(f"Not enough valid data points to fit regime model for {feature_name}. Returning zeros.")
            return np.zeros(len(feature_series), dtype=int)

        if self.model_type == "hmm":
            model = GaussianHMM(
                n_components=self.n_states_per_feature,
                covariance_type="diag",
                n_iter=100,
                random_state=self.random_state
            )
        else:
            model = GaussianMixture(
                n_components=self.n_states_per_feature,
                covariance_type="diag",
                random_state=self.random_state
            )

        model.fit(vals)
        self.fitted_models[feature_name] = model

        # Predict states
        full_vals = feature_series.fillna(0.0).values.reshape(-1, 1)
        states = model.predict(full_vals)

        # Ensure state 1 has a higher mean feature value than state 0 for consistency
        means = model.means_.flatten() if hasattr(model, "means_") else np.array([0, 1])
        if len(means) >= 2 and means[0] > means[1]:
            states = 1 - states

        return states

    def build_composite_state(self, df_features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Combines 2-state regimes across multiple features into a composite regime state ID (0 to 2^N - 1).
        E.g. for 3 features, binary combinations (S0, S1, S2) form state ID = S0*4 + S1*2 + S2*1 (0..7).
        """
        df = df_features.copy()
        regime_cols = []

        for col in feature_cols:
            if col not in df.columns:
                continue
            reg_col = f"{col}_regime"
            df[reg_col] = self.fit_predict_feature(df[col], col)
            regime_cols.append(reg_col)

        # Calculate composite integer state index
        composite_state = np.zeros(len(df), dtype=int)
        for i, reg_col in enumerate(regime_cols):
            weight = 2 ** (len(regime_cols) - 1 - i)
            composite_state += df[reg_col].values * weight

        df["composite_state_id"] = composite_state
        return df
