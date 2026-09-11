import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

logger = logging.getLogger("state_machine.regime_classifier")

class DynamicRegimeClassifier:
    """
    Implements Dynamic Regime Clustering using a joint Multivariate Gaussian Hidden Markov Model (HMM)
    or Gaussian Mixture Model (GMM) fitted across multiple normalized features simultaneously.
    Uses BIC/AIC to select optimal state count (~3 states).
    """

    def __init__(self, n_components: int = 3, model_type: str = "hmm", random_state: int = 42):
        self.n_components = n_components
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model: Optional[Union[GaussianHMM, GaussianMixture]] = None

    def fit_predict_multivariate(self, df_features: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, float, float]:
        """
        Fits a multivariate Gaussian HMM or GMM jointly across all target features.
        Returns dataframe with 'regime_state_id', BIC, and AIC scores.
        """
        df = df_features.copy()
        valid_cols = [c for c in feature_cols if c in df.columns]

        if not valid_cols:
            logger.warning("No valid feature columns provided for multivariate regime clustering.")
            df["regime_state_id"] = 0
            return df, 0.0, 0.0

        feature_matrix = df[valid_cols].fillna(0.0).values

        if len(feature_matrix) < self.n_components * 2:
            logger.warning(f"Not enough observations ({len(feature_matrix)}) to fit {self.n_components}-state multivariate model. Assigning zero state.")
            df["regime_state_id"] = 0
            return df, 0.0, 0.0

        if self.model_type == "hmm":
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=200,
                random_state=self.random_state
            )
            model.fit(feature_matrix)
            states = model.predict(feature_matrix)
            log_likelihood = model.score(feature_matrix)

            # Calculate AIC / BIC
            n_features = len(valid_cols)
            n_params = self.n_components * (self.n_components - 1) + 2 * self.n_components * n_features
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(len(feature_matrix)) * n_params
        else:
            model = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                random_state=self.random_state
            )
            model.fit(feature_matrix)
            states = model.predict(feature_matrix)
            aic = model.aic(feature_matrix)
            bic = model.bic(feature_matrix)

        self.model = model
        df["regime_state_id"] = states
        df["composite_state_id"] = states

        return df, float(aic), float(bic)

    def find_optimal_states(self, df_features: pd.DataFrame, feature_cols: List[str], max_states: int = 5) -> Tuple[int, Dict[int, float]]:
        """
        Uses BIC criteria to select the optimal number of states (typically 2 to 4).
        """
        bic_scores = {}
        for k in range(2, max_states + 1):
            clf = DynamicRegimeClassifier(n_components=k, model_type=self.model_type, random_state=self.random_state)
            _, aic, bic = clf.fit_predict_multivariate(df_features, feature_cols)
            bic_scores[k] = bic

        optimal_k = min(bic_scores, key=bic_scores.get) if bic_scores else 3
        return optimal_k, bic_scores
