import pytest
import pandas as pd
import numpy as np

from src.state_machine.regime_classifier import DynamicRegimeClassifier
from src.state_machine.state_machine import PointInTimeStateMachine

@pytest.fixture
def mock_features():
    np.random.seed(42)
    dates = [f"2024-01-{i:02d}" for i in range(1, 21)]
    return pd.DataFrame({
        "trade_date": dates,
        "rpi_raw": np.random.normal(0.1, 0.05, 20),
        "iv_30d": np.random.normal(0.15, 0.02, 20),
        "expiry_pressure_ratio": np.random.normal(1.2, 0.3, 20)
    })

def test_dynamic_regime_classifier(mock_features):
    classifier = DynamicRegimeClassifier(n_states_per_feature=2, model_type="hmm")
    df_result = classifier.build_composite_state(mock_features, ["rpi_raw", "iv_30d", "expiry_pressure_ratio"])

    assert "rpi_raw_regime" in df_result.columns
    assert "iv_30d_regime" in df_result.columns
    assert "expiry_pressure_ratio_regime" in df_result.columns
    assert "composite_state_id" in df_result.columns
    assert set(df_result["composite_state_id"].unique()).issubset(set(range(8)))

def test_point_in_time_state_machine(mock_features):
    sm = PointInTimeStateMachine(min_periods=5, n_states_per_feature=2)
    df_states, trans_matrix = sm.process_state_classification(mock_features, ["rpi_raw", "iv_30d"])

    assert not df_states.empty
    assert trans_matrix.shape == (4, 4)  # 2^2 = 4 states
    row_sums = trans_matrix.sum(axis=1)
    for r in row_sums:
        assert np.isclose(r, 1.0) or np.isclose(r, 0.0)

def test_pit_expanding_quantiles(mock_features):
    sm = PointInTimeStateMachine(min_periods=5)
    df_quantiles = sm.compute_expanding_quantiles(mock_features, ["rpi_raw"], quantiles=[0.33, 0.67])

    assert "rpi_raw_expanding_q33" in df_quantiles.columns
    assert "rpi_raw_expanding_q67" in df_quantiles.columns
    assert pd.isna(df_quantiles["rpi_raw_expanding_q33"].iloc[0])  # NaN due to min_periods=5
    assert not pd.isna(df_quantiles["rpi_raw_expanding_q33"].iloc[10])
