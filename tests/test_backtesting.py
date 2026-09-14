import pytest
import pandas as pd
import numpy as np

from src.backtesting.cost_model import IndianTransactionCostModel
from src.backtesting.state_returns import StateConditionalReturnAnalysis
from src.backtesting.significance import StatisticalSignificanceTesting
from src.backtesting.walk_forward import WalkForwardValidator
from src.backtesting.engine import VectorBTBacktestEngine

@pytest.fixture
def mock_states_returns():
    np.random.seed(42)
    dates = [f"2024-01-{i:02d}" for i in range(1, 31)]
    df = pd.DataFrame({
        "trade_date": dates,
        "close": np.linspace(21000, 22000, 30) + np.random.normal(0, 50, 30),
        "composite_state_id": np.random.choice([0, 1, 2], size=30),
        "rpi_raw": np.random.normal(0.1, 0.05, 30),
        "iv_30d": np.random.normal(0.15, 0.02, 30),
        "expiry_pressure_ratio": np.random.normal(1.2, 0.3, 30)
    })
    return df

def test_indian_cost_model():
    cost_model = IndianTransactionCostModel(slippage_bps=2.0)
    trade_value = 1000000.0  # Rs 10 Lakh trade
    cost_info = cost_model.calculate_round_turn_cost(trade_value, is_buy=False)

    assert "total_cost" in cost_info
    assert "cost_bps" in cost_info
    assert cost_info["stt"] > 0
    assert cost_info["gst"] > 0
    assert cost_info["brokerage"] == 20.0

def test_state_conditional_return_analysis(mock_states_returns):
    analyzer = StateConditionalReturnAnalysis()
    df_fwd = analyzer.compute_forward_returns(mock_states_returns, price_col="close")

    assert "fwd_ret_1d" in df_fwd.columns
    assert "fwd_ret_5d" in df_fwd.columns
    assert "fwd_ret_10d" in df_fwd.columns

    df_stats = analyzer.analyze_conditional_returns(df_fwd, state_col="composite_state_id")
    assert not df_stats.empty
    assert "mean_1d" in df_stats.columns
    assert "hit_rate_1d" in df_stats.columns

def test_statistical_significance_testing(mock_states_returns):
    analyzer = StateConditionalReturnAnalysis()
    df_fwd = analyzer.compute_forward_returns(mock_states_returns, price_col="close")

    tester = StatisticalSignificanceTesting(max_lags=2, fdr_alpha=0.05)
    df_sig = tester.test_state_returns_significance(df_fwd, state_col="composite_state_id")

    assert not df_sig.empty
    assert "newey_west_tstat" in df_sig.columns
    assert "p_value_fdr_bh" in df_sig.columns
    assert "is_significant_fdr" in df_sig.columns

def test_walk_forward_validator(mock_states_returns):
    wf = WalkForwardValidator(train_window_days=10, test_window_days=5)
    df_oos = wf.run_walk_forward_state_generation(mock_states_returns, ["rpi_raw", "iv_30d"])

    assert not df_oos.empty
    assert "composite_state_id" in df_oos.columns

def test_vectorbt_backtest_engine(mock_states_returns):
    engine = VectorBTBacktestEngine()
    metrics = engine.run_state_strategy_backtest(mock_states_returns, price_col="close")

    assert "total_return" in metrics
    assert "net_sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "calmar_ratio" in metrics
