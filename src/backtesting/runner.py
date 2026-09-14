import os
import argparse
import logging
import pandas as pd
import numpy as np
from src.utils.config import load_config, setup_logger
from src.backtesting.cost_model import IndianTransactionCostModel
from src.backtesting.state_returns import StateConditionalReturnAnalysis
from src.backtesting.significance import StatisticalSignificanceTesting
from src.backtesting.walk_forward import WalkForwardValidator
from src.backtesting.engine import VectorBTBacktestEngine

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Backtesting Layer Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("backtesting_runner", log_file=config["logging"]["log_file"], level=config["logging"]["level"])

    processed_dir = config["data_ingestion"]["processed_dir"]
    states_file = os.path.join(processed_dir, "states.parquet")

    if not os.path.exists(states_file):
        logger.error(f"States file not found at {states_file}. Run state machine classification first.")
        return

    logger.info(f"Loading state-classified store from {states_file}...")
    df_states = pd.read_parquet(states_file)

    # 1. Compute Forward Returns
    analyzer = StateConditionalReturnAnalysis()
    df_returns = analyzer.compute_forward_returns(df_states, price_col="close")

    # 2. Compute State-Conditional Return Distribution Statistics
    df_state_stats = analyzer.analyze_conditional_returns(df_returns)
    logger.info(f"State-Conditional Return Statistics:\n{df_state_stats}")

    # 3. Newey-West HAC t-test and Benjamini-Hochberg FDR Control
    tester = StatisticalSignificanceTesting(max_lags=5, fdr_alpha=0.05)
    df_sig = tester.test_state_returns_significance(df_returns)
    logger.info(f"Significance Testing (Newey-West t-stats & BH FDR):\n{df_sig}")

    # 4. Walk-Forward OOS State Generation
    wf_validator = WalkForwardValidator(train_window_days=1, test_window_days=1)
    target_feature_cols = [c for c in ["rpi_raw", "iv_30d", "expiry_pressure_ratio"] if c in df_states.columns]
    df_oos_states = wf_validator.run_walk_forward_state_generation(df_states, target_feature_cols)

    # 5. VectorBT Backtest Engine Simulation
    cost_model = IndianTransactionCostModel()
    engine = VectorBTBacktestEngine(cost_model=cost_model)
    backtest_metrics = engine.run_state_strategy_backtest(df_oos_states)
    logger.info(f"Backtest Strategy Net Performance Metrics:\n{backtest_metrics}")

    # 6. Save Backtest Analytics to processed directory
    output_backtest_file = os.path.join(processed_dir, "backtest_results.parquet")
    df_returns.to_parquet(output_backtest_file, index=False)
    logger.info(f"Phase 4 Backtesting complete! Analytics saved to {output_backtest_file}")

if __name__ == "__main__":
    main()
