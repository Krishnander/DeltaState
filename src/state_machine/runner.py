import os
import argparse
import logging
import pandas as pd
import numpy as np
from src.utils.config import load_config, setup_logger
from src.state_machine.state_machine import PointInTimeStateMachine

def main():
    parser = argparse.ArgumentParser(description="State Classification Layer Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("state_machine_runner", log_file=config["logging"]["log_file"], level=config["logging"]["level"])

    processed_dir = config["data_ingestion"]["processed_dir"]
    features_file = os.path.join(processed_dir, "features.parquet")

    if not os.path.exists(features_file):
        logger.error(f"Features file not found at {features_file}. Run feature engineering first.")
        return

    logger.info(f"Loading feature store from {features_file}...")
    df_features = pd.read_parquet(features_file)

    # Core signal features for regime classification (3 key signals -> 2^3 = 8 states)
    target_feature_cols = [c for c in ["rpi_raw", "iv_30d", "expiry_pressure_ratio"] if c in df_features.columns]
    if not target_feature_cols:
        target_feature_cols = [c for c in df_features.columns if c not in ["trade_date"] and not c.endswith("_zscore") and not c.endswith("_quantile")][:3]

    logger.info(f"Classifying dynamic regime states using signals: {target_feature_cols}")

    sm = PointInTimeStateMachine(min_periods=1, n_states_per_feature=2)
    df_states, trans_matrix = sm.process_state_classification(df_features, target_feature_cols)

    output_states_file = os.path.join(processed_dir, "states.parquet")
    df_states.to_parquet(output_states_file, index=False)

    logger.info(f"State Classification complete! Saved classified states to {output_states_file} with {len(df_states)} rows.")
    logger.info(f"Transition Matrix (shape {trans_matrix.shape}):\n{trans_matrix}")

if __name__ == "__main__":
    main()
