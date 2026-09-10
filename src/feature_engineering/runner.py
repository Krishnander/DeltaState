import os
import argparse
import logging
import pandas as pd
from src.utils.config import load_config, setup_logger
from src.feature_engineering.rpi import RetailPositioningIndex
from src.feature_engineering.iv_surface import ConstantMaturityIVSurface
from src.feature_engineering.expiry_pressure import ExpiryPressureMetrics
from src.feature_engineering.sentiment_features import SentimentFeatureExtractor
from src.feature_engineering.pit_normalizer import PITExpandingNormalizer

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Orchestration Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("feature_engineering_runner", log_file=config["logging"]["log_file"], level=config["logging"]["level"])

    raw_dir = config["data_ingestion"]["raw_dir"]
    processed_dir = config["data_ingestion"]["processed_dir"]
    target_symbol = config["universe"]["target"]
    os.makedirs(processed_dir, exist_ok=True)

    logger.info("Starting Feature Engineering pipeline execution...")

    # 1. Load ingested datasets from raw Parquet data lake
    participant_oi_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(raw_dir, "participant_oi")) for f in fn if f.endswith(".parquet")]
    fno_bhavcopy_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(raw_dir, "bhavcopy")) for f in fn if f.startswith("fno_bhavcopy_") and f.endswith(".parquet")]
    equity_bhavcopy_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(raw_dir, "bhavcopy")) for f in fn if f.startswith("equity_bhavcopy_") and f.endswith(".parquet")]
    sentiment_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(raw_dir, "sentiment")) for f in fn if f.endswith(".parquet")]

    df_participant_oi = pd.concat([pd.read_parquet(f) for f in participant_oi_files], ignore_index=True) if participant_oi_files else pd.DataFrame()
    df_fno_bhavcopy = pd.concat([pd.read_parquet(f) for f in fno_bhavcopy_files], ignore_index=True) if fno_bhavcopy_files else pd.DataFrame()
    df_equity_bhavcopy = pd.concat([pd.read_parquet(f) for f in equity_bhavcopy_files], ignore_index=True) if equity_bhavcopy_files else pd.DataFrame()
    df_sentiment = pd.concat([pd.read_parquet(f) for f in sentiment_files], ignore_index=True) if sentiment_files else pd.DataFrame()

    def clean_df(df):
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.loc[:, ~df.columns.duplicated()].copy()
        if "trade_date" in df.columns:
            if isinstance(df["trade_date"], pd.DataFrame):
                df["trade_date"] = df["trade_date"].iloc[:, 0]
            df["trade_date"] = df["trade_date"].astype(str)
        return df

    df_participant_oi = clean_df(df_participant_oi)
    df_fno_bhavcopy = clean_df(df_fno_bhavcopy)
    df_equity_bhavcopy = clean_df(df_equity_bhavcopy)
    df_sentiment = clean_df(df_sentiment)

    # Filter F&O Bhavcopy for target symbol (e.g. NIFTY) to prevent cross-symbol IV/Expiry contamination
    if not df_fno_bhavcopy.empty:
        sym_col = "SYMBOL" if "SYMBOL" in df_fno_bhavcopy.columns else "symbol" if "symbol" in df_fno_bhavcopy.columns else None
        if sym_col:
            df_target_fno = df_fno_bhavcopy[df_fno_bhavcopy[sym_col] == target_symbol].copy()
        else:
            df_target_fno = df_fno_bhavcopy
    else:
        df_target_fno = pd.DataFrame()

    # 2. Compute Signals
    rpi_calculator = RetailPositioningIndex()
    df_rpi = rpi_calculator.compute(df_participant_oi)

    iv_calculator = ConstantMaturityIVSurface()
    df_iv = iv_calculator.compute_atm_iv_surface(df_target_fno)

    expiry_calculator = ExpiryPressureMetrics()
    df_expiry = expiry_calculator.compute(df_target_fno, df_equity_bhavcopy)

    sentiment_calculator = SentimentFeatureExtractor()
    df_sent = sentiment_calculator.compute_sentiment_features(df_sentiment)

    # 3. Merge raw feature signals on trade_date
    df_features = df_rpi.merge(df_iv, on="trade_date", how="outer")
    if not df_sent.empty:
        df_features = df_features.merge(df_sent, on="trade_date", how="outer")
    if not df_expiry.empty and "symbol" in df_expiry.columns:
        df_nifty_expiry = df_expiry[df_expiry["symbol"] == target_symbol].copy()
        if not df_nifty_expiry.empty:
            df_features = df_features.merge(df_nifty_expiry[["trade_date", "expiry_pressure_ratio"]], on="trade_date", how="outer")

    df_features = df_features.sort_values("trade_date").reset_index(drop=True)

    # 4. Normalize with PIT Expanding Window
    feature_cols = [c for c in ["rpi_raw", "iv_30d", "iv_slope_30_7", "expiry_pressure_ratio", "sentiment_signal"] if c in df_features.columns]
    normalizer = PITExpandingNormalizer(min_periods=1)
    df_final_features = normalizer.normalize_features(df_features, feature_cols)

    output_path = os.path.join(processed_dir, "features.parquet")
    df_final_features.to_parquet(output_path, index=False)
    logger.info(f"Feature store successfully constructed and saved to {output_path} with {len(df_final_features)} rows.")

if __name__ == "__main__":
    main()
