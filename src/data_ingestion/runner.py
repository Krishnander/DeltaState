import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from src.utils.config import load_config, setup_logger
from src.data_ingestion.participant_oi import ParticipantOIFetcher
from src.data_ingestion.bhavcopy import BhavcopyFetcher

def main():
    parser = argparse.ArgumentParser(description="Batch Data Ingestion Runner")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("data_ingestion_runner", log_file=config["logging"]["log_file"], level=config["logging"]["level"])

    start_date_str = args.start_date or config["data_ingestion"]["start_date"]
    end_date_str = args.end_date or config["data_ingestion"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    logger.info(f"Starting batch ingestion from {start_date_str} to {end_date_str}")

    oi_fetcher = ParticipantOIFetcher(raw_dir=config["data_ingestion"]["raw_dir"])
    bhavcopy_fetcher = BhavcopyFetcher(raw_dir=config["data_ingestion"]["raw_dir"])

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            date_list.append(current_date)
        current_date += timedelta(days=1)

    success_count = 0
    fail_count = 0

    for dt in tqdm(date_list, desc="Ingesting trading days"):
        date_str = dt.strftime("%Y-%m-%d")
        logger.info(f"Processing trade date: {date_str}")

        oi_df = oi_fetcher.fetch_for_date(date_str)
        bhav_df = bhavcopy_fetcher.fetch_for_date(date_str, target_symbol=config["universe"]["target"])

        if oi_df is not None or bhav_df is not None:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"Ingestion run complete. Successful/Cached dates: {success_count}, Failed/Holiday dates: {fail_count}")

if __name__ == "__main__":
    main()
