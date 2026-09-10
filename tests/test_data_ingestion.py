import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.data_ingestion.base import BaseFetcher
from src.data_ingestion.participant_oi import ParticipantOIFetcher
from src.data_ingestion.bhavcopy import BhavcopyFetcher

@pytest.fixture
def tmp_dirs(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    return str(raw_dir), str(processed_dir)

@pytest.fixture
def mock_oi_df():
    return pd.DataFrame({
        "Client": [1000, 1500],
        "FII": [5000, 4800],
        "DII": [2000, 2100],
        "Pro": [1200, 1100],
        "Total": [9200, 9500]
    })

@pytest.fixture
def mock_bhavcopy_df():
    return pd.DataFrame({
        "INSTRUMENT": ["FUTIDX", "OPTIDX", "FUTSTK"],
        "SYMBOL": ["NIFTY", "NIFTY", "RELIANCE"],
        "EXPIRY_DT": ["2024-01-25", "2024-01-25", "2024-01-25"],
        "STRIKE_PR": [0.0, 21500.0, 0.0],
        "OPTION_TY": ["XX", "CE", "XX"],
        "OPEN": [21500.0, 150.0, 2700.0],
        "HIGH": [21600.0, 200.0, 2750.0],
        "LOW": [21400.0, 100.0, 2680.0],
        "CLOSE": [21550.0, 180.0, 2720.0],
        "OPEN_INT": [100000, 50000, 20000],
        "CHG_IN_OI": [5000, 2000, -500]
    })

def test_base_fetcher_idempotent_save(tmp_dirs, mock_oi_df):
    raw_dir, processed_dir = tmp_dirs
    fetcher = BaseFetcher(raw_dir=raw_dir, processed_dir=processed_dir)
    file_path = os.path.join(raw_dir, "test.parquet")

    # Initial save
    assert fetcher.save_parquet_idempotent(mock_oi_df, file_path) is True
    assert os.path.exists(file_path)

    # Read back and compare
    read_df = pd.read_parquet(file_path)
    assert len(read_df) == len(mock_oi_df)
    assert "Client" in read_df.columns

    # Overwrite (idempotent)
    assert fetcher.save_parquet_idempotent(mock_oi_df, file_path) is True

def test_participant_oi_fetcher_caching(tmp_dirs, mock_oi_df):
    raw_dir, processed_dir = tmp_dirs
    fetcher = ParticipantOIFetcher(raw_dir=raw_dir, processed_dir=processed_dir)

    with patch("nselib.derivatives.participant_wise_open_interest", return_value=mock_oi_df) as mock_nselib:
        # First call - fetches from mock_nselib
        df1 = fetcher.fetch_for_date("2024-01-15")
        assert df1 is not None
        assert mock_nselib.call_count == 1

        # Second call - should hit local cache, mock_nselib should NOT be called again
        df2 = fetcher.fetch_for_date("2024-01-15")
        assert df2 is not None
        assert mock_nselib.call_count == 1

def test_bhavcopy_fetcher_filtering(tmp_dirs, mock_bhavcopy_df):
    raw_dir, processed_dir = tmp_dirs
    fetcher = BhavcopyFetcher(raw_dir=raw_dir, processed_dir=processed_dir)

    with patch("nselib.derivatives.fno_bhav_copy", return_value=mock_bhavcopy_df):
        df_nifty = fetcher.fetch_for_date("2024-01-15", target_symbol="NIFTY")
        assert df_nifty is not None
        assert len(df_nifty) == 2
        assert set(df_nifty["SYMBOL"].unique()) == {"NIFTY"}
