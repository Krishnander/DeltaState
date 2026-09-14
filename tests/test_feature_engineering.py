import pytest
import pandas as pd
import numpy as np

from src.feature_engineering.rpi import RetailPositioningIndex
from src.feature_engineering.iv_surface import ConstantMaturityIVSurface
from src.feature_engineering.expiry_pressure import ExpiryPressureMetrics
from src.feature_engineering.sentiment_features import SentimentFeatureExtractor
from src.feature_engineering.pit_normalizer import PITExpandingNormalizer

@pytest.fixture
def mock_participant_oi():
    return pd.DataFrame({
        "trade_date": ["2024-01-15", "2024-01-15", "2024-01-16", "2024-01-16"],
        "Client Type": ["Client", "FII", "Client", "FII"],
        "Future Index Long": [10000, 50000, 12000, 48000],
        "Future Index Short": [20000, 10000, 15000, 12000]
    })

@pytest.fixture
def mock_fno_bhavcopy():
    return pd.DataFrame({
        "trade_date": ["2024-01-15"] * 4 + ["2024-01-16"] * 4,
        "symbol": ["NIFTY"] * 8,
        "expiry_date": ["2024-01-25", "2024-01-25", "2024-02-29", "2024-02-29"] * 2,
        "strike": [21500, 21600, 21500, 21600] * 2,
        "option_type": ["CE", "PE", "CE", "PE"] * 2,
        "close_price": [150.0, 120.0, 300.0, 280.0] * 2,
        "open_interest": [50000, 40000, 30000, 20000] * 2,
        "underlying_price": [21520.0] * 8
    })

@pytest.fixture
def mock_sentiment():
    return pd.DataFrame({
        "timestamp": ["2024-01-15T10:00:00", "2024-01-15T11:00:00", "2024-01-16T10:00:00"],
        "label": ["bullish", "bearish", "bullish"]
    })

def test_rpi_calculation(mock_participant_oi):
    rpi_calc = RetailPositioningIndex()
    df_rpi = rpi_calc.compute(mock_participant_oi)

    assert not df_rpi.empty
    assert "rpi_raw" in df_rpi.columns
    assert len(df_rpi) == 2  # 2 trade dates
    # Check Client Net OI on 2024-01-15: 10000 - 20000 = -10000
    assert df_rpi.loc[df_rpi["trade_date"] == "2024-01-15", "client_net_fut_oi"].iloc[0] == -10000

def test_iv_surface_calculation(mock_fno_bhavcopy):
    iv_calc = ConstantMaturityIVSurface()
    df_iv = iv_calc.compute_atm_iv_surface(mock_fno_bhavcopy)

    assert not df_iv.empty
    assert "iv_7d" in df_iv.columns
    assert "iv_30d" in df_iv.columns
    assert "iv_60d" in df_iv.columns

def test_expiry_pressure_calculation(mock_fno_bhavcopy):
    exp_calc = ExpiryPressureMetrics()
    df_exp = exp_calc.compute(mock_fno_bhavcopy)

    assert not df_exp.empty
    assert "expiry_pressure_ratio" in df_exp.columns
    assert df_exp["expiry_pressure_ratio"].iloc[0] > 0

def test_sentiment_feature_extraction(mock_sentiment):
    sent_calc = SentimentFeatureExtractor()
    df_sent = sent_calc.compute_sentiment_features(mock_sentiment)

    assert not df_sent.empty
    assert "sentiment_signal" in df_sent.columns

def test_pit_expanding_normalizer():
    df_features = pd.DataFrame({
        "trade_date": [f"2024-01-{i:02d}" for i in range(1, 11)],
        "rpi_raw": np.linspace(-0.5, 0.5, 10)
    })

    normalizer = PITExpandingNormalizer(min_periods=3)
    df_norm = normalizer.normalize_features(df_features, ["rpi_raw"])

    assert "rpi_raw_pit_zscore" in df_norm.columns
    assert "rpi_raw_pit_quantile" in df_norm.columns
    # First 2 periods should be NaN due to min_periods=3
    assert pd.isna(df_norm["rpi_raw_pit_quantile"].iloc[0])
    assert not pd.isna(df_norm["rpi_raw_pit_quantile"].iloc[5])
