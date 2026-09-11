# Options Microstructure Research Engine

A modular, quantitative research engine for batch processing Indian equity and options derivatives (NSE F&O) with Point-in-Time (PIT) discipline.

## Project Directory Structure

```text
options-microstructure-research/
│
├── data/
│   ├── raw/                  # Raw bhavcopy, participant OI, option chain
│   └── processed/            # PIT-aligned feature sets
│
├── src/
│   ├── data_ingestion/       # Data download, validation, and caching scripts
│   ├── feature_engineering/  # RPI, Expiry Pressure, IV Surface, Sentiment
│   ├── state_machine/        # HMM/GMM regime clustering and state classification
│   ├── backtesting/          # Backtest engine, cost modeling
│   ├── research/             # Statistical analysis, notebooks
│   └── utils/                # Helper functions (config, logging)
│
├── notebooks/                # Jupyter notebooks for EDA and research
├── reports/                  # Generated research reports and figures
├── tests/                    # Unit and integration tests
├── config.yaml               # Configuration for data paths, thresholds
├── pyproject.toml            # Project dependencies (Poetry)
└── README.md                 # Project documentation
```
