# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Bot
```bash
# Run single strategy backtest with defaults
python main.py

# Run with custom parameters
python main.py --start-date 2020-01-01 --capital 5000000 --no-plots

# Run strategy comparison
python compare_strategies.py

# Run specific strategies
python compare_strategies.py --strategies conservative,aggressive,value_investing
```

### Docker Usage
```bash
# Build Docker image
docker build -t stock-analysis-bot .

# Run with Docker
docker run -v $(pwd)/results:/app/results stock-analysis-bot

# Run with custom parameters
docker run -v $(pwd)/results:/app/results stock-analysis-bot python main.py --capital 2000000
```

### Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Architecture Overview

This is a Japanese stock fundamental analysis trading bot with a modular architecture:

### Core Components

**Data Layer**
- `data_fetcher.py`: Handles Yahoo Finance API integration via yfinance for stock prices and fundamental data
- Stock universe defined in `config.py` (25 major Japanese stocks with `.T` suffix for TSE)

**Analysis Layer**
- `fundamental_analyzer.py`: Implements multi-metric scoring system using P/E, P/B, ROE, debt ratios, profit margins, dividend yields
- Weighted composite scoring with configurable thresholds and weights in `config.py`

**Strategy Layer**
- `trading_strategy.py`: Portfolio management with equal weighting, rebalancing, stop-loss/take-profit rules
- `strategy_factory.py`: 8 built-in strategies (Conservative, Aggressive, Value, Growth, Dividend Focus, etc.)
- Each strategy has different portfolio sizes, rebalancing frequencies, and fundamental weight preferences

**Execution Layer**
- `backtester.py`: Historical simulation engine with 3-5 years of data
- `compare_strategies.py`: Multi-strategy comparison framework
- `visualizer.py`: Comprehensive charting and reporting (performance vs Nikkei 225, drawdowns, heatmaps)

### Key Design Patterns

**Configuration Management**: All parameters centralized in `config.py` including:
- Trading rules (stop-loss: -15%, take-profit: +30%, rebalancing: 90 days)
- Fundamental weights and scoring thresholds
- Stock universe and file paths

**Results Management**: Timestamped directory structure (`results/yyyyMMddHHmmSS/`) containing:
- CSV exports (portfolio_history, trade_history, fundamental_analysis, performance_metrics)
- Text reports and PNG visualizations
- Strategy comparison data when running multiple strategies

**Modular Strategy System**: Strategies are defined as configuration dictionaries with portfolio_size, rebalance_frequency, stop_loss, take_profit, and custom fundamental weights.

## Data Flow

1. **Data Acquisition**: `data_fetcher.py` retrieves price and fundamental data from Yahoo Finance
2. **Fundamental Scoring**: `fundamental_analyzer.py` calculates composite scores for stock ranking  
3. **Portfolio Construction**: `trading_strategy.py` selects top-ranked stocks based on strategy criteria
4. **Simulation**: `backtester.py` runs historical trades with rebalancing and risk management
5. **Analysis**: Performance metrics calculated (returns, Sharpe ratio, max drawdown, alpha vs Nikkei 225)
6. **Visualization**: `visualizer.py` generates comprehensive charts and saves timestamped results

## Entry Points

- `main.py`: Single strategy execution with command-line arguments
- `compare_strategies.py`: Multi-strategy comparison tool
- Both support `--start-date`, `--end-date`, `--capital`, `--no-plots` arguments

## File Structure Notes

- All results auto-saved to timestamped directories under `results/`
- Logs saved to `logs/` directory with execution timestamps
- `data/` directory created automatically for cached stock data
- Docker support via `Dockerfile` for consistent execution environment