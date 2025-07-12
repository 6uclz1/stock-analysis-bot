# 🇯🇵 Japanese Stock Fundamental Analysis Trading Bot

A comprehensive Python-based trading bot that analyzes Japanese stocks using fundamental analysis, features multiple trading strategies, and provides advanced backtesting capabilities with strategy comparison.

## 🌟 Latest Updates

- **Multiple Strategy Framework**: 8 different trading strategies (Conservative, Aggressive, Value Investing, Growth Investing, etc.)
- **Strategy Comparison Tool**: Compare performance across multiple strategies simultaneously
- **Timestamped Results**: Organized result management with `yyyyMMddHHmmSS` directory structure
- **Enhanced Visualization**: Comprehensive charts, risk-return analysis, and strategy comparison plots
- **Docker Support**: Containerized execution environment for consistent results

## 🎯 Features

### 📊 Fundamental Analysis
- **Multi-metric scoring system** using P/E ratio, P/B ratio, ROE, debt ratios, profit margins, and dividend yields
- **Weighted composite scoring** for ranking stocks
- **Automatic filtering** of stocks with insufficient fundamental data
- **Sector diversification** analysis
- **Customizable weights** for different fundamental metrics

### 🎯 Multiple Trading Strategies
- **8 Built-in Strategies**: Conservative, Aggressive, Value Investing, Growth Investing, Dividend Focus, Quality Focus, Momentum, and Balanced approaches
- **Strategy Comparison**: Test multiple strategies simultaneously and compare performance
- **Customizable Parameters**: Different portfolio sizes, rebalancing frequencies, and risk management rules per strategy
- **Strategy Optimization**: Find the best performing strategy for your risk profile

### 💰 Advanced Portfolio Management
- **Medium to long-term** investment approach (3 months to 1 year holding periods)
- **Flexible portfolio sizes** (5-15 stocks depending on strategy)
- **Dynamic rebalancing** (30-120 days depending on strategy)
- **Advanced risk management** with customizable stop-loss and take-profit rules
- **Portfolio optimization** based on fundamental scores and strategy-specific criteria

### 🔬 Backtesting Engine
- **Historical simulation** with 3-5 years of data
- **Comprehensive performance metrics**:
  - Total and annualized returns
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
  - Alpha vs Nikkei 225 benchmark
- **Trade tracking** and portfolio history

### 📈 Visualization & Reporting
- **Performance charts** vs benchmark (Nikkei 225)
- **Strategy comparison charts** with risk-return analysis
- **Portfolio evolution** comparison across strategies
- **Drawdown analysis** and risk metrics
- **Monthly returns heatmap**
- **Fundamental analysis scores visualization**
- **Trading activity analysis**
- **Strategy characteristics radar chart**
- **Timestamped result directories** for organized output
- **Comprehensive text and CSV reports**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Docker (optional, recommended for consistent environment)
- Internet connection for data fetching

### Option 1: Docker (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/your-username/japanese-stock-analysis-bot.git
cd japanese-stock-analysis-bot
```

2. **Run with Docker**
```bash
docker build -t stock-analysis-bot .
docker run -v $(pwd)/results:/app/results stock-analysis-bot
```

### Option 2: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/japanese-stock-analysis-bot.git
cd japanese-stock-analysis-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run single strategy backtest**
```bash
python main.py
```

4. **Run strategy comparison**
```bash
python compare_strategies.py
```

The bot will automatically:
- Download historical data for Japanese stocks
- Perform fundamental analysis
- Run backtesting simulation(s)
- Generate performance reports and visualizations
- Save results in timestamped directories

## 📋 Usage Examples

### Single Strategy Backtesting
```bash
# Run with default settings (1M yen, 2023-present)
python main.py

# Specify custom date range
python main.py --start-date 2020-01-01 --end-date 2023-12-31

# Use different initial capital (5M yen)
python main.py --capital 5000000

# Skip visualizations (faster execution)
python main.py --no-plots
```

### Strategy Comparison
```bash
# Compare all available strategies
python compare_strategies.py

# Compare specific strategies
python compare_strategies.py --strategies value_investing,growth_investing,dividend_focus

# Custom comparison with different capital
python compare_strategies.py --capital 2000000 --start-date 2023-01-01

# Skip plots for faster execution
python compare_strategies.py --no-plots
```

### Docker Usage
```bash
# Run strategy comparison in Docker
docker run -v $(pwd)/results:/app/results stock-analysis-bot

# Run single strategy with custom parameters
docker run -v $(pwd)/results:/app/results stock-analysis-bot python main.py --capital 2000000
```

### Advanced Configuration
Edit `config.py` to customize:
- Stock universe (add/remove stocks)
- Fundamental analysis weights
- Trading parameters (stop-loss, take-profit)
- Portfolio size and rebalancing frequency
- Scoring thresholds

## 📁 Project Structure

```
japanese-stock-analysis-bot/
│
├── main.py                    # Single strategy execution script
├── compare_strategies.py      # Strategy comparison script
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── README.md               # This file
├── README_JP.md           # Japanese version README
│
├── data_fetcher.py           # Stock data retrieval using yfinance
├── fundamental_analyzer.py   # Fundamental analysis and scoring
├── trading_strategy.py       # Portfolio management and trading logic
├── backtester.py            # Backtesting engine
├── visualizer.py            # Charts and visualization
├── strategy_factory.py      # Multiple strategy definitions
├── strategy_comparison.py   # Strategy comparison engine
│
├── data/                    # Downloaded stock data (auto-created)
├── results/                 # Analysis results (timestamped directories)
│   ├── 20231215143022/     # Example timestamped result directory
│   ├── 20231215143155/     # Another timestamped result directory
│   └── ...
└── logs/                   # Application logs (auto-created)
```

## 🔧 Configuration

### Stock Universe
The bot analyzes 25 major Japanese stocks by default, including:
- Toyota Motor (7203.T)
- Sony Group (6758.T)
- SoftBank Group (9984.T)
- Nintendo (7974.T)
- And 21 more...

### Fundamental Metrics & Weights
- **P/E Ratio (20%)**: Lower is better for value investing
- **P/B Ratio (15%)**: Price relative to book value
- **ROE (20%)**: Return on equity efficiency
- **Debt Ratio (15%)**: Financial stability
- **Profit Margin (15%)**: Profitability
- **Dividend Yield (15%)**: Income generation

### Trading Parameters (Default Strategy)
- **Portfolio Size**: 8 stocks (configurable)
- **Rebalancing**: Every 90 days
- **Stop Loss**: -15%
- **Take Profit**: +30%
- **Initial Capital**: ¥1,000,000 (configurable)

### Available Trading Strategies
1. **Conservative**: 10 stocks, 120-day rebalancing, -10% stop-loss, +25% take-profit
2. **Aggressive**: 5 stocks, 30-day rebalancing, -25% stop-loss, +60% take-profit
3. **Value Investing**: 10 stocks, focus on low P/E and P/B ratios
4. **Growth Investing**: 6 stocks, emphasis on ROE and profit margins
5. **Dividend Focus**: 12 stocks, prioritizes dividend yield
6. **Quality Focus**: 8 stocks, balanced approach with quality metrics
7. **Momentum**: 6 stocks, shorter rebalancing periods
8. **Balanced**: 8 stocks, equal weight across all metrics

## 📊 Sample Output

### Single Strategy Output
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║           🇯🇵 Japanese Stock Fundamental Analysis Trading Bot 🤖              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📋 CONFIGURATION:
==================================================
Start Date:           2023-01-01
End Date:             Current date
Initial Capital:      ¥1,000,000
Portfolio Size:       8 stocks
Rebalancing:          Every 90 days
==================================================

🎯 PERFORMANCE SUMMARY:
Initial Capital:         ¥1,000,000
Final Value:             ¥1,277,034
Total Return:            +27.70%
Annualized Return:       +66.11%
Sharpe Ratio:            6.36
Max Drawdown:            -2.05%
Win Rate:                100.0%

🏆 BENCHMARK COMPARISON:
Strategy Return:         +66.11%
Nikkei 225 Return:      +43.25%
Alpha (Outperformance): +22.86%
```

### Strategy Comparison Output
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║        🇯🇵 Japanese Stock Strategy Comparison Tool 📊                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

STRATEGY PERFORMANCE SUMMARY:
----------------------------------------------------------------------------------------------------
        Strategy Total Return (%) Annual Return (%) Sharpe Ratio Max Drawdown (%) Alpha vs Nikkei (%)
growth_investing            31.69             77.04         4.75            -8.15               33.79
 value_investing            26.16             61.97         6.84            -1.85               18.72

TOP PERFORMERS:
--------------------------------------------------
Best Annual Return:  growth_investing (77.04%)
Best Sharpe Ratio:   value_investing (6.84)
Best Alpha:          growth_investing (33.79%)
```

## 📈 Generated Reports

The bot automatically generates timestamped result directories (`results/yyyyMMddHHmmSS/`) containing:

### Single Strategy Results
1. **Performance Report** (`performance_report.txt`)
   - Detailed metrics and statistics
   - Top holdings analysis
   - Trading summary

2. **CSV Data Files**
   - `portfolio_history.csv` - Daily portfolio values
   - `trade_history.csv` - All executed trades
   - `fundamental_analysis.csv` - Stock scores and rankings
   - `performance_metrics.csv` - Summary statistics

3. **Visualizations** (PNG files)
   - Portfolio performance vs benchmark
   - Drawdown analysis
   - Monthly returns heatmap
   - Fundamental scores comparison
   - Trading activity analysis

### Strategy Comparison Results
1. **Comparison Report** (`strategy_comparison_report.txt`)
   - Performance summary across all strategies
   - Top performers identification
   - Strategy characteristics comparison

2. **Strategy Data Files**
   - `strategy_comparison_summary.csv` - All strategy metrics
   - Individual strategy subdirectories with detailed results

3. **Comparison Visualizations**
   - Strategy performance comparison charts
   - Risk-return scatter plot
   - Portfolio evolution comparison
   - Strategy characteristics radar chart

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This bot is designed for research and educational purposes
- **Not Investment Advice**: Do not use for actual trading without thorough testing and validation
- **Historical Performance**: Past performance does not guarantee future results
- **Data Limitations**: Uses publicly available data which may have delays or inaccuracies
- **Risk Warning**: All investments carry risk of loss

## 🛠️ Customization

### Adding New Stocks
Edit `config.py` and add stock symbols to `JAPANESE_STOCKS` list:
```python
JAPANESE_STOCKS = [
    '7203.T',  # Toyota Motor
    '6758.T',  # Sony Group
    'XXXX.T',  # Your new stock (Tokyo Stock Exchange)
    # ... more stocks
]
```

### Modifying Analysis Weights
Adjust fundamental analysis weights in `config.py`:
```python
FUNDAMENTAL_WEIGHTS = {
    'per_score': 0.25,      # Increase P/E weight
    'pbr_score': 0.10,      # Decrease P/B weight
    'roe_score': 0.25,      # Increase ROE weight
    # ... other weights
}
```

### Changing Trading Rules
Modify trading parameters in `config.py`:
```python
STOP_LOSS_THRESHOLD = -0.20      # -20% stop loss
TAKE_PROFIT_THRESHOLD = 0.50     # +50% take profit
REBALANCE_FREQUENCY = 60         # Rebalance every 60 days
```

### Creating Custom Strategies
Add new strategies in `strategy_factory.py`:
```python
def create_custom_strategy(self):
    return {
        'name': 'custom_strategy',
        'description': 'Your custom strategy description',
        'portfolio_size': 10,
        'rebalance_frequency': 45,
        'stop_loss': -0.15,
        'take_profit': 0.35,
        'weights': {
            'per_score': 0.30,
            'pbr_score': 0.20,
            'roe_score': 0.25,
            'debt_ratio_score': 0.10,
            'profit_margin_score': 0.10,
            'dividend_yield_score': 0.05
        }
    }
```

## 🔍 Available Commands

### Strategy Comparison Commands
```bash
# List all available strategies
python compare_strategies.py --help

# Run specific strategies
python compare_strategies.py --strategies conservative,aggressive,value_investing

# Custom date range and capital
python compare_strategies.py --start-date 2022-01-01 --capital 5000000
```

### Available Strategies
- `conservative` - Low risk, stable returns
- `aggressive` - High risk, high potential returns
- `value_investing` - Focus on undervalued stocks
- `growth_investing` - Focus on growth potential
- `dividend_focus` - Prioritize dividend income
- `quality_focus` - Balanced quality approach
- `momentum` - Short-term momentum strategy
- `balanced` - Equal weight across metrics

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional fundamental metrics
- Machine learning integration
- Real-time data feeds
- Options trading strategies
- International markets support
- Additional strategy types
- Performance optimization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **yfinance** for providing free access to Yahoo Finance data
- **Japanese stock market** data providers
- **Open source community** for the excellent Python libraries

---

**Happy Trading! 🚀📈**

*Remember: This is for educational purposes only. Always do your own research and consider consulting with a financial advisor before making investment decisions.*