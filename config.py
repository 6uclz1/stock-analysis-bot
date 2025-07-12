"""
Configuration file for Japanese Stock Fundamental Analysis Trading Bot
"""

# Data settings
DATA_START_DATE = "2019-01-01"
DATA_END_DATE = None  # None for current date

# Trading settings
PORTFOLIO_SIZE = 8  # Number of stocks in portfolio
REBALANCE_FREQUENCY = 90  # Days between rebalancing
STOP_LOSS_THRESHOLD = -0.15  # -15% stop loss
TAKE_PROFIT_THRESHOLD = 0.30  # +30% take profit

# Fundamental analysis weights
FUNDAMENTAL_WEIGHTS = {
    'per_score': 0.20,      # Price-to-Earnings Ratio
    'pbr_score': 0.15,      # Price-to-Book Ratio  
    'roe_score': 0.20,      # Return on Equity
    'debt_ratio_score': 0.15,  # Debt-to-Equity Ratio
    'profit_margin_score': 0.15,  # Profit Margin
    'dividend_yield_score': 0.15   # Dividend Yield
}

# Scoring thresholds
PER_GOOD_THRESHOLD = 15
PER_BAD_THRESHOLD = 25
PBR_GOOD_THRESHOLD = 1.0
PBR_BAD_THRESHOLD = 2.0
ROE_GOOD_THRESHOLD = 0.15  # 15%
ROE_BAD_THRESHOLD = 0.05   # 5%
DEBT_RATIO_GOOD_THRESHOLD = 0.3  # 30%
DEBT_RATIO_BAD_THRESHOLD = 0.6   # 60%
PROFIT_MARGIN_GOOD_THRESHOLD = 0.1  # 10%
PROFIT_MARGIN_BAD_THRESHOLD = 0.02  # 2%
DIVIDEND_YIELD_GOOD_THRESHOLD = 0.03  # 3%
DIVIDEND_YIELD_BAD_THRESHOLD = 0.01   # 1%

# Sample Japanese stock symbols (with .T suffix for Tokyo Stock Exchange)
JAPANESE_STOCKS = [
    '7203.T',  # Toyota Motor
    '6758.T',  # Sony Group
    '8306.T',  # Mitsubishi UFJ Financial
    '9984.T',  # SoftBank Group
    '6501.T',  # Hitachi
    '8058.T',  # Mitsubishi Corp
    '4519.T',  # Chugai Pharmaceutical
    '6954.T',  # Fanuc
    '8801.T',  # Mitsui Fudosan
    '7974.T',  # Nintendo
    '4502.T',  # Takeda Pharmaceutical
    '8316.T',  # Sumitomo Mitsui Financial
    '9432.T',  # NTT
    '6902.T',  # Denso
    '7751.T',  # Canon
    '4063.T',  # Shin-Etsu Chemical
    '6981.T',  # Murata Manufacturing
    '6361.T',  # Ebara Corp
    '8035.T',  # Tokyo Electron
    '4578.T',  # Otsuka Holdings
    '7267.T',  # Honda Motor
    '8053.T',  # Sumitomo Corp
    '8031.T',  # Mitsui & Co
    '9983.T',  # Fast Retailing
    '4661.T',  # Oriental Land
]

# Benchmark symbol
BENCHMARK_SYMBOL = '^N225'  # Nikkei 225

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# File paths
DATA_DIR = 'data'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# Results management with timestamps
def get_results_dir_with_timestamp():
    """Get results directory with timestamp for current execution"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return f'{RESULTS_DIR}/{timestamp}'