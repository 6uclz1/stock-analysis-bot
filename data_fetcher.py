"""
Data fetching module for Japanese stocks using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class JapaneseStockDataFetcher:
    """Fetches and manages Japanese stock data using yfinance"""
    
    def __init__(self):
        self.data_dir = config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical stock data for a given symbol
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
                
            # Add symbol column
            data['Symbol'] = symbol
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_fundamental_data(self, symbol: str) -> Dict:
        """
        Fetch fundamental data for a given symbol
        """
        try:
            logger.info(f"Fetching fundamental data for {symbol}")
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'symbol': symbol,
                'pe_ratio': info.get('trailingPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'profit_margin': info.get('profitMargins', np.nan),
                'dividend_yield': info.get('dividendYield', np.nan),
                'market_cap': info.get('marketCap', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                'current_ratio': info.get('currentRatio', np.nan),
                'book_value': info.get('bookValue', np.nan),
                'total_cash': info.get('totalCash', np.nan),
                'total_debt': info.get('totalDebt', np.nan),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {'symbol': symbol}
    
    def fetch_multiple_stocks(self, symbols: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data for multiple stocks
        """
        all_data = []
        
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, start_date, end_date)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            return combined_data
        else:
            return pd.DataFrame()
    
    def fetch_multiple_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental data for multiple stocks
        """
        fundamentals_list = []
        
        for symbol in symbols:
            fundamentals = self.fetch_fundamental_data(symbol)
            fundamentals_list.append(fundamentals)
        
        fundamentals_df = pd.DataFrame(fundamentals_list)
        return fundamentals_df
    
    def fetch_benchmark_data(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch benchmark index data (Nikkei 225)
        """
        return self.fetch_stock_data(config.BENCHMARK_SYMBOL, start_date, end_date)
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """
        Save data to CSV file
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=True)
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file
        """
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading data from {filepath}")
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            logger.warning(f"File {filepath} not found")
            return pd.DataFrame()
    
    def get_stock_list(self) -> List[str]:
        """
        Get list of Japanese stocks to analyze
        """
        return config.JAPANESE_STOCKS
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics
        """
        data = data.copy()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        
        # Calculate rolling metrics
        data['Volatility_30d'] = data['Daily_Return'].rolling(30).std() * np.sqrt(252)
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        return data