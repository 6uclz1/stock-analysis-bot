"""
Backtesting engine for fundamental analysis trading strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from data_fetcher import JapaneseStockDataFetcher
from fundamental_analyzer import FundamentalAnalyzer
from trading_strategy import TradingStrategy
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting engine for fundamental analysis trading strategy"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.data_fetcher = JapaneseStockDataFetcher()
        self.analyzer = FundamentalAnalyzer()
        self.strategy = TradingStrategy(initial_capital)
        self.initial_capital = initial_capital
        
    def prepare_data(self, start_date: str, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare stock price data and fundamental data for backtesting
        """
        logger.info("Preparing data for backtesting...")
        
        # Get list of stocks to analyze
        stocks = self.data_fetcher.get_stock_list()
        
        # Fetch historical price data
        logger.info("Fetching historical price data...")
        price_data = self.data_fetcher.fetch_multiple_stocks(stocks, start_date, end_date)
        
        if price_data.empty:
            raise ValueError("No price data available for backtesting")
        
        # Fetch fundamental data (using most recent data as proxy for historical)
        logger.info("Fetching fundamental data...")
        fundamental_data = self.data_fetcher.fetch_multiple_fundamentals(stocks)
        
        # Filter stocks with valid fundamental data
        fundamental_data = self.analyzer.filter_valid_stocks(fundamental_data)
        
        # Keep only stocks that have both price and fundamental data
        valid_symbols = set(price_data['Symbol'].unique()) & set(fundamental_data['symbol'].unique())
        
        price_data = price_data[price_data['Symbol'].isin(valid_symbols)]
        fundamental_data = fundamental_data[fundamental_data['symbol'].isin(valid_symbols)]
        
        logger.info(f"Prepared data for {len(valid_symbols)} stocks")
        
        return price_data, fundamental_data
    
    def get_stock_prices_on_date(self, price_data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """
        Get stock prices for a specific date
        """
        date_str = date.strftime('%Y-%m-%d')
        
        # Get prices for the specific date or the closest previous trading day
        prices = {}
        
        for symbol in price_data['Symbol'].unique():
            symbol_data = price_data[price_data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_index()
            
            # Find the closest date <= target date
            available_dates = symbol_data.index
            valid_dates = available_dates[available_dates <= date]
            
            if len(valid_dates) > 0:
                closest_date = valid_dates[-1]
                prices[symbol] = symbol_data.loc[closest_date, 'Close']
        
        return prices
    
    def run_backtest(self, start_date: str, end_date: Optional[str] = None) -> Dict:
        """
        Run the complete backtesting process
        """
        logger.info(f"Starting backtest from {start_date} to {end_date or 'present'}")
        
        # Prepare data
        price_data, fundamental_data = self.prepare_data(start_date, end_date)
        
        if fundamental_data.empty:
            raise ValueError("No valid fundamental data for backtesting")
        
        # Analyze fundamentals and get stock rankings
        analysis_results = self.analyzer.analyze_stocks(fundamental_data)
        selected_stocks = self.analyzer.get_top_stocks(analysis_results, self.strategy.portfolio_size)
        
        logger.info(f"Selected stocks for backtesting: {selected_stocks}")
        
        # Set up backtesting parameters
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = datetime.now()
        
        # Generate rebalancing dates
        rebalance_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            rebalance_dates.append(current_date)
            current_date += timedelta(days=self.strategy.rebalance_frequency)
        
        # Get all unique dates from price data for daily tracking
        all_dates = sorted(price_data.index.unique())
        logger.info(f"Found {len(all_dates)} unique dates in price data")
        if all_dates:
            logger.info(f"Date range: {all_dates[0]} to {all_dates[-1]}")
        
        # Convert to timezone-naive for comparison
        start_dt_naive = start_dt
        end_dt_naive = end_dt
        backtest_dates = []
        for d in all_dates:
            # Convert pandas timestamp to datetime if needed
            d_ts = pd.Timestamp(d)
            d_naive = d_ts.tz_localize(None) if d_ts.tz is not None else d_ts
            d_date = d_naive.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date_only = start_dt_naive.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date_only = end_dt_naive.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if start_date_only <= d_date <= end_date_only:
                backtest_dates.append(d)
        
        logger.info(f"Start date: {start_dt_naive}, End date: {end_dt_naive}")
        logger.info(f"Filtered to {len(backtest_dates)} trading days")
        
        logger.info(f"Backtesting over {len(backtest_dates)} trading days with {len(rebalance_dates)} rebalancing periods")
        
        # Run backtest simulation
        for i, date in enumerate(backtest_dates):
            stock_prices = self.get_stock_prices_on_date(price_data, date)
            
            if not stock_prices:
                continue
            
            # Check if it's a rebalancing date
            if date in rebalance_dates or i == 0:  # Rebalance on first day too
                logger.info(f"Rebalancing on {date.strftime('%Y-%m-%d')}")
                self.strategy.rebalance_portfolio(selected_stocks, stock_prices, date)
            
            # Check stop-loss and take-profit conditions
            self.strategy.check_stop_loss_take_profit(stock_prices, date)
            
            # Record portfolio state
            self.strategy.record_portfolio_state(stock_prices, date)
        
        # Get benchmark performance
        benchmark_data = self.data_fetcher.fetch_benchmark_data(start_date, end_date)
        benchmark_performance = self.calculate_benchmark_performance(benchmark_data, start_dt, end_dt)
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(benchmark_performance)
        
        logger.info("Backtest completed successfully")
        
        return {
            'performance_metrics': performance_metrics,
            'portfolio_history': self.strategy.get_portfolio_history_df(),
            'trade_history': self.strategy.get_trade_history_df(),
            'portfolio_summary': self.strategy.get_portfolio_summary(),
            'analysis_results': analysis_results,
            'benchmark_performance': benchmark_performance
        }
    
    def calculate_benchmark_performance(self, benchmark_data: pd.DataFrame, 
                                      start_date: datetime, end_date: datetime) -> Dict:
        """
        Calculate benchmark (Nikkei 225) performance
        """
        if benchmark_data.empty:
            return {'total_return': 0, 'annualized_return': 0}
        
        # Filter benchmark data to backtest period
        benchmark_data = benchmark_data.sort_index()
        start_date_tz = pd.Timestamp(start_date).tz_localize(benchmark_data.index.tz)
        end_date_tz = pd.Timestamp(end_date).tz_localize(benchmark_data.index.tz)
        mask = (benchmark_data.index >= start_date_tz) & (benchmark_data.index <= end_date_tz)
        benchmark_period = benchmark_data[mask]
        
        if len(benchmark_period) < 2:
            return {'total_return': 0, 'annualized_return': 0}
        
        start_price = benchmark_period['Close'].iloc[0]
        end_price = benchmark_period['Close'].iloc[-1]
        
        total_return = (end_price - start_price) / start_price
        
        # Calculate annualized return
        days = (benchmark_period.index[-1] - benchmark_period.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'start_price': start_price,
            'end_price': end_price,
            'data': benchmark_period
        }
    
    def calculate_performance_metrics(self, benchmark_performance: Dict) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        portfolio_history = self.strategy.get_portfolio_history_df()
        
        if portfolio_history.empty:
            return {}
        
        # Calculate returns
        portfolio_history = portfolio_history.sort_values('date')
        portfolio_history['daily_return'] = portfolio_history['total_value'].pct_change()
        
        initial_value = self.initial_capital
        final_value = portfolio_history['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate time period
        start_date = portfolio_history['date'].iloc[0]
        end_date = portfolio_history['date'].iloc[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        # Annualized return
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        daily_returns = portfolio_history['daily_return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming risk-free rate of 0%)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        portfolio_history['cumulative_max'] = portfolio_history['total_value'].cummax()
        portfolio_history['drawdown'] = (portfolio_history['total_value'] - portfolio_history['cumulative_max']) / portfolio_history['cumulative_max']
        max_drawdown = portfolio_history['drawdown'].min()
        
        # Win rate (percentage of profitable trades)
        trade_history = self.strategy.get_trade_history_df()
        if not trade_history.empty:
            sell_trades = trade_history[trade_history['action'] == 'SELL']
            profitable_trades = 0
            total_trades = 0
            
            for _, trade in sell_trades.iterrows():
                symbol = trade['symbol']
                # Find corresponding buy trade (simplified - assumes FIFO)
                buy_trades = trade_history[
                    (trade_history['symbol'] == symbol) & 
                    (trade_history['action'] == 'BUY') & 
                    (trade_history['date'] < trade['date'])
                ]
                
                if not buy_trades.empty:
                    # Use most recent buy price
                    buy_price = buy_trades['price'].iloc[-1]
                    if trade['price'] > buy_price:
                        profitable_trades += 1
                    total_trades += 1
            
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Comparison with benchmark
        benchmark_total_return = benchmark_performance.get('total_return', 0)
        benchmark_annualized_return = benchmark_performance.get('annualized_return', 0)
        alpha = annualized_return - benchmark_annualized_return
        
        metrics = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'num_trades': len(trade_history),
            'backtest_days': days,
            'backtest_years': years,
            'benchmark_total_return': benchmark_total_return,
            'benchmark_total_return_pct': benchmark_total_return * 100,
            'benchmark_annualized_return': benchmark_annualized_return,
            'benchmark_annualized_return_pct': benchmark_annualized_return * 100,
            'alpha': alpha,
            'alpha_pct': alpha * 100
        }
        
        return metrics
    
    def generate_performance_report(self, backtest_results: Dict) -> str:
        """
        Generate a comprehensive performance report
        """
        metrics = backtest_results['performance_metrics']
        
        if not metrics:
            return "No performance metrics available"
        
        report = []
        report.append("=" * 80)
        report.append("           JAPANESE STOCK FUNDAMENTAL ANALYSIS BACKTEST REPORT")
        report.append("=" * 80)
        
        # Performance Summary
        report.append("\nPERFORMANCE SUMMARY:")
        report.append("-" * 40)
        report.append(f"Initial Capital:         ¥{metrics['initial_capital']:,.0f}")
        report.append(f"Final Value:             ¥{metrics['final_value']:,.0f}")
        report.append(f"Total Return:            {metrics['total_return_pct']:+.2f}%")
        report.append(f"Annualized Return:       {metrics['annualized_return_pct']:+.2f}%")
        report.append(f"Volatility (Annual):     {metrics['volatility_pct']:.2f}%")
        report.append(f"Sharpe Ratio:            {metrics['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown:            {metrics['max_drawdown_pct']:.2f}%")
        report.append(f"Win Rate:                {metrics['win_rate_pct']:.1f}%")
        
        # Benchmark Comparison
        report.append("\nBENCHMARK COMPARISON (Nikkei 225):")
        report.append("-" * 40)
        report.append(f"Strategy Return:         {metrics['annualized_return_pct']:+.2f}%")
        report.append(f"Benchmark Return:        {metrics['benchmark_annualized_return_pct']:+.2f}%")
        report.append(f"Alpha (Outperformance):  {metrics['alpha_pct']:+.2f}%")
        
        # Trading Statistics
        report.append("\nTRADING STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Trades:            {metrics['num_trades']}")
        report.append(f"Backtest Period:         {metrics['backtest_days']} days ({metrics['backtest_years']:.1f} years)")
        
        # Top Holdings
        portfolio_summary = backtest_results.get('portfolio_summary', {})
        if 'positions' in portfolio_summary:
            report.append("\nCURRENT TOP HOLDINGS:")
            report.append("-" * 40)
            positions = portfolio_summary['positions']
            sorted_positions = sorted(positions.items(), 
                                    key=lambda x: x[1]['current_value'], reverse=True)
            
            for symbol, pos in sorted_positions[:5]:
                pnl_pct = pos['unrealized_pnl_pct'] * 100
                report.append(f"{symbol:<10} ¥{pos['current_value']:>10,.0f} ({pnl_pct:+.1f}%)")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)