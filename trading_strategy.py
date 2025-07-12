"""
Trading strategy and portfolio management module
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class TradingStrategy:
    """Implements fundamental-based trading strategy with portfolio management"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio = {}  # {symbol: {'shares': int, 'avg_price': float, 'entry_date': datetime}}
        self.trade_history = []
        self.portfolio_history = []
        self.rebalance_frequency = config.REBALANCE_FREQUENCY
        self.portfolio_size = config.PORTFOLIO_SIZE
        self.stop_loss = config.STOP_LOSS_THRESHOLD
        self.take_profit = config.TAKE_PROFIT_THRESHOLD
        
    def calculate_position_size(self, stock_price: float, num_stocks: int) -> int:
        """
        Calculate position size based on equal weighting
        """
        target_position_value = self.current_capital / num_stocks
        shares = int(target_position_value / stock_price)
        return max(shares, 0)
    
    def execute_trade(self, symbol: str, action: str, shares: int, price: float, 
                     date: datetime, reason: str = ""):
        """
        Execute a trade and update portfolio
        """
        trade_value = shares * price
        
        if action == "BUY":
            if trade_value <= self.current_capital:
                self.current_capital -= trade_value
                
                if symbol in self.portfolio:
                    # Update existing position
                    current_shares = self.portfolio[symbol]['shares']
                    current_value = current_shares * self.portfolio[symbol]['avg_price']
                    new_avg_price = (current_value + trade_value) / (current_shares + shares)
                    
                    self.portfolio[symbol]['shares'] += shares
                    self.portfolio[symbol]['avg_price'] = new_avg_price
                else:
                    # New position
                    self.portfolio[symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'entry_date': date
                    }
                
                logger.info(f"BUY: {shares} shares of {symbol} at ¥{price:.2f}")
                
            else:
                logger.warning(f"Insufficient capital for {symbol} purchase")
                return False
                
        elif action == "SELL":
            if symbol in self.portfolio and self.portfolio[symbol]['shares'] >= shares:
                self.current_capital += trade_value
                self.portfolio[symbol]['shares'] -= shares
                
                if self.portfolio[symbol]['shares'] == 0:
                    del self.portfolio[symbol]
                
                logger.info(f"SELL: {shares} shares of {symbol} at ¥{price:.2f}")
            else:
                logger.warning(f"Insufficient shares for {symbol} sale")
                return False
        
        # Record trade
        trade_record = {
            'date': date,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'value': trade_value,
            'reason': reason,
            'cash_balance': self.current_capital
        }
        
        self.trade_history.append(trade_record)
        return True
    
    def rebalance_portfolio(self, selected_stocks: List[str], stock_prices: Dict[str, float], 
                          date: datetime) -> None:
        """
        Rebalance portfolio to target allocation
        """
        logger.info(f"Rebalancing portfolio on {date.strftime('%Y-%m-%d')}")
        
        # Sell stocks not in selected list
        stocks_to_sell = [symbol for symbol in self.portfolio.keys() if symbol not in selected_stocks]
        
        for symbol in stocks_to_sell:
            if symbol in stock_prices:
                shares = self.portfolio[symbol]['shares']
                price = stock_prices[symbol]
                self.execute_trade(symbol, "SELL", shares, price, date, "Rebalance - Exit")
        
        # Calculate target positions for selected stocks
        available_stocks = [s for s in selected_stocks if s in stock_prices]
        
        if not available_stocks:
            logger.warning("No available stocks for rebalancing")
            return
        
        total_portfolio_value = self.get_portfolio_value(stock_prices)
        target_value_per_stock = total_portfolio_value / len(available_stocks)
        
        # Buy/adjust positions for selected stocks
        for symbol in available_stocks:
            price = stock_prices[symbol]
            target_shares = int(target_value_per_stock / price)
            current_shares = self.portfolio.get(symbol, {}).get('shares', 0)
            
            shares_diff = target_shares - current_shares
            
            if shares_diff > 0:
                # Buy more shares
                self.execute_trade(symbol, "BUY", shares_diff, price, date, "Rebalance - Entry/Add")
            elif shares_diff < 0:
                # Sell excess shares
                self.execute_trade(symbol, "SELL", abs(shares_diff), price, date, "Rebalance - Reduce")
    
    def check_stop_loss_take_profit(self, stock_prices: Dict[str, float], date: datetime) -> None:
        """
        Check and execute stop-loss and take-profit orders
        """
        symbols_to_check = list(self.portfolio.keys())
        
        for symbol in symbols_to_check:
            if symbol not in stock_prices:
                continue
                
            current_price = stock_prices[symbol]
            avg_price = self.portfolio[symbol]['avg_price']
            shares = self.portfolio[symbol]['shares']
            
            return_pct = (current_price - avg_price) / avg_price
            
            # Check stop loss
            if return_pct <= self.stop_loss:
                self.execute_trade(symbol, "SELL", shares, current_price, date, 
                                 f"Stop Loss ({return_pct:.2%})")
                logger.info(f"Stop loss triggered for {symbol}: {return_pct:.2%}")
            
            # Check take profit
            elif return_pct >= self.take_profit:
                self.execute_trade(symbol, "SELL", shares, current_price, date, 
                                 f"Take Profit ({return_pct:.2%})")
                logger.info(f"Take profit triggered for {symbol}: {return_pct:.2%}")
    
    def get_portfolio_value(self, stock_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        """
        portfolio_value = self.current_capital
        
        for symbol, position in self.portfolio.items():
            if symbol in stock_prices:
                portfolio_value += position['shares'] * stock_prices[symbol]
        
        return portfolio_value
    
    def record_portfolio_state(self, stock_prices: Dict[str, float], date: datetime) -> None:
        """
        Record current portfolio state for tracking
        """
        total_value = self.get_portfolio_value(stock_prices)
        
        # Calculate individual position values
        positions = {}
        for symbol, position in self.portfolio.items():
            if symbol in stock_prices:
                current_value = position['shares'] * stock_prices[symbol]
                positions[symbol] = {
                    'shares': position['shares'],
                    'avg_price': position['avg_price'],
                    'current_price': stock_prices[symbol],
                    'current_value': current_value,
                    'unrealized_pnl': current_value - (position['shares'] * position['avg_price']),
                    'unrealized_pnl_pct': (stock_prices[symbol] - position['avg_price']) / position['avg_price']
                }
        
        portfolio_state = {
            'date': date,
            'total_value': total_value,
            'cash': self.current_capital,
            'positions': positions,
            'num_positions': len(self.portfolio),
            'return_from_start': (total_value - self.initial_capital) / self.initial_capital
        }
        
        self.portfolio_history.append(portfolio_state)
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get current portfolio summary
        """
        if not self.portfolio_history:
            return {}
        
        latest = self.portfolio_history[-1]
        
        # Calculate performance metrics
        total_return = latest['return_from_start']
        
        if len(self.portfolio_history) > 1:
            returns = [state['return_from_start'] for state in self.portfolio_history]
            daily_returns = np.diff(returns)
            volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
            sharpe_ratio = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': latest['total_value'],
            'cash': latest['cash'],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(self.trade_history),
            'num_positions': latest['num_positions'],
            'positions': latest['positions']
        }
        
        return summary
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """
        Get portfolio history as DataFrame
        """
        if not self.portfolio_history:
            return pd.DataFrame()
        
        history_data = []
        for state in self.portfolio_history:
            row = {
                'date': state['date'],
                'total_value': state['total_value'],
                'cash': state['cash'],
                'num_positions': state['num_positions'],
                'return_from_start': state['return_from_start']
            }
            history_data.append(row)
        
        return pd.DataFrame(history_data)