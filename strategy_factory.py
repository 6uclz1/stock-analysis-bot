"""
Strategy factory for creating different trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from fundamental_analyzer import FundamentalAnalyzer
import config

class StrategyFactory:
    """Factory for creating different trading strategies"""
    
    def __init__(self):
        self.analyzer = FundamentalAnalyzer()
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return [
            'fundamental_conservative',
            'fundamental_aggressive', 
            'value_investing',
            'growth_investing',
            'dividend_focus',
            'momentum_value',
            'low_volatility',
            'quality_growth'
        ]
    
    def create_strategy(self, strategy_name: str) -> Dict:
        """Create strategy configuration"""
        strategies = {
            'fundamental_conservative': self._fundamental_conservative(),
            'fundamental_aggressive': self._fundamental_aggressive(),
            'value_investing': self._value_investing(),
            'growth_investing': self._growth_investing(),
            'dividend_focus': self._dividend_focus(),
            'momentum_value': self._momentum_value(),
            'low_volatility': self._low_volatility(),
            'quality_growth': self._quality_growth()
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Strategy {strategy_name} not found. Available: {list(strategies.keys())}")
        
        return strategies[strategy_name]
    
    def _fundamental_conservative(self) -> Dict:
        """Conservative fundamental strategy - original strategy"""
        return {
            'name': 'Fundamental Conservative',
            'description': 'Conservative fundamental analysis with balanced risk management',
            'weights': {
                'per_score': 0.20,
                'pbr_score': 0.15,
                'roe_score': 0.20,
                'debt_ratio_score': 0.15,
                'profit_margin_score': 0.15,
                'dividend_yield_score': 0.15
            },
            'portfolio_size': 8,
            'rebalance_frequency': 90,
            'stop_loss': -0.15,
            'take_profit': 0.30,
            'scoring_thresholds': {
                'per_good': 15, 'per_bad': 25,
                'pbr_good': 1.0, 'pbr_bad': 2.0,
                'roe_good': 0.15, 'roe_bad': 0.05,
                'debt_good': 0.3, 'debt_bad': 0.6,
                'margin_good': 0.1, 'margin_bad': 0.02,
                'dividend_good': 0.03, 'dividend_bad': 0.01
            }
        }
    
    def _fundamental_aggressive(self) -> Dict:
        """Aggressive fundamental strategy"""
        return {
            'name': 'Fundamental Aggressive',
            'description': 'Aggressive fundamental analysis with higher risk tolerance',
            'weights': {
                'per_score': 0.15,
                'pbr_score': 0.10,
                'roe_score': 0.30,
                'debt_ratio_score': 0.10,
                'profit_margin_score': 0.25,
                'dividend_yield_score': 0.10
            },
            'portfolio_size': 5,
            'rebalance_frequency': 60,
            'stop_loss': -0.20,
            'take_profit': 0.50,
            'scoring_thresholds': {
                'per_good': 20, 'per_bad': 35,
                'pbr_good': 1.5, 'pbr_bad': 3.0,
                'roe_good': 0.20, 'roe_bad': 0.10,
                'debt_good': 0.4, 'debt_bad': 0.8,
                'margin_good': 0.15, 'margin_bad': 0.05,
                'dividend_good': 0.02, 'dividend_bad': 0.005
            }
        }
    
    def _value_investing(self) -> Dict:
        """Value investing strategy - focus on low P/E and P/B"""
        return {
            'name': 'Value Investing',
            'description': 'Focus on undervalued stocks with low P/E and P/B ratios',
            'weights': {
                'per_score': 0.35,
                'pbr_score': 0.30,
                'roe_score': 0.15,
                'debt_ratio_score': 0.10,
                'profit_margin_score': 0.05,
                'dividend_yield_score': 0.05
            },
            'portfolio_size': 10,
            'rebalance_frequency': 120,
            'stop_loss': -0.10,
            'take_profit': 0.25,
            'scoring_thresholds': {
                'per_good': 10, 'per_bad': 20,
                'pbr_good': 0.8, 'pbr_bad': 1.5,
                'roe_good': 0.12, 'roe_bad': 0.05,
                'debt_good': 0.3, 'debt_bad': 0.6,
                'margin_good': 0.08, 'margin_bad': 0.02,
                'dividend_good': 0.03, 'dividend_bad': 0.01
            }
        }
    
    def _growth_investing(self) -> Dict:
        """Growth investing strategy - focus on ROE and profit margins"""
        return {
            'name': 'Growth Investing',
            'description': 'Focus on companies with high growth potential and profitability',
            'weights': {
                'per_score': 0.10,
                'pbr_score': 0.05,
                'roe_score': 0.40,
                'debt_ratio_score': 0.15,
                'profit_margin_score': 0.25,
                'dividend_yield_score': 0.05
            },
            'portfolio_size': 6,
            'rebalance_frequency': 45,
            'stop_loss': -0.25,
            'take_profit': 0.60,
            'scoring_thresholds': {
                'per_good': 25, 'per_bad': 40,
                'pbr_good': 2.0, 'pbr_bad': 4.0,
                'roe_good': 0.25, 'roe_bad': 0.15,
                'debt_good': 0.3, 'debt_bad': 0.5,
                'margin_good': 0.20, 'margin_bad': 0.10,
                'dividend_good': 0.02, 'dividend_bad': 0.005
            }
        }
    
    def _dividend_focus(self) -> Dict:
        """Dividend-focused strategy"""
        return {
            'name': 'Dividend Focus',
            'description': 'Focus on high dividend yield stocks with stable fundamentals',
            'weights': {
                'per_score': 0.15,
                'pbr_score': 0.15,
                'roe_score': 0.15,
                'debt_ratio_score': 0.20,
                'profit_margin_score': 0.15,
                'dividend_yield_score': 0.20
            },
            'portfolio_size': 12,
            'rebalance_frequency': 180,
            'stop_loss': -0.12,
            'take_profit': 0.20,
            'scoring_thresholds': {
                'per_good': 12, 'per_bad': 20,
                'pbr_good': 1.0, 'pbr_bad': 2.0,
                'roe_good': 0.12, 'roe_bad': 0.05,
                'debt_good': 0.2, 'debt_bad': 0.4,
                'margin_good': 0.08, 'margin_bad': 0.03,
                'dividend_good': 0.04, 'dividend_bad': 0.02
            }
        }
    
    def _momentum_value(self) -> Dict:
        """Momentum + Value hybrid strategy"""
        return {
            'name': 'Momentum Value',
            'description': 'Combines value metrics with momentum indicators',
            'weights': {
                'per_score': 0.25,
                'pbr_score': 0.20,
                'roe_score': 0.25,
                'debt_ratio_score': 0.10,
                'profit_margin_score': 0.15,
                'dividend_yield_score': 0.05
            },
            'portfolio_size': 7,
            'rebalance_frequency': 30,
            'stop_loss': -0.18,
            'take_profit': 0.35,
            'scoring_thresholds': {
                'per_good': 12, 'per_bad': 22,
                'pbr_good': 0.9, 'pbr_bad': 1.8,
                'roe_good': 0.18, 'roe_bad': 0.08,
                'debt_good': 0.25, 'debt_bad': 0.5,
                'margin_good': 0.12, 'margin_bad': 0.04,
                'dividend_good': 0.025, 'dividend_bad': 0.01
            }
        }
    
    def _low_volatility(self) -> Dict:
        """Low volatility strategy - focus on stable companies"""
        return {
            'name': 'Low Volatility',
            'description': 'Focus on stable, low-risk companies with consistent performance',
            'weights': {
                'per_score': 0.20,
                'pbr_score': 0.20,
                'roe_score': 0.15,
                'debt_ratio_score': 0.25,
                'profit_margin_score': 0.10,
                'dividend_yield_score': 0.10
            },
            'portfolio_size': 15,
            'rebalance_frequency': 120,
            'stop_loss': -0.08,
            'take_profit': 0.15,
            'scoring_thresholds': {
                'per_good': 8, 'per_bad': 15,
                'pbr_good': 0.7, 'pbr_bad': 1.2,
                'roe_good': 0.10, 'roe_bad': 0.05,
                'debt_good': 0.15, 'debt_bad': 0.3,
                'margin_good': 0.06, 'margin_bad': 0.02,
                'dividend_good': 0.03, 'dividend_bad': 0.015
            }
        }
    
    def _quality_growth(self) -> Dict:
        """Quality growth strategy - high ROE, low debt, strong margins"""
        return {
            'name': 'Quality Growth',
            'description': 'High-quality companies with strong fundamentals and growth',
            'weights': {
                'per_score': 0.15,
                'pbr_score': 0.10,
                'roe_score': 0.35,
                'debt_ratio_score': 0.20,
                'profit_margin_score': 0.15,
                'dividend_yield_score': 0.05
            },
            'portfolio_size': 8,
            'rebalance_frequency': 75,
            'stop_loss': -0.20,
            'take_profit': 0.40,
            'scoring_thresholds': {
                'per_good': 18, 'per_bad': 30,
                'pbr_good': 1.2, 'pbr_bad': 2.5,
                'roe_good': 0.22, 'roe_bad': 0.12,
                'debt_good': 0.2, 'debt_bad': 0.4,
                'margin_good': 0.15, 'margin_bad': 0.08,
                'dividend_good': 0.02, 'dividend_bad': 0.005
            }
        }
    
    def apply_strategy_to_analyzer(self, strategy_config: Dict, analyzer: FundamentalAnalyzer):
        """Apply strategy configuration to fundamental analyzer"""
        # Update weights
        analyzer.weights = strategy_config['weights']
        
        # Update scoring thresholds (would need to modify FundamentalAnalyzer class)
        # For now, we'll return the thresholds to be used in a modified analyzer
        return strategy_config['scoring_thresholds']