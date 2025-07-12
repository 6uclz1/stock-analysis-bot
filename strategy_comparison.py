"""
Strategy comparison module for testing multiple strategies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

from backtester import Backtester
from strategy_factory import StrategyFactory
from trading_strategy import TradingStrategy
from fundamental_analyzer import FundamentalAnalyzer
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class StrategyComparison:
    """Compare multiple trading strategies"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.strategy_factory = StrategyFactory()
        self.results = {}
        
    def run_strategy_comparison(self, strategies: List[str], start_date: str, 
                              end_date: Optional[str] = None) -> Dict:
        """
        Run backtest comparison for multiple strategies
        """
        logger.info(f"Starting strategy comparison for {len(strategies)} strategies")
        
        comparison_results = {}
        
        for strategy_name in strategies:
            logger.info(f"Testing strategy: {strategy_name}")
            
            try:
                # Get strategy configuration
                strategy_config = self.strategy_factory.create_strategy(strategy_name)
                
                # Create customized backtester with strategy settings
                backtester = self._create_strategy_backtester(strategy_config)
                
                # Run backtest
                results = backtester.run_backtest(start_date, end_date)
                
                # Add strategy info to results
                results['strategy_config'] = strategy_config
                results['strategy_name'] = strategy_name
                
                comparison_results[strategy_name] = results
                
                logger.info(f"Completed strategy: {strategy_name}")
                
            except Exception as e:
                logger.error(f"Error testing strategy {strategy_name}: {str(e)}")
                continue
        
        self.results = comparison_results
        return comparison_results
    
    def _create_strategy_backtester(self, strategy_config: Dict) -> Backtester:
        """Create backtester with custom strategy configuration"""
        # Create backtester
        backtester = Backtester(self.initial_capital)
        
        # Update analyzer weights
        backtester.analyzer.weights = strategy_config['weights']
        
        # Update trading strategy parameters
        backtester.strategy.portfolio_size = strategy_config['portfolio_size']
        backtester.strategy.rebalance_frequency = strategy_config['rebalance_frequency']
        backtester.strategy.stop_loss = strategy_config['stop_loss']
        backtester.strategy.take_profit = strategy_config['take_profit']
        
        return backtester
    
    def generate_comparison_report(self) -> str:
        """Generate comparison report for all strategies"""
        if not self.results:
            return "No strategy results available for comparison"
        
        report = []
        report.append("=" * 100)
        report.append("                    STRATEGY COMPARISON REPORT")
        report.append("=" * 100)
        
        # Create summary table
        summary_data = []
        for strategy_name, results in self.results.items():
            metrics = results.get('performance_metrics', {})
            if metrics:
                summary_data.append({
                    'Strategy': strategy_name,
                    'Total Return (%)': f"{metrics.get('total_return_pct', 0):.2f}",
                    'Annual Return (%)': f"{metrics.get('annualized_return_pct', 0):.2f}",
                    'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                    'Max Drawdown (%)': f"{metrics.get('max_drawdown_pct', 0):.2f}",
                    'Win Rate (%)': f"{metrics.get('win_rate_pct', 0):.1f}",
                    'Alpha vs Nikkei (%)': f"{metrics.get('alpha_pct', 0):.2f}",
                    'Volatility (%)': f"{metrics.get('volatility_pct', 0):.2f}"
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            
            report.append("\nSTRATEGY PERFORMANCE SUMMARY:")
            report.append("-" * 100)
            report.append(df_summary.to_string(index=False))
            
            # Find best performing strategies
            report.append("\n\nTOP PERFORMERS:")
            report.append("-" * 50)
            
            # Best by total return
            best_return = df_summary.loc[df_summary['Annual Return (%)'].str.replace('%', '').astype(float).idxmax()]
            report.append(f"Best Annual Return:  {best_return['Strategy']} ({best_return['Annual Return (%)']}%)")
            
            # Best by Sharpe ratio
            best_sharpe = df_summary.loc[df_summary['Sharpe Ratio'].astype(float).idxmax()]
            report.append(f"Best Sharpe Ratio:   {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']})")
            
            # Best by alpha
            best_alpha = df_summary.loc[df_summary['Alpha vs Nikkei (%)'].str.replace('%', '').astype(float).idxmax()]
            report.append(f"Best Alpha:          {best_alpha['Strategy']} ({best_alpha['Alpha vs Nikkei (%)']}%)")
            
            # Lowest drawdown
            lowest_dd = df_summary.loc[df_summary['Max Drawdown (%)'].str.replace('%', '').astype(float).idxmin()]
            report.append(f"Lowest Drawdown:     {lowest_dd['Strategy']} ({lowest_dd['Max Drawdown (%)']}%)")
        
        # Strategy details
        report.append("\n\nSTRATEGY DETAILS:")
        report.append("-" * 100)
        
        for strategy_name, results in self.results.items():
            config_info = results.get('strategy_config', {})
            metrics = results.get('performance_metrics', {})
            
            report.append(f"\n{strategy_name.upper()}:")
            report.append(f"Description: {config_info.get('description', 'N/A')}")
            report.append(f"Portfolio Size: {config_info.get('portfolio_size', 'N/A')} stocks")
            report.append(f"Rebalance Frequency: {config_info.get('rebalance_frequency', 'N/A')} days")
            report.append(f"Stop Loss: {config_info.get('stop_loss', 0)*100:.1f}%")
            report.append(f"Take Profit: {config_info.get('take_profit', 0)*100:.1f}%")
            
            if metrics:
                report.append(f"Final Value: ¥{metrics.get('final_value', 0):,.0f}")
                report.append(f"Total Trades: {metrics.get('num_trades', 0)}")
        
        report.append("\n" + "=" * 100)
        
        return "\n".join(report)
    
    def create_comparison_visualizations(self, save_dir: Optional[str] = None):
        """Create comparison visualizations"""
        if not self.results:
            logger.warning("No results available for visualization")
            return
        
        if save_dir is None:
            save_dir = config.get_results_dir_with_timestamp()
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Performance comparison chart
        self._plot_performance_comparison(save_dir)
        
        # Risk-return scatter plot
        self._plot_risk_return_scatter(save_dir)
        
        # Portfolio evolution comparison
        self._plot_portfolio_evolution(save_dir)
        
        # Strategy characteristics radar chart
        self._plot_strategy_radar(save_dir)
        
        logger.info(f"Comparison visualizations saved to {save_dir}")
    
    def _plot_performance_comparison(self, save_dir: str):
        """Plot performance comparison bar chart"""
        metrics_data = []
        
        for strategy_name, results in self.results.items():
            metrics = results.get('performance_metrics', {})
            if metrics:
                metrics_data.append({
                    'Strategy': strategy_name,
                    'Annual Return': metrics.get('annualized_return_pct', 0),
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Alpha': metrics.get('alpha_pct', 0)
                })
        
        if not metrics_data:
            return
        
        df = pd.DataFrame(metrics_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Annual returns
        bars1 = ax1.bar(df['Strategy'], df['Annual Return'], color='steelblue', alpha=0.8)
        ax1.set_title('Annual Returns by Strategy', fontweight='bold')
        ax1.set_ylabel('Annual Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Sharpe ratios
        bars2 = ax2.bar(df['Strategy'], df['Sharpe Ratio'], color='green', alpha=0.8)
        ax2.set_title('Sharpe Ratios by Strategy', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Alpha vs benchmark
        colors = ['red' if x < 0 else 'green' for x in df['Alpha']]
        bars3 = ax3.bar(df['Strategy'], df['Alpha'], color=colors, alpha=0.8)
        ax3.set_title('Alpha vs Nikkei 225 by Strategy', fontweight='bold')
        ax3.set_ylabel('Alpha (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height >= 0 else -0.5),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Portfolio sizes
        portfolio_sizes = [self.results[name]['strategy_config']['portfolio_size'] 
                          for name in df['Strategy']]
        bars4 = ax4.bar(df['Strategy'], portfolio_sizes, color='orange', alpha=0.8)
        ax4.set_title('Portfolio Size by Strategy', fontweight='bold')
        ax4.set_ylabel('Number of Stocks')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'strategy_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_risk_return_scatter(self, save_dir: str):
        """Plot risk-return scatter plot"""
        risk_return_data = []
        
        for strategy_name, results in self.results.items():
            metrics = results.get('performance_metrics', {})
            if metrics:
                risk_return_data.append({
                    'Strategy': strategy_name,
                    'Return': metrics.get('annualized_return_pct', 0),
                    'Volatility': metrics.get('volatility_pct', 0),
                    'Sharpe': metrics.get('sharpe_ratio', 0)
                })
        
        if not risk_return_data:
            return
        
        df = pd.DataFrame(risk_return_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot with color based on Sharpe ratio
        scatter = ax.scatter(df['Volatility'], df['Return'], 
                           c=df['Sharpe'], s=200, alpha=0.7, 
                           cmap='RdYlGn', edgecolors='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=15)
        
        # Add strategy labels
        for i, row in df.iterrows():
            ax.annotate(row['Strategy'], 
                       (row['Volatility'], row['Return']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, ha='left')
        
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Annual Return (%)')
        ax.set_title('Risk-Return Profile by Strategy', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'risk_return_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_portfolio_evolution(self, save_dir: str):
        """Plot portfolio value evolution for all strategies"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for i, (strategy_name, results) in enumerate(self.results.items()):
            portfolio_history = results.get('portfolio_history')
            if portfolio_history is not None and not portfolio_history.empty:
                portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
                ax.plot(portfolio_history['date'], portfolio_history['total_value'],
                       label=strategy_name, color=colors[i], linewidth=2)
        
        # Add benchmark if available
        benchmark_data = None
        for results in self.results.values():
            if 'benchmark_performance' in results and 'data' in results['benchmark_performance']:
                benchmark_data = results['benchmark_performance']['data']
                break
        
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalize benchmark to initial capital
            start_value = self.initial_capital
            benchmark_start = benchmark_data['Close'].iloc[0]
            normalized_benchmark = (benchmark_data['Close'] / benchmark_start) * start_value
            
            ax.plot(benchmark_data.index, normalized_benchmark,
                   label='Nikkei 225', color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (¥)')
        ax.set_title('Portfolio Evolution Comparison', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'portfolio_evolution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_strategy_radar(self, save_dir: str):
        """Plot radar chart comparing strategy characteristics"""
        # Strategy characteristics data
        char_data = []
        
        for strategy_name, results in self.results.items():
            metrics = results.get('performance_metrics', {})
            config_info = results.get('strategy_config', {})
            
            if metrics:
                # Normalize metrics to 0-10 scale
                char_data.append({
                    'Strategy': strategy_name,
                    'Return': min(max(metrics.get('annualized_return_pct', 0) / 5, 0), 10),
                    'Sharpe': min(max(metrics.get('sharpe_ratio', 0) * 2, 0), 10),
                    'Low Risk': min(max(10 - abs(metrics.get('max_drawdown_pct', 0)) / 2, 0), 10),
                    'Win Rate': min(max(metrics.get('win_rate_pct', 0) / 10, 0), 10),
                    'Alpha': min(max((metrics.get('alpha_pct', 0) + 20) / 4, 0), 10),
                    'Consistency': min(max(10 - metrics.get('volatility_pct', 0) / 2, 0), 10)
                })
        
        if not char_data:
            return
        
        df = pd.DataFrame(char_data)
        
        # Setup radar chart
        categories = ['Return', 'Sharpe', 'Low Risk', 'Win Rate', 'Alpha', 'Consistency']
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each category
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[cat] for cat in categories]
            values += [values[0]]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Strategy'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'])
        ax.grid(True)
        
        ax.set_title('Strategy Characteristics Comparison', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'strategy_radar_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_comparison_results(self, save_dir: Optional[str] = None):
        """Save comparison results to CSV files"""
        if save_dir is None:
            save_dir = config.get_results_dir_with_timestamp()
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Strategy performance summary
        summary_data = []
        for strategy_name, results in self.results.items():
            metrics = results.get('performance_metrics', {})
            config_info = results.get('strategy_config', {})
            
            if metrics:
                summary_data.append({
                    'strategy_name': strategy_name,
                    'description': config_info.get('description', ''),
                    'portfolio_size': config_info.get('portfolio_size', 0),
                    'rebalance_frequency': config_info.get('rebalance_frequency', 0),
                    'stop_loss': config_info.get('stop_loss', 0),
                    'take_profit': config_info.get('take_profit', 0),
                    **metrics
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(save_dir, 'strategy_comparison_summary.csv'), index=False)
        
        # Individual strategy results
        for strategy_name, results in self.results.items():
            strategy_dir = os.path.join(save_dir, f"strategy_{strategy_name.lower().replace(' ', '_')}")
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Save portfolio history
            portfolio_history = results.get('portfolio_history')
            if portfolio_history is not None and not portfolio_history.empty:
                portfolio_history.to_csv(os.path.join(strategy_dir, 'portfolio_history.csv'), index=False)
            
            # Save trade history
            trade_history = results.get('trade_history')
            if trade_history is not None and not trade_history.empty:
                trade_history.to_csv(os.path.join(strategy_dir, 'trade_history.csv'), index=False)
        
        logger.info(f"Comparison results saved to {save_dir}")