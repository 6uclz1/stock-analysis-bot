"""
Visualization module for backtesting results and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import config

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Visualizer:
    """Creates visualizations for backtesting results and analysis"""
    
    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configure matplotlib for Japanese fonts (if available)
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_portfolio_performance(self, portfolio_history: pd.DataFrame, 
                                 benchmark_data: Optional[pd.DataFrame] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot portfolio value over time vs benchmark
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Portfolio value over time
        portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
        
        ax1.plot(portfolio_history['date'], portfolio_history['total_value'], 
                linewidth=2, label='Portfolio Value', color='blue')
        
        # Add benchmark if available
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalize benchmark to same starting value
            start_value = portfolio_history['total_value'].iloc[0]
            benchmark_start = benchmark_data['Close'].iloc[0]
            normalized_benchmark = (benchmark_data['Close'] / benchmark_start) * start_value
            
            ax1.plot(benchmark_data.index, normalized_benchmark, 
                    linewidth=2, label='Nikkei 225 (Normalized)', color='red', alpha=0.7)
        
        ax1.set_title('Portfolio Performance Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value (¥)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Format y-axis with comma separators
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x:,.0f}'))
        
        # Portfolio returns
        portfolio_history['return_pct'] = portfolio_history['return_from_start'] * 100
        ax2.plot(portfolio_history['date'], portfolio_history['return_pct'], 
                linewidth=2, color='green')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax2.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_drawdown(self, portfolio_history: pd.DataFrame, 
                     save_path: Optional[str] = None) -> None:
        """
        Plot drawdown over time
        """
        portfolio_history = portfolio_history.copy()
        portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
        
        # Calculate drawdown
        portfolio_history['cumulative_max'] = portfolio_history['total_value'].cummax()
        portfolio_history['drawdown'] = (
            (portfolio_history['total_value'] - portfolio_history['cumulative_max']) / 
            portfolio_history['cumulative_max'] * 100
        )
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.fill_between(portfolio_history['date'], portfolio_history['drawdown'], 0, 
                       alpha=0.6, color='red', label='Drawdown')
        ax.plot(portfolio_history['date'], portfolio_history['drawdown'], 
               color='darkred', linewidth=1)
        
        ax.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        # Highlight maximum drawdown
        max_dd_idx = portfolio_history['drawdown'].idxmin()
        max_dd_date = portfolio_history.loc[max_dd_idx, 'date']
        max_dd_value = portfolio_history.loc[max_dd_idx, 'drawdown']
        
        ax.annotate(f'Max DD: {max_dd_value:.2f}%', 
                   xy=(max_dd_date, max_dd_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_monthly_returns(self, portfolio_history: pd.DataFrame, 
                            save_path: Optional[str] = None) -> None:
        """
        Plot monthly returns heatmap
        """
        portfolio_history = portfolio_history.copy()
        portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
        portfolio_history = portfolio_history.set_index('date')
        
        # Calculate monthly returns
        monthly_returns = portfolio_history['total_value'].resample('M').last().pct_change() * 100
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) < 2:
            print("Insufficient data for monthly returns heatmap")
            return
        
        # Create year-month matrix
        monthly_data = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_table = monthly_data.pivot(index='Year', columns='Month', values='Return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'}, ax=ax)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_fundamental_scores(self, analysis_results: pd.DataFrame, 
                               top_n: int = 15, save_path: Optional[str] = None) -> None:
        """
        Plot fundamental analysis scores
        """
        top_stocks = analysis_results.head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Composite scores bar chart
        bars = ax1.barh(range(len(top_stocks)), top_stocks['composite_score'], 
                       color='steelblue', alpha=0.8)
        ax1.set_yticks(range(len(top_stocks)))
        ax1.set_yticklabels(top_stocks['symbol'])
        ax1.set_xlabel('Composite Fundamental Score')
        ax1.set_title(f'Top {top_n} Stocks by Fundamental Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add score labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', ha='left', va='center')
        
        # Individual metric scores heatmap for top stocks
        score_columns = ['per_score', 'pbr_score', 'roe_score', 
                        'debt_ratio_score', 'profit_margin_score', 'dividend_yield_score']
        
        score_data = top_stocks[['symbol'] + score_columns].set_index('symbol')
        score_data.columns = ['P/E', 'P/B', 'ROE', 'Debt', 'Profit Margin', 'Dividend']
        
        sns.heatmap(score_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Score (0-10)'}, ax=ax2)
        
        ax2.set_title('Individual Metric Scores', fontweight='bold')
        ax2.set_xlabel('Fundamental Metrics')
        ax2.set_ylabel('Stock Symbol')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sector_allocation(self, analysis_results: pd.DataFrame, 
                              selected_stocks: List[str], save_path: Optional[str] = None) -> None:
        """
        Plot sector allocation of selected stocks
        """
        selected_data = analysis_results[analysis_results['symbol'].isin(selected_stocks)]
        
        if 'sector' not in selected_data.columns or selected_data['sector'].isna().all():
            print("Sector information not available")
            return
        
        sector_counts = selected_data['sector'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sector_counts)))
        wedges, texts, autotexts = ax.pie(sector_counts.values, labels=sector_counts.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        ax.set_title('Portfolio Sector Allocation', fontsize=14, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_trade_analysis(self, trade_history: pd.DataFrame, 
                           save_path: Optional[str] = None) -> None:
        """
        Plot trading activity analysis
        """
        if trade_history.empty:
            print("No trade history available")
            return
        
        trade_history = trade_history.copy()
        trade_history['date'] = pd.to_datetime(trade_history['date'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Trade frequency over time
        trade_counts = trade_history.set_index('date').resample('M').size()
        ax1.plot(trade_counts.index, trade_counts.values, marker='o', linewidth=2)
        ax1.set_title('Trading Frequency Over Time', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Trades per Month')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Buy vs Sell distribution
        action_counts = trade_history['action'].value_counts()
        ax2.bar(action_counts.index, action_counts.values, color=['green', 'red'], alpha=0.7)
        ax2.set_title('Buy vs Sell Trades', fontweight='bold')
        ax2.set_ylabel('Number of Trades')
        
        # Trade value distribution
        ax3.hist(trade_history['value'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_title('Trade Value Distribution', fontweight='bold')
        ax3.set_xlabel('Trade Value (¥)')
        ax3.set_ylabel('Frequency')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x:,.0f}'))
        
        # Most traded stocks
        symbol_counts = trade_history['symbol'].value_counts().head(10)
        ax4.barh(range(len(symbol_counts)), symbol_counts.values, color='orange', alpha=0.7)
        ax4.set_yticks(range(len(symbol_counts)))
        ax4.set_yticklabels(symbol_counts.index)
        ax4.set_title('Most Traded Stocks (Top 10)', fontweight='bold')
        ax4.set_xlabel('Number of Trades')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comprehensive_report(self, backtest_results: Dict, 
                                   save_dir: Optional[str] = None) -> None:
        """
        Create a comprehensive visual report
        """
        if save_dir is None:
            save_dir = config.get_results_dir_with_timestamp()
        
        os.makedirs(save_dir, exist_ok=True)
        
        portfolio_history = backtest_results.get('portfolio_history')
        trade_history = backtest_results.get('trade_history')
        analysis_results = backtest_results.get('analysis_results')
        benchmark_performance = backtest_results.get('benchmark_performance', {})
        
        benchmark_data = benchmark_performance.get('data')
        
        print("Generating comprehensive visual report...")
        
        # 1. Portfolio performance
        if portfolio_history is not None and not portfolio_history.empty:
            self.plot_portfolio_performance(
                portfolio_history, benchmark_data,
                save_path=os.path.join(save_dir, 'portfolio_performance.png')
            )
        
        # 2. Drawdown analysis
        if portfolio_history is not None and not portfolio_history.empty:
            self.plot_drawdown(
                portfolio_history,
                save_path=os.path.join(save_dir, 'drawdown_analysis.png')
            )
        
        # 3. Monthly returns heatmap
        if portfolio_history is not None and not portfolio_history.empty:
            self.plot_monthly_returns(
                portfolio_history,
                save_path=os.path.join(save_dir, 'monthly_returns.png')
            )
        
        # 4. Fundamental analysis scores
        if analysis_results is not None and not analysis_results.empty:
            self.plot_fundamental_scores(
                analysis_results,
                save_path=os.path.join(save_dir, 'fundamental_scores.png')
            )
        
        # 5. Trading analysis
        if trade_history is not None and not trade_history.empty:
            self.plot_trade_analysis(
                trade_history,
                save_path=os.path.join(save_dir, 'trade_analysis.png')
            )
        
        print(f"Visual report saved to: {save_dir}")
    
    def save_results_to_csv(self, backtest_results: Dict, save_dir: Optional[str] = None) -> None:
        """
        Save all results to CSV files
        """
        if save_dir is None:
            save_dir = config.get_results_dir_with_timestamp()
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save portfolio history
        portfolio_history = backtest_results.get('portfolio_history')
        if portfolio_history is not None and not portfolio_history.empty:
            portfolio_history.to_csv(os.path.join(save_dir, 'portfolio_history.csv'), index=False)
        
        # Save trade history
        trade_history = backtest_results.get('trade_history')
        if trade_history is not None and not trade_history.empty:
            trade_history.to_csv(os.path.join(save_dir, 'trade_history.csv'), index=False)
        
        # Save analysis results
        analysis_results = backtest_results.get('analysis_results')
        if analysis_results is not None and not analysis_results.empty:
            analysis_results.to_csv(os.path.join(save_dir, 'fundamental_analysis.csv'), index=False)
        
        # Save performance metrics
        performance_metrics = backtest_results.get('performance_metrics', {})
        if performance_metrics:
            metrics_df = pd.DataFrame([performance_metrics])
            metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)
        
        print(f"CSV results saved to: {save_dir}")