"""
Fundamental analysis module for scoring Japanese stocks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """Analyzes fundamental metrics and scores stocks"""
    
    def __init__(self):
        self.weights = config.FUNDAMENTAL_WEIGHTS
        
    def score_per(self, per_value: float) -> float:
        """
        Score P/E ratio (lower is better for value investing)
        """
        if pd.isna(per_value) or per_value <= 0:
            return 0.0
        
        if per_value <= config.PER_GOOD_THRESHOLD:
            return 10.0
        elif per_value <= config.PER_BAD_THRESHOLD:
            # Linear interpolation between good and bad thresholds
            score = 10.0 - 8.0 * (per_value - config.PER_GOOD_THRESHOLD) / (config.PER_BAD_THRESHOLD - config.PER_GOOD_THRESHOLD)
            return max(score, 2.0)
        else:
            return 2.0
    
    def score_pbr(self, pbr_value: float) -> float:
        """
        Score P/B ratio (lower is better for value investing)
        """
        if pd.isna(pbr_value) or pbr_value <= 0:
            return 0.0
            
        if pbr_value <= config.PBR_GOOD_THRESHOLD:
            return 10.0
        elif pbr_value <= config.PBR_BAD_THRESHOLD:
            score = 10.0 - 8.0 * (pbr_value - config.PBR_GOOD_THRESHOLD) / (config.PBR_BAD_THRESHOLD - config.PBR_GOOD_THRESHOLD)
            return max(score, 2.0)
        else:
            return 2.0
    
    def score_roe(self, roe_value: float) -> float:
        """
        Score Return on Equity (higher is better)
        """
        if pd.isna(roe_value):
            return 0.0
            
        if roe_value >= config.ROE_GOOD_THRESHOLD:
            return 10.0
        elif roe_value >= config.ROE_BAD_THRESHOLD:
            score = 2.0 + 8.0 * (roe_value - config.ROE_BAD_THRESHOLD) / (config.ROE_GOOD_THRESHOLD - config.ROE_BAD_THRESHOLD)
            return min(score, 10.0)
        else:
            return 2.0
    
    def score_debt_ratio(self, debt_to_equity: float) -> float:
        """
        Score debt-to-equity ratio (lower is better)
        """
        if pd.isna(debt_to_equity):
            return 5.0  # Neutral score for missing data
            
        if debt_to_equity <= config.DEBT_RATIO_GOOD_THRESHOLD:
            return 10.0
        elif debt_to_equity <= config.DEBT_RATIO_BAD_THRESHOLD:
            score = 10.0 - 8.0 * (debt_to_equity - config.DEBT_RATIO_GOOD_THRESHOLD) / (config.DEBT_RATIO_BAD_THRESHOLD - config.DEBT_RATIO_GOOD_THRESHOLD)
            return max(score, 2.0)
        else:
            return 2.0
    
    def score_profit_margin(self, profit_margin: float) -> float:
        """
        Score profit margin (higher is better)
        """
        if pd.isna(profit_margin):
            return 0.0
            
        if profit_margin >= config.PROFIT_MARGIN_GOOD_THRESHOLD:
            return 10.0
        elif profit_margin >= config.PROFIT_MARGIN_BAD_THRESHOLD:
            score = 2.0 + 8.0 * (profit_margin - config.PROFIT_MARGIN_BAD_THRESHOLD) / (config.PROFIT_MARGIN_GOOD_THRESHOLD - config.PROFIT_MARGIN_BAD_THRESHOLD)
            return min(score, 10.0)
        else:
            return 2.0
    
    def score_dividend_yield(self, dividend_yield: float) -> float:
        """
        Score dividend yield (higher is better, but not too high)
        """
        if pd.isna(dividend_yield) or dividend_yield == 0:
            return 5.0  # Neutral score for no dividend
            
        if config.DIVIDEND_YIELD_GOOD_THRESHOLD <= dividend_yield <= 0.08:  # Sweet spot: 3-8%
            return 10.0
        elif dividend_yield >= config.DIVIDEND_YIELD_BAD_THRESHOLD:
            if dividend_yield < config.DIVIDEND_YIELD_GOOD_THRESHOLD:
                score = 2.0 + 8.0 * (dividend_yield - config.DIVIDEND_YIELD_BAD_THRESHOLD) / (config.DIVIDEND_YIELD_GOOD_THRESHOLD - config.DIVIDEND_YIELD_BAD_THRESHOLD)
                return min(score, 10.0)
            elif dividend_yield > 0.08:  # Too high might indicate distress
                return 6.0
            else:
                return 10.0
        else:
            return 2.0
    
    def calculate_fundamental_score(self, fundamentals: Dict) -> Tuple[float, Dict]:
        """
        Calculate composite fundamental score for a stock
        """
        scores = {}
        
        # Calculate individual scores
        scores['per_score'] = self.score_per(fundamentals.get('pe_ratio'))
        scores['pbr_score'] = self.score_pbr(fundamentals.get('pb_ratio'))
        scores['roe_score'] = self.score_roe(fundamentals.get('roe'))
        scores['debt_ratio_score'] = self.score_debt_ratio(fundamentals.get('debt_to_equity'))
        scores['profit_margin_score'] = self.score_profit_margin(fundamentals.get('profit_margin'))
        scores['dividend_yield_score'] = self.score_dividend_yield(fundamentals.get('dividend_yield'))
        
        # Calculate weighted composite score
        composite_score = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            if metric in self.weights and not pd.isna(score):
                weight = self.weights[metric]
                composite_score += score * weight
                total_weight += weight
        
        # Normalize by actual total weight (in case some metrics are missing)
        if total_weight > 0:
            composite_score = composite_score / total_weight * 10.0  # Scale to 0-10
        
        return composite_score, scores
    
    def analyze_stocks(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze multiple stocks and return ranked results
        """
        logger.info(f"Analyzing {len(fundamentals_df)} stocks")
        
        results = []
        
        for _, row in fundamentals_df.iterrows():
            fundamentals = row.to_dict()
            composite_score, individual_scores = self.calculate_fundamental_score(fundamentals)
            
            result = {
                'symbol': fundamentals['symbol'],
                'composite_score': composite_score,
                **individual_scores,
                **{k: v for k, v in fundamentals.items() if k != 'symbol'}
            }
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Sort by composite score (descending)
        results_df = results_df.sort_values('composite_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        logger.info(f"Analysis complete. Top stock: {results_df.iloc[0]['symbol']} (Score: {results_df.iloc[0]['composite_score']:.2f})")
        
        return results_df
    
    def get_top_stocks(self, analysis_results: pd.DataFrame, n: int) -> List[str]:
        """
        Get top N stocks based on fundamental analysis
        """
        top_stocks = analysis_results.head(n)['symbol'].tolist()
        logger.info(f"Selected top {n} stocks: {top_stocks}")
        return top_stocks
    
    def filter_valid_stocks(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out stocks with insufficient fundamental data
        """
        initial_count = len(fundamentals_df)
        
        # Remove stocks with too many missing fundamental metrics
        required_metrics = ['pe_ratio', 'pb_ratio', 'roe', 'profit_margin']
        
        # Count non-null values for required metrics
        fundamentals_df['valid_metrics_count'] = fundamentals_df[required_metrics].notna().sum(axis=1)
        
        # Keep stocks with at least 2 out of 4 required metrics
        filtered_df = fundamentals_df[fundamentals_df['valid_metrics_count'] >= 2].copy()
        
        # Remove stocks with negative or zero P/E or P/B ratios
        filtered_df = filtered_df[
            (filtered_df['pe_ratio'].isna()) | (filtered_df['pe_ratio'] > 0)
        ]
        filtered_df = filtered_df[
            (filtered_df['pb_ratio'].isna()) | (filtered_df['pb_ratio'] > 0)
        ]
        
        final_count = len(filtered_df)
        logger.info(f"Filtered stocks: {initial_count} -> {final_count}")
        
        return filtered_df.drop('valid_metrics_count', axis=1)
    
    def generate_analysis_report(self, analysis_results: pd.DataFrame) -> str:
        """
        Generate a text report of the analysis
        """
        report = []
        report.append("=== FUNDAMENTAL ANALYSIS REPORT ===\n")
        
        top_10 = analysis_results.head(10)
        
        report.append("TOP 10 STOCKS BY FUNDAMENTAL SCORE:")
        report.append("-" * 60)
        report.append(f"{'Rank':<4} {'Symbol':<10} {'Score':<6} {'P/E':<6} {'P/B':<6} {'ROE':<6}")
        report.append("-" * 60)
        
        for _, row in top_10.iterrows():
            pe_str = f"{row['pe_ratio']:.1f}" if pd.notna(row['pe_ratio']) else "N/A"
            pb_str = f"{row['pb_ratio']:.1f}" if pd.notna(row['pb_ratio']) else "N/A"
            roe_str = f"{row['roe']*100:.1f}%" if pd.notna(row['roe']) else "N/A"
            
            report.append(
                f"{row['rank']:<4} {row['symbol']:<10} {row['composite_score']:<6.2f} "
                f"{pe_str:<6} {pb_str:<6} {roe_str:<6}"
            )
        
        report.append("\n" + "=" * 60)
        
        # Summary statistics
        report.append(f"Total stocks analyzed: {len(analysis_results)}")
        report.append(f"Average fundamental score: {analysis_results['composite_score'].mean():.2f}")
        report.append(f"Score range: {analysis_results['composite_score'].min():.2f} - {analysis_results['composite_score'].max():.2f}")
        
        return "\n".join(report)