#!/usr/bin/env python3
"""
Strategy comparison script for Japanese Stock Fundamental Analysis Trading Bot

Usage:
    python compare_strategies.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--capital AMOUNT] [--strategies STRATEGY1,STRATEGY2,...]

Examples:
    python compare_strategies.py                                    # Compare all strategies
    python compare_strategies.py --strategies value_investing,growth_investing,dividend_focus
    python compare_strategies.py --start-date 2023-01-01 --capital 2000000
"""

import argparse
import logging
import os
import sys
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_comparison import StrategyComparison
from strategy_factory import StrategyFactory
import config

def setup_logging():
    """Setup logging configuration"""
    log_dir = config.LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    return logger

def parse_arguments():
    """Parse command line arguments"""
    factory = StrategyFactory()
    available_strategies = factory.get_available_strategies()
    
    parser = argparse.ArgumentParser(
        description='Japanese Stock Strategy Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Strategies:
{chr(10).join([f"  - {s}" for s in available_strategies])}

Examples:
  %(prog)s                                    Compare all strategies
  %(prog)s --strategies value_investing,growth_investing
  %(prog)s --start-date 2023-01-01 --capital 2000000
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2023-01-01',
        help='Start date for backtesting (YYYY-MM-DD). Default: 2023-01-01'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for backtesting (YYYY-MM-DD). Default: current date'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=1000000,
        help='Initial capital in Japanese Yen. Default: 1,000,000'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        default=None,
        help=f'Comma-separated list of strategies to compare. Default: all strategies. Available: {", ".join(available_strategies)}'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots and visualizations'
    )
    
    return parser.parse_args()

def validate_strategies(strategy_names: list) -> list:
    """Validate strategy names"""
    factory = StrategyFactory()
    available = factory.get_available_strategies()
    
    valid_strategies = []
    for name in strategy_names:
        name = name.strip()
        if name in available:
            valid_strategies.append(name)
        else:
            print(f"Warning: Unknown strategy '{name}'. Available strategies: {', '.join(available)}")
    
    return valid_strategies

def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║        🇯🇵 Japanese Stock Strategy Comparison Tool 📊                          ║
║                                                                               ║
║  🔄 Multiple Strategies  📈 Performance Comparison  🎯 Optimization           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_configuration(args, strategies):
    """Print current configuration"""
    print("\n📋 COMPARISON CONFIGURATION:")
    print("=" * 60)
    print(f"Start Date:           {args.start_date}")
    print(f"End Date:             {args.end_date or 'Current date'}")
    print(f"Initial Capital:      ¥{args.capital:,.0f}")
    print(f"Strategies to Test:   {len(strategies)}")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")
    print("=" * 60)

def main():
    """Main execution function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        logger = setup_logging()
        
        # Print banner and configuration
        print_banner()
        
        # Determine strategies to test
        factory = StrategyFactory()
        if args.strategies:
            strategy_names = args.strategies.split(',')
            strategies = validate_strategies(strategy_names)
        else:
            strategies = factory.get_available_strategies()
        
        if not strategies:
            print("❌ No valid strategies specified. Exiting.")
            sys.exit(1)
        
        print_configuration(args, strategies)
        
        # Validate inputs
        if args.capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        logger.info("Starting Japanese Stock Strategy Comparison")
        logger.info(f"Testing {len(strategies)} strategies")
        logger.info(f"Backtest period: {args.start_date} to {args.end_date or 'present'}")
        logger.info(f"Initial capital: ¥{args.capital:,.0f}")
        
        # Initialize strategy comparison
        print("\n🚀 Initializing strategy comparison...")
        comparison = StrategyComparison(initial_capital=args.capital)
        
        # Run strategy comparison
        print("📈 Running strategy comparison... This may take several minutes...")
        print("   - Testing multiple strategies")
        print("   - Fetching stock data for each strategy")
        print("   - Running backtests")
        print("   - Calculating performance metrics")
        
        results = comparison.run_strategy_comparison(strategies, args.start_date, args.end_date)
        
        if not results:
            print("❌ No strategy results obtained. Please check logs for errors.")
            sys.exit(1)
        
        # Generate comparison report
        print("\n📊 STRATEGY COMPARISON COMPLETED!")
        print("=" * 100)
        
        comparison_report = comparison.generate_comparison_report()
        print(comparison_report)
        
        # Save results
        print("\n💾 Saving comparison results...")
        # Create timestamped results directory
        results_dir = config.get_results_dir_with_timestamp()
        os.makedirs(results_dir, exist_ok=True)
        
        comparison.save_comparison_results(results_dir)
        
        # Save comparison report
        report_filename = "strategy_comparison_report.txt"
        report_path = os.path.join(results_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(comparison_report)
        
        print(f"   ✓ Results saved to: {results_dir}")
        print(f"   ✓ Comparison report: {report_path}")
        
        # Generate visualizations
        if not args.no_plots:
            print("\n📊 Generating comparison visualizations...")
            try:
                comparison.create_comparison_visualizations(results_dir)
                print("   ✓ Visualizations generated successfully")
                print(f"   ✓ Charts saved to: {results_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {str(e)}")
                print(f"   ⚠️  Warning: Could not generate plots - {str(e)}")
        
        # Summary of best strategies
        print(f"\n🎯 SUMMARY:")
        
        # Find best performing strategy
        best_strategy = None
        best_return = float('-inf')
        best_sharpe = float('-inf')
        best_alpha = float('-inf')
        
        for strategy_name, result in results.items():
            metrics = result.get('performance_metrics', {})
            annual_return = metrics.get('annualized_return_pct', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            alpha = metrics.get('alpha_pct', 0)
            
            if annual_return > best_return:
                best_return = annual_return
                best_strategy = strategy_name
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
            
            if alpha > best_alpha:
                best_alpha = alpha
        
        if best_strategy:
            print(f"   Best Overall Strategy: {best_strategy} ({best_return:+.2f}% annual return)")
            print(f"   Best Sharpe Ratio:     {best_sharpe:.2f}")
            print(f"   Best Alpha vs Nikkei:  {best_alpha:+.2f}%")
        
        print(f"\n✅ Strategy comparison completed successfully!")
        print(f"📁 Check the '{results_dir}' folder for detailed results")
        
        logger.info("Japanese Stock Strategy Comparison completed successfully")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        logger.info("Operation cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"\n❌ {error_msg}")
        logger.error(error_msg, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()