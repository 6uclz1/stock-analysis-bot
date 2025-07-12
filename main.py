#!/usr/bin/env python3
"""
Main execution script for Japanese Stock Fundamental Analysis Trading Bot

Usage:
    python main.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--capital AMOUNT]

Examples:
    python main.py                                    # Run with default settings
    python main.py --start-date 2020-01-01           # Start from specific date
    python main.py --capital 5000000                 # Use 5 million yen initial capital
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

from backtester import Backtester
from visualizer import Visualizer
import config

def setup_logging():
    """Setup logging configuration"""
    log_dir = config.LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    parser = argparse.ArgumentParser(
        description='Japanese Stock Fundamental Analysis Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    Run with default settings
  %(prog)s --start-date 2020-01-01           Start from specific date  
  %(prog)s --capital 5000000                 Use 5 million yen initial capital
  %(prog)s --start-date 2020-01-01 --end-date 2023-12-31 --capital 2000000
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=config.DATA_START_DATE,
        help=f'Start date for backtesting (YYYY-MM-DD). Default: {config.DATA_START_DATE}'
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
        '--no-plots',
        action='store_true',
        help='Skip generating plots and visualizations'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='Save results to CSV files'
    )
    
    return parser.parse_args()

def validate_dates(start_date: str, end_date: str = None):
    """Validate date inputs"""
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if end_dt <= start_dt:
                raise ValueError("End date must be after start date")
            if end_dt > datetime.now():
                raise ValueError("End date cannot be in the future")
        
        if start_dt > datetime.now():
            raise ValueError("Start date cannot be in the future")
            
        # Check if start date is too far in the past (data availability)
        min_date = datetime(2010, 1, 1)
        if start_dt < min_date:
            raise ValueError(f"Start date cannot be before {min_date.strftime('%Y-%m-%d')}")
            
    except ValueError as e:
        if "time data" in str(e):
            raise ValueError("Date must be in YYYY-MM-DD format")
        else:
            raise e

def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           🇯🇵 Japanese Stock Fundamental Analysis Trading Bot 🤖              ║
║                                                                               ║
║  📊 Fundamental Analysis  💰 Portfolio Management  📈 Backtesting             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_configuration(args):
    """Print current configuration"""
    print("\n📋 CONFIGURATION:")
    print("=" * 50)
    print(f"Start Date:           {args.start_date}")
    print(f"End Date:             {args.end_date or 'Current date'}")
    print(f"Initial Capital:      ¥{args.capital:,.0f}")
    print(f"Portfolio Size:       {config.PORTFOLIO_SIZE} stocks")
    print(f"Rebalance Frequency:  {config.REBALANCE_FREQUENCY} days")
    print(f"Stop Loss:            {config.STOP_LOSS_THRESHOLD:.1%}")
    print(f"Take Profit:          {config.TAKE_PROFIT_THRESHOLD:.1%}")
    print(f"Stock Universe:       {len(config.JAPANESE_STOCKS)} Japanese stocks")
    print("=" * 50)

def main():
    """Main execution function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        logger = setup_logging()
        
        # Print banner and configuration
        print_banner()
        print_configuration(args)
        
        # Validate inputs
        validate_dates(args.start_date, args.end_date)
        
        if args.capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        logger.info("Starting Japanese Stock Fundamental Analysis Trading Bot")
        logger.info(f"Backtest period: {args.start_date} to {args.end_date or 'present'}")
        logger.info(f"Initial capital: ¥{args.capital:,.0f}")
        
        # Initialize backtester
        print("\n🚀 Initializing backtesting engine...")
        backtester = Backtester(initial_capital=args.capital)
        
        # Run backtest
        print("📈 Running backtest... This may take a few minutes...")
        print("   - Fetching stock data")
        print("   - Analyzing fundamentals") 
        print("   - Simulating trades")
        print("   - Calculating performance metrics")
        
        backtest_results = backtester.run_backtest(args.start_date, args.end_date)
        
        # Generate performance report
        print("\n📊 BACKTEST COMPLETED!")
        print("=" * 80)
        
        performance_report = backtester.generate_performance_report(backtest_results)
        print(performance_report)
        
        # Save results if requested
        if args.save_results:
            print("\n💾 Saving results...")
            # Create timestamped results directory
            results_dir = config.get_results_dir_with_timestamp()
            os.makedirs(results_dir, exist_ok=True)
            
            visualizer = Visualizer()
            visualizer.save_results_to_csv(backtest_results, results_dir)
            
            # Save performance report
            report_filename = "performance_report.txt"
            report_path = os.path.join(results_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(performance_report)
            
            print(f"   ✓ Results saved to: {results_dir}")
            print(f"   ✓ Performance report: {report_path}")
        
        # Generate visualizations
        if not args.no_plots:
            print("\n📊 Generating visualizations...")
            try:
                # Use the same timestamped directory for visualizations
                if args.save_results:
                    viz_results_dir = results_dir
                else:
                    viz_results_dir = config.get_results_dir_with_timestamp()
                    os.makedirs(viz_results_dir, exist_ok=True)
                
                visualizer = Visualizer()
                visualizer.create_comprehensive_report(backtest_results, viz_results_dir)
                print("   ✓ Visualizations generated successfully")
                print(f"   ✓ Charts saved to: {viz_results_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {str(e)}")
                print(f"   ⚠️  Warning: Could not generate plots - {str(e)}")
        
        # Summary
        performance_metrics = backtest_results.get('performance_metrics', {})
        if performance_metrics:
            print(f"\n🎯 SUMMARY:")
            print(f"   Total Return:     {performance_metrics.get('total_return_pct', 0):+.2f}%")
            print(f"   Annual Return:    {performance_metrics.get('annualized_return_pct', 0):+.2f}%")
            print(f"   Sharpe Ratio:     {performance_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown:     {performance_metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"   Alpha vs Nikkei:  {performance_metrics.get('alpha_pct', 0):+.2f}%")
        
        print(f"\n✅ Analysis completed successfully!")
        if args.save_results or not args.no_plots:
            final_results_dir = results_dir if args.save_results else viz_results_dir
            print(f"📁 Check the '{final_results_dir}' folder for detailed results")
        else:
            print(f"📁 Results saved to timestamped directories in '{config.RESULTS_DIR}'")
        
        logger.info("Japanese Stock Fundamental Analysis Trading Bot completed successfully")
        
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