# -*- coding: utf-8 -*-
"""
===================================
Stock Watchlist AI Analysis System - Main Scheduler
===================================

Responsibilities:
1. Coordinate modules to complete stock analysis pipeline
2. Implement low-concurrency thread pool scheduling
3. Global exception handling to ensure single-stock failures do not affect the whole run
4. Provide command-line entry point

Usage:
    python main.py              # Normal run
    python main.py --debug      # Debug mode
    python main.py --dry-run    # Fetch data only, no analysis

Trading Philosophy (integrated into analysis):
- Strict entry: Never chase highs; do not buy when deviation rate > 5%
- Trend trading: Only trade when MA5 > MA10 > MA20 (bullish alignment)
- Efficiency first: Focus on stocks with good chip concentration
- Buy-point preference: Low-volume pullback to MA5/MA10 support
"""
import os
from src.config import setup_env
setup_env()

# Proxy config - controlled via USE_PROXY env var, disabled by default
# GitHub Actions environment automatically skips proxy configuration
if os.getenv("GITHUB_ACTIONS") != "true" and os.getenv("USE_PROXY", "false").lower() == "true":
    # Local dev environment, enable proxy (configure PROXY_HOST and PROXY_PORT in .env)
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "10809")
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from src.config import get_config, Config
from src.logging_config import setup_logging


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Stock Watchlist AI Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                    # Normal run
  python main.py --debug            # Debug mode
  python main.py --dry-run          # Fetch data only, no AI analysis
  python main.py --stocks 600519,000001  # Analyze specific stocks
  python main.py --no-notify        # Do not send notifications
  python main.py --single-notify    # Per-stock push mode (push immediately after each stock)
  python main.py --schedule         # Enable scheduled task mode
  python main.py --market-review    # Run market review only
        '''
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Fetch data only, skip AI analysis'
    )

    parser.add_argument(
        '--stocks',
        type=str,
        help='Stock codes to analyze, comma-separated (overrides config)'
    )

    parser.add_argument(
        '--no-notify',
        action='store_true',
        help='Do not send push notifications'
    )

    parser.add_argument(
        '--single-notify',
        action='store_true',
        help='Per-stock push mode: push immediately after each stock instead of summary'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of concurrent threads (default: from config)'
    )

    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Enable scheduled task mode, run daily'
    )

    parser.add_argument(
        '--market-review',
        action='store_true',
        help='Run market review analysis only'
    )

    parser.add_argument(
        '--no-market-review',
        action='store_true',
        help='Skip market review analysis'
    )

    parser.add_argument(
        '--webui',
        action='store_true',
        help='Launch Web management UI'
    )

    parser.add_argument(
        '--webui-only',
        action='store_true',
        help='Launch Web service only, no auto analysis'
    )

    parser.add_argument(
        '--serve',
        action='store_true',
        help='Start FastAPI backend service (with analysis tasks)'
    )

    parser.add_argument(
        '--serve-only',
        action='store_true',
        help='Start FastAPI backend service only, no auto analysis'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='FastAPI service port (default: 8000)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='FastAPI service listen address (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--no-context-snapshot',
        action='store_true',
        help='Do not save analysis context snapshots'
    )

    # === Backtest ===
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest (evaluate historical analysis results)'
    )

    parser.add_argument(
        '--backtest-code',
        type=str,
        default=None,
        help='Backtest specific stock code only'
    )

    parser.add_argument(
        '--backtest-days',
        type=int,
        default=None,
        help='Backtest evaluation window (trading days, default: from config)'
    )

    parser.add_argument(
        '--backtest-force',
        action='store_true',
        help='Force backtest (recalculate even if results exist)'
    )

    return parser.parse_args()


def run_full_analysis(
    config: Config,
    args: argparse.Namespace,
    stock_codes: Optional[List[str]] = None
):
    """
    Execute the full analysis pipeline (individual stocks + market review).

    This is the main function called by the scheduled task.
    """
    try:
        from src.core.pipeline import StockAnalysisPipeline
        from src.core.market_review import run_market_review

        # CLI arg --single-notify overrides config (#55)
        if getattr(args, 'single_notify', False):
            config.single_stock_notify = True

        # Create pipeline scheduler
        save_context_snapshot = None
        if getattr(args, 'no_context_snapshot', False):
            save_context_snapshot = False
        query_id = uuid.uuid4().hex
        pipeline = StockAnalysisPipeline(
            config=config,
            max_workers=args.workers,
            query_id=query_id,
            query_source="cli",
            save_context_snapshot=save_context_snapshot
        )

        # 1. Run individual stock analysis
        results = pipeline.run(
            stock_codes=stock_codes,
            dry_run=args.dry_run,
            send_notification=not args.no_notify
        )

        # Issue #128: Analysis interval - add delay between stock analysis and market review
        analysis_delay = getattr(config, 'analysis_delay', 0)
        if analysis_delay > 0 and config.market_review_enabled and not args.no_market_review:
            logger.info(f"Waiting {analysis_delay}s before market review (to avoid API rate limiting)...")
            time.sleep(analysis_delay)

        # 2. Run market review (if enabled and not in stock-only mode)
        market_report = ""
        if config.market_review_enabled and not args.no_market_review:
            # Call once and capture the result
            review_result = run_market_review(
                notifier=pipeline.notifier,
                analyzer=pipeline.analyzer,
                search_service=pipeline.search_service,
                send_notification=not args.no_notify
            )
            # If there is a result, assign to market_report for Feishu doc generation
            if review_result:
                market_report = review_result

        # Output summary
        if results:
            logger.info("\n===== Analysis Results Summary =====")
            for r in sorted(results, key=lambda x: x.sentiment_score, reverse=True):
                emoji = r.get_emoji()
                logger.info(
                    f"{emoji} {r.name}({r.code}): {r.operation_advice} | "
                    f"Score {r.sentiment_score} | {r.trend_prediction}"
                )

        logger.info("\nTask execution complete")

        # === Generate Feishu cloud document ===
        try:
            from src.feishu_doc import FeishuDocManager

            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and (results or market_report):
                logger.info("Creating Feishu cloud document...")

                # 1. Prepare title e.g. "2024-01-01 13:01 Market Review"
                tz_cn = timezone(timedelta(hours=8))
                now = datetime.now(tz_cn)
                doc_title = f"{now.strftime('%Y-%m-%d %H:%M')} Market Review"

                # 2. Prepare content (combine stock analysis and market review)
                full_content = ""

                # Add market review content (if available)
                if market_report:
                    full_content += f"# 📈 Market Review\n\n{market_report}\n\n---\n\n"

                # Add stock decision dashboard (generated by NotificationService)
                if results:
                    dashboard_content = pipeline.notifier.generate_dashboard_report(results)
                    full_content += f"# 🚀 Stock Decision Dashboard\n\n{dashboard_content}"

                # 3. Create document
                doc_url = feishu_doc.create_daily_doc(doc_title, full_content)
                if doc_url:
                    logger.info(f"Feishu cloud document created: {doc_url}")
                    # Optional: also push document link to group chat
                    if not args.no_notify:
                        pipeline.notifier.send(f"[{now.strftime('%Y-%m-%d %H:%M')}] Review document created: {doc_url}")

        except Exception as e:
            logger.error(f"Feishu document generation failed: {e}")

        # === Auto backtest ===
        try:
            if getattr(config, 'backtest_enabled', False):
                from src.services.backtest_service import BacktestService

                logger.info("Starting auto backtest...")
                service = BacktestService()
                stats = service.run_backtest(
                    force=False,
                    eval_window_days=getattr(config, 'backtest_eval_window_days', 10),
                    min_age_days=getattr(config, 'backtest_min_age_days', 14),
                    limit=200,
                )
                logger.info(
                    f"Auto backtest complete: processed={stats.get('processed')} saved={stats.get('saved')} "
                    f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
                )
        except Exception as e:
            logger.warning(f"Auto backtest failed (ignored): {e}")

    except Exception as e:
        logger.exception(f"Analysis pipeline execution failed: {e}")


def start_api_server(host: str, port: int, config: Config) -> None:
    """
    Start the FastAPI service in a background thread.

    Args:
        host: Listen address
        port: Listen port
        config: Configuration object
    """
    import threading
    import uvicorn
    
    def run_server():
        level_name = (config.log_level or "INFO").lower()
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            log_level=level_name,
            log_config=None,
        )
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info(f"FastAPI service started: http://{host}:{port}")


def start_bot_stream_clients(config: Config) -> None:
    """Start bot stream clients when enabled in config."""
    # Start DingTalk Stream client
    if config.dingtalk_stream_enabled:
        try:
            from bot.platforms import start_dingtalk_stream_background, DINGTALK_STREAM_AVAILABLE
            if DINGTALK_STREAM_AVAILABLE:
                if start_dingtalk_stream_background():
                    logger.info("[Main] Dingtalk Stream client started in background.")
                else:
                    logger.warning("[Main] Dingtalk Stream client failed to start.")
            else:
                logger.warning("[Main] Dingtalk Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install dingtalk-stream")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Dingtalk Stream client: {exc}")

    # Start Feishu Stream client
    if getattr(config, 'feishu_stream_enabled', False):
        try:
            from bot.platforms import start_feishu_stream_background, FEISHU_SDK_AVAILABLE
            if FEISHU_SDK_AVAILABLE:
                if start_feishu_stream_background():
                    logger.info("[Main] Feishu Stream client started in background.")
                else:
                    logger.warning("[Main] Feishu Stream client failed to start.")
            else:
                logger.warning("[Main] Feishu Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install lark-oapi")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Feishu Stream client: {exc}")


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 means success).
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load config (before setting up logging, to get log directory)
    config = get_config()

    # Set up logging (output to console and file)
    setup_logging(log_prefix="stock_analysis", debug=args.debug, log_dir=config.log_dir)
    
    logger.info("=" * 60)
    logger.info("Stock Watchlist AI Analysis System Started")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Validate config
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)
    
    # Parse stock list
    stock_codes = None
    if args.stocks:
        stock_codes = [code.strip() for code in args.stocks.split(',') if code.strip()]
        logger.info(f"Using command-line stock list: {stock_codes}")
    
    # === Map --webui / --webui-only args to --serve / --serve-only ===
    if args.webui:
        args.serve = True
    if args.webui_only:
        args.serve_only = True

    # Backward compat: legacy WEBUI_ENABLED env var
    if config.webui_enabled and not (args.serve or args.serve_only):
        args.serve = True

    # === Start Web service (if enabled) ===
    start_serve = (args.serve or args.serve_only) and os.getenv("GITHUB_ACTIONS") != "true"

    # Backward compat: legacy WEBUI_HOST/WEBUI_PORT if not overridden via --host/--port
    if start_serve:
        if args.host == '0.0.0.0' and os.getenv('WEBUI_HOST'):
            args.host = os.getenv('WEBUI_HOST')
        if args.port == 8000 and os.getenv('WEBUI_PORT'):
            args.port = int(os.getenv('WEBUI_PORT'))
    
    bot_clients_started = False
    if start_serve:
        try:
            start_api_server(host=args.host, port=args.port, config=config)
            bot_clients_started = True
        except Exception as e:
            logger.error(f"Failed to start FastAPI service: {e}")
    
    if bot_clients_started:
        start_bot_stream_clients(config)
    
    # === Web service only mode: do not auto-run analysis ===
    if args.serve_only:
        logger.info("Mode: Web service only")
        logger.info(f"Web service running: http://{args.host}:{args.port}")
        logger.info("Trigger analysis via /api/v1/analysis/stock/{code}")
        logger.info(f"API docs: http://{args.host}:{args.port}/docs")
        logger.info("Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nUser interrupted, exiting")
        return 0

    try:
        # Mode 0: Backtest
        if getattr(args, 'backtest', False):
            logger.info("Mode: Backtest")
            from src.services.backtest_service import BacktestService

            service = BacktestService()
            stats = service.run_backtest(
                code=getattr(args, 'backtest_code', None),
                force=getattr(args, 'backtest_force', False),
                eval_window_days=getattr(args, 'backtest_days', None),
            )
            logger.info(
                f"Backtest complete: processed={stats.get('processed')} saved={stats.get('saved')} "
                f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
            )
            return 0

        # Mode 1: Market review only
        if args.market_review:
            from src.analyzer import GeminiAnalyzer
            from src.core.market_review import run_market_review
            from src.notification import NotificationService
            from src.search_service import SearchService

            logger.info("Mode: Market review only")
            notifier = NotificationService()
            
            # Initialize search service and analyzer (if configured)
            search_service = None
            analyzer = None
            
            if config.bocha_api_keys or config.tavily_api_keys or config.brave_api_keys or config.serpapi_keys:
                search_service = SearchService(
                    bocha_keys=config.bocha_api_keys,
                    tavily_keys=config.tavily_api_keys,
                    brave_keys=config.brave_api_keys,
                    serpapi_keys=config.serpapi_keys
                )
            
            if config.gemini_api_key or config.openai_api_key:
                analyzer = GeminiAnalyzer(api_key=config.gemini_api_key)
                if not analyzer.is_available():
                    logger.warning("AI analyzer not available after init, please check API Key config")
                    analyzer = None
            else:
                logger.warning("No API Key detected (Gemini/OpenAI), will use template-only report generation")
            
            run_market_review(
                notifier=notifier, 
                analyzer=analyzer, 
                search_service=search_service,
                send_notification=not args.no_notify
            )
            return 0
        
        # Mode 2: Scheduled task mode
        if args.schedule or config.schedule_enabled:
            logger.info("Mode: Scheduled task")
            logger.info(f"Daily execution time: {config.schedule_time}")
            
            from src.scheduler import run_with_schedule
            
            def scheduled_task():
                run_full_analysis(config, args, stock_codes)
            
            run_with_schedule(
                task=scheduled_task,
                schedule_time=config.schedule_time,
                run_immediately=True  # Run once on startup
            )
            return 0
        
        # Mode 3: Normal single run
        run_full_analysis(config, args, stock_codes)
        
        logger.info("\nExecution complete")
        
        # If service is enabled and not in scheduled mode, keep the process running
        keep_running = start_serve and not (args.schedule or config.schedule_enabled)
        if keep_running:
            logger.info("API service running (press Ctrl+C to exit)...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nUser interrupted, exiting")
        return 130

    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
