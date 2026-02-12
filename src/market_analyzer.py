# -*- coding: utf-8 -*-
"""
===================================
Market Review Analysis Module
===================================

Responsibilities:
1. Fetch major market index data (SSE, SZSE, ChiNext)
2. Search market news to form review intelligence
3. Use LLM to generate daily market review reports
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.search_service import SearchService
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """Market index data"""
    code: str                    # Index code
    name: str                    # Index name
    current: float = 0.0         # Current price
    change: float = 0.0          # Price change (points)
    change_pct: float = 0.0      # Change percentage (%)
    open: float = 0.0            # Opening price
    high: float = 0.0            # Highest price
    low: float = 0.0             # Lowest price
    prev_close: float = 0.0      # Previous close
    volume: float = 0.0          # Volume (lots)
    amount: float = 0.0          # Turnover (CNY)
    amplitude: float = 0.0       # Amplitude (%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """Market overview data"""
    date: str                           # Date
    indices: List[MarketIndex] = field(default_factory=list)  # Major indices
    up_count: int = 0                   # Number of gainers
    down_count: int = 0                 # Number of decliners
    flat_count: int = 0                 # Number of unchanged
    limit_up_count: int = 0             # Number of limit-up stocks
    limit_down_count: int = 0           # Number of limit-down stocks
    total_amount: float = 0.0           # Total turnover (billion CNY)
    # north_flow: float = 0.0           # Northbound capital net inflow (billion CNY) - deprecated, API unavailable

    # Sector performance rankings
    top_sectors: List[Dict] = field(default_factory=list)     # Top 5 gaining sectors
    bottom_sectors: List[Dict] = field(default_factory=list)  # Top 5 declining sectors


class MarketAnalyzer:
    """
    Market Review Analyzer

    Features:
    1. Fetch real-time major index quotes
    2. Fetch market advance/decline statistics
    3. Fetch sector performance rankings
    4. Search market news
    5. Generate market review reports
    """
    
    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        """
        Initialize the market analyzer.

        Args:
            search_service: Search service instance
            analyzer: AI analyzer instance (used to call LLM)
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()

    def get_market_overview(self) -> MarketOverview:
        """
        Fetch market overview data.

        Returns:
            MarketOverview: Market overview data object
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)

        # 1. Fetch major index quotes
        overview.indices = self._get_main_indices()

        # 2. Fetch advance/decline statistics
        self._get_market_statistics(overview)

        # 3. Fetch sector performance rankings
        self._get_sector_rankings(overview)

        # 4. Fetch northbound capital (optional)
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """Fetch real-time quotes for major indices."""
        indices = []

        try:
            logger.info("[Market] Fetching real-time major index quotes...")

            # Use DataFetcherManager to fetch index quotes
            # Manager will automatically try: Akshare -> Tushare -> Yfinance
            data_list = self.data_manager.get_main_indices()

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[Market] All data sources failed; will rely on news search for analysis")
            else:
                logger.info(f"[Market] Fetched quotes for {len(indices)} indices")

        except Exception as e:
            logger.error(f"[Market] Failed to fetch index quotes: {e}")

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """Fetch market advance/decline statistics."""
        try:
            logger.info("[Market] Fetching market advance/decline statistics...")

            stats = self.data_manager.get_market_stats()

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(f"[Market] Up:{overview.up_count} Down:{overview.down_count} Flat:{overview.flat_count} "
                          f"LimitUp:{overview.limit_up_count} LimitDown:{overview.limit_down_count} "
                          f"Turnover:{overview.total_amount:.0f}B")

        except Exception as e:
            logger.error(f"[Market] Failed to fetch advance/decline statistics: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        """Fetch sector performance rankings."""
        try:
            logger.info("[Market] Fetching sector performance rankings...")

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(f"[Market] Leading sectors: {[s['name'] for s in overview.top_sectors]}")
                logger.info(f"[Market] Lagging sectors: {[s['name'] for s in overview.bottom_sectors]}")

        except Exception as e:
            logger.error(f"[Market] Failed to fetch sector rankings: {e}")
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """Fetch northbound capital inflow."""
    #     try:
    #         logger.info("[Market] Fetching northbound capital...")
    #
    #         # Fetch northbound capital data
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #
    #         if df is not None and not df.empty:
    #             # Get the latest record
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # Convert to billion CNY
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #
    #             logger.info(f"[Market] Northbound capital net inflow: {overview.north_flow:.2f}B")
    #
    #     except Exception as e:
    #         logger.warning(f"[Market] Failed to fetch northbound capital: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        Search for market news.

        Returns:
            List of news items
        """
        if not self.search_service:
            logger.warning("[Market] Search service not configured; skipping news search")
            return []
        
        all_news = []
        today = datetime.now()
        date_str = today.strftime('%Y年%m月%d日')

        # Multi-dimensional search
        search_queries = [
            "A-share market review",
            "stock market analysis",
            "A-share market hotspot sectors",
        ]
        
        try:
            logger.info("[Market] Starting market news search...")

            for query in search_queries:
                # Use search_stock_news method, passing "大盘" (market) as the stock name
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name="大盘",
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[Market] Search '{query}' returned {len(response.results)} results")

            logger.info(f"[Market] Total market news collected: {len(all_news)}")

        except Exception as e:
            logger.error(f"[Market] Failed to search market news: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        Use LLM to generate a market review report.

        Args:
            overview: Market overview data
            news: Market news list (list of SearchResult objects)

        Returns:
            Market review report text
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[Market] AI analyzer not configured or unavailable; using template report")
            return self._generate_template_review(overview, news)
        
        # Build prompt
        prompt = self._build_review_prompt(overview, news)

        try:
            logger.info("[Market] Calling LLM to generate review report...")
            
            generation_config = {
                'temperature': 0.7,
                'max_output_tokens': 2048,
            }
            
            # Call based on the API type used by the analyzer
            if self.analyzer._use_openai:
                # Use OpenAI-compatible API
                review = self.analyzer._call_openai_api(prompt, generation_config)
            else:
                # Use Gemini API
                response = self.analyzer._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                review = response.text.strip() if response and response.text else None
            
            if review:
                logger.info(f"[Market] Review report generated successfully, length: {len(review)} chars")
                # Inject structured data tables into LLM prose sections
                return self._inject_data_into_review(review, overview)
            else:
                logger.warning("[Market] LLM returned empty response")
                return self._generate_template_review(overview, news)
                
        except Exception as e:
            logger.error(f"[Market] LLM failed to generate review report: {e}")
            return self._generate_template_review(overview, news)
    
    def _inject_data_into_review(self, review: str, overview: MarketOverview) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        import re

        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)

        # Inject market stats after "### 1. Market Summary" section (before next ###)
        if stats_block:
            review = self._insert_after_section(review, r'###\s*1\.\s*Market Summary', stats_block)

        # Inject indices table after "### 2. Index Commentary" section
        if indices_block:
            review = self._insert_after_section(review, r'###\s*2\.\s*Index Commentary', indices_block)

        # Inject sector rankings after "### 4. Sector Analysis" section
        if sector_block:
            review = self._insert_after_section(review, r'###\s*4\.\s*Sector Analysis', sector_block)

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        lines = [
            f"> 📈 Up **{overview.up_count}** / Down **{overview.down_count}** / "
            f"Flat **{overview.flat_count}** | "
            f"Limit Up **{overview.limit_up_count}** / Limit Down **{overview.limit_down_count}** | "
            f"Turnover **{overview.total_amount:.0f}** B"
        ]
        return "\n".join(lines)

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """Build indices table block (without amplitude)."""
        if not overview.indices:
            return ""
        lines = [
            "| Index | Latest | Change% | Turnover(B) |",
            "|-------|--------|---------|-------------|"]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct < 0 else "🟢" if idx.change_pct > 0 else "⚪"
            amount_raw = idx.amount or 0.0
            amount_yi = amount_raw / 1e8 if amount_raw > 1e6 else amount_raw
            lines.append(f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | {amount_yi:.0f} |")
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build sector ranking block."""
        if not overview.top_sectors and not overview.bottom_sectors:
            return ""
        lines = []
        if overview.top_sectors:
            top = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:5]]
            )
            lines.append(f"> 🔥 Leading: {top}")
        if overview.bottom_sectors:
            bot = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:5]]
            )
            lines.append(f"> 💧 Lagging: {bot}")
        return "\n".join(lines)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """Build the review report prompt."""
        # Index quote information (concise format, no emojis)
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # Sector information
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])

        # News information - supports both SearchResult objects and dicts
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # Compatible with both SearchResult objects and dicts
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        prompt = f"""You are a professional A-share/HK/US stock market analyst. Please generate a concise market review report based on the following data.

[IMPORTANT] Output requirements:
- Must output in plain Markdown text format
- Do NOT output JSON format
- Do NOT output code blocks
- Use emojis sparingly, only in headings (max 1 per heading)

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_text if indices_text else "No index data available (API error)"}

## Market Overview
- Up: {overview.up_count} | Down: {overview.down_count} | Flat: {overview.flat_count}
- Limit Up: {overview.limit_up_count} | Limit Down: {overview.limit_down_count}
- Total Turnover: {overview.total_amount:.0f} billion CNY

## Sector Performance
Leading: {top_sectors_text if top_sectors_text else "No data available"}
Lagging: {bottom_sectors_text if bottom_sectors_text else "No data available"}

## Market News
{news_text if news_text else "No relevant news available"}

{"Note: Due to market data retrieval failure, please mainly perform qualitative analysis based on [Market News]. Do not fabricate specific index values." if not indices_text else ""}

---

# Output Format Template (please strictly follow this format)

## 📊 {overview.date} Market Review

### 1. Market Summary
(2-3 sentences summarizing today's overall market performance, including index movements and volume changes)

### 2. Index Commentary
(Analyze the characteristics of SSE, SZSE, ChiNext and other indices)

### 3. Capital Flows
(Interpret the implications of turnover and capital flows)

### 4. Sector Analysis
(Analyze the logic and drivers behind leading and lagging sectors)

### 5. Market Outlook
(Based on current trends and news, provide forecast for the next trading day)

### 6. Risk Warnings
(Key risk factors to watch)

---

Please output the market review report directly, without any additional explanatory text.
"""
        return prompt
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """Generate a review report using a template (fallback when LLM is unavailable)."""

        # Determine market trend
        sh_index = next((idx for idx in overview.indices if idx.code == '000001'), None)
        if sh_index:
            if sh_index.change_pct > 1:
                market_mood = "strong rally"
            elif sh_index.change_pct > 0:
                market_mood = "slight gain"
            elif sh_index.change_pct > -1:
                market_mood = "slight decline"
            else:
                market_mood = "notable decline"
        else:
            market_mood = "consolidation"
        
        # Index quotes (concise format)
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # Sector information
        top_text = ", ".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = ", ".join([s['name'] for s in overview.bottom_sectors[:3]])
        
        report = f"""## 📊 {overview.date} Market Review

### 1. Market Summary
Today's A-share market showed an overall **{market_mood}** trend.

### 2. Major Indices
{indices_text}

### 3. Market Statistics
| Metric | Value |
|--------|-------|
| Up | {overview.up_count} |
| Down | {overview.down_count} |
| Limit Up | {overview.limit_up_count} |
| Limit Down | {overview.limit_down_count} |
| Total Turnover | {overview.total_amount:.0f}B |

### 4. Sector Performance
- **Leading**: {top_text}
- **Lagging**: {bottom_text}

### 5. Risk Warnings
Markets carry risk; invest with caution. The above data is for reference only and does not constitute investment advice.

---
*Review time: {datetime.now().strftime('%H:%M')}*
"""
        return report
    
    def run_daily_review(self) -> str:
        """
        Run the daily market review workflow.

        Returns:
            Review report text
        """
        logger.info("========== Starting Market Review Analysis ==========")
        
        # 1. Fetch market overview
        overview = self.get_market_overview()

        # 2. Search market news
        news = self.search_market_news()

        # 3. Generate review report
        report = self.generate_market_review(overview, news)
        
        logger.info("========== Market Review Analysis Complete ==========")
        
        return report


# Test entry point
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # Test fetching market overview
    overview = analyzer.get_market_overview()
    print(f"\n=== Market Overview ===")
    print(f"Date: {overview.date}")
    print(f"Index count: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"Up: {overview.up_count} | Down: {overview.down_count}")
    print(f"Turnover: {overview.total_amount:.0f}B")

    # Test generating template report
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== Review Report ===")
    print(report)
