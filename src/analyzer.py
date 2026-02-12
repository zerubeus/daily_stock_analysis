# -*- coding: utf-8 -*-
"""
===================================
Stock Watchlist AI Analysis System - AI Analysis Layer
===================================

Responsibilities:
1. Encapsulate Gemini API call logic
2. Leverage Google Search Grounding for realtime news
3. Generate analysis reports combining technical and news data
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from json_repair import repair_json

from src.config import get_config

logger = logging.getLogger(__name__)


# Stock name mapping (common stocks)
STOCK_NAME_MAP = {
    # === A-shares ===
    '600519': 'Kweichow Moutai',
    '000001': 'Ping An Bank',
    '300750': 'CATL',
    '002594': 'BYD',
    '600036': 'China Merchants Bank',
    '601318': 'Ping An Insurance',
    '000858': 'Wuliangye',
    '600276': 'Hengrui Medicine',
    '601012': 'LONGi Green Energy',
    '002475': 'Luxshare Precision',
    '300059': 'East Money',
    '002415': 'Hikvision',
    '600900': 'Yangtze Power',
    '601166': 'Industrial Bank',
    '600028': 'Sinopec',

    # === US stocks ===
    'AAPL': 'Apple',
    'TSLA': 'Tesla',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet A',
    'GOOG': 'Alphabet C',
    'AMZN': 'Amazon',
    'NVDA': 'NVIDIA',
    'META': 'Meta',
    'AMD': 'AMD',
    'INTC': 'Intel',
    'BABA': 'Alibaba',
    'PDD': 'PDD Holdings',
    'JD': 'JD.com',
    'BIDU': 'Baidu',
    'NIO': 'NIO',
    'XPEV': 'XPeng',
    'LI': 'Li Auto',
    'COIN': 'Coinbase',
    'MSTR': 'MicroStrategy',

    # === HK stocks (5-digit codes) ===
    '00700': 'Tencent',
    '03690': 'Meituan',
    '01810': 'Xiaomi',
    '09988': 'Alibaba-SW',
    '09618': 'JD.com-SW',
    '09888': 'Baidu-SW',
    '01024': 'Kuaishou',
    '00981': 'SMIC',
    '02015': 'Li Auto-W',
    '09868': 'XPeng-W',
    '00005': 'HSBC Holdings',
    '01299': 'AIA Group',
    '00941': 'China Mobile',
    '00883': 'CNOOC',
}


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """
    Get stock name from multiple sources.

    Priority:
    1. From the passed context (realtime data)
    2. From the static STOCK_NAME_MAP
    3. From DataFetcherManager (various data sources)
    4. Return default name (Stock+code)

    Args:
        stock_code: Stock code
        context: Analysis context (optional)
        data_manager: DataFetcherManager instance (optional)

    Returns:
        Stock name
    """
    # 1. From context (realtime quote data)
    if context:
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('Stock'):
                return name

        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. From static mapping
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. From data sources
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"Failed to init DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"Failed to get stock name from data source: {e}")

    # 4. Return default name
    return f'Stock {stock_code}'


@dataclass
class AnalysisResult:
    """
    AI Analysis Result - Decision Dashboard version

    Encapsulates Gemini response including decision dashboard and detailed analysis
    """
    code: str
    name: str

    # ========== Core Metrics ==========
    sentiment_score: int  # Overall score 0-100 (>70 Strong Bullish, >60 Bullish, 40-60 Neutral, <40 Bearish)
    trend_prediction: str  # Trend: Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish
    operation_advice: str  # Advice: Buy/Add Position/Hold/Reduce/Sell/Wait
    decision_type: str = "hold"  # Decision type: buy/hold/sell (for statistics)
    confidence_level: str = "Medium"  # Confidence: High/Medium/Low

    # ========== Decision Dashboard ==========
    dashboard: Optional[Dict[str, Any]] = None  # Complete dashboard data

    # ========== Trend Analysis ==========
    trend_analysis: str = ""  # Price trend analysis (support, resistance, trendlines)
    short_term_outlook: str = ""  # Short-term outlook (1-3 days)
    medium_term_outlook: str = ""  # Medium-term outlook (1-2 weeks)

    # ========== Technical Analysis ==========
    technical_analysis: str = ""  # Technical indicators summary
    ma_analysis: str = ""  # MA analysis (bullish/bearish alignment, golden/death cross)
    volume_analysis: str = ""  # Volume analysis (heavy/light volume, institutional activity)
    pattern_analysis: str = ""  # Candlestick pattern analysis

    # ========== Fundamental Analysis ==========
    fundamental_analysis: str = ""  # Fundamental analysis
    sector_position: str = ""  # Sector position and industry trends
    company_highlights: str = ""  # Company highlights / risks

    # ========== Sentiment / News Analysis ==========
    news_summary: str = ""  # Recent news / announcements summary
    market_sentiment: str = ""  # Market sentiment analysis
    hot_topics: str = ""  # Related hot topics

    # ========== Comprehensive Analysis ==========
    analysis_summary: str = ""  # Analysis summary
    key_points: str = ""  # Key insights (3-5 points)
    risk_warning: str = ""  # Risk warnings
    buy_reason: str = ""  # Buy/sell rationale

    # ========== Metadata ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # Market snapshot (for display)
    raw_response: Optional[str] = None  # Raw response (for debugging)
    search_performed: bool = False  # Whether web search was performed
    data_sources: str = ""  # Data source description
    success: bool = True
    error_message: Optional[str] = None

    # ========== Price Data (snapshot at analysis time) ==========
    current_price: Optional[float] = None  # Stock price at analysis time
    change_pct: Optional[float] = None     # Change percentage at analysis time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # Decision dashboard data
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
        }

    def get_core_conclusion(self) -> str:
        """Get core conclusion (one sentence)"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """Get position-specific advice"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """Get sniper price targets"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """Get action checklist"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """Get risk alerts"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """Return emoji based on operation advice"""
        emoji_map = {
            'Buy': '🟢',
            'Add Position': '🟢',
            'Strong Buy': '💚',
            'Hold': '🟡',
            'Wait': '⚪',
            'Reduce': '🟠',
            'Sell': '🔴',
            'Strong Sell': '❌',
        }
        advice = self.operation_advice or ''
        # Direct match first
        if advice in emoji_map:
            return emoji_map[advice]
        # Handle compound advice like "Sell/Wait" — use the first part
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        # Score-based fallback
        score = self.sentiment_score
        if score >= 80:
            return '💚'
        elif score >= 65:
            return '🟢'
        elif score >= 55:
            return '🟡'
        elif score >= 45:
            return '⚪'
        elif score >= 35:
            return '🟠'
        else:
            return '🔴'

    def get_confidence_stars(self) -> str:
        """Return confidence level as stars"""
        star_map = {'High': '⭐⭐⭐', 'Medium': '⭐⭐', 'Low': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI Analyzer

    Responsibilities:
    1. Call Google Gemini API for stock analysis
    2. Generate analysis reports combining pre-searched news and technical data
    3. Parse AI-returned JSON format results

    Usage:
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """

    # ========================================
    # System Prompt - Decision Dashboard v2.0
    # ========================================
    # Output format upgrade: from simple signals to Decision Dashboard
    # Core modules: Core Conclusion + Data Perspective + Intelligence + Battle Plan
    # ========================================

    SYSTEM_PROMPT = """You are an A-share trend trading analyst, responsible for generating professional Decision Dashboard analysis reports.

## Core Trading Philosophy (Must Be Strictly Followed)

### 1. Strict Entry (No Chasing)
- **Never chase highs**: When price deviates from MA5 by more than 5%, absolutely do not buy
- **Deviation formula**: (Current Price - MA5) / MA5 × 100%
- Deviation < 2%: Optimal buy zone
- Deviation 2-5%: Small position entry acceptable
- Deviation > 5%: Strictly forbidden to chase! Classify as "Wait"

### 2. Trend Trading (Follow the Trend)
- **Bullish alignment requirement**: MA5 > MA10 > MA20
- Only trade stocks in bullish alignment; never touch bearish alignment
- Diverging MAs trending up is better than converging MAs
- Trend strength: Check if MA spacing is widening

### 3. Efficiency First (Chip Structure)
- Monitor chip concentration: 90% concentration < 15% indicates concentrated chips
- Profit ratio analysis: Be cautious of profit-taking when 70-90% of chips are profitable
- Average cost vs current price: Current price 5-15% above average cost is healthy

### 4. Buy Point Preference (Pullback to Support)
- **Best buy point**: Low-volume pullback to MA5 with support
- **Second best**: Pullback to MA10 with support
- **Wait**: When price breaks below MA20

### 5. Risk Screening Focus
- Share reduction announcements (shareholders, executives)
- Profit warnings / significant decline
- Regulatory penalties / investigations
- Industry policy headwinds
- Large share unlocks

## Output Format: Decision Dashboard JSON

Please strictly output in the following JSON format, forming a complete Decision Dashboard:

```json
{
    "stock_name": "Stock name",
    "sentiment_score": 0-100 integer,
    "trend_prediction": "Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish",
    "operation_advice": "Buy/Add Position/Hold/Reduce/Sell/Wait",
    "decision_type": "buy/hold/sell",
    "confidence_level": "High/Medium/Low",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "One-sentence core conclusion (under 30 words, directly tell user what to do)",
            "signal_type": "🟢Buy Signal/🟡Hold & Wait/🔴Sell Signal/⚠️Risk Warning",
            "time_sensitivity": "Act Now/Today/This Week/No Rush",
            "position_advice": {
                "no_position": "For those without position: specific action guidance",
                "has_position": "For holders: specific action guidance"
            }
        },

        "data_perspective": {
            "trend_status": {
                "ma_alignment": "MA alignment status description",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": current price value,
                "ma5": MA5 value,
                "ma10": MA10 value,
                "ma20": MA20 value,
                "bias_ma5": deviation percentage value,
                "bias_status": "Safe/Warning/Danger",
                "support_level": support price,
                "resistance_level": resistance price
            },
            "volume_analysis": {
                "volume_ratio": volume ratio value,
                "volume_status": "Heavy/Light/Normal",
                "turnover_rate": turnover rate percentage,
                "volume_meaning": "Volume interpretation (e.g.: low-volume pullback indicates reduced selling pressure)"
            },
            "chip_structure": {
                "profit_ratio": profit ratio,
                "avg_cost": average cost,
                "concentration": chip concentration,
                "chip_health": "Healthy/Normal/Caution"
            }
        },

        "intelligence": {
            "latest_news": "[Latest News] Recent important news summary",
            "risk_alerts": ["Risk 1: specific description", "Risk 2: specific description"],
            "positive_catalysts": ["Catalyst 1: specific description", "Catalyst 2: specific description"],
            "earnings_outlook": "Earnings outlook analysis (based on annual report preview, performance flash, etc.)",
            "sentiment_summary": "One-sentence sentiment summary"
        },

        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "Ideal buy point: XX (near MA5)",
                "secondary_buy": "Secondary buy point: XX (near MA10)",
                "stop_loss": "Stop loss: XX (below MA20 or X%)",
                "take_profit": "Target: XX (previous high / round number)"
            },
            "position_strategy": {
                "suggested_position": "Suggested position: X/10",
                "entry_plan": "Staged entry strategy description",
                "risk_control": "Risk control strategy description"
            },
            "action_checklist": [
                "✅/⚠️/❌ Check 1: Bullish alignment",
                "✅/⚠️/❌ Check 2: Deviation < 5%",
                "✅/⚠️/❌ Check 3: Volume confirmation",
                "✅/⚠️/❌ Check 4: No major negative news",
                "✅/⚠️/❌ Check 5: Healthy chip structure"
            ]
        }
    },

    "analysis_summary": "100-word comprehensive analysis summary",
    "key_points": "3-5 key insights, comma-separated",
    "risk_warning": "Risk warnings",
    "buy_reason": "Trading rationale, referencing trading philosophy",

    "trend_analysis": "Price trend pattern analysis",
    "short_term_outlook": "Short-term outlook (1-3 days)",
    "medium_term_outlook": "Medium-term outlook (1-2 weeks)",
    "technical_analysis": "Technical analysis summary",
    "ma_analysis": "Moving average analysis",
    "volume_analysis": "Volume analysis",
    "pattern_analysis": "Candlestick pattern analysis",
    "fundamental_analysis": "Fundamental analysis",
    "sector_position": "Sector and industry analysis",
    "company_highlights": "Company highlights / risks",
    "news_summary": "News summary",
    "market_sentiment": "Market sentiment",
    "hot_topics": "Related hot topics",

    "search_performed": true/false,
    "data_sources": "Data source description"
}
```

## Scoring Standards

### Strong Buy (80-100):
- ✅ Bullish alignment: MA5 > MA10 > MA20
- ✅ Low deviation: <2%, optimal buy zone
- ✅ Low-volume pullback or breakout on heavy volume
- ✅ Healthy chip concentration
- ✅ Positive news catalysts

### Buy (60-79):
- ✅ Bullish or weak bullish alignment
- ✅ Deviation <5%
- ✅ Normal volume
- ⚪ One minor condition may be unmet

### Wait (40-59):
- ⚠️ Deviation >5% (chasing risk)
- ⚠️ MAs entangled, unclear trend
- ⚠️ Risk events present

### Sell/Reduce (0-39):
- ❌ Bearish alignment
- ❌ Break below MA20
- ❌ Heavy volume decline
- ❌ Major negative news

## Decision Dashboard Core Principles

1. **Core conclusion first**: One sentence clarifying buy/sell/wait
2. **Position-specific advice**: Different advice for holders vs non-holders
3. **Precise price targets**: Must provide specific prices, no vague statements
4. **Visual checklist**: Use ✅⚠️❌ to clearly show each check result
5. **Risk priority**: Risk items from news/sentiment must be prominently highlighted"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI analyzer

        Priority: Gemini > OpenAI Compatible API

        Args:
            api_key: Gemini API Key (optional, defaults from config)
        """
        config = get_config()
        self._api_key = api_key or config.gemini_api_key
        self._model = None
        self._current_model_name = None
        self._using_fallback = False
        self._use_openai = False
        self._openai_client = None

        # Check if Gemini API Key is valid (filter placeholders)
        gemini_key_valid = self._api_key and not self._api_key.startswith('your_') and len(self._api_key) > 10

        # Try Gemini first
        if gemini_key_valid:
            try:
                self._init_model()
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}, trying OpenAI Compatible API")
                self._init_openai_fallback()
        else:
            logger.info("Gemini API Key not configured, trying OpenAI Compatible API")
            self._init_openai_fallback()

        # Neither configured
        if not self._model and not self._openai_client:
            logger.warning("No AI API Key configured, AI analysis will be unavailable")

    def _init_openai_fallback(self) -> None:
        """
        Initialize OpenAI Compatible API as fallback

        Supports all OpenAI-format APIs, including:
        - OpenAI official
        - DeepSeek
        - Qwen
        - Moonshot, etc.
        """
        config = get_config()

        # Check if OpenAI API Key is valid (filter placeholders)
        openai_key_valid = (
            config.openai_api_key and
            not config.openai_api_key.startswith('your_') and
            len(config.openai_api_key) > 10
        )

        if not openai_key_valid:
            logger.debug("OpenAI Compatible API not configured or invalid")
            return

        # Separate import and client creation for better error messages
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai library not installed, run: pip install openai")
            return

        try:
            # base_url is optional, defaults to OpenAI official endpoint
            client_kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url and config.openai_base_url.startswith('http'):
                client_kwargs["base_url"] = config.openai_base_url

            self._openai_client = OpenAI(**client_kwargs)
            self._current_model_name = config.openai_model
            self._use_openai = True
            logger.info(f"OpenAI Compatible API initialized (base_url: {config.openai_base_url}, model: {config.openai_model})")
        except ImportError as e:
            # Missing dependency (e.g., socksio)
            if 'socksio' in str(e).lower() or 'socks' in str(e).lower():
                logger.error(f"OpenAI client needs SOCKS proxy support, run: pip install httpx[socks] or pip install socksio")
            else:
                logger.error(f"OpenAI dependency missing: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'socks' in error_msg or 'socksio' in error_msg or 'proxy' in error_msg:
                logger.error(f"OpenAI proxy config error: {e}, for SOCKS proxy run: pip install httpx[socks]")
            else:
                logger.error(f"OpenAI Compatible API init failed: {e}")

    def _init_model(self) -> None:
        """
        Initialize Gemini model

        Configuration:
        - Uses gemini-3-flash-preview or gemini-2.5-flash model
        - Google Search disabled (uses external Tavily/SerpAPI search)
        """
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)

            config = get_config()
            model_name = config.gemini_model
            fallback_model = config.gemini_model_fallback

            # Google Search Grounding disabled (known compatibility issues)
            # Using external search services (Tavily/SerpAPI) for news

            # Try primary model
            try:
                self._model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = model_name
                self._using_fallback = False
                logger.info(f"Gemini model initialized (model: {model_name})")
            except Exception as model_error:
                # Try fallback model
                logger.warning(f"Primary model {model_name} init failed: {model_error}, trying fallback {fallback_model}")
                self._model = genai.GenerativeModel(
                    model_name=fallback_model,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = fallback_model
                self._using_fallback = True
                logger.info(f"Gemini fallback model initialized (model: {fallback_model})")

        except Exception as e:
            logger.error(f"Gemini model init failed: {e}")
            self._model = None

    def _switch_to_fallback_model(self) -> bool:
        """
        Switch to fallback model

        Returns:
            Whether switch was successful
        """
        try:
            import google.generativeai as genai
            config = get_config()
            fallback_model = config.gemini_model_fallback

            logger.warning(f"[LLM] Switching to fallback model: {fallback_model}")
            self._model = genai.GenerativeModel(
                model_name=fallback_model,
                system_instruction=self.SYSTEM_PROMPT,
            )
            self._current_model_name = fallback_model
            self._using_fallback = True
            logger.info(f"[LLM] Fallback model {fallback_model} initialized")
            return True
        except Exception as e:
            logger.error(f"[LLM] Failed to switch to fallback model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if analyzer is available"""
        return self._model is not None or self._openai_client is not None

    def _call_openai_api(self, prompt: str, generation_config: dict) -> str:
        """
        Call OpenAI Compatible API

        Args:
            prompt: Prompt text
            generation_config: Generation config

        Returns:
            Response text
        """
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay

        def _build_base_request_kwargs() -> dict:
            kwargs = {
                "model": self._current_model_name,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": generation_config.get('temperature', config.openai_temperature),
            }
            return kwargs

        def _is_unsupported_param_error(error_message: str, param_name: str) -> bool:
            lower_msg = error_message.lower()
            return ('400' in lower_msg or "unsupported parameter" in lower_msg or "unsupported param" in lower_msg) and param_name in lower_msg

        if not hasattr(self, "_token_param_mode"):
            self._token_param_mode = {}

        max_output_tokens = generation_config.get('max_output_tokens', 8192)
        model_name = self._current_model_name
        mode = self._token_param_mode.get(model_name, "max_tokens")

        def _kwargs_with_mode(mode_value):
            kwargs = _build_base_request_kwargs()
            if mode_value is not None:
                kwargs[mode_value] = max_output_tokens
            return kwargs

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 60)
                    logger.info(f"[OpenAI] Retry {attempt + 1}, waiting {delay:.1f}s...")
                    time.sleep(delay)

                try:
                    response = self._openai_client.chat.completions.create(**_kwargs_with_mode(mode))
                except Exception as e:
                    error_str = str(e)
                    if mode == "max_tokens" and _is_unsupported_param_error(error_str, "max_tokens"):
                        mode = "max_completion_tokens"
                        self._token_param_mode[model_name] = mode
                        response = self._openai_client.chat.completions.create(**_kwargs_with_mode(mode))
                    elif mode == "max_completion_tokens" and _is_unsupported_param_error(error_str, "max_completion_tokens"):
                        mode = None
                        self._token_param_mode[model_name] = mode
                        response = self._openai_client.chat.completions.create(**_kwargs_with_mode(mode))
                    else:
                        raise

                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    raise ValueError("OpenAI API returned empty response")
                    
            except Exception as e:
                error_str = str(e)
                is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'quota' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[OpenAI] API rate limited, attempt {attempt + 1}/{max_retries}: {error_str[:100]}")
                else:
                    logger.warning(f"[OpenAI] API call failed, attempt {attempt + 1}/{max_retries}: {error_str[:100]}")
                
                if attempt == max_retries - 1:
                    raise
        
        raise Exception("OpenAI API call failed, max retries reached")
    
    def _call_api_with_retry(self, prompt: str, generation_config: dict) -> str:
        """
        Call AI API with retry and model switching

        Priority: Gemini > Gemini fallback > OpenAI Compatible API

        Handles 429 rate limit errors:
        1. Exponential backoff retry
        2. Switch to fallback model after multiple failures
        3. Try OpenAI after Gemini completely fails

        Args:
            prompt: Prompt text
            generation_config: Generation config

        Returns:
            Response text
        """
        # If already in OpenAI mode, call OpenAI directly
        if self._use_openai:
            return self._call_openai_api(prompt, generation_config)
        
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        last_error = None
        tried_fallback = getattr(self, '_using_fallback', False)
        
        for attempt in range(max_retries):
            try:
                # Add delay before request (prevent rapid requests triggering rate limit)
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff: 5, 10, 20, 40...
                    delay = min(delay, 60)  # Max 60 seconds
                    logger.info(f"[Gemini] Retry {attempt + 1}, waiting {delay:.1f}s...")
                    time.sleep(delay)
                
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": 120}
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("Gemini returned empty response")
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check for 429 rate limit error
                is_rate_limit = '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower()

                if is_rate_limit:
                    logger.warning(f"[Gemini] API rate limited (429), attempt {attempt + 1}/{max_retries}: {error_str[:100]}")

                    # If retried half the attempts and haven't switched fallback, try switching
                    if attempt >= max_retries // 2 and not tried_fallback:
                        if self._switch_to_fallback_model():
                            tried_fallback = True
                            logger.info("[Gemini] Switched to fallback model, continuing retries")
                        else:
                            logger.warning("[Gemini] Fallback switch failed, continuing with current model")
                else:
                    # Non-rate-limit error, log and continue retrying
                    logger.warning(f"[Gemini] API call failed, attempt {attempt + 1}/{max_retries}: {error_str[:100]}")

        # All Gemini retries failed, try OpenAI Compatible API
        if self._openai_client:
            logger.warning("[Gemini] All retries failed, switching to OpenAI Compatible API")
            try:
                return self._call_openai_api(prompt, generation_config)
            except Exception as openai_error:
                logger.error(f"[OpenAI] Fallback API also failed: {openai_error}")
                raise last_error or openai_error
        elif config.openai_api_key and config.openai_base_url:
            # Try lazy-loading OpenAI init
            logger.warning("[Gemini] All retries failed, trying to init OpenAI Compatible API")
            self._init_openai_fallback()
            if self._openai_client:
                try:
                    return self._call_openai_api(prompt, generation_config)
                except Exception as openai_error:
                    logger.error(f"[OpenAI] Fallback API also failed: {openai_error}")
                    raise last_error or openai_error

        # All methods failed
        raise last_error or Exception("All AI API calls failed, max retries reached")
    
    def analyze(
        self,
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a single stock

        Flow:
        1. Format input data (technical + news)
        2. Call Gemini API (with retry and model switching)
        3. Parse JSON response
        4. Return structured result

        Args:
            context: Context data from storage.get_analysis_context()
            news_context: Pre-searched news content (optional)

        Returns:
            AnalysisResult object
        """
        code = context.get('code', 'Unknown')
        config = get_config()

        # Pre-request delay (prevent consecutive requests triggering rate limit)
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] Waiting {request_delay:.1f}s before request...")
            time.sleep(request_delay)

        # Prefer stock name from context (passed by main.py)
        name = context.get('stock_name')
        if not name or name.startswith('Stock'):
            # Fallback: from realtime data
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # Last resort: from mapping
                name = STOCK_NAME_MAP.get(code, f'Stock {code}')
        
        # Return default result if model not available
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='Neutral',
                operation_advice='Hold',
                confidence_level='Low',
                analysis_summary='AI analysis not enabled (API Key not configured)',
                risk_warning='Please configure Gemini API Key and retry',
                success=False,
                error_message='Gemini API Key not configured',
            )
        
        try:
            # Format input (technical data + news)
            prompt = self._format_prompt(context, name, news_context)

            # Get model name
            model_name = getattr(self, '_current_model_name', None)
            if not model_name:
                model_name = getattr(self._model, '_model_name', 'unknown')
                if hasattr(self._model, 'model_name'):
                    model_name = self._model.model_name

            logger.info(f"========== AI Analysis {name}({code}) ==========")
            logger.info(f"[LLM Config] Model: {model_name}")
            logger.info(f"[LLM Config] Prompt length: {len(prompt)} chars")
            logger.info(f"[LLM Config] Includes news: {'Yes' if news_context else 'No'}")

            # Log prompt (INFO level: preview, DEBUG level: full)
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt Preview]\n{prompt_preview}")
            logger.debug(f"=== Full Prompt ({len(prompt)} chars) ===\n{prompt}\n=== End Prompt ===")

            # Set generation config (temperature from config file)
            config = get_config()
            generation_config = {
                "temperature": config.gemini_temperature,
                "max_output_tokens": 8192,
            }

            # Log based on actual API in use
            api_provider = "OpenAI" if self._use_openai else "Gemini"
            logger.info(f"[LLM Call] Calling {api_provider} API...")

            # Call API with retry
            start_time = time.time()
            response_text = self._call_api_with_retry(prompt, generation_config)
            elapsed = time.time() - start_time

            # Log response info
            logger.info(f"[LLM Response] {api_provider} API success, {elapsed:.2f}s, {len(response_text)} chars")

            # Log response preview (INFO) and full response (DEBUG)
            response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            logger.info(f"[LLM Response Preview]\n{response_preview}")
            logger.debug(f"=== {api_provider} Full Response ({len(response_text)} chars) ===\n{response_text}\n=== End Response ===")

            # Parse response
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            result.market_snapshot = self._build_market_snapshot(context)

            logger.info(f"[LLM Parse] {name}({code}) analysis complete: {result.trend_prediction}, score {result.sentiment_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI analysis {name}({code}) failed: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='Neutral',
                operation_advice='Hold',
                confidence_level='Low',
                analysis_summary=f'Analysis error: {str(e)[:100]}',
                risk_warning='Analysis failed, please retry later or analyze manually',
                success=False,
                error_message=str(e),
            )
    
    def _format_prompt(
        self,
        context: Dict[str, Any],
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        Format analysis prompt (Decision Dashboard v2.0)

        Includes: technical indicators, realtime quotes (volume ratio/turnover rate),
        chip distribution, trend analysis, news

        Args:
            context: Technical data context (with enhanced data)
            name: Stock name (default, may be overridden by context)
            news_context: Pre-searched news content
        """
        code = context.get('code', 'Unknown')

        # Prefer stock name from context (from realtime_quote)
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'Stock {code}':
            stock_name = STOCK_NAME_MAP.get(code, f'Stock {code}')

        today = context.get('today', {})

        # ========== Build Decision Dashboard formatted input ==========
        prompt = f"""# Decision Dashboard Analysis Request

## 📊 Stock Basic Information
| Item | Data |
|------|------|
| Stock Code | **{code}** |
| Stock Name | **{stock_name}** |
| Analysis Date | {context.get('date', 'Unknown')} |

---

## 📈 Technical Data

### Today's Market
| Indicator | Value |
|-----------|-------|
| Close | {today.get('close', 'N/A')} |
| Open | {today.get('open', 'N/A')} |
| High | {today.get('high', 'N/A')} |
| Low | {today.get('low', 'N/A')} |
| Change% | {today.get('pct_chg', 'N/A')}% |
| Volume | {self._format_volume(today.get('volume'))} |
| Turnover | {self._format_amount(today.get('amount'))} |

### Moving Average System (Key Indicators)
| MA | Value | Description |
|----|-------|-------------|
| MA5 | {today.get('ma5', 'N/A')} | Short-term trend line |
| MA10 | {today.get('ma10', 'N/A')} | Short-mid term trend line |
| MA20 | {today.get('ma20', 'N/A')} | Mid-term trend line |
| MA Pattern | {context.get('ma_status', 'Unknown')} | Bullish/Bearish/Entangled |
"""

        # Add realtime market data (volume ratio, turnover rate, etc.)
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### Enhanced Realtime Data
| Indicator | Value | Interpretation |
|-----------|-------|----------------|
| Current Price | {rt.get('price', 'N/A')} | |
| **Volume Ratio** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **Turnover Rate** | **{rt.get('turnover_rate', 'N/A')}%** | |
| PE Ratio (TTM) | {rt.get('pe_ratio', 'N/A')} | |
| PB Ratio | {rt.get('pb_ratio', 'N/A')} | |
| Market Cap | {self._format_amount(rt.get('total_mv'))} | |
| Float Market Cap | {self._format_amount(rt.get('circ_mv'))} | |
| 60-Day Change | {rt.get('change_60d', 'N/A')}% | Mid-term performance |
"""

        # Add chip distribution data
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
### Chip Distribution (Efficiency Metrics)
| Indicator | Value | Healthy Standard |
|-----------|-------|------------------|
| **Profit Ratio** | **{profit_ratio:.1%}** | Caution at 70-90% |
| Average Cost | {chip.get('avg_cost', 'N/A')} | Price should be 5-15% above |
| 90% Chip Concentration | {chip.get('concentration_90', 0):.2%} | <15% = concentrated |
| 70% Chip Concentration | {chip.get('concentration_70', 0):.2%} | |
| Chip Status | {chip.get('chip_status', 'Unknown')} | |
"""

        # Add trend analysis results (based on trading philosophy)
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 Over 5%, strictly no chasing!" if trend.get('bias_ma5', 0) > 5 else "✅ Safe range"
            prompt += f"""
### Trend Analysis (Based on Trading Philosophy)
| Indicator | Value | Assessment |
|-----------|-------|------------|
| Trend Status | {trend.get('trend_status', 'Unknown')} | |
| MA Alignment | {trend.get('ma_alignment', 'Unknown')} | MA5>MA10>MA20 = Bullish |
| Trend Strength | {trend.get('trend_strength', 0)}/100 | |
| **Bias(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| Bias(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| Volume Status | {trend.get('volume_status', 'Unknown')} | {trend.get('volume_trend', '')} |
| System Signal | {trend.get('buy_signal', 'Unknown')} | |
| System Score | {trend.get('signal_score', 0)}/100 | |

#### System Analysis Reasons
**Buy Reasons**:
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['None'])) if trend.get('signal_reasons') else '- None'}

**Risk Factors**:
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['None'])) if trend.get('risk_factors') else '- None'}
"""

        # Add day-over-day comparison data
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
### Volume & Price Changes
- Volume change vs yesterday: {volume_change}x
- Price change vs yesterday: {context.get('price_change_ratio', 'N/A')}%
"""

        # Add news search results (key section)
        prompt += """
---

## 📰 News & Sentiment Intelligence
"""
        if news_context:
            prompt += f"""
Below are the news search results for **{stock_name}({code})** from the past 7 days. Please focus on extracting:
1. 🚨 **Risk Alerts**: Share reductions, penalties, negative news
2. 🎯 **Positive Catalysts**: Earnings, contracts, policy support
3. 📊 **Earnings Outlook**: Annual report previews, earnings flash reports

```
{news_context}
```
"""
        else:
            prompt += """
No recent news found for this stock. Please analyze primarily based on technical data.
"""

        # Inject missing data warning
        if context.get('data_missing'):
            prompt += """
⚠️ **Missing Data Warning**
Due to API limitations, complete realtime quotes and technical indicator data are unavailable.
Please **ignore N/A values in the tables above** and focus on the **📰 News & Sentiment Intelligence** section for fundamental and sentiment analysis.
When answering technical questions (e.g., moving averages, bias), state "data unavailable" directly. **Never fabricate data**.
"""

        # Explicit output requirements
        prompt += f"""
---

## ✅ Analysis Task

Please generate a **Decision Dashboard** for **{stock_name}({code})**, strictly in JSON format.

### ⚠️ Important: Stock Name Confirmation
If the stock name shown above is "Stock {code}" or incorrect, please **clearly output the correct full name** at the beginning of your analysis.

### Key Questions (Must Answer Clearly):
1. ❓ Does the stock meet MA5>MA10>MA20 bullish alignment?
2. ❓ Is the current bias within safe range (<5%)? — If over 5%, must label "strictly no chasing"
3. ❓ Is volume confirming the trend (low-volume pullback / breakout on heavy volume)?
4. ❓ Is the chip structure healthy?
5. ❓ Any major negative news? (share reductions, penalties, earnings miss, etc.)

### Decision Dashboard Requirements:
- **Stock Name**: Must output the correct full name (e.g., "Kweichow Moutai" not "Stock 600519")
- **Core Conclusion**: One sentence clearly stating buy/sell/wait
- **Position-Specific Advice**: What to do if no position vs holding position
- **Specific Price Targets**: Buy price, stop loss, target price (precise)
- **Checklist**: Mark each item with ✅/⚠️/❌

Please output the complete Decision Dashboard in JSON format."""

        return prompt
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """Format volume display"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f}B shares"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f}K shares"
        else:
            return f"{volume:.0f} shares"

    def _format_amount(self, amount: Optional[float]) -> str:
        """Format turnover amount display"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f}B"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f}M"
        else:
            return f"{amount:.0f}"

    def _format_percent(self, value: Optional[float]) -> str:
        """Format percentage display"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_price(self, value: Optional[float]) -> str:
        """Format price display"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build market snapshot for display"""
        today = context.get('today', {}) or {}
        realtime = context.get('realtime', {}) or {}
        yesterday = context.get('yesterday', {}) or {}

        prev_close = yesterday.get('close')
        close = today.get('close')
        high = today.get('high')
        low = today.get('low')

        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                change_amount = None

        snapshot = {
            "date": context.get('date', 'Unknown'),
            "close": self._format_price(close),
            "open": self._format_price(today.get('open')),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get('pct_chg')),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get('volume')),
            "amount": self._format_amount(today.get('amount')),
        }

        if realtime:
            snapshot.update({
                "price": self._format_price(realtime.get('price')),
                "volume_ratio": realtime.get('volume_ratio', 'N/A'),
                "turnover_rate": self._format_percent(realtime.get('turnover_rate')),
                "source": getattr(realtime.get('source'), 'value', realtime.get('source', 'N/A')),
            })

        return snapshot

    def _parse_response(
        self,
        response_text: str,
        code: str,
        name: str
    ) -> AnalysisResult:
        """
        Parse Gemini response (Decision Dashboard version)

        Attempts to extract JSON analysis result including dashboard field.
        Falls back to smart extraction or default result on failure.
        """
        try:
            # Clean response text: remove markdown code block markers
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # Try to find JSON content
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # Try to fix common JSON issues
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)
                
                # Extract dashboard data
                dashboard = data.get('dashboard', None)

                # Prefer AI-returned stock name (if original name is invalid)
                ai_stock_name = data.get('stock_name')
                if ai_stock_name and (name.startswith('Stock') or name == code or 'Unknown' in name):
                    name = ai_stock_name

                # Parse all fields with defaults to prevent missing values
                # Infer decision_type from operation_advice if missing
                decision_type = data.get('decision_type', '')
                if not decision_type:
                    op = data.get('operation_advice', 'Hold')
                    if op in ['Buy', 'Add Position', 'Strong Buy']:
                        decision_type = 'buy'
                    elif op in ['Sell', 'Reduce', 'Strong Sell']:
                        decision_type = 'sell'
                    else:
                        decision_type = 'hold'
                
                return AnalysisResult(
                    code=code,
                    name=name,
                    # Core metrics
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', 'Neutral'),
                    operation_advice=data.get('operation_advice', 'Hold'),
                    decision_type=decision_type,
                    confidence_level=data.get('confidence_level', 'Medium'),
                    # Decision dashboard
                    dashboard=dashboard,
                    # Trend analysis
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # Technical
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # Fundamental
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # Sentiment / News
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # Comprehensive
                    analysis_summary=data.get('analysis_summary', 'Analysis complete'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # Metadata
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', 'Technical data'),
                    success=True,
                )
            else:
                # No JSON found, try to extract from plain text
                logger.warning(f"Cannot extract JSON from response, using text analysis")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}, trying text extraction")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON format issues"""
        import re

        # Remove comments
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Ensure booleans are lowercase
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        # fix by json-repair
        json_str = repair_json(json_str)
        
        return json_str
    
    def _parse_text_response(
        self,
        response_text: str,
        code: str,
        name: str
    ) -> AnalysisResult:
        """Extract analysis info from plain text response as best as possible"""
        sentiment_score = 50
        trend = 'Neutral'
        advice = 'Hold'

        text_lower = response_text.lower()

        # Simple sentiment detection
        positive_keywords = ['bullish', 'buy', 'strong buy', 'add position', 'breakout', 'uptrend', 'catalyst']
        negative_keywords = ['bearish', 'sell', 'reduce', 'breakdown', 'downtrend', 'risk', 'warning']

        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)

        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = 'Bullish'
            advice = 'Buy'
            decision_type = 'buy'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = 'Bearish'
            advice = 'Sell'
            decision_type = 'sell'
        else:
            decision_type = 'hold'

        summary = response_text[:500] if response_text else 'No analysis result'

        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            decision_type=decision_type,
            confidence_level='Low',
            analysis_summary=summary,
            key_points='JSON parsing failed, for reference only',
            risk_warning='Analysis results may be inaccurate, consider other information',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self,
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        Batch analyze multiple stocks

        Note: Delays between analyses to avoid API rate limits

        Args:
            contexts: List of context data
            delay_between: Delay between analyses (seconds)

        Returns:
            List of AnalysisResult
        """
        results = []

        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"Waiting {delay_between}s before continuing...")
                time.sleep(delay_between)

            result = self.analyze(context)
            results.append(result)

        return results


# Convenience function
def get_analyzer() -> GeminiAnalyzer:
    """Get Gemini analyzer instance"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.DEBUG)

    # Mock context data
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': 'Bullish alignment',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }

    analyzer = GeminiAnalyzer()

    if analyzer.is_available():
        print("=== AI Analysis Test ===")
        result = analyzer.analyze(test_context)
        print(f"Result: {result.to_dict()}")
    else:
        print("Gemini API not configured, skipping test")
