# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 LLM 调用逻辑（通过 LiteLLM 统一调用 Gemini/Anthropic/OpenAI 等）
2. 结合技术面和消息面生成分析报告
3. 解析 LLM 响应为结构化 AnalysisResult
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config
from src.output_normalizer import normalize_analysis_payload, should_prefer_ai_stock_name

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',

    # === 美股 ===
    'AAPL': '苹果',
    'TSLA': '特斯拉',
    'MSFT': '微软',
    'GOOGL': '谷歌A',
    'GOOG': '谷歌C',
    'AMZN': '亚马逊',
    'NVDA': '英伟达',
    'META': 'Meta',
    'AMD': 'AMD',
    'INTC': '英特尔',
    'BABA': '阿里巴巴',
    'PDD': '拼多多',
    'JD': '京东',
    'BIDU': '百度',
    'NIO': '蔚来',
    'XPEV': '小鹏汽车',
    'LI': '理想汽车',
    'COIN': 'Coinbase',
    'MSTR': 'MicroStrategy',

    # === 港股 (5位数字) ===
    '00700': '腾讯控股',
    '03690': '美团',
    '01810': '小米集团',
    '09988': '阿里巴巴',
    '09618': '京东集团',
    '09888': '百度集团',
    '01024': '快手',
    '00981': '中芯国际',
    '02015': '理想汽车',
    '09868': '小鹏汽车',
    '00005': '汇丰控股',
    '01299': '友邦保险',
    '00941': '中国移动',
    '00883': '中国海洋石油',
}


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """
    多来源获取股票中文名称

    获取策略（按优先级）：
    1. 从传入的 context 中获取（realtime 数据）
    2. 从静态映射表 STOCK_NAME_MAP 获取
    3. 从 DataFetcherManager 获取（各数据源）
    4. 返回默认名称（股票+代码）

    Args:
        stock_code: 股票代码
        context: 分析上下文（可选）
        data_manager: DataFetcherManager 实例（可选）

    Returns:
        股票中文名称
    """
    # 1. 从上下文获取（实时行情数据）
    if context:
        # 优先从 stock_name 字段获取
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith(('股票', 'Stock ')):
                return name

        # 其次从 realtime 数据获取
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. 从静态映射表获取
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. 从数据源获取
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"无法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                # 更新缓存
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"从数据源获取股票名称失败: {e}")

    # 4. 返回默认名称
    return f'Stock {stock_code}'


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版

    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str

    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    decision_type: str = "hold"  # 决策类型：buy/hold/sell（用于统计）
    confidence_level: str = "Medium"  # 置信度：高/中/低

    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据

    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）

    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析

    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点

    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题

    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由

    # ========== 元数据 ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # 当日行情快照（展示用）
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None

    # ========== 价格数据（分析时快照）==========
    current_price: Optional[float] = None  # 分析时的股价
    change_pct: Optional[float] = None     # 分析时的涨跌幅(%)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
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
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            'Buy': '🟢',
            'Add': '🟢',
            'Strong Buy': '💚',
            'Hold': '🟡',
            'Wait': '⚪',
            'Reduce': '🟠',
            'Sell': '🔴',
            'Strong Sell': '❌',
            # Chinese fallback for historical data
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        advice = self.operation_advice or ''
        # Direct match first
        if advice in emoji_map:
            return emoji_map[advice]
        # Handle compound advice like "卖出/观望" — use the first part
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
        """返回置信度星级"""
        star_map = {'High': '⭐⭐⭐', 'Medium': '⭐⭐', 'Low': '⭐',
                    '高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器

    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果

    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """

    # ========================================
    # 系统提示词 - 决策仪表盘 v2.0
    # ========================================
    # 输出格式升级：从简单信号升级为决策仪表盘
    # 核心模块：核心结论 + 数据透视 + 舆情情报 + 作战计划
    # ========================================

    SYSTEM_PROMPT = """You are a trend-trading focused A-share investment analyst responsible for generating professional Decision Dashboard analysis reports.

## Language Requirement
ALL output text must be in English. Do NOT use Chinese characters in any field values.
Translate Chinese stock names, news summaries, and all analysis text to English.

## Core Trading Philosophy (Must Strictly Follow)

### 1. Strict Entry Strategy (No Chasing)
- **Never chase rallies**: When price deviates from MA5 by more than 5%, absolutely do not buy
- **Bias formula**: (Current Price - MA5) / MA5 × 100%
- Bias < 2%: Ideal buy zone
- Bias 2-5%: Small position entry OK
- Bias > 5%: Do NOT chase! Immediately rate as "Wait"

### 2. Trend Trading (Follow the Trend)
- **Bull alignment required**: MA5 > MA10 > MA20
- Only trade stocks in bull alignment; avoid bear alignment entirely
- Diverging upward MAs preferred over converging MAs
- Trend strength: check if MA spacing is widening

### 3. Efficiency First (Chip Structure)
- Watch chip concentration: 90% concentration < 15% means chips are concentrated
- Profit ratio analysis: 70-90% profitable positions warrant caution for profit-taking
- Average cost vs price: price 5-15% above average cost is healthy

### 4. Entry Point Preference (Pullback to Support)
- **Best entry**: Light volume pullback to MA5 with support
- **Secondary entry**: Pullback to MA10 with support
- **Wait**: When price breaks below MA20

### 5. Risk Screening Focus
- Share reduction announcements (major shareholders, executives)
- Earnings pre-loss / significant decline
- Regulatory penalties / investigations
- Adverse industry policies
- Large lock-up expirations

### 6. Valuation Focus (PE/PB)
- Check if PE ratio is reasonable during analysis
- If PE is significantly elevated (far above industry average or historical mean), flag in risk section
- High-growth stocks may tolerate higher PE, but need earnings support

### 7. Strong Trend Stock Relaxation
- Stocks in strong trends (bull alignment + high trend strength + volume confirmation) may relax bias requirements
- Light position tracking OK for such stocks, but always set stop-loss; never blindly chase

## Output Format: Decision Dashboard JSON

Strictly output in the following JSON format — a complete Decision Dashboard:

```json
{
    "stock_name": "Stock name",
    "sentiment_score": 0-100 integer,
    "trend_prediction": "Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish",
    "operation_advice": "Buy/Add/Hold/Reduce/Sell/Wait",
    "decision_type": "buy/hold/sell",
    "confidence_level": "High/Medium/Low",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "One-sentence core conclusion (concise, tell user what to do)",
            "signal_type": "🟢Buy Signal/🟡Hold & Wait/🔴Sell Signal/⚠️Risk Alert",
            "time_sensitivity": "Act Now/Today/This Week/No Rush",
            "position_advice": {
                "no_position": "Advice for those without positions: specific action",
                "has_position": "Advice for position holders: specific action"
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
                "bias_ma5": bias percentage value,
                "bias_status": "Safe/Warning/Danger",
                "support_level": support price,
                "resistance_level": resistance price
            },
            "volume_analysis": {
                "volume_ratio": volume ratio value,
                "volume_status": "Heavy/Light/Normal",
                "turnover_rate": turnover rate percentage,
                "volume_meaning": "Volume interpretation (e.g.: light pullback = selling pressure easing)"
            },
            "chip_structure": {
                "profit_ratio": profit ratio,
                "avg_cost": average cost,
                "concentration": chip concentration,
                "chip_health": "Healthy/Fair/Caution"
            }
        },

        "intelligence": {
            "latest_news": "[Latest News] Recent important news summary",
            "risk_alerts": ["Risk 1: specific description", "Risk 2: specific description"],
            "positive_catalysts": ["Catalyst 1: specific description", "Catalyst 2: specific description"],
            "earnings_outlook": "Earnings outlook analysis (based on annual report forecasts, earnings flash reports, etc.)",
            "sentiment_summary": "One-sentence sentiment summary"
        },

        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "Ideal buy point: XX (near MA5)",
                "secondary_buy": "Secondary buy point: XX (near MA10)",
                "stop_loss": "Stop loss: XX (below MA20 or X%)",
                "take_profit": "Target: XX (prior high / round number)"
            },
            "position_strategy": {
                "suggested_position": "Suggested position: X/10",
                "entry_plan": "Staged entry strategy description",
                "risk_control": "Risk control strategy description"
            },
            "action_checklist": [
                "✅/⚠️/❌ Check 1: Bull alignment",
                "✅/⚠️/❌ Check 2: Bias within safe range (relaxed for strong trends)",
                "✅/⚠️/❌ Check 3: Volume confirmation",
                "✅/⚠️/❌ Check 4: No major negative news",
                "✅/⚠️/❌ Check 5: Healthy chip structure",
                "✅/⚠️/❌ Check 6: Reasonable PE valuation"
            ]
        }
    },

    "analysis_summary": "100-word comprehensive analysis summary",
    "key_points": "3-5 key takeaways, comma separated",
    "risk_warning": "Risk warning",
    "buy_reason": "Trading rationale, referencing trading philosophy",

    "trend_analysis": "Price trend analysis",
    "short_term_outlook": "Short-term 1-3 day outlook",
    "medium_term_outlook": "Medium-term 1-2 week outlook",
    "technical_analysis": "Technical analysis summary",
    "ma_analysis": "Moving average system analysis",
    "volume_analysis": "Volume analysis",
    "pattern_analysis": "Candlestick pattern analysis",
    "fundamental_analysis": "Fundamental analysis",
    "sector_position": "Sector/industry analysis",
    "company_highlights": "Company highlights/risks",
    "news_summary": "News summary",
    "market_sentiment": "Market sentiment",
    "hot_topics": "Related hot topics",

    "search_performed": true/false,
    "data_sources": "Data source description"
}
```

## Scoring Criteria

### Strong Buy (80-100):
- ✅ Bull alignment: MA5 > MA10 > MA20
- ✅ Low bias: <2%, ideal buy zone
- ✅ Light volume pullback or breakout on volume
- ✅ Concentrated & healthy chip structure
- ✅ Positive news catalysts

### Buy (60-79):
- ✅ Bull alignment or weak bull
- ✅ Bias <5%
- ✅ Normal volume
- ⚪ One minor condition may be unmet

### Wait (40-59):
- ⚠️ Bias >5% (chasing risk)
- ⚠️ MAs intertwined, unclear trend
- ⚠️ Risk events present

### Sell/Reduce (0-39):
- ❌ Bear alignment
- ❌ Broke below MA20
- ❌ Heavy volume decline
- ❌ Major negative news

## Decision Dashboard Core Principles

1. **Core conclusion first**: One sentence — buy, sell, or wait
2. **Position-based advice**: Different advice for no-position vs holding
3. **Precise sniper points**: Must give specific prices, no vague language
4. **Visual checklist**: Use ✅⚠️❌ to clearly mark each check result
5. **Risk priority**: Risk alerts from intelligence must be prominently flagged"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Analyzer via LiteLLM.

        Args:
            api_key: Ignored (kept for backward compatibility). Keys are loaded from config.
        """
        self._router = None
        self._litellm_available = False
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys), AI analysis will be unavailable")

    @staticmethod
    def _get_api_keys_for_model(model: str, config: Config) -> List[str]:
        """Return API keys for a litellm model based on provider prefix."""
        if model.startswith("gemini/") or model.startswith("vertex_ai/"):
            return [k for k in config.gemini_api_keys if k and len(k) >= 8]
        if model.startswith("anthropic/"):
            return [k for k in config.anthropic_api_keys if k and len(k) >= 8]
        return [k for k in config.openai_api_keys if k and len(k) >= 8]

    @staticmethod
    def _extra_litellm_params(model: str, config: Config) -> dict:
        """Build extra litellm params (api_base, headers) for OpenAI-compatible models."""
        params: Dict[str, Any] = {}
        if not model.startswith("gemini/") and not model.startswith("anthropic/") and not model.startswith("vertex_ai/"):
            if config.openai_base_url:
                params["api_base"] = config.openai_base_url
            if config.openai_base_url and "aihubmix.com" in config.openai_base_url:
                params["extra_headers"] = {"APP-Code": "GPIJ3886"}
        return params

    def _init_litellm(self) -> None:
        """Initialize litellm Router (multi-key) or flag single-key availability."""
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model:
            logger.warning("Analyzer LLM: LITELLM_MODEL not configured")
            return

        keys = self._get_api_keys_for_model(litellm_model, config)
        if not keys:
            logger.warning(f"Analyzer LLM: No API keys found for model {litellm_model}")
            return

        self._litellm_available = True

        if len(keys) > 1:
            extra_params = self._extra_litellm_params(litellm_model, config)
            model_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {
                        "model": litellm_model,
                        "api_key": k,
                        **extra_params,
                    },
                }
                for k in keys
            ]
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            models_in_router = list(dict.fromkeys(m["litellm_params"]["model"] for m in model_list))
            logger.info(f"Analyzer LLM: Router initialized with {len(keys)} keys for {litellm_model} (models: {models_in_router})")
        else:
            logger.info(f"Analyzer LLM: litellm initialized (model={litellm_model})")

    def is_available(self) -> bool:
        """Check if LiteLLM is properly configured with at least one API key."""
        return self._router is not None or self._litellm_available

    def _call_litellm(self, prompt: str, generation_config: dict) -> str:
        """Call LLM via litellm with fallback across configured models.

        Args:
            prompt: User prompt text.
            generation_config: Dict with optional keys: temperature, max_output_tokens, max_tokens.

        Returns:
            Response text from LLM.
        """
        config = get_config()
        max_tokens = (
            generation_config.get('max_output_tokens')
            or generation_config.get('max_tokens')
            or 8192
        )
        temperature = generation_config.get('temperature', 0.7)

        models_to_try = [config.litellm_model] + (config.litellm_fallback_models or [])
        models_to_try = [m for m in models_to_try if m]

        last_error = None
        for model in models_to_try:
            keys = self._get_api_keys_for_model(model, config)
            if not keys:
                logger.debug(f"[LiteLLM] Skipping {model}: no API keys")
                continue
            try:
                model_short = model.split("/")[-1] if "/" in model else model
                call_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                extra = get_thinking_extra_body(model_short)
                if extra:
                    call_kwargs["extra_body"] = extra

                if self._router and model == config.litellm_model:
                    response = self._router.completion(**call_kwargs)
                else:
                    call_kwargs["api_key"] = keys[0]
                    call_kwargs.update(self._extra_litellm_params(model, config))
                    response = litellm.completion(**call_kwargs)

                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                raise ValueError("LLM returned empty response")

            except Exception as e:
                logger.warning(f"[LiteLLM] {model} failed: {e}")
                last_error = e
                continue

        raise Exception(f"All LLM models failed (tried {len(models_to_try)} model(s)). Last error: {last_error}")
    
    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析单只股票
        
        流程：
        1. 格式化输入数据（技术面 + 新闻）
        2. 调用 Gemini API（带重试和模型切换）
        3. 解析 JSON 响应
        4. 返回结构化结果
        
        Args:
            context: 从 storage.get_analysis_context() 获取的上下文数据
            news_context: 预先搜索的新闻内容（可选）
            
        Returns:
            AnalysisResult 对象
        """
        code = context.get('code', 'Unknown')
        config = get_config()
        
        # 请求前增加延时（防止连续请求触发限流）
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        # 优先从上下文获取股票名称（由 main.py 传入）
        name = context.get('stock_name')
        if not name or name.startswith(('股票', 'Stock ')):
            # 备选：从 realtime 中获取
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # 最后从映射表获取
                name = STOCK_NAME_MAP.get(code, f'Stock {code}')
        
        # 如果模型不可用，返回默认结果
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='Neutral',
                operation_advice='Hold',
                confidence_level='Low',
                analysis_summary='AI analysis unavailable (LLM API Key not configured)',
                risk_warning='Please configure an LLM API Key (GEMINI_API_KEY/ANTHROPIC_API_KEY/OPENAI_API_KEY) and retry',
                success=False,
                error_message='LLM API Key not configured',
            )
        
        try:
            # 格式化输入（包含技术面数据和新闻）
            prompt = self._format_prompt(context, name, news_context)
            
            config = get_config()
            model_name = config.litellm_model or "unknown"
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")
            logger.info(f"[LLM配置] Prompt 长度: {len(prompt)} 字符")
            logger.info(f"[LLM配置] 是否包含新闻: {'是' if news_context else '否'}")

            # 记录完整 prompt 到日志（INFO级别记录摘要，DEBUG记录完整）
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt 预览]\n{prompt_preview}")
            logger.debug(f"=== 完整 Prompt ({len(prompt)}字符) ===\n{prompt}\n=== End Prompt ===")

            # 设置生成配置
            generation_config = {
                "temperature": config.gemini_temperature,
                "max_output_tokens": 8192,
            }

            logger.info(f"[LLM调用] 开始调用 {model_name}...")

            # 使用 litellm 调用
            start_time = time.time()
            response_text = self._call_litellm(prompt, generation_config)
            elapsed = time.time() - start_time

            # 记录响应信息
            logger.info(f"[LLM返回] {model_name} 响应成功, 耗时 {elapsed:.2f}s, 响应长度 {len(response_text)} 字符")
            
            # 记录响应预览（INFO级别）和完整响应（DEBUG级别）
            response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            logger.info(f"[LLM返回 预览]\n{response_preview}")
            logger.debug(f"=== {model_name} 完整响应 ({len(response_text)}字符) ===\n{response_text}\n=== End Response ===")
            
            # 解析响应
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            result.market_snapshot = self._build_market_snapshot(context)

            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
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

        Includes: technical indicators, realtime quotes (vol ratio/turnover rate), chip distribution, trend analysis, news

        Args:
            context: Technical data context (with enhanced data)
            name: Stock name (default, may be overridden by context)
            news_context: Pre-searched news content
        """
        code = context.get('code', 'Unknown')
        
        # 优先使用上下文中的股票名称（从 realtime_quote 获取）
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name in (f'股票{code}', f'Stock {code}'):
            stock_name = STOCK_NAME_MAP.get(code, f'Stock {code}')
            
        today = context.get('today', {})
        
        # ========== Build Decision Dashboard formatted input ==========
        prompt = f"""# Decision Dashboard Analysis Request

## 📊 Stock Information
| Item | Data |
|------|------|
| Stock Code | **{code}** |
| Stock Name | **{stock_name}** |
| Analysis Date | {context.get('date', 'Unknown')} |

---

## 📈 Technical Data

### Today's Market
| Indicator | Value |
|------|------|
| Close | {today.get('close', 'N/A')} CNY |
| Open | {today.get('open', 'N/A')} CNY |
| High | {today.get('high', 'N/A')} CNY |
| Low | {today.get('low', 'N/A')} CNY |
| Change% | {today.get('pct_chg', 'N/A')}% |
| Volume | {self._format_volume(today.get('volume'))} |
| Turnover | {self._format_amount(today.get('amount'))} |

### Moving Average System (Key Indicators)
| MA | Value | Description |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | Short-term trend |
| MA10 | {today.get('ma10', 'N/A')} | Mid-short trend |
| MA20 | {today.get('ma20', 'N/A')} | Mid-term trend |
| MA Pattern | {context.get('ma_status', 'Unknown')} | Bull/Bear/Mixed |
"""
        
        # Add realtime quote data (vol ratio, turnover rate, etc.)
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### Realtime Enhanced Data
| Indicator | Value | Note |
|------|------|------|
| Current Price | {rt.get('price', 'N/A')} CNY | |
| **Vol Ratio** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **Turnover Rate** | **{rt.get('turnover_rate', 'N/A')}%** | |
| PE Ratio (TTM) | {rt.get('pe_ratio', 'N/A')} | |
| PB Ratio | {rt.get('pb_ratio', 'N/A')} | |
| Total Mkt Cap | {self._format_amount(rt.get('total_mv'))} | |
| Float Mkt Cap | {self._format_amount(rt.get('circ_mv'))} | |
| 60D Change | {rt.get('change_60d', 'N/A')}% | Mid-term perf |
"""
        
        # Add chip distribution data
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
### Chip Distribution (Efficiency Metrics)
| Indicator | Value | Healthy Range |
|------|------|----------|
| **Profit Ratio** | **{profit_ratio:.1%}** | Caution at 70-90% |
| Avg Cost | {chip.get('avg_cost', 'N/A')} CNY | Price should be 5-15% above |
| 90% Chip Conc | {chip.get('concentration_90', 0):.2%} | <15% = concentrated |
| 70% Chip Conc | {chip.get('concentration_70', 0):.2%} | |
| Chip Status | {chip.get('chip_status', 'Unknown')} | |
"""
        
        # Add trend analysis results (based on trading philosophy)
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 Over 5%, do NOT chase!" if trend.get('bias_ma5', 0) > 5 else "✅ Safe range"
            prompt += f"""
### Trend Analysis (Based on Trading Philosophy)
| Indicator | Value | Assessment |
|------|------|------|
| Trend Status | {trend.get('trend_status', 'Unknown')} | |
| MA Alignment | {trend.get('ma_alignment', 'Unknown')} | MA5>MA10>MA20 = Bull |
| Trend Strength | {trend.get('trend_strength', 0)}/100 | |
| **Bias(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| Bias(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| Volume Status | {trend.get('volume_status', 'Unknown')} | {trend.get('volume_trend', '')} |
| System Signal | {trend.get('buy_signal', 'Unknown')} | |
| System Score | {trend.get('signal_score', 0)}/100 | |

#### System Analysis Reasoning
**Buy Reasons**:
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['None'])) if trend.get('signal_reasons') else '- None'}

**Risk Factors**:
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['None'])) if trend.get('risk_factors') else '- None'}
"""
        
        # Add yesterday comparison data
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
### Volume & Price Changes
- Volume vs yesterday: {volume_change}x
- Price vs yesterday: {context.get('price_change_ratio', 'N/A')}%
"""
        
        # Add news search results (key section)
        prompt += """
---

## 📰 Intelligence
"""
        if news_context:
            prompt += f"""
Below are news search results for **{stock_name}({code})** from the past 7 days. Focus on extracting:
1. 🚨 **Risk Alerts**: share reduction, penalties, negative news
2. 🎯 **Positive Catalysts**: earnings, contracts, policy
3. 📊 **Earnings Outlook**: annual forecasts, earnings flash reports

```
{news_context}
```
"""
        else:
            prompt += """
No recent news found for this stock. Please analyze primarily based on technical data.
"""

        # Inject data missing warning
        if context.get('data_missing'):
            prompt += """
⚠️ **Data Missing Warning**
Due to API limitations, complete realtime quotes and technical indicator data are unavailable.
Please **ignore N/A values in the tables above** and focus on the **[📰 Intelligence]** section for fundamental and sentiment analysis.
When addressing technical questions (MAs, bias), state “data missing, cannot determine” -- **never fabricate data**.
"""

        # Explicit output requirements
        prompt += f"""
---

## ✅ Analysis Task

Generate a Decision Dashboard for **{stock_name}({code})**, strictly in JSON format.
"""
        if context.get('is_index_etf'):
            prompt += """
> ⚠️ **Index/ETF Constraint**: This instrument is an index-tracking ETF or market index.
> - Risk analysis focuses only on: **index trend, tracking error, market liquidity**
> - Do NOT include fund company litigation, reputation, or management changes in risk alerts
> - Earnings outlook based on **overall index constituent performance**, not fund company financials
> - `risk_alerts` must not contain fund manager operational risks

"""
        prompt += f"""
### ⚠️ Important: Correct Stock Name Format
Use the format "Stock Name (Code)", for example "Kweichow Moutai (600519)".
If the stock name shown above is "Stock {code}", "Stock{code}", or otherwise incorrect, please output the correct full stock name at the beginning of your analysis.

### Key Focus (Must Answer Clearly):
1. ❓ Does MA5>MA10>MA20 bull alignment hold?
2. ❓ Is bias within safe range (<5%)? — If >5%, must flag "Do NOT chase!"
3. ❓ Is volume confirming (light pullback / breakout volume)?
4. ❓ Is chip structure healthy?
5. ❓ Any major negative news? (share reduction, penalties, earnings miss, etc.)

### Dashboard Requirements:
- **Stock Name**: Must output the correct full name (not "Stock 600519")
- **Core Conclusion**: One sentence — buy, sell, or wait
- **Position Advice**: What to do if no position vs holding
- **Precise Price Points**: Buy price, stop-loss, target (to the cent)
- **Checklist**: Mark each item with ✅/⚠️/❌

Output the complete Decision Dashboard in JSON format."""
        
        return prompt
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """格式化成交量显示"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 100M shares"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 10K shares"
        else:
            return f"{volume:.0f} shares"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """格式化成交额显示"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 100M CNY"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 10K CNY"
        else:
            return f"{amount:.0f} CNY"

    def _format_percent(self, value: Optional[float]) -> str:
        """格式化百分比显示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_price(self, value: Optional[float]) -> str:
        """格式化价格显示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """构建当日行情快照（展示用）"""
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
        解析 Gemini 响应（决策仪表盘版）
        
        尝试从响应中提取 JSON 格式的分析结果，包含 dashboard 字段
        如果解析失败，尝试智能提取或返回默认结果
        """
        try:
            # 清理响应文本：移除 markdown 代码块标记
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # 尝试找到 JSON 内容
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # 尝试修复常见的 JSON 问题
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)
                data = normalize_analysis_payload(data)
                
                # 提取 dashboard 数据
                dashboard = data.get('dashboard', None)

                # 优先使用 AI 返回的股票名称（如果原名称无效或包含代码）
                ai_stock_name = data.get('stock_name')
                if ai_stock_name and should_prefer_ai_stock_name(name, ai_stock_name, code):
                    name = ai_stock_name

                # 解析所有字段，使用默认值防止缺失
                # 解析 decision_type，如果没有则根据 operation_advice 推断
                decision_type = data.get('decision_type', '')
                if not decision_type:
                    op = data.get('operation_advice', 'Hold')
                    if op in ['Buy', 'Add', 'Strong Buy', '买入', '加仓', '强烈买入']:
                        decision_type = 'buy'
                    elif op in ['Sell', 'Reduce', 'Strong Sell', '卖出', '减仓', '强烈卖出']:
                        decision_type = 'sell'
                    else:
                        decision_type = 'hold'
                
                return AnalysisResult(
                    code=code,
                    name=name,
                    # 核心指标
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', 'Neutral'),
                    operation_advice=data.get('operation_advice', 'Hold'),
                    decision_type=decision_type,
                    confidence_level=data.get('confidence_level', 'Medium'),
                    # 决策仪表盘
                    dashboard=dashboard,
                    # 走势分析
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # 技术面
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # 基本面
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # 情绪面/消息面
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # 综合
                    analysis_summary=data.get('analysis_summary', 'Analysis complete'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # 元数据
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', 'Technical data'),
                    success=True,
                )
            else:
                # 没有找到 JSON，尝试从纯文本中提取信息
                logger.warning(f"无法从响应中提取 JSON，使用原始文本分析")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}，尝试从文本提取")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import re
        
        # 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保布尔值是小写
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
        """从纯文本响应中尽可能提取分析信息"""
        # 尝试识别关键词来判断情绪
        sentiment_score = 50
        trend = 'Neutral'
        advice = 'Hold'
        
        text_lower = response_text.lower()
        
        # 简单的情绪识别
        positive_keywords = ['看多', '买入', '上涨', '突破', '强势', '利好', '加仓', 'bullish', 'buy']
        negative_keywords = ['看空', '卖出', '下跌', '跌破', '弱势', '利空', '减仓', 'bearish', 'sell']
        
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
        
        # 截取前500字符作为摘要
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
            key_points='JSON parse failed, for reference only',
            risk_warning='Results may be inaccurate, cross-reference with other data',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        注意：为避免 API 速率限制，每次分析之间会有延迟
        
        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    """获取 LLM 分析器实例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟上下文数据
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
        'ma_status': '多头排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析测试 ===")
        result = analyzer.analyze(test_context)
        print(f"分析结果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳过测试")
