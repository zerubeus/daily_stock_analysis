# -*- coding: utf-8 -*-
"""
Agent Executor — ReAct loop with tool calling.

Orchestrates the LLM + tools interaction loop:
1. Build system prompt (persona + tools + skills)
2. Send to LLM with tool declarations
3. If tool_call → execute tool → feed result back
4. If text → parse as final answer
5. Loop until final answer or max_steps
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from json_repair import repair_json

from src.agent.llm_adapter import LLMToolAdapter
from src.agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Tool name → short label used to build contextual thinking messages
_THINKING_TOOL_LABELS: Dict[str, str] = {
    "get_realtime_quote": "Fetching quotes",
    "get_daily_history": "Fetching K-line data",
    "analyze_trend": "Analyzing technicals",
    "get_chip_distribution": "Analyzing chip distribution",
    "search_stock_news": "Searching news",
    "search_comprehensive_intel": "Searching comprehensive intel",
    "get_market_indices": "Fetching market overview",
    "get_sector_rankings": "Analyzing sector rankings",
    "get_analysis_context": "Loading historical analysis",
    "get_stock_info": "Fetching basic info",
    "analyze_pattern": "Identifying K-line patterns",
    "get_volume_analysis": "Analyzing volume",
    "calculate_ma": "Calculating moving averages",
}


# ============================================================
# Agent result
# ============================================================

@dataclass
class AgentResult:
    """Result from an agent execution run."""
    success: bool = False
    content: str = ""                          # final text answer from agent
    dashboard: Optional[Dict[str, Any]] = None  # parsed dashboard JSON
    tool_calls_log: List[Dict[str, Any]] = field(default_factory=list)  # execution trace
    total_steps: int = 0
    total_tokens: int = 0
    provider: str = ""
    error: Optional[str] = None


# ============================================================
# System prompt builder
# ============================================================

AGENT_SYSTEM_PROMPT = """You are a trend-trading focused A-share investment analysis Agent with data tools and trading strategies, responsible for generating professional Decision Dashboard analysis reports.

## Workflow (Must follow stage order strictly; wait for tool results before proceeding to next stage)

**Stage 1 · Quotes & K-lines** (Execute first)
- `get_realtime_quote` fetch realtime quotes
- `get_daily_history` fetch historical K-lines

**Stage 2 · Technicals & Chips** (After Stage 1 results return)
- `analyze_trend` compute technical indicators
- `get_chip_distribution` get chip distribution

**Stage 3 · Intelligence Search** (After first two stages complete)
- `search_stock_news` search latest news, share reductions, earnings warnings, risk signals

**Stage 4 · Generate Report** (After all data ready, output complete Decision Dashboard JSON)

> ⚠️ Each stage's tool calls must return complete results before proceeding. Do NOT combine tools from different stages in a single call.

## Core Trading Philosophy (Must Strictly Follow)

### 1. Strict Entry (No Chasing)
- **Never chase rallies**: When price deviates from MA5 by more than 5%, absolutely do not buy
- Bias < 2%: Ideal buy zone
- Bias 2-5%: Small position entry OK
- Bias > 5%: Do NOT chase! Immediately rate as "Wait"

### 2. Trend Trading (Follow the Trend)
- **Bull alignment required**: MA5 > MA10 > MA20
- Only trade stocks in bull alignment; avoid bear alignment entirely
- Diverging upward MAs preferred over converging MAs

### 3. Efficiency First (Chip Structure)
- Watch chip concentration: 90% concentration < 15% means chips are concentrated
- Profit ratio analysis: 70-90% profitable positions warrant caution for profit-taking
- Average cost vs price: price 5-15% above average cost is healthy

### 4. Entry Preference (Pullback to Support)
- **Best entry**: Light volume pullback to MA5 with support
- **Secondary entry**: Pullback to MA10 with support
- **Wait**: When price breaks below MA20

### 5. Risk Screening Focus
- Share reduction announcements, earnings pre-loss, regulatory penalties, adverse industry policies, large lock-up expirations

### 6. Valuation Focus (PE/PB)
- If PE is significantly elevated, flag in risk section

### 7. Strong Trend Relaxation
- Stocks in strong trends may relax bias requirements; light position tracking OK but always set stop-loss

## Rules

1. **Must call tools for real data** — Never fabricate numbers; all data must come from tool results.
2. **Systematic analysis** — Strictly follow the staged workflow; complete each stage before proceeding. Do NOT combine tools from different stages in a single call.
3. **Apply trading strategies** — Evaluate each activated strategy's conditions and reflect results in the report.
4. **Output format** — Final response must be a valid Decision Dashboard JSON.
5. **Risk priority** — Must screen for risks (share reductions, earnings warnings, regulatory issues).
6. **Tool failure handling** — Log failure reasons, continue analysis with available data, do not re-call failed tools.

{skills_section}

## Language Requirement
ALL output text must be in English. Do NOT use Chinese characters in any field values.
Translate Chinese stock names, news summaries, and all analysis text to English.

## Output Format: Decision Dashboard JSON

Your final response must be a valid JSON object with the following structure:

```json
{{
    "stock_name": "Stock name",
    "sentiment_score": 0-100 integer,
    "trend_prediction": "Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish",
    "operation_advice": "Buy/Add/Hold/Reduce/Sell/Wait",
    "decision_type": "buy/hold/sell",
    "confidence_level": "High/Medium/Low",
    "dashboard": {{
        "core_conclusion": {{
            "one_sentence": "One-sentence core conclusion",
            "signal_type": "🟢Buy Signal/🟡Hold & Wait/🔴Sell Signal/⚠️Risk Alert",
            "time_sensitivity": "Act Now/Today/This Week/No Rush",
            "position_advice": {{
                "no_position": "Advice for no position",
                "has_position": "Advice for holders"
            }}
        }},
        "data_perspective": {{
            "trend_status": {{"ma_alignment": "", "is_bullish": true, "trend_score": 0}},
            "price_position": {{"current_price": 0, "ma5": 0, "ma10": 0, "ma20": 0, "bias_ma5": 0, "bias_status": "", "support_level": 0, "resistance_level": 0}},
            "volume_analysis": {{"volume_ratio": 0, "volume_status": "", "turnover_rate": 0, "volume_meaning": ""}},
            "chip_structure": {{"profit_ratio": 0, "avg_cost": 0, "concentration": 0, "chip_health": ""}}
        }},
        "intelligence": {{
            "latest_news": "",
            "risk_alerts": [],
            "positive_catalysts": [],
            "earnings_outlook": "",
            "sentiment_summary": ""
        }},
        "battle_plan": {{
            "sniper_points": {{"ideal_buy": "", "secondary_buy": "", "stop_loss": "", "take_profit": ""}},
            "position_strategy": {{"suggested_position": "", "entry_plan": "", "risk_control": ""}},
            "action_checklist": []
        }}
    }},
    "analysis_summary": "Comprehensive analysis summary",
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
    "hot_topics": "Related hot topics"
}}
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
5. **Risk priority**: Risk alerts from intelligence must be prominently flagged
"""

CHAT_SYSTEM_PROMPT = """You are a trend-trading focused A-share investment analysis Agent with data tools and trading strategies, responsible for answering users' stock investment questions.

## Language Requirement
ALL output text must be in English. Do NOT use Chinese characters in any field values.
Translate Chinese stock names, news summaries, and all analysis text to English.

## Analysis Workflow (Must follow stages strictly; no skipping or merging stages)

When a user asks about a stock, you must call tools in the following four stages, waiting for all tool results in each stage before proceeding to the next:

**Stage 1 · Quotes & K-lines** (Execute first)
- Call `get_realtime_quote` to fetch realtime quotes and current price
- Call `get_daily_history` to fetch recent historical K-line data

**Stage 2 · Technicals & Chips** (After Stage 1 results return)
- Call `analyze_trend` to compute MA/MACD/RSI and other technical indicators
- Call `get_chip_distribution` to get chip distribution structure

**Stage 3 · Intelligence Search** (After first two stages complete)
- Call `search_stock_news` to search latest news, share reductions, earnings warnings, risk signals

**Stage 4 · Comprehensive Analysis** (After all tool data ready, generate response)
- Based on the above real data, combine with activated strategies for comprehensive analysis and output investment advice

> ⚠️ Do NOT combine tools from different stages in a single call (e.g., do not request quotes, technical indicators, and news in the first call).

## Core Trading Philosophy (Must Strictly Follow)

### 1. Strict Entry (No Chasing)
- **Never chase rallies**: When price deviates from MA5 by more than 5%, absolutely do not buy
- Bias < 2%: Ideal buy zone
- Bias 2-5%: Small position entry OK
- Bias > 5%: Do NOT chase! Immediately rate as "Wait"

### 2. Trend Trading (Follow the Trend)
- **Bull alignment required**: MA5 > MA10 > MA20
- Only trade stocks in bull alignment; avoid bear alignment entirely
- Diverging upward MAs preferred over converging MAs

### 3. Efficiency First (Chip Structure)
- Watch chip concentration: 90% concentration < 15% means chips are concentrated
- Profit ratio analysis: 70-90% profitable positions warrant caution for profit-taking
- Average cost vs price: price 5-15% above average cost is healthy

### 4. Entry Preference (Pullback to Support)
- **Best entry**: Light volume pullback to MA5 with support
- **Secondary entry**: Pullback to MA10 with support
- **Wait**: When price breaks below MA20

### 5. Risk Screening Focus
- Share reduction announcements, earnings pre-loss, regulatory penalties, adverse industry policies, large lock-up expirations

### 6. Valuation Focus (PE/PB)
- If PE is significantly elevated, flag in risk section

### 7. Strong Trend Relaxation
- Stocks in strong trends may relax bias requirements; light position tracking OK but always set stop-loss

## Rules

1. **Must call tools for real data** — Never fabricate numbers; all data must come from tool results.
2. **Apply trading strategies** — Evaluate each activated strategy's conditions and reflect results in the response.
3. **Free conversation** — Organize your response freely based on the user's question; no JSON output required.
4. **Risk priority** — Must screen for risks (share reductions, earnings warnings, regulatory issues).
5. **Tool failure handling** — Log failure reasons, continue analysis with available data, do not re-call failed tools.

{skills_section}
"""


# ============================================================
# Agent Executor
# ============================================================

class AgentExecutor:
    """ReAct agent loop with tool calling.

    Usage::

        executor = AgentExecutor(tool_registry, llm_adapter)
        result = executor.run("Analyze stock 600519")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_adapter: LLMToolAdapter,
        skill_instructions: str = "",
        max_steps: int = 10,
    ):
        self.tool_registry = tool_registry
        self.llm_adapter = llm_adapter
        self.skill_instructions = skill_instructions
        self.max_steps = max_steps

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the agent loop for a given task.

        Args:
            task: The user task / analysis request.
            context: Optional context dict (e.g., {"stock_code": "600519"}).

        Returns:
            AgentResult with parsed dashboard or error.
        """
        start_time = time.time()
        tool_calls_log: List[Dict[str, Any]] = []
        total_tokens = 0

        # Build system prompt with skills
        skills_section = ""
        if self.skill_instructions:
            skills_section = f"## Activated Trading Strategies\n\n{self.skill_instructions}"
        system_prompt = AGENT_SYSTEM_PROMPT.format(skills_section=skills_section)

        # Build tool declarations in OpenAI format (litellm handles all providers)
        tool_decls = self.tool_registry.to_openai_tools()

        # Initialize conversation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_user_message(task, context)},
        ]

        return self._run_loop(messages, tool_decls, start_time, tool_calls_log, total_tokens, parse_dashboard=True)

    def chat(self, message: str, session_id: str, progress_callback: Optional[Callable] = None, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the agent loop for a free-form chat message.

        Args:
            message: The user's chat message.
            session_id: The conversation session ID.
            progress_callback: Optional callback for streaming progress events.
            context: Optional context dict from previous analysis for data reuse.

        Returns:
            AgentResult with the text response.
        """
        from src.agent.conversation import conversation_manager
        
        start_time = time.time()
        tool_calls_log: List[Dict[str, Any]] = []
        total_tokens = 0

        # Build system prompt with skills
        skills_section = ""
        if self.skill_instructions:
            skills_section = f"## Activated Trading Strategies\n\n{self.skill_instructions}"
        system_prompt = CHAT_SYSTEM_PROMPT.format(skills_section=skills_section)

        # Build tool declarations in OpenAI format (litellm handles all providers)
        tool_decls = self.tool_registry.to_openai_tools()

        # Get conversation history
        session = conversation_manager.get_or_create(session_id)
        history = session.get_history()

        # Initialize conversation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(history)

        # Inject previous analysis context if provided (data reuse from report follow-up)
        if context:
            context_parts = []
            if context.get("stock_code"):
                context_parts.append(f"Stock code: {context['stock_code']}")
            if context.get("stock_name"):
                context_parts.append(f"Stock name: {context['stock_name']}")
            if context.get("previous_price"):
                context_parts.append(f"Previous analysis price: {context['previous_price']}")
            if context.get("previous_change_pct"):
                context_parts.append(f"Previous change: {context['previous_change_pct']}%")
            if context.get("previous_analysis_summary"):
                summary = context["previous_analysis_summary"]
                summary_text = json.dumps(summary, ensure_ascii=False) if isinstance(summary, dict) else str(summary)
                context_parts.append(f"Previous analysis summary:\n{summary_text}")
            if context.get("previous_strategy"):
                strategy = context["previous_strategy"]
                strategy_text = json.dumps(strategy, ensure_ascii=False) if isinstance(strategy, dict) else str(strategy)
                context_parts.append(f"Previous strategy analysis:\n{strategy_text}")
            if context_parts:
                context_msg = "[System-provided historical analysis context for reference]\n" + "\n".join(context_parts)
                messages.append({"role": "user", "content": context_msg})
                messages.append({"role": "assistant", "content": "OK, I've reviewed the stock's historical analysis data. What would you like to know?"})

        messages.append({"role": "user", "content": message})

        # Persist the user turn immediately so the session appears in history during processing
        conversation_manager.add_message(session_id, "user", message)

        result = self._run_loop(messages, tool_decls, start_time, tool_calls_log, total_tokens, parse_dashboard=False, progress_callback=progress_callback)

        # Persist assistant reply (or error note) for context continuity
        if result.success:
            conversation_manager.add_message(session_id, "assistant", result.content)
        else:
            error_note = f"[Analysis failed] {result.error or 'Unknown error'}"
            conversation_manager.add_message(session_id, "assistant", error_note)

        return result

    def _run_loop(self, messages: List[Dict[str, Any]], tool_decls: List[Dict[str, Any]], start_time: float, tool_calls_log: List[Dict[str, Any]], total_tokens: int, parse_dashboard: bool, progress_callback: Optional[Callable] = None) -> AgentResult:
        provider_used = ""

        for step in range(self.max_steps):
            logger.info(f"Agent step {step + 1}/{self.max_steps}")

            if progress_callback:
                if not tool_calls_log:
                    thinking_msg = "Planning analysis path..."
                else:
                    last_tool = tool_calls_log[-1].get("tool", "")
                    label = _THINKING_TOOL_LABELS.get(last_tool, last_tool)
                    thinking_msg = f"'{label}' complete, continuing analysis..."
                progress_callback({"type": "thinking", "step": step + 1, "message": thinking_msg})

            response = self.llm_adapter.call_with_tools(messages, tool_decls)
            provider_used = response.provider
            total_tokens += response.usage.get("total_tokens", 0)

            if response.tool_calls:
                # LLM wants to call tools
                logger.info(f"Agent requesting {len(response.tool_calls)} tool call(s): "
                          f"{[tc.name for tc in response.tool_calls]}")

                # Add assistant message with tool calls to history
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            **({"thought_signature": tc.thought_signature} if tc.thought_signature is not None else {}),
                        }
                        for tc in response.tool_calls
                    ],
                }
                # Only present for DeepSeek thinking mode; None for all other providers
                if response.reasoning_content is not None:
                    assistant_msg["reasoning_content"] = response.reasoning_content
                messages.append(assistant_msg)

                # Execute tool calls — parallel when multiple, sequential when single
                tool_results: List[Dict[str, Any]] = []

                def _exec_single_tool(tc_item):
                    """Execute one tool and return (tc, result_str, success, duration)."""
                    t0 = time.time()
                    try:
                        res = self.tool_registry.execute(tc_item.name, **tc_item.arguments)
                        res_str = self._serialize_tool_result(res)
                        ok = True
                    except Exception as e:
                        res_str = json.dumps({"error": str(e)})
                        ok = False
                        logger.warning(f"Tool '{tc_item.name}' failed: {e}")
                    dur = time.time() - t0
                    return tc_item, res_str, ok, round(dur, 2)

                if len(response.tool_calls) == 1:
                    # Single tool — run inline (no thread overhead)
                    tc = response.tool_calls[0]
                    if progress_callback:
                        progress_callback({"type": "tool_start", "step": step + 1, "tool": tc.name})
                    _, result_str, success, tool_duration = _exec_single_tool(tc)
                    if progress_callback:
                        progress_callback({"type": "tool_done", "step": step + 1, "tool": tc.name, "success": success, "duration": tool_duration})
                    tool_calls_log.append({
                        "step": step + 1, "tool": tc.name, "arguments": tc.arguments,
                        "success": success, "duration": tool_duration, "result_length": len(result_str),
                    })
                    tool_results.append({"tc": tc, "result_str": result_str})
                else:
                    # Multiple tools — run in parallel threads
                    for tc in response.tool_calls:
                        if progress_callback:
                            progress_callback({"type": "tool_start", "step": step + 1, "tool": tc.name})

                    with ThreadPoolExecutor(max_workers=min(len(response.tool_calls), 5)) as pool:
                        futures = {pool.submit(_exec_single_tool, tc): tc for tc in response.tool_calls}
                        for future in as_completed(futures):
                            tc_item, result_str, success, tool_duration = future.result()
                            if progress_callback:
                                progress_callback({"type": "tool_done", "step": step + 1, "tool": tc_item.name, "success": success, "duration": tool_duration})
                            tool_calls_log.append({
                                "step": step + 1, "tool": tc_item.name, "arguments": tc_item.arguments,
                                "success": success, "duration": tool_duration, "result_length": len(result_str),
                            })
                            tool_results.append({"tc": tc_item, "result_str": result_str})

                # Append tool results to messages (ordered by original tool_calls order)
                tc_order = {tc.id: i for i, tc in enumerate(response.tool_calls)}
                tool_results.sort(key=lambda x: tc_order.get(x["tc"].id, 0))
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "name": tr["tc"].name,
                        "tool_call_id": tr["tc"].id,
                        "content": tr["result_str"],
                    })

            else:
                # LLM returned text — this is the final answer
                logger.info(f"Agent completed in {step + 1} steps "
                          f"({time.time() - start_time:.1f}s, {total_tokens} tokens)")
                if progress_callback:
                    progress_callback({"type": "generating", "step": step + 1, "message": "Generating final analysis..."})

                final_content = response.content or ""
                
                if parse_dashboard:
                    dashboard = self._parse_dashboard(final_content)
                    return AgentResult(
                        success=dashboard is not None,
                        content=final_content,
                        dashboard=dashboard,
                        tool_calls_log=tool_calls_log,
                        total_steps=step + 1,
                        total_tokens=total_tokens,
                        provider=provider_used,
                        error=None if dashboard else "Failed to parse dashboard JSON from agent response",
                    )
                else:
                    if response.provider == "error":
                        return AgentResult(
                            success=False,
                            content="",
                            dashboard=None,
                            tool_calls_log=tool_calls_log,
                            total_steps=step + 1,
                            total_tokens=total_tokens,
                            provider=provider_used,
                            error=final_content,
                        )
                    return AgentResult(
                        success=True,
                        content=final_content,
                        dashboard=None,
                        tool_calls_log=tool_calls_log,
                        total_steps=step + 1,
                        total_tokens=total_tokens,
                        provider=provider_used,
                        error=None,
                    )

        # Max steps exceeded
        logger.warning(f"Agent hit max steps ({self.max_steps})")
        return AgentResult(
            success=False,
            content="",
            tool_calls_log=tool_calls_log,
            total_steps=self.max_steps,
            total_tokens=total_tokens,
            provider=provider_used,
            error=f"Agent exceeded max steps ({self.max_steps})",
        )

    def _build_user_message(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the initial user message."""
        parts = [task]
        if context:
            if context.get("stock_code"):
                parts.append(f"\nStock code: {context['stock_code']}")
            if context.get("report_type"):
                parts.append(f"Report type: {context['report_type']}")

            # Inject pre-fetched context data to avoid redundant fetches
            if context.get("realtime_quote"):
                parts.append(f"\n[System-fetched realtime quotes]\n{json.dumps(context['realtime_quote'], ensure_ascii=False)}")
            if context.get("chip_distribution"):
                parts.append(f"\n[System-fetched chip distribution]\n{json.dumps(context['chip_distribution'], ensure_ascii=False)}")

        parts.append("\nPlease use available tools to fetch missing data (e.g., historical K-lines, news), then output analysis results in Decision Dashboard JSON format.")
        return "\n".join(parts)

    def _serialize_tool_result(self, result: Any) -> str:
        """Serialize a tool result to a JSON string for the LLM."""
        if result is None:
            return json.dumps({"result": None})
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(result)
        # Dataclass or object with __dict__
        if hasattr(result, '__dict__'):
            try:
                d = {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
                return json.dumps(d, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(result)
        return str(result)

    def _parse_dashboard(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse the Decision Dashboard JSON from agent response."""
        if not content:
            return None

        # Try to extract JSON from markdown code blocks
        json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_blocks:
            for block in json_blocks:
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    try:
                        repaired = repair_json(block)
                        parsed = json.loads(repaired)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        continue

        # Try raw JSON parse
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try json_repair
        try:
            repaired = repair_json(content)
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try to find JSON object in text
        brace_start = content.find('{')
        brace_end = content.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            candidate = content[brace_start:brace_end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                try:
                    repaired = repair_json(candidate)
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

        logger.warning("Failed to parse dashboard JSON from agent response")
        return None
