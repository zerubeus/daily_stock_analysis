# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Jinja2 Report Renderer
===================================

Renders reports from Jinja2 templates. Falls back to caller's logic on template
missing or render error. Template path is relative to project root.
Any expensive data preparation should be injected by the caller via extra_context.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analyzer import AnalysisResult
from src.config import get_config
from src.output_normalizer import (
    normalize_dashboard,
    normalize_operation_advice,
    normalize_trend_prediction,
)

logger = logging.getLogger(__name__)


def _get_signal_level(result: AnalysisResult) -> tuple:
    """Return (signal_text, emoji, color_tag) for a result."""
    advice = result.operation_advice
    score = result.sentiment_score
    advice_map = {
        "Strong Buy": ("Strong Buy", "💚", "strong_buy"),
        "Buy": ("Buy", "🟢", "buy"),
        "Add": ("Buy", "🟢", "buy"),
        "Hold": ("Hold", "🟡", "hold"),
        "Wait": ("Wait", "⚪", "wait"),
        "Reduce": ("Reduce", "🟠", "reduce"),
        "Sell": ("Sell", "🔴", "sell"),
        "Strong Sell": ("Sell", "🔴", "sell"),
        # Chinese fallback for historical data.
        "强烈买入": ("Strong Buy", "💚", "strong_buy"),
        "买入": ("Buy", "🟢", "buy"),
        "加仓": ("Buy", "🟢", "buy"),
        "持有": ("Hold", "🟡", "hold"),
        "观望": ("Wait", "⚪", "wait"),
        "减仓": ("Reduce", "🟠", "reduce"),
        "卖出": ("Sell", "🔴", "sell"),
        "强烈卖出": ("Sell", "🔴", "sell"),
    }
    if advice in advice_map:
        return advice_map[advice]
    if score >= 80:
        return ("Strong Buy", "💚", "strong_buy")
    elif score >= 65:
        return ("Buy", "🟢", "buy")
    elif score >= 55:
        return ("Hold", "🟡", "hold")
    elif score >= 45:
        return ("Wait", "⚪", "wait")
    elif score >= 35:
        return ("Reduce", "🟠", "reduce")
    elif score < 35:
        return ("Sell", "🔴", "sell")
    return ("Wait", "⚪", "wait")


def _display_operation_advice(value: Any) -> str:
    """Normalize operation advice labels to English for rendering."""
    normalized = normalize_operation_advice(value)
    return normalized if isinstance(normalized, str) else (str(value) if value is not None else "")


def _display_trend_prediction(value: Any) -> str:
    """Normalize trend prediction labels to English for rendering."""
    normalized = normalize_trend_prediction(value)
    return normalized if isinstance(normalized, str) else (str(value) if value is not None else "")


def _escape_md(text: str) -> str:
    """Escape markdown special chars (*ST etc)."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_")


def _clean_sniper_value(val: Any) -> str:
    """Format sniper point value for display (strip label prefixes)."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip() if val else ""
    if not s or s == "N/A":
        return s or "N/A"
    prefixes = [
        "理想买入点：", "次优买入点：", "止损位：", "目标位：",
        "理想买入点:", "次优买入点:", "止损位:", "目标位:",
        "Ideal buy point: ", "Ideal buy point:",
        "Secondary buy point: ", "Secondary buy point:",
        "Stop loss: ", "Stop loss:",
        "Target: ", "Target:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _normalize_history_by_code(history_by_code: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize stored history rows for English report rendering."""
    if not isinstance(history_by_code, dict):
        return {}

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for code, entries in history_by_code.items():
        normalized_entries: List[Dict[str, Any]] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            row = dict(entry)
            row["operation_advice"] = _display_operation_advice(row.get("operation_advice"))
            row["trend_prediction"] = _display_trend_prediction(row.get("trend_prediction"))
            normalized_entries.append(row)
        normalized[str(code)] = normalized_entries
    return normalized


def _resolve_templates_dir() -> Path:
    """Resolve template directory relative to project root."""
    config = get_config()
    base = Path(__file__).resolve().parent.parent.parent
    templates_dir = Path(config.report_templates_dir)
    if not templates_dir.is_absolute():
        return base / templates_dir
    return templates_dir


def render(
    platform: str,
    results: List[AnalysisResult],
    report_date: Optional[str] = None,
    summary_only: bool = False,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render report using Jinja2 template.

    Args:
        platform: One of: markdown, wechat, brief
        results: List of AnalysisResult
        report_date: Report date string (default: today)
        summary_only: Whether to output summary only
        extra_context: Additional template context

    Returns:
        Rendered string, or None on error (caller should fallback).
    """
    from datetime import datetime

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        logger.warning("jinja2 not installed, report renderer disabled")
        return None

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    templates_dir = _resolve_templates_dir()
    template_name = f"report_{platform}.j2"
    template_path = templates_dir / template_name
    if not template_path.exists():
        logger.debug("Report template not found: %s", template_path)
        return None

    # Build template context with pre-computed signal levels (sorted by score)
    sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)
    sorted_enriched = []
    for r in sorted_results:
        st, se, _ = _get_signal_level(r)
        rn = r.name if r.name and not r.name.startswith(("股票", "Stock ")) else f"Stock {r.code}"
        sorted_enriched.append({
            "result": r,
            "dashboard": normalize_dashboard(r.dashboard) if r.dashboard else {},
            "signal_text": st,
            "signal_emoji": se,
            "display_advice": _display_operation_advice(r.operation_advice),
            "display_trend": _display_trend_prediction(r.trend_prediction),
            "stock_name": _escape_md(rn),
        })

    buy_count = sum(1 for r in results if getattr(r, "decision_type", "") == "buy")
    sell_count = sum(1 for r in results if getattr(r, "decision_type", "") == "sell")
    hold_count = sum(1 for r in results if getattr(r, "decision_type", "") in ("hold", ""))

    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def failed_checks(checklist: List[str]) -> List[str]:
        return [c for c in (checklist or []) if c.startswith("❌") or c.startswith("⚠️")]

    context: Dict[str, Any] = {
        "report_date": report_date,
        "report_timestamp": report_timestamp,
        "results": sorted_results,
        "enriched": sorted_enriched,  # Sorted by sentiment_score desc
        "summary_only": summary_only,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "escape_md": _escape_md,
        "clean_sniper": _clean_sniper_value,
        "failed_checks": failed_checks,
        "history_by_code": {},
    }
    if extra_context:
        context.update(extra_context)
        context["history_by_code"] = _normalize_history_by_code(context.get("history_by_code"))

    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
        )
        template = env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        logger.warning("Report render failed for %s: %s", template_name, e)
        return None
