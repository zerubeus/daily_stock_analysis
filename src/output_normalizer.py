# -*- coding: utf-8 -*-
"""Normalize analysis output values to English for report rendering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


_OPERATION_ADVICE_MAP = {
    "强烈买入": "Strong Buy",
    "买入": "Buy",
    "加仓": "Add",
    "持有": "Hold",
    "观望": "Wait",
    "减仓": "Reduce",
    "卖出": "Sell",
    "强烈卖出": "Strong Sell",
}

_TREND_PREDICTION_MAP = {
    "强烈看多": "Strong Bullish",
    "看多": "Bullish",
    "震荡": "Neutral",
    "看空": "Bearish",
    "强烈看空": "Strong Bearish",
}

_CONFIDENCE_LEVEL_MAP = {
    "高": "High",
    "中": "Medium",
    "低": "Low",
}

_SIGNAL_TYPE_MAP = {
    "🟢买入信号": "🟢Buy Signal",
    "🟡持有观望": "🟡Hold & Wait",
    "🔴卖出信号": "🔴Sell Signal",
    "⚠️风险警告": "⚠️Risk Alert",
}

_TIME_SENSITIVITY_MAP = {
    "立即行动": "Act Now",
    "今日内": "Today",
    "本周内": "This Week",
    "不急": "No Rush",
}

_BIAS_STATUS_MAP = {
    "安全": "Safe",
    "警戒": "Warning",
    "危险": "Danger",
}

_VOLUME_STATUS_MAP = {
    "放量": "Heavy",
    "缩量": "Light",
    "平量": "Normal",
}

_CHIP_HEALTH_MAP = {
    "健康": "Healthy",
    "一般": "Fair",
    "警惕": "Caution",
}


def contains_cjk(text: Optional[str]) -> bool:
    """Return True when a string contains CJK ideographs."""
    if not text:
        return False
    return any("\u4e00" <= char <= "\u9fff" for char in str(text))


def normalize_operation_advice(value: Any) -> Any:
    """Normalize operation advice labels to English."""
    return _OPERATION_ADVICE_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def normalize_trend_prediction(value: Any) -> Any:
    """Normalize trend prediction labels to English."""
    return _TREND_PREDICTION_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def normalize_confidence_level(value: Any) -> Any:
    """Normalize confidence labels to English."""
    return _CONFIDENCE_LEVEL_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def normalize_signal_type(value: Any) -> Any:
    """Normalize dashboard signal type labels to English."""
    return _SIGNAL_TYPE_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def normalize_time_sensitivity(value: Any) -> Any:
    """Normalize dashboard time-sensitivity labels to English."""
    return _TIME_SENSITIVITY_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def normalize_bias_status(value: Any) -> Any:
    """Normalize bias-status labels to English."""
    return _BIAS_STATUS_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def normalize_volume_status(value: Any) -> Any:
    """Normalize volume-status labels to English."""
    return _VOLUME_STATUS_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def normalize_chip_health(value: Any) -> Any:
    """Normalize chip-health labels to English."""
    return _CHIP_HEALTH_MAP.get(str(value).strip(), value) if isinstance(value, str) else value


def should_prefer_ai_stock_name(current_name: str, ai_stock_name: str, code: str) -> bool:
    """Prefer AI stock name when the existing one is placeholder-like or Chinese."""
    candidate = (ai_stock_name or "").strip()
    current = (current_name or "").strip()
    if not candidate:
        return False
    if not current or current == code or current.startswith(("股票", "Stock ")) or "Unknown" in current:
        return True
    return contains_cjk(current) and not contains_cjk(candidate)


def normalize_dashboard(dashboard: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize known enum-like dashboard fields to English."""
    if not isinstance(dashboard, dict):
        return dashboard

    normalized = deepcopy(dashboard)
    core = normalized.get("core_conclusion")
    if isinstance(core, dict):
        core["signal_type"] = normalize_signal_type(core.get("signal_type"))
        core["time_sensitivity"] = normalize_time_sensitivity(core.get("time_sensitivity"))

    data_perspective = normalized.get("data_perspective")
    if isinstance(data_perspective, dict):
        price_position = data_perspective.get("price_position")
        if isinstance(price_position, dict):
            price_position["bias_status"] = normalize_bias_status(price_position.get("bias_status"))

        volume_analysis = data_perspective.get("volume_analysis")
        if isinstance(volume_analysis, dict):
            volume_analysis["volume_status"] = normalize_volume_status(volume_analysis.get("volume_status"))

        chip_structure = data_perspective.get("chip_structure")
        if isinstance(chip_structure, dict):
            chip_structure["chip_health"] = normalize_chip_health(chip_structure.get("chip_health"))

    return normalized


def normalize_analysis_payload(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize top-level and nested dashboard enum-like fields to English."""
    if not isinstance(data, dict):
        return data

    normalized = deepcopy(data)
    normalized["trend_prediction"] = normalize_trend_prediction(normalized.get("trend_prediction"))
    normalized["operation_advice"] = normalize_operation_advice(normalized.get("operation_advice"))
    normalized["confidence_level"] = normalize_confidence_level(normalized.get("confidence_level"))
    normalized["dashboard"] = normalize_dashboard(normalized.get("dashboard"))
    return normalized
