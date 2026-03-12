# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Report renderer tests
===================================

Tests for Jinja2 report rendering and fallback behavior.
"""

import sys
import unittest
from unittest.mock import MagicMock

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.analyzer import AnalysisResult
from src.services.report_renderer import render


def _make_result(
    code: str = "600519",
    name: str = "Kweichow Moutai",
    sentiment_score: int = 72,
    operation_advice: str = "持有",
    analysis_summary: str = "Steady setup",
    decision_type: str = "hold",
    dashboard: dict = None,
) -> AnalysisResult:
    if dashboard is None:
        dashboard = {
            "core_conclusion": {"one_sentence": "Hold and wait", "time_sensitivity": "本周内"},
            "intelligence": {"risk_alerts": []},
            "battle_plan": {"sniper_points": {"stop_loss": "110"}},
        }
    return AnalysisResult(
        code=code,
        name=name,
        trend_prediction="看多",
        sentiment_score=sentiment_score,
        operation_advice=operation_advice,
        analysis_summary=analysis_summary,
        decision_type=decision_type,
        dashboard=dashboard,
    )


class TestReportRenderer(unittest.TestCase):
    """Report renderer tests."""

    def test_render_markdown_summary_only(self) -> None:
        """Markdown platform renders with summary_only."""
        r = _make_result()
        out = render("markdown", [r], summary_only=True)
        self.assertIsNotNone(out)
        self.assertIn("Decision Dashboard", out)
        self.assertIn("Kweichow Moutai", out)
        self.assertIn("Hold", out)
        self.assertIn("Bullish", out)

    def test_render_markdown_full(self) -> None:
        """Markdown platform renders full report."""
        r = _make_result()
        out = render("markdown", [r], summary_only=False)
        self.assertIsNotNone(out)
        self.assertIn("Core Conclusion", out)
        self.assertIn("Battle Plan", out)
        self.assertIn("This Week", out)

    def test_render_wechat(self) -> None:
        """Wechat platform renders."""
        r = _make_result()
        out = render("wechat", [r])
        self.assertIsNotNone(out)
        self.assertIn("Kweichow Moutai", out)
        self.assertIn("Hold", out)

    def test_render_brief(self) -> None:
        """Brief platform renders 3-5 sentence summary."""
        r = _make_result()
        out = render("brief", [r])
        self.assertIsNotNone(out)
        self.assertIn("Decision Brief", out)
        self.assertIn("Kweichow Moutai", out)

    def test_render_unknown_platform_returns_none(self) -> None:
        """Unknown platform returns None (caller fallback)."""
        r = _make_result()
        out = render("unknown_platform", [r])
        self.assertIsNone(out)

    def test_render_empty_results_returns_content(self) -> None:
        """Empty results still produces header."""
        out = render("markdown", [], summary_only=True)
        self.assertIsNotNone(out)
        self.assertIn("0", out)

    def test_render_normalizes_history_rows(self) -> None:
        """History comparison rows render English labels for stored Chinese values."""
        r = _make_result()
        out = render(
            "markdown",
            [r],
            summary_only=False,
            extra_context={
                "history_by_code": {
                    "600519": [
                        {
                            "created_at": "2026-03-11 20:00:00",
                            "sentiment_score": 65,
                            "operation_advice": "持有",
                            "trend_prediction": "看多",
                        }
                    ]
                }
            },
        )
        self.assertIsNotNone(out)
        self.assertIn("Historical Signal Comparison", out)
        self.assertIn("| Hold | Bullish |", out)
