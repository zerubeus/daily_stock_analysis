# -*- coding: utf-8 -*-
"""Market strategy blueprints for CN/US daily market recap."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StrategyDimension:
    """Single strategy dimension used by market recap prompts."""

    name: str
    objective: str
    checkpoints: List[str]


@dataclass(frozen=True)
class MarketStrategyBlueprint:
    """Region specific market strategy blueprint."""

    region: str
    title: str
    positioning: str
    principles: List[str]
    dimensions: List[StrategyDimension]
    action_framework: List[str]

    def to_prompt_block(self) -> str:
        """Render blueprint as prompt instructions."""
        principles_text = "\n".join([f"- {item}" for item in self.principles])
        action_text = "\n".join([f"- {item}" for item in self.action_framework])

        dims = []
        for dim in self.dimensions:
            checkpoints = "\n".join([f"  - {cp}" for cp in dim.checkpoints])
            dims.append(f"- {dim.name}: {dim.objective}\n{checkpoints}")
        dimensions_text = "\n".join(dims)

        return (
            f"## Strategy Blueprint: {self.title}\n"
            f"{self.positioning}\n\n"
            f"### Strategy Principles\n{principles_text}\n\n"
            f"### Analysis Dimensions\n{dimensions_text}\n\n"
            f"### Action Framework\n{action_text}"
        )

    def to_markdown_block(self) -> str:
        """Render blueprint as markdown section for template fallback report."""
        dims = "\n".join([f"- **{dim.name}**: {dim.objective}" for dim in self.dimensions])
        section_title = "### 6. Strategy Framework"
        return f"{section_title}\n{dims}\n"


CN_BLUEPRINT = MarketStrategyBlueprint(
    region="cn",
    title="A股市场三段式复盘策略 (A-share Three-Stage Market Recap Strategy)",
    positioning="Focus on index trend, capital flows, and sector rotation to build the next-session trading plan.",
    principles=[
        "Read index direction first, then volume structure, then sector persistence.",
        "Every conclusion should map to position sizing, pace, and risk-control actions.",
        "Use same-day market data and the last 3 days of news; do not speculate beyond verified information.",
    ],
    dimensions=[
        StrategyDimension(
            name="Trend Structure",
            objective="Determine whether the market is in an advancing, ranging, or defensive phase.",
            checkpoints=[
                "Are the SSE, SZSE, and ChiNext moving in the same direction",
                "Was the move confirmed by expanding-up volume or contracting-down volume",
                "Were key support or resistance levels broken",
            ],
        ),
        StrategyDimension(
            name="Capital Sentiment",
            objective="Identify short-term risk appetite and sentiment temperature.",
            checkpoints=[
                "Advance/decline breadth and limit-up/limit-down structure",
                "Whether turnover expanded",
                "Whether high-flyers started to show distribution or disagreement",
            ],
        ),
        StrategyDimension(
            name="Core Sectors",
            objective="Extract tradable leadership themes and avoid weak areas.",
            checkpoints=[
                "Whether leading sectors have event-driven catalysts",
                "Whether sector leadership is driven by clear leading stocks",
                "Whether weakness in laggards is broadening",
            ],
        ),
    ],
    action_framework=[
        "进攻 / Risk-on: index alignment to the upside, expanding turnover, and strengthening leadership themes.",
        "均衡 / Balanced: mixed index signals or low-volume consolidation; keep size controlled and wait for confirmation.",
        "防守 / Defensive: weakening indices with broadening laggards; prioritize risk control and position reduction.",
    ],
)

US_BLUEPRINT = MarketStrategyBlueprint(
    region="us",
    title="US Market Regime Strategy",
    positioning="Focus on index trend, macro narrative, and sector rotation to define next-session risk posture.",
    principles=[
        "Read market regime from S&P 500, Nasdaq, and Dow alignment first.",
        "Separate beta move from theme-driven alpha rotation.",
        "Translate recap into actionable risk-on/risk-off stance with clear invalidation points.",
    ],
    dimensions=[
        StrategyDimension(
            name="Trend Regime",
            objective="Classify the market as momentum, range, or risk-off.",
            checkpoints=[
                "Are SPX/NDX/DJI directionally aligned",
                "Did volume confirm the move",
                "Are key index levels reclaimed or lost",
            ],
        ),
        StrategyDimension(
            name="Macro & Flows",
            objective="Map policy/rates narrative into equity risk appetite.",
            checkpoints=[
                "Treasury yield and USD implications",
                "Breadth and leadership concentration",
                "Defensive vs growth factor rotation",
            ],
        ),
        StrategyDimension(
            name="Sector Themes",
            objective="Identify persistent leaders and vulnerable laggards.",
            checkpoints=[
                "AI/semiconductor/software trend persistence",
                "Energy/financials sensitivity to macro data",
                "Volatility signals from VIX and large-cap earnings",
            ],
        ),
    ],
    action_framework=[
        "Risk-on: broad index breakout with expanding participation.",
        "Neutral: mixed index signals; focus on selective relative strength.",
        "Risk-off: failed breakouts and rising volatility; prioritize capital preservation.",
    ],
)


def get_market_strategy_blueprint(region: str) -> MarketStrategyBlueprint:
    """Return strategy blueprint by market region."""
    return US_BLUEPRINT if region == "us" else CN_BLUEPRINT
