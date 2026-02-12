# -*- coding: utf-8 -*-
"""
===================================
Trend Trading Analyzer - Based on User Trading Philosophy
===================================

Core Trading Principles:
1. Strict Entry - No chasing highs, pursue high success rate per trade
2. Trend Trading - MA5>MA10>MA20 bullish alignment, follow the trend
3. Efficiency First - Focus on stocks with good chip structure
4. Entry Preference - Buy on pullbacks near MA5/MA10

Technical Criteria:
- Bullish Alignment: MA5 > MA10 > MA20
- Bias Rate: (Close - MA5) / MA5 < 5% (no chasing highs)
- Volume Pattern: Low volume pullbacks preferred
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TrendStatus(Enum):
    """Trend status enum"""
    STRONG_BULL = "Strong Bull"      # MA5 > MA10 > MA20, spread widening
    BULL = "Bullish"                 # MA5 > MA10 > MA20
    WEAK_BULL = "Weak Bull"          # MA5 > MA10, but MA10 < MA20
    CONSOLIDATION = "Consolidation"  # MAs entangled
    WEAK_BEAR = "Weak Bear"          # MA5 < MA10, but MA10 > MA20
    BEAR = "Bearish"                 # MA5 < MA10 < MA20
    STRONG_BEAR = "Strong Bear"      # MA5 < MA10 < MA20, spread widening


class VolumeStatus(Enum):
    """Volume status enum"""
    HEAVY_VOLUME_UP = "Heavy Volume Up"       # Volume and price rising together
    HEAVY_VOLUME_DOWN = "Heavy Volume Down"   # Heavy selling
    SHRINK_VOLUME_UP = "Light Volume Up"      # Rising on low volume
    SHRINK_VOLUME_DOWN = "Light Volume Pullback"  # Pullback on low volume (good)
    NORMAL = "Normal Volume"


class BuySignal(Enum):
    """Buy signal enum"""
    STRONG_BUY = "Strong Buy"     # Multiple conditions met
    BUY = "Buy"                   # Basic conditions met
    HOLD = "Hold"                 # Can continue holding
    WAIT = "Wait"                 # Wait for better timing
    SELL = "Sell"                 # Trend weakening
    STRONG_SELL = "Strong Sell"   # Trend broken


class MACDStatus(Enum):
    """MACD status enum"""
    GOLDEN_CROSS_ZERO = "Golden Cross Above Zero"  # DIF crosses above DEA, above zero line
    GOLDEN_CROSS = "Golden Cross"                  # DIF crosses above DEA
    BULLISH = "Bullish"                            # DIF>DEA>0
    CROSSING_UP = "Crossing Up Zero"               # DIF crosses above zero line
    CROSSING_DOWN = "Crossing Down Zero"           # DIF crosses below zero line
    BEARISH = "Bearish"                            # DIF<DEA<0
    DEATH_CROSS = "Death Cross"                    # DIF crosses below DEA


class RSIStatus(Enum):
    """RSI status enum"""
    OVERBOUGHT = "Overbought"    # RSI > 70
    STRONG_BUY = "Strong Buy"    # 50 < RSI < 70
    NEUTRAL = "Neutral"          # 40 <= RSI <= 60
    WEAK = "Weak"                # 30 < RSI < 40
    OVERSOLD = "Oversold"        # RSI < 30


@dataclass
class TrendAnalysisResult:
    """Trend analysis result"""
    code: str

    # Trend assessment
    trend_status: TrendStatus = TrendStatus.CONSOLIDATION
    ma_alignment: str = ""           # MA alignment description
    trend_strength: float = 0.0      # Trend strength 0-100

    # MA data
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    current_price: float = 0.0

    # Bias rate (deviation from MA5)
    bias_ma5: float = 0.0            # (Close - MA5) / MA5 * 100
    bias_ma10: float = 0.0
    bias_ma20: float = 0.0

    # Volume analysis
    volume_status: VolumeStatus = VolumeStatus.NORMAL
    volume_ratio_5d: float = 0.0     # Daily volume / 5-day avg volume
    volume_trend: str = ""           # Volume trend description

    # Support and resistance
    support_ma5: bool = False        # Whether MA5 acts as support
    support_ma10: bool = False       # Whether MA10 acts as support
    resistance_levels: List[float] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)

    # MACD indicator
    macd_dif: float = 0.0          # DIF fast line
    macd_dea: float = 0.0          # DEA slow line
    macd_bar: float = 0.0           # MACD histogram
    macd_status: MACDStatus = MACDStatus.BULLISH
    macd_signal: str = ""            # MACD signal description

    # RSI indicator
    rsi_6: float = 0.0              # RSI(6) short-term
    rsi_12: float = 0.0             # RSI(12) mid-term
    rsi_24: float = 0.0             # RSI(24) long-term
    rsi_status: RSIStatus = RSIStatus.NEUTRAL
    rsi_signal: str = ""              # RSI signal description

    # Buy signal
    buy_signal: BuySignal = BuySignal.WAIT
    signal_score: int = 0            # Overall score 0-100
    signal_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'trend_status': self.trend_status.value,
            'ma_alignment': self.ma_alignment,
            'trend_strength': self.trend_strength,
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'ma60': self.ma60,
            'current_price': self.current_price,
            'bias_ma5': self.bias_ma5,
            'bias_ma10': self.bias_ma10,
            'bias_ma20': self.bias_ma20,
            'volume_status': self.volume_status.value,
            'volume_ratio_5d': self.volume_ratio_5d,
            'volume_trend': self.volume_trend,
            'support_ma5': self.support_ma5,
            'support_ma10': self.support_ma10,
            'buy_signal': self.buy_signal.value,
            'signal_score': self.signal_score,
            'signal_reasons': self.signal_reasons,
            'risk_factors': self.risk_factors,
            'macd_dif': self.macd_dif,
            'macd_dea': self.macd_dea,
            'macd_bar': self.macd_bar,
            'macd_status': self.macd_status.value,
            'macd_signal': self.macd_signal,
            'rsi_6': self.rsi_6,
            'rsi_12': self.rsi_12,
            'rsi_24': self.rsi_24,
            'rsi_status': self.rsi_status.value,
            'rsi_signal': self.rsi_signal,
        }


class StockTrendAnalyzer:
    """
    Stock Trend Analyzer

    Implements the user's trading philosophy:
    1. Trend Assessment - MA5>MA10>MA20 bullish alignment
    2. Bias Rate Detection - No chasing highs, don't buy if deviation from MA5 > 5%
    3. Volume Analysis - Prefer low volume pullbacks
    4. Entry Identification - Pullback to MA5/MA10 support
    5. MACD Indicator - Trend confirmation and golden/death cross signals
    6. RSI Indicator - Overbought/oversold assessment
    """
    
    # Trading parameter configuration
    BIAS_THRESHOLD = 5.0        # Bias rate threshold (%), no buying above this
    VOLUME_SHRINK_RATIO = 0.7   # Low volume threshold (daily vol / 5-day avg)
    VOLUME_HEAVY_RATIO = 1.5    # Heavy volume threshold
    MA_SUPPORT_TOLERANCE = 0.02  # MA support tolerance (2%)

    # MACD parameters (standard 12/26/9)
    MACD_FAST = 12              # Fast line period
    MACD_SLOW = 26             # Slow line period
    MACD_SIGNAL = 9             # Signal line period

    # RSI parameters
    RSI_SHORT = 6               # Short-term RSI period
    RSI_MID = 12               # Mid-term RSI period
    RSI_LONG = 24              # Long-term RSI period
    RSI_OVERBOUGHT = 70        # Overbought threshold
    RSI_OVERSOLD = 30          # Oversold threshold
    
    def __init__(self):
        """Initialize analyzer"""
        pass
    
    def analyze(self, df: pd.DataFrame, code: str) -> TrendAnalysisResult:
        """
        Analyze stock trend

        Args:
            df: DataFrame containing OHLCV data
            code: Stock code

        Returns:
            TrendAnalysisResult analysis result
        """
        result = TrendAnalysisResult(code=code)
        
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"{code} insufficient data for trend analysis")
            result.risk_factors.append("Insufficient data for analysis")
            return result
        
        # Ensure data is sorted by date
        df = df.sort_values('date').reset_index(drop=True)

        # Calculate moving averages
        df = self._calculate_mas(df)

        # Calculate MACD and RSI
        df = self._calculate_macd(df)
        df = self._calculate_rsi(df)

        # Get latest data
        latest = df.iloc[-1]
        result.current_price = float(latest['close'])
        result.ma5 = float(latest['MA5'])
        result.ma10 = float(latest['MA10'])
        result.ma20 = float(latest['MA20'])
        result.ma60 = float(latest.get('MA60', 0))

        # 1. Trend assessment
        self._analyze_trend(df, result)

        # 2. Bias rate calculation
        self._calculate_bias(result)

        # 3. Volume analysis
        self._analyze_volume(df, result)

        # 4. Support and resistance analysis
        self._analyze_support_resistance(df, result)

        # 5. MACD analysis
        self._analyze_macd(df, result)

        # 6. RSI analysis
        self._analyze_rsi(df, result)

        # 7. Generate buy signal
        self._generate_signal(result)

        return result
    
    def _calculate_mas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        df = df.copy()
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        if len(df) >= 60:
            df['MA60'] = df['close'].rolling(window=60).mean()
        else:
            df['MA60'] = df['MA20']  # Use MA20 as fallback when data is insufficient
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicator

        Formula:
        - EMA(12): 12-day exponential moving average
        - EMA(26): 26-day exponential moving average
        - DIF = EMA(12) - EMA(26)
        - DEA = EMA(DIF, 9)
        - MACD = (DIF - DEA) * 2
        """
        df = df.copy()

        # Calculate fast and slow EMAs
        ema_fast = df['close'].ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.MACD_SLOW, adjust=False).mean()

        # Calculate DIF (fast line)
        df['MACD_DIF'] = ema_fast - ema_slow

        # Calculate DEA (signal line)
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=self.MACD_SIGNAL, adjust=False).mean()

        # Calculate histogram
        df['MACD_BAR'] = (df['MACD_DIF'] - df['MACD_DEA']) * 2

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI indicator

        Formula:
        - RS = Average gain / Average loss
        - RSI = 100 - (100 / (1 + RS))
        """
        df = df.copy()

        for period in [self.RSI_SHORT, self.RSI_MID, self.RSI_LONG]:
            # Calculate price changes
            delta = df['close'].diff()

            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain/loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Fill NaN values
            rsi = rsi.fillna(50)  # Default neutral value

            # Add to DataFrame
            col_name = f'RSI_{period}'
            df[col_name] = rsi

        return df
    
    def _analyze_trend(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        Analyze trend status

        Core logic: Assess MA alignment and trend strength
        """
        ma5, ma10, ma20 = result.ma5, result.ma10, result.ma20
        
        # Assess MA alignment
        if ma5 > ma10 > ma20:
            # Check if spread is widening (strong trend)
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA5'] - prev['MA20']) / prev['MA20'] * 100 if prev['MA20'] > 0 else 0
            curr_spread = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BULL
                result.ma_alignment = "Strong bullish alignment, MAs diverging upward"
                result.trend_strength = 90
            else:
                result.trend_status = TrendStatus.BULL
                result.ma_alignment = "Bullish alignment MA5>MA10>MA20"
                result.trend_strength = 75
                
        elif ma5 > ma10 and ma10 <= ma20:
            result.trend_status = TrendStatus.WEAK_BULL
            result.ma_alignment = "Weak bullish, MA5>MA10 but MA10≤MA20"
            result.trend_strength = 55
            
        elif ma5 < ma10 < ma20:
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA20'] - prev['MA5']) / prev['MA5'] * 100 if prev['MA5'] > 0 else 0
            curr_spread = (ma20 - ma5) / ma5 * 100 if ma5 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BEAR
                result.ma_alignment = "Strong bearish alignment, MAs diverging downward"
                result.trend_strength = 10
            else:
                result.trend_status = TrendStatus.BEAR
                result.ma_alignment = "Bearish alignment MA5<MA10<MA20"
                result.trend_strength = 25
                
        elif ma5 < ma10 and ma10 >= ma20:
            result.trend_status = TrendStatus.WEAK_BEAR
            result.ma_alignment = "Weak bearish, MA5<MA10 but MA10≥MA20"
            result.trend_strength = 40
            
        else:
            result.trend_status = TrendStatus.CONSOLIDATION
            result.ma_alignment = "MAs entangled, trend unclear"
            result.trend_strength = 50
    
    def _calculate_bias(self, result: TrendAnalysisResult) -> None:
        """
        Calculate bias rate

        Bias = (Price - MA) / MA * 100%

        Strict entry: Do not chase when bias exceeds 5%
        """
        price = result.current_price
        
        if result.ma5 > 0:
            result.bias_ma5 = (price - result.ma5) / result.ma5 * 100
        if result.ma10 > 0:
            result.bias_ma10 = (price - result.ma10) / result.ma10 * 100
        if result.ma20 > 0:
            result.bias_ma20 = (price - result.ma20) / result.ma20 * 100
    
    def _analyze_volume(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        Analyze volume

        Preference: Low volume pullback > Heavy volume rise > Low volume rise > Heavy volume decline
        """
        if len(df) < 5:
            return
        
        latest = df.iloc[-1]
        vol_5d_avg = df['volume'].iloc[-6:-1].mean()
        
        if vol_5d_avg > 0:
            result.volume_ratio_5d = float(latest['volume']) / vol_5d_avg
        
        # Determine price change
        prev_close = df.iloc[-2]['close']
        price_change = (latest['close'] - prev_close) / prev_close * 100
        
        # Volume status assessment
        if result.volume_ratio_5d >= self.VOLUME_HEAVY_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_UP
                result.volume_trend = "Heavy volume rise, strong bullish momentum"
            else:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_DOWN
                result.volume_trend = "Heavy volume decline, watch for risk"
        elif result.volume_ratio_5d <= self.VOLUME_SHRINK_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_UP
                result.volume_trend = "Low volume rise, weak upward momentum"
            else:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_DOWN
                result.volume_trend = "Low volume pullback, washout pattern (positive)"
        else:
            result.volume_status = VolumeStatus.NORMAL
            result.volume_trend = "Normal volume"
    
    def _analyze_support_resistance(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        Analyze support and resistance levels

        Entry preference: Pullback to MA5/MA10 finding support
        """
        price = result.current_price
        
        # Check if price finds support near MA5
        if result.ma5 > 0:
            ma5_distance = abs(price - result.ma5) / result.ma5
            if ma5_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma5:
                result.support_ma5 = True
                result.support_levels.append(result.ma5)
        
        # Check if price finds support near MA10
        if result.ma10 > 0:
            ma10_distance = abs(price - result.ma10) / result.ma10
            if ma10_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma10:
                result.support_ma10 = True
                if result.ma10 not in result.support_levels:
                    result.support_levels.append(result.ma10)
        
        # MA20 as key support
        if result.ma20 > 0 and price >= result.ma20:
            result.support_levels.append(result.ma20)
        
        # Recent high as resistance
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            if recent_high > price:
                result.resistance_levels.append(recent_high)

    def _analyze_macd(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        Analyze MACD indicator

        Core signals:
        - Golden cross above zero: Strongest buy signal
        - Golden cross: DIF crosses above DEA
        - Death cross: DIF crosses below DEA
        """
        if len(df) < self.MACD_SLOW:
            result.macd_signal = "Insufficient data"
            return

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Get MACD data
        result.macd_dif = float(latest['MACD_DIF'])
        result.macd_dea = float(latest['MACD_DEA'])
        result.macd_bar = float(latest['MACD_BAR'])

        # Determine golden/death cross
        prev_dif_dea = prev['MACD_DIF'] - prev['MACD_DEA']
        curr_dif_dea = result.macd_dif - result.macd_dea

        # Golden cross: DIF crosses above DEA
        is_golden_cross = prev_dif_dea <= 0 and curr_dif_dea > 0

        # Death cross: DIF crosses below DEA
        is_death_cross = prev_dif_dea >= 0 and curr_dif_dea < 0

        # Zero line crossing
        prev_zero = prev['MACD_DIF']
        curr_zero = result.macd_dif
        is_crossing_up = prev_zero <= 0 and curr_zero > 0
        is_crossing_down = prev_zero >= 0 and curr_zero < 0

        # Determine MACD status
        if is_golden_cross and curr_zero > 0:
            result.macd_status = MACDStatus.GOLDEN_CROSS_ZERO
            result.macd_signal = "⭐ Golden cross above zero, strong buy signal!"
        elif is_crossing_up:
            result.macd_status = MACDStatus.CROSSING_UP
            result.macd_signal = "⚡ DIF crossed above zero, trend strengthening"
        elif is_golden_cross:
            result.macd_status = MACDStatus.GOLDEN_CROSS
            result.macd_signal = "✅ Golden cross, uptrend"
        elif is_death_cross:
            result.macd_status = MACDStatus.DEATH_CROSS
            result.macd_signal = "❌ Death cross, downtrend"
        elif is_crossing_down:
            result.macd_status = MACDStatus.CROSSING_DOWN
            result.macd_signal = "⚠️ DIF crossed below zero, trend weakening"
        elif result.macd_dif > 0 and result.macd_dea > 0:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = "✓ Bullish alignment, continued uptrend"
        elif result.macd_dif < 0 and result.macd_dea < 0:
            result.macd_status = MACDStatus.BEARISH
            result.macd_signal = "⚠ Bearish alignment, continued downtrend"
        else:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = " MACD neutral zone"

    def _analyze_rsi(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        Analyze RSI indicator

        Core assessment:
        - RSI > 70: Overbought, caution on chasing highs
        - RSI < 30: Oversold, watch for rebound
        - 40-60: Neutral zone
        """
        if len(df) < self.RSI_LONG:
            result.rsi_signal = "Insufficient data"
            return

        latest = df.iloc[-1]

        # Get RSI data
        result.rsi_6 = float(latest[f'RSI_{self.RSI_SHORT}'])
        result.rsi_12 = float(latest[f'RSI_{self.RSI_MID}'])
        result.rsi_24 = float(latest[f'RSI_{self.RSI_LONG}'])

        # Use mid-term RSI(12) as primary indicator
        rsi_mid = result.rsi_12

        # Determine RSI status
        if rsi_mid > self.RSI_OVERBOUGHT:
            result.rsi_status = RSIStatus.OVERBOUGHT
            result.rsi_signal = f"⚠️ RSI overbought ({rsi_mid:.1f}>70), high short-term pullback risk"
        elif rsi_mid > 60:
            result.rsi_status = RSIStatus.STRONG_BUY
            result.rsi_signal = f"✅ RSI strong ({rsi_mid:.1f}), solid bullish momentum"
        elif rsi_mid >= 40:
            result.rsi_status = RSIStatus.NEUTRAL
            result.rsi_signal = f" RSI neutral ({rsi_mid:.1f}), consolidating"
        elif rsi_mid >= self.RSI_OVERSOLD:
            result.rsi_status = RSIStatus.WEAK
            result.rsi_signal = f"⚡ RSI weak ({rsi_mid:.1f}), watch for rebound"
        else:
            result.rsi_status = RSIStatus.OVERSOLD
            result.rsi_signal = f"⭐ RSI oversold ({rsi_mid:.1f}<30), high rebound potential"

    def _generate_signal(self, result: TrendAnalysisResult) -> None:
        """
        Generate buy signal

        Composite scoring system:
        - Trend (30 pts): Bullish alignment scores high
        - Bias rate (20 pts): Close to MA5 scores high
        - Volume (15 pts): Low volume pullback scores high
        - Support (10 pts): MA support scores high
        - MACD (15 pts): Golden cross and bullish scores high
        - RSI (10 pts): Oversold and strong scores high
        """
        score = 0
        reasons = []
        risks = []

        # === Trend score (30 pts) ===
        trend_scores = {
            TrendStatus.STRONG_BULL: 30,
            TrendStatus.BULL: 26,
            TrendStatus.WEAK_BULL: 18,
            TrendStatus.CONSOLIDATION: 12,
            TrendStatus.WEAK_BEAR: 8,
            TrendStatus.BEAR: 4,
            TrendStatus.STRONG_BEAR: 0,
        }
        trend_score = trend_scores.get(result.trend_status, 12)
        score += trend_score

        if result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            reasons.append(f"✅ {result.trend_status.value}, go long with the trend")
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            risks.append(f"⚠️ {result.trend_status.value}, not suitable for long")

        # === Bias rate score (20 pts) ===
        bias = result.bias_ma5
        if bias < 0:
            # Price below MA5 (pulling back)
            if bias > -3:
                score += 20
                reasons.append(f"✅ Price slightly below MA5 ({bias:.1f}%), pullback buy point")
            elif bias > -5:
                score += 16
                reasons.append(f"✅ Price pulling back to MA5 ({bias:.1f}%), watch support")
            else:
                score += 8
                risks.append(f"⚠️ Deviation too large ({bias:.1f}%), potential breakdown")
        elif bias < 2:
            score += 18
            reasons.append(f"✅ Price near MA5 ({bias:.1f}%), good entry timing")
        elif bias < self.BIAS_THRESHOLD:
            score += 14
            reasons.append(f"⚡ Price slightly above MA5 ({bias:.1f}%), small position entry OK")
        else:
            score += 4
            risks.append(f"❌ Deviation too high ({bias:.1f}%>5%), strictly no chasing!")

        # === Volume score (15 pts) ===
        volume_scores = {
            VolumeStatus.SHRINK_VOLUME_DOWN: 15,  # Low volume pullback best
            VolumeStatus.HEAVY_VOLUME_UP: 12,     # Heavy volume rise second
            VolumeStatus.NORMAL: 10,
            VolumeStatus.SHRINK_VOLUME_UP: 6,     # Low volume rise weaker
            VolumeStatus.HEAVY_VOLUME_DOWN: 0,    # Heavy volume decline worst
        }
        vol_score = volume_scores.get(result.volume_status, 8)
        score += vol_score

        if result.volume_status == VolumeStatus.SHRINK_VOLUME_DOWN:
            reasons.append("✅ Low volume pullback, market maker washout")
        elif result.volume_status == VolumeStatus.HEAVY_VOLUME_DOWN:
            risks.append("⚠️ Heavy volume decline, watch for risk")

        # === Support score (10 pts) ===
        if result.support_ma5:
            score += 5
            reasons.append("✅ MA5 support holding")
        if result.support_ma10:
            score += 5
            reasons.append("✅ MA10 support holding")

        # === MACD score (15 pts) ===
        macd_scores = {
            MACDStatus.GOLDEN_CROSS_ZERO: 15,  # Golden cross above zero strongest
            MACDStatus.GOLDEN_CROSS: 12,      # Golden cross
            MACDStatus.CROSSING_UP: 10,       # Crossing above zero
            MACDStatus.BULLISH: 8,            # Bullish
            MACDStatus.BEARISH: 2,            # Bearish
            MACDStatus.CROSSING_DOWN: 0,       # Crossing below zero
            MACDStatus.DEATH_CROSS: 0,        # Death cross
        }
        macd_score = macd_scores.get(result.macd_status, 5)
        score += macd_score

        if result.macd_status in [MACDStatus.GOLDEN_CROSS_ZERO, MACDStatus.GOLDEN_CROSS]:
            reasons.append(f"✅ {result.macd_signal}")
        elif result.macd_status in [MACDStatus.DEATH_CROSS, MACDStatus.CROSSING_DOWN]:
            risks.append(f"⚠️ {result.macd_signal}")
        else:
            reasons.append(result.macd_signal)

        # === RSI score (10 pts) ===
        rsi_scores = {
            RSIStatus.OVERSOLD: 10,       # Oversold best
            RSIStatus.STRONG_BUY: 8,     # Strong
            RSIStatus.NEUTRAL: 5,        # Neutral
            RSIStatus.WEAK: 3,            # Weak
            RSIStatus.OVERBOUGHT: 0,       # Overbought worst
        }
        rsi_score = rsi_scores.get(result.rsi_status, 5)
        score += rsi_score

        if result.rsi_status in [RSIStatus.OVERSOLD, RSIStatus.STRONG_BUY]:
            reasons.append(f"✅ {result.rsi_signal}")
        elif result.rsi_status == RSIStatus.OVERBOUGHT:
            risks.append(f"⚠️ {result.rsi_signal}")
        else:
            reasons.append(result.rsi_signal)

        # === Final assessment ===
        result.signal_score = score
        result.signal_reasons = reasons
        result.risk_factors = risks

        # Generate buy signal (thresholds adjusted for 100-point scale)
        if score >= 75 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            result.buy_signal = BuySignal.STRONG_BUY
        elif score >= 60 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL, TrendStatus.WEAK_BULL]:
            result.buy_signal = BuySignal.BUY
        elif score >= 45:
            result.buy_signal = BuySignal.HOLD
        elif score >= 30:
            result.buy_signal = BuySignal.WAIT
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            result.buy_signal = BuySignal.STRONG_SELL
        else:
            result.buy_signal = BuySignal.SELL
    
    def format_analysis(self, result: TrendAnalysisResult) -> str:
        """
        Format analysis result as text

        Args:
            result: Analysis result

        Returns:
            Formatted analysis text
        """
        lines = [
            f"=== {result.code} Trend Analysis ===",
            f"",
            f"📊 Trend: {result.trend_status.value}",
            f"   MA Alignment: {result.ma_alignment}",
            f"   Trend Strength: {result.trend_strength}/100",
            f"",
            f"📈 MA Data:",
            f"   Price: {result.current_price:.2f}",
            f"   MA5:  {result.ma5:.2f} (Bias {result.bias_ma5:+.2f}%)",
            f"   MA10: {result.ma10:.2f} (Bias {result.bias_ma10:+.2f}%)",
            f"   MA20: {result.ma20:.2f} (Bias {result.bias_ma20:+.2f}%)",
            f"",
            f"📊 Volume Analysis: {result.volume_status.value}",
            f"   Vol Ratio(vs 5d): {result.volume_ratio_5d:.2f}",
            f"   Volume Trend: {result.volume_trend}",
            f"",
            f"📈 MACD Indicator: {result.macd_status.value}",
            f"   DIF: {result.macd_dif:.4f}",
            f"   DEA: {result.macd_dea:.4f}",
            f"   MACD: {result.macd_bar:.4f}",
            f"   Signal: {result.macd_signal}",
            f"",
            f"📊 RSI Indicator: {result.rsi_status.value}",
            f"   RSI(6): {result.rsi_6:.1f}",
            f"   RSI(12): {result.rsi_12:.1f}",
            f"   RSI(24): {result.rsi_24:.1f}",
            f"   Signal: {result.rsi_signal}",
            f"",
            f"🎯 Recommendation: {result.buy_signal.value}",
            f"   Overall Score: {result.signal_score}/100",
        ]

        if result.signal_reasons:
            lines.append(f"")
            lines.append(f"✅ Buy Reasons:")
            for reason in result.signal_reasons:
                lines.append(f"   {reason}")

        if result.risk_factors:
            lines.append(f"")
            lines.append(f"⚠️ Risk Factors:")
            for risk in result.risk_factors:
                lines.append(f"   {risk}")

        return "\n".join(lines)


def analyze_stock(df: pd.DataFrame, code: str) -> TrendAnalysisResult:
    """
    Convenience function: Analyze a single stock

    Args:
        df: DataFrame containing OHLCV data
        code: Stock code

    Returns:
        TrendAnalysisResult analysis result
    """
    analyzer = StockTrendAnalyzer()
    return analyzer.analyze(df, code)


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Simulated data test
    import numpy as np

    dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
    np.random.seed(42)

    # Simulate bullish alignment data
    base_price = 10.0
    prices = [base_price]
    for i in range(59):
        change = np.random.randn() * 0.02 + 0.003  # Slight upward trend
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices],
    })
    
    analyzer = StockTrendAnalyzer()
    result = analyzer.analyze(df, '000001')
    print(analyzer.format_analysis(result))
