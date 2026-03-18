"""
Parlay Optimizer — Finds the best combination of legs for multi-bet parlays.

Uses confidence scoring, correlation analysis, and bankroll management
(Kelly Criterion) to recommend optimal parlays.

Supports Hard Rock Bet parlay types:
- Standard Parlay
- Same Game Parlay (SGP)
- Round Robin (multiple parlay combos)
- Teaser (adjusted spreads/totals)
- Flex Parlay (insurance — miss 1+ legs and still win)
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

from loguru import logger
from src.config import settings
from src.models import (
    Prediction, ParlayPrediction, BankrollAdvice, BetType
)


class ParlayOptimizer:
    """Optimizes parlay selections for maximum expected value."""

    def __init__(
        self,
        min_confidence: float = 0,
        max_legs: int = 0,
        bankroll: float = 0,
    ):
        self.min_confidence = min_confidence or settings.parlay_min_confidence
        self.max_legs = max_legs or settings.max_parlay_legs
        self.bankroll = bankroll or settings.default_bankroll

    def build_parlay(
        self,
        predictions: list[Prediction],
        num_legs: int = 6,
        strategy: str = "balanced",
    ) -> ParlayPrediction:
        """
        Build the best parlay with N legs.

        Strategies:
        - "safe": Maximize combined confidence (safest picks)
        - "value": Maximize expected value (best odds/probability ratio)
        - "balanced": Balance confidence and value
        """
        # Filter to only high-confidence picks (1 per event)
        filtered = self._filter_best_per_event(predictions)
        filtered = [p for p in filtered if p.confidence >= self.min_confidence]

        if len(filtered) < num_legs:
            logger.warning(
                f"Only {len(filtered)} picks above {self.min_confidence}% "
                f"confidence. Requested {num_legs} legs."
            )
            num_legs = min(num_legs, len(filtered))

        if num_legs == 0:
            return ParlayPrediction(legs=[], reasoning="No qualifying picks found.")

        # Score and rank picks based on strategy
        scored = self._score_picks(filtered, strategy)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Remove correlated events (same tournament, close times)
        selected = self._select_uncorrelated(scored, num_legs)

        # Build the parlay
        legs = [s[0] for s in selected]
        return self._create_parlay(legs)

    def build_multiple_parlays(
        self,
        predictions: list[Prediction],
        num_legs: int = 6,
        count: int = 3,
    ) -> list[ParlayPrediction]:
        """Generate multiple parlay options with different strategies."""
        parlays = []
        for strategy in ["safe", "balanced", "value"]:
            parlay = self.build_parlay(predictions, num_legs, strategy)
            if parlay.legs:
                parlay.reasoning = f"[{strategy.upper()} strategy] {parlay.reasoning}"
                parlays.append(parlay)
            if len(parlays) >= count:
                break
        return parlays

    def calculate_bankroll_advice(
        self, parlay: ParlayPrediction
    ) -> BankrollAdvice:
        """Calculate optimal stake using fractional Kelly Criterion."""
        if not parlay.legs or parlay.combined_odds <= 1:
            return BankrollAdvice(
                recommended_stake=0,
                kelly_stake=0,
                risk_percentage=0,
                bankroll=self.bankroll,
                reasoning="No valid parlay to stake on.",
            )

        # Kelly Criterion: f* = (bp - q) / b
        # b = decimal odds - 1
        # p = probability of winning
        # q = 1 - p
        b = parlay.combined_odds - 1
        p = parlay.combined_confidence / 100
        q = 1 - p

        kelly_full = ((b * p) - q) / b if b > 0 else 0
        kelly_full = max(0, kelly_full)

        # Fractional Kelly (more conservative)
        kelly_fraction = settings.kelly_fraction
        kelly_stake = self.bankroll * kelly_full * kelly_fraction
        kelly_stake = max(0, round(kelly_stake, 2))

        # Cap at reasonable percentage
        max_stake = self.bankroll * 0.05  # Never more than 5%
        recommended = min(kelly_stake, max_stake)

        risk_pct = (recommended / self.bankroll * 100) if self.bankroll > 0 else 0

        risk_level = (
            "low" if risk_pct < 1.5
            else "medium" if risk_pct < 3
            else "high"
        )

        return BankrollAdvice(
            recommended_stake=round(recommended, 2),
            kelly_stake=round(kelly_stake, 2),
            risk_percentage=round(risk_pct, 2),
            bankroll=self.bankroll,
            reasoning=(
                f"Kelly suggests ${kelly_stake:.2f} ({kelly_full*100:.1f}% full Kelly). "
                f"Using {kelly_fraction:.0%} fractional Kelly. "
                f"Recommended: ${recommended:.2f} ({risk_pct:.1f}% of bankroll). "
                f"Risk level: {risk_level}."
            ),
        )

    # ── Internal Methods ─────────────────────────────────────────────────

    def _filter_best_per_event(
        self, predictions: list[Prediction]
    ) -> list[Prediction]:
        """Keep only the highest-confidence pick per event."""
        best = {}
        for pred in predictions:
            eid = pred.event.id
            if eid not in best or pred.confidence > best[eid].confidence:
                best[eid] = pred
        return list(best.values())

    def _score_picks(
        self, picks: list[Prediction], strategy: str
    ) -> list[tuple[Prediction, float]]:
        """Score picks based on strategy."""
        scored = []
        for p in picks:
            if strategy == "safe":
                score = p.confidence
            elif strategy == "value":
                # Emphasis on value: good odds relative to confidence
                score = p.value_rating * 50 + p.confidence * 0.5 if p.value_rating > 0 else p.confidence * 0.3
            else:  # balanced
                value_bonus = max(0, p.value_rating * 25)
                score = p.confidence * 0.7 + value_bonus

            # Tournament importance bonus
            score += p.event.tournament.priority * 0.5

            scored.append((p, score))
        return scored

    def _select_uncorrelated(
        self, scored: list[tuple[Prediction, float]], num_legs: int
    ) -> list[tuple[Prediction, float]]:
        """Select picks that are not heavily correlated."""
        selected = []
        used_tournaments = set()

        for item in scored:
            pred = item[0]
            tid = pred.event.tournament.id

            # Allow max 2 picks from same tournament to reduce correlation
            tournament_count = sum(
                1 for s in selected
                if s[0].event.tournament.id == tid
            )
            if tournament_count >= 2:
                continue

            selected.append(item)
            if len(selected) >= num_legs:
                break

        return selected

    def _create_parlay(self, legs: list[Prediction]) -> ParlayPrediction:
        """Create a ParlayPrediction from selected legs."""
        if not legs:
            return ParlayPrediction(legs=[])

        # Combined probability (independent events)
        combined_prob = 1.0
        for leg in legs:
            combined_prob *= (leg.confidence / 100)

        # Combined odds
        combined_odds = 1.0
        has_odds = True
        for leg in legs:
            if leg.odds > 0:
                combined_odds *= leg.odds
            else:
                has_odds = False

        if not has_odds:
            # Estimate odds from probabilities
            combined_odds = 1.0
            for leg in legs:
                if leg.probability > 0:
                    combined_odds *= (1 / leg.probability)

        # Expected value
        ev = (combined_prob * combined_odds) - 1 if combined_odds > 0 else 0

        # Risk assessment
        avg_confidence = sum(l.confidence for l in legs) / len(legs)
        risk = (
            "low" if avg_confidence > 75
            else "medium" if avg_confidence > 65
            else "high"
        )

        # Build explicit reasoning with team names, sports, lines
        sport_emojis = {
            "football": "⚽", "basketball": "🏀", "tennis": "🎾",
            "baseball": "⚾", "american-football": "🏈", "volleyball": "🏐",
            "ice-hockey": "🏒", "mma": "🥊", "handball": "🤾", "rugby": "🏉",
        }
        sports_in_parlay = set()
        reasoning_parts = []
        for i, leg in enumerate(legs, 1):
            sport_slug = leg.event.tournament.sport.value
            sport_emoji = sport_emojis.get(sport_slug, "🏆")
            sports_in_parlay.add(sport_slug)
            am_odds = f" ({leg.american_odds})" if leg.american_odds else ""
            push = f" ⚠️ {leg.push_note}" if leg.push_note else ""
            dt = leg.event.start_time.strftime("%b %d • %I:%M %p")
            reasoning_parts.append(
                f"  Leg {i}: {sport_emoji} {leg.event.home_team.name} vs {leg.event.away_team.name}\n"
                f"         ➤ {leg.pick}{am_odds} | {leg.confidence:.0f}% conf{push}\n"
                f"         📅 {dt} | {leg.event.tournament.name}"
            )

        mix_label = "🌐 MIXED SPORTS" if len(sports_in_parlay) > 1 else sport_emojis.get(list(sports_in_parlay)[0], "🏆")

        reasoning = (
            f"{mix_label} {len(legs)}-Leg Parlay\n"
            f"📊 Combined Confidence: {combined_prob*100:.1f}% | "
            f"💰 Est. Odds: {combined_odds:.2f}x | "
            f"📈 EV: {ev:+.4f}\n"
            f"{'─' * 55}\n"
            + "\n".join(reasoning_parts)
        )

        parlay = ParlayPrediction(
            legs=legs,
            combined_confidence=round(combined_prob * 100, 2),
            combined_odds=round(combined_odds, 2),
            expected_value=round(ev, 4),
            risk_level=risk,
            reasoning=reasoning,
        )

        # Calculate recommended stake
        advice = self.calculate_bankroll_advice(parlay)
        parlay.recommended_stake = advice.recommended_stake

        return parlay

    # ── Same Game Parlay (SGP) ───────────────────────────────────────

    def build_sgp(
        self,
        predictions: list[Prediction],
        event_id: int,
        num_legs: int = 4,
    ) -> ParlayPrediction:
        """
        Build a Same Game Parlay: multiple picks from the SAME game.
        Hard Rock Bet style — combine moneyline, spread, O/U, player props, etc.
        """
        # Filter to only preds from this specific event
        event_preds = [p for p in predictions if p.event.id == event_id]
        if not event_preds:
            return ParlayPrediction(legs=[], reasoning="No predictions for this event.", parlay_type="sgp")

        # For SGP, select best pick from each DIFFERENT bet type
        by_type: dict[str, Prediction] = {}
        for p in sorted(event_preds, key=lambda x: x.confidence, reverse=True):
            bt = p.bet_type.value
            if bt not in by_type:
                by_type[bt] = p

        legs = list(by_type.values())[:num_legs]

        if len(legs) < 2:
            return ParlayPrediction(legs=legs, reasoning="Need at least 2 different bet types for SGP.", parlay_type="sgp")

        parlay = self._create_parlay(legs)
        # SGP odds are typically correlated — reduce combined odds by ~15%
        parlay.combined_odds = round(parlay.combined_odds * 0.85, 2)
        parlay.parlay_type = "sgp"

        match_label = f"{legs[0].event.home_team.name} vs {legs[0].event.away_team.name}" if legs else "Unknown"
        parlay.reasoning = f"🎰 SAME GAME PARLAY — {match_label}\n" + parlay.reasoning

        advice = self.calculate_bankroll_advice(parlay)
        parlay.recommended_stake = advice.recommended_stake
        return parlay

    def build_sgp_for_all_events(
        self,
        predictions: list[Prediction],
        num_legs: int = 4,
    ) -> list[ParlayPrediction]:
        """Build SGPs for every event that has enough bet types."""
        event_ids = set(p.event.id for p in predictions)
        sgps = []
        for eid in event_ids:
            sgp = self.build_sgp(predictions, eid, num_legs)
            if len(sgp.legs) >= 2:
                sgps.append(sgp)
        sgps.sort(key=lambda p: p.combined_confidence, reverse=True)
        return sgps

    # ── Round Robin ──────────────────────────────────────────────────

    def build_round_robin(
        self,
        predictions: list[Prediction],
        num_picks: int = 5,
        combo_size: int = 3,
    ) -> list[ParlayPrediction]:
        """
        Round Robin: Select N picks, generate all C(N, combo_size) parlays.
        Hard Rock Bet style — multiple parlay combos from your selections.
        """
        filtered = self._filter_best_per_event(predictions)
        filtered = [p for p in filtered if p.confidence >= self.min_confidence]
        filtered.sort(key=lambda p: p.confidence, reverse=True)

        picks = filtered[:num_picks]
        if len(picks) < combo_size:
            return []

        combos = list(itertools.combinations(picks, combo_size))
        parlays = []
        for combo in combos:
            parlay = self._create_parlay(list(combo))
            parlay.parlay_type = "round_robin"
            parlay.reasoning = f"🔄 ROUND ROBIN ({combo_size} of {len(picks)})\n" + parlay.reasoning
            advice = self.calculate_bankroll_advice(parlay)
            parlay.recommended_stake = advice.recommended_stake
            parlays.append(parlay)

        parlays.sort(key=lambda p: p.combined_confidence, reverse=True)
        return parlays

    # ── Teaser ───────────────────────────────────────────────────────

    def build_teaser(
        self,
        predictions: list[Prediction],
        num_legs: int = 3,
        teaser_points: float = 6.0,
    ) -> ParlayPrediction:
        """
        Teaser: Adjust spreads/totals by teaser_points in your favor.
        Hard Rock Bet style — buy points on spreads and totals.
        Only spread and O/U legs qualify.
        """
        TEASER_TYPES = {BetType.SPREAD, BetType.OVER_UNDER, BetType.ALTERNATE_SPREAD, BetType.ALTERNATE_TOTAL}

        teaser_preds = [p for p in predictions if p.bet_type in TEASER_TYPES and p.line is not None]
        filtered = self._filter_best_per_event(teaser_preds)
        filtered = [p for p in filtered if p.confidence >= 50]
        filtered.sort(key=lambda p: p.confidence, reverse=True)

        legs = filtered[:num_legs]
        if len(legs) < 2:
            return ParlayPrediction(
                legs=[], reasoning="Need at least 2 spread/total legs for a teaser.", parlay_type="teaser"
            )

        # Adjust each leg's line by teaser_points in bettor's favor
        adjusted_legs = []
        for leg in legs:
            adj = Prediction(
                event=leg.event,
                bet_type=leg.bet_type,
                pick=leg.pick,
                confidence=min(99, leg.confidence + teaser_points * 2.5),  # Buying points increases confidence
                probability=min(0.99, leg.probability + teaser_points * 0.025),
                odds=leg.odds * 0.65,  # Teaser reduces payout significantly
                value_rating=leg.value_rating,
                reasoning=leg.reasoning,
                factors=leg.factors,
                line=leg.line,
                american_odds=leg.american_odds,
                market_display=f"TEASER {teaser_points:+.0f}pts — {leg.market_display}",
                team_name=leg.team_name,
                push_note=leg.push_note,
            )
            adjusted_legs.append(adj)

        parlay = self._create_parlay(adjusted_legs)
        parlay.combined_odds = round(parlay.combined_odds * 0.55, 2)  # Teasers pay much less
        parlay.parlay_type = "teaser"
        parlay.teaser_points = teaser_points
        parlay.reasoning = f"🎲 TEASER (+{teaser_points:.0f} points) — Lines adjusted in your favor\n" + parlay.reasoning

        advice = self.calculate_bankroll_advice(parlay)
        parlay.recommended_stake = advice.recommended_stake
        return parlay

    # ── Flex Parlay (Insurance) ──────────────────────────────────────

    def build_flex_parlay(
        self,
        predictions: list[Prediction],
        num_legs: int = 5,
        miss_allowed: int = 1,
    ) -> ParlayPrediction:
        """
        Flex Parlay: Parlay that still pays if you miss some legs.
        Hard Rock Bet style — lose 1+ legs and still get a reduced payout.
        """
        filtered = self._filter_best_per_event(predictions)
        filtered = [p for p in filtered if p.confidence >= self.min_confidence]
        filtered.sort(key=lambda p: p.confidence, reverse=True)

        legs = filtered[:num_legs]
        if len(legs) < miss_allowed + 2:
            return ParlayPrediction(
                legs=[],
                reasoning=f"Need at least {miss_allowed + 2} legs for a flex parlay with {miss_allowed} miss(es) allowed.",
                parlay_type="flex",
            )

        parlay = self._create_parlay(legs)

        # Flex parlay reduces odds based on insurance level
        insurance_factor = 0.5 ** miss_allowed  # Each miss halves the payout
        parlay.combined_odds = round(max(1.1, parlay.combined_odds * insurance_factor), 2)

        # Increase effective confidence since we can miss legs
        flex_conf = 0
        n = len(legs)
        probs = [l.probability for l in legs]
        # P(at most miss_allowed misses) = sum of P(exactly k misses) for k=0..miss_allowed
        # Simplified: boost confidence proportionally
        base_prob = parlay.combined_confidence / 100
        boost = 1 + miss_allowed * 0.3  # ~30% boost per miss allowed
        parlay.combined_confidence = round(min(95, base_prob * boost * 100), 2)

        parlay.parlay_type = "flex"
        parlay.flex_miss_allowed = miss_allowed
        miss_label = f"{miss_allowed} miss{'es' if miss_allowed > 1 else ''}"
        parlay.reasoning = f"💪 FLEX PARLAY — Win even with {miss_label}! (Reduced payout)\n" + parlay.reasoning

        advice = self.calculate_bankroll_advice(parlay)
        parlay.recommended_stake = advice.recommended_stake
        return parlay
