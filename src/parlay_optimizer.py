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
import re
from typing import Optional

from loguru import logger
from src.analyzer import StatisticalAnalyzer
from src.config import settings
from src.markets import decimal_to_american
from src.models import (
    Prediction, ParlayPrediction, BankrollAdvice, BetType, Sport
)


class ParlayOptimizer:
    """Optimizes parlay selections for maximum expected value."""

    def __init__(
        self,
        min_confidence: Optional[float] = None,
        max_legs: Optional[int] = None,
        bankroll: Optional[float] = None,
    ):
        # `None` means "use the configured default". The previous signature
        # defaulted these to 0 and then did `min_confidence or default`, so
        # passing 0 — a legitimate "no confidence floor" — was silently
        # replaced by the 70% default, and every leg got filtered out.
        self.min_confidence = (
            settings.parlay_min_confidence if min_confidence is None
            else min_confidence
        )
        self.max_legs = (
            settings.max_parlay_legs if max_legs is None else max_legs
        )
        self.bankroll = (
            settings.default_bankroll if bankroll is None else bankroll
        )

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
        # Restrict to bettable prices first, then take one leg per event.
        # Order matters: filtering per event first would pick the 94%
        # alternate line for every game and leave nothing in the band.
        candidates = self._bettable(predictions)
        filtered = self._filter_best_per_event(candidates)
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

    # Parlay legs are drawn from this probability band.
    #
    # Without it, `_filter_best_per_event` hands back whatever market has the
    # highest confidence, which is always an extreme alternate line — "Over
    # 0.5 Goals" at 94%. A four-leg ticket of -1900 shots pays 1.26 and is
    # not a bet anyone wants. These are the odds people actually parlay.
    LEG_MIN_PROB = 0.35
    LEG_MAX_PROB = 0.80

    @staticmethod
    def _discount_odds(decimal_odds: float, factor: float) -> float:
        """Scale the *profit* on a price, keeping it a valid decimal price.

        Multiplying decimal odds directly is wrong: decimal odds are
        1 + profit, so scaling the whole thing can drop below 1.0, which says
        a winning bet returns less than the stake. A teaser of short legs did
        exactly that and produced combined odds of 0.18.
        """
        if decimal_odds <= 1.0:
            return 1.01
        return round(1.0 + (decimal_odds - 1.0) * factor, 3)

    def _bettable(self, picks: list[Prediction]) -> list[Prediction]:
        """Keep only legs in the band people actually parlay."""
        banded = [
            p for p in picks
            if self.LEG_MIN_PROB <= p.probability <= self.LEG_MAX_PROB
        ]
        # If the band is empty, fall back rather than returning nothing.
        return banded or picks

    def _drop_contradictions(
        self, legs: list[Prediction]
    ) -> list[Prediction]:
        """Remove legs that cannot both win.

        Picking the best leg per bet type can pair up markets that flatly
        conflict — "Over 2.5" from the totals market alongside "Under 3.5"
        from the alternates, or "BTTS No" with a 2-1 correct score. A ticket
        containing both is dead on submission.

        Legs are considered in confidence order, and any later leg that
        contradicts one already kept is dropped.
        """
        kept: list[Prediction] = []
        for leg in sorted(legs, key=lambda p: p.confidence, reverse=True):
            if not any(self._conflicts(leg, other) for other in kept):
                kept.append(leg)
        return kept

    def _conflicts(self, a: Prediction, b: Prediction) -> bool:
        """True when two legs on the same game cannot both win."""
        if a.event.id != b.event.id:
            return False

        def is_over(p: Prediction) -> Optional[bool]:
            text = p.pick.lower()
            if "over" in text:
                return True
            if "under" in text:
                return False
            return None

        TOTAL_TYPES = {
            BetType.OVER_UNDER, BetType.ALTERNATE_TOTAL, BetType.TEAM_TOTAL,
        }
        # Totals on the same side of the same subject: an Over at a higher
        # line cannot coexist with an Under at a lower one.
        if a.bet_type in TOTAL_TYPES and b.bet_type in TOTAL_TYPES:
            if a.team_name == b.team_name:
                a_over, b_over = is_over(a), is_over(b)
                if (
                    a_over is not None and b_over is not None
                    and a_over != b_over
                    and a.line is not None and b.line is not None
                ):
                    over_line = a.line if a_over else b.line
                    under_line = b.line if a_over else a.line
                    if over_line >= under_line:
                        return True

        # Two different named winners of the same game.
        RESULT_TYPES = {
            BetType.MONEYLINE, BetType.THREE_WAY, BetType.GAME_RESULT_90,
        }
        if a.bet_type in RESULT_TYPES and b.bet_type in RESULT_TYPES:
            if a.pick != b.pick:
                return True

        # BTTS cannot be both yes and no; nor can it sit against a clean sheet.
        if a.bet_type == BetType.BOTH_TEAMS_SCORE and b.bet_type == BetType.CLEAN_SHEET:
            return "yes" in a.pick.lower()
        if b.bet_type == BetType.BOTH_TEAMS_SCORE and a.bet_type == BetType.CLEAN_SHEET:
            return "yes" in b.pick.lower()

        return False

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

        legs = self._drop_contradictions(list(by_type.values()))[:num_legs]

        if len(legs) < 2:
            return ParlayPrediction(legs=legs, reasoning="Need at least 2 different bet types for SGP.", parlay_type="sgp")

        parlay = self._create_parlay(legs)

        # Same-game legs are not independent, and the old code only adjusted
        # the payout for that — it cut the odds 15% while still multiplying
        # the leg probabilities as though the legs were unrelated. That
        # understates the true chance of the ticket landing (legs like "home
        # win" and "home -1.5" move together), so the stake advice built on it
        # was wrong in both directions at once.
        #
        # Correlated legs are handled by shrinking the product toward the
        # weakest leg: independence is the floor, and a perfectly correlated
        # ticket can be no more likely than its least likely leg.
        independent = parlay.combined_confidence / 100
        weakest = min(leg.probability for leg in legs)
        correlation = 0.35  # same game, mixed market types
        adjusted = independent + correlation * (weakest - independent)
        parlay.combined_confidence = round(100 * adjusted, 2)

        # Books price that correlation into the payout as well.
        parlay.combined_odds = self._discount_odds(parlay.combined_odds, 0.85)
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
        filtered = self._filter_best_per_event(self._bettable(predictions))
        filtered = [p for p in filtered if p.confidence >= self.min_confidence]
        filtered.sort(key=lambda p: p.confidence, reverse=True)

        picks = filtered[:num_picks]
        if len(picks) < combo_size:
            return []

        combos = list(itertools.combinations(picks, combo_size))
        parlays = []
        # A round robin is every one of these tickets bet together, so the
        # stake advice has to be split across them. Sizing each ticket as if
        # it were the only bet on the slip — the old behaviour — multiplied
        # the real outlay by the number of combinations.
        per_ticket_divisor = max(1, len(combos))
        for combo in combos:
            parlay = self._create_parlay(list(combo))
            parlay.parlay_type = "round_robin"
            advice = self.calculate_bankroll_advice(parlay)
            parlay.recommended_stake = round(
                advice.recommended_stake / per_ticket_divisor, 2
            )
            parlay.reasoning = (
                f"🔄 ROUND ROBIN ({combo_size} of {len(picks)}) — "
                f"{len(combos)} tickets on this slip; stake shown is per "
                f"ticket, so the total outlay is "
                f"{len(combos)} × ${parlay.recommended_stake:.2f} = "
                f"${parlay.recommended_stake * len(combos):.2f}.\n"
                + parlay.reasoning
            )
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

        # Teasers only exist in the high-scoring, point-based sports. Buying
        # "6 points" onto a soccer or hockey goal line means moving it by six
        # goals, which is not a bet any book offers — and the old code applied
        # it to every sport, producing 97%-confidence tickets off goal lines.
        TEASER_SPORTS = {Sport.AMERICAN_FOOTBALL, Sport.BASKETBALL}

        teaser_preds = [
            p for p in predictions
            if p.bet_type in TEASER_TYPES
            and p.line is not None
            and p.event.tournament.sport in TEASER_SPORTS
        ]
        if not teaser_preds:
            return ParlayPrediction(
                legs=[],
                reasoning=(
                    "Teasers apply to American football and basketball only — "
                    "no qualifying spread or total legs on the board."
                ),
                parlay_type="teaser",
            )

        # Teasing is only worth anything near a coin-flip line — buying six
        # points onto a market already at 94% adds nothing and costs payout.
        # Sorting by confidence picked exactly those useless legs, so take the
        # lines closest to even money instead.
        teaser_preds = [p for p in teaser_preds if 0.40 <= p.probability <= 0.70]
        filtered = self._filter_best_per_event(teaser_preds)
        filtered.sort(key=lambda p: abs(p.probability - 0.5))

        legs = filtered[:num_legs]
        if len(legs) < 2:
            return ParlayPrediction(
                legs=[], reasoning="Need at least 2 spread/total legs for a teaser.", parlay_type="teaser"
            )

        # Move each leg's line by teaser_points in the bettor's favour, and
        # re-price it off the game model at the new line.
        #
        # The old version left `line` and `pick` untouched and simply added
        # `teaser_points * 2.5` percentage points to the confidence. So a
        # displayed teaser leg still showed the untweaked number, the gain was
        # the same 15 points whether the sport was the NFL or the NHL, and no
        # actual line was ever bought. Here the line really moves and the
        # probability comes from the model at that line.
        analyzer = StatisticalAnalyzer()
        adjusted_legs = []
        for leg in legs:
            model = analyzer.build_model(leg.event)
            old_line = leg.line if leg.line is not None else 0.0
            is_total = leg.bet_type in (
                BetType.OVER_UNDER, BetType.ALTERNATE_TOTAL
            )
            wants_over = "over" in leg.pick.lower()

            if is_total:
                # Teasing a total moves the line away from the side you took.
                new_line = (
                    old_line - teaser_points if wants_over
                    else old_line + teaser_points
                )
                if model is not None:
                    ou = model.over_under(new_line)
                    new_prob = ou["over"] if wants_over else ou["under"]
                else:
                    new_prob = leg.probability
            else:
                # Teasing a handicap adds points to the side you took.
                new_line = old_line + teaser_points
                if model is not None:
                    is_home = leg.team_name == leg.event.home_team.name
                    hc = model.handicap(new_line if is_home else -new_line)
                    new_prob = hc["home"] if is_home else hc["away"]
                else:
                    new_prob = leg.probability

            new_prob = min(0.99, max(0.01, new_prob))
            new_pick = re.sub(
                r"[+-]?\d+(?:\.\d+)?",
                f"{new_line:+g}" if not is_total else f"{new_line:g}",
                leg.pick,
                count=1,
            )

            adjusted_legs.append(Prediction(
                event=leg.event,
                bet_type=leg.bet_type,
                pick=f"{new_pick} (teased {teaser_points:+g})",
                confidence=round(new_prob * 100, 1),
                probability=new_prob,
                odds=self._discount_odds(leg.odds, 0.65),
                value_rating=0.0,      # Re-priced by us, so no claimed edge
                reasoning=(
                    f"Teased {teaser_points:+g} from {old_line:g} to "
                    f"{new_line:g}; re-priced from the game model at the new "
                    f"line.\n{leg.reasoning}"
                ),
                factors=leg.factors,
                line=new_line,
                american_odds=decimal_to_american(
                    self._discount_odds(leg.odds, 0.65)
                ),
                market_display=(
                    f"TEASER {teaser_points:+g}pts — {leg.market_display}"
                ),
                team_name=leg.team_name,
                push_note=leg.push_note,
                price_source="model",
            ))

        parlay = self._create_parlay(adjusted_legs)
        # Discount the profit, not the whole price — see _discount_odds.
        parlay.combined_odds = self._discount_odds(parlay.combined_odds, 0.55)
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
        filtered = self._filter_best_per_event(self._bettable(predictions))
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
        parlay.combined_odds = self._discount_odds(
            parlay.combined_odds, insurance_factor
        )

        # P(at most `miss_allowed` legs lose), computed exactly.
        #
        # The old code described this calculation in a comment and then did
        # not do it — it multiplied the all-legs-win probability by
        # (1 + 0.3 * misses), an arbitrary factor, and left the leg
        # probabilities it had gathered unused. Legs have different
        # probabilities, so this is a Poisson-binomial: walk the legs and keep
        # a running distribution over how many have lost so far.
        probs = [min(0.999, max(0.001, leg.probability)) for leg in legs]
        dist = [1.0]  # dist[k] = P(exactly k misses so far)
        for p in probs:
            nxt = [0.0] * (len(dist) + 1)
            for k, acc in enumerate(dist):
                nxt[k] += acc * p              # this leg wins
                nxt[k + 1] += acc * (1.0 - p)  # this leg misses
            dist = nxt
        parlay.combined_confidence = round(
            100 * sum(dist[: miss_allowed + 1]), 2
        )

        parlay.parlay_type = "flex"
        parlay.flex_miss_allowed = miss_allowed
        miss_label = f"{miss_allowed} miss{'es' if miss_allowed > 1 else ''}"
        parlay.reasoning = f"💪 FLEX PARLAY — Win even with {miss_label}! (Reduced payout)\n" + parlay.reasoning

        advice = self.calculate_bankroll_advice(parlay)
        parlay.recommended_stake = advice.recommended_stake
        return parlay
