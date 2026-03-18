"""
AI Prediction Agent — The brain that combines statistics with LLM reasoning.

Orchestrates data fetching, statistical analysis, and AI-powered insights
to generate premium bet predictions with detailed market breakdowns.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

from loguru import logger

from src.config import settings
from src.models import (
    Sport, BetType, MatchEvent, Prediction, ParlayPrediction, SPORT_EMOJIS,
)
from src.sofascore_client import SofaScoreClient
from src.analyzer import StatisticalAnalyzer
from src.parlay_optimizer import ParlayOptimizer

# LLM integration (optional — works without it using pure stats)
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = bool(settings.openai_api_key)
except ImportError:
    HAS_OPENAI = False


class PredictionAgent:
    """Premium AI Prediction Agent."""

    def __init__(self, bankroll: float = 0):
        self.client = SofaScoreClient()
        self.analyzer = StatisticalAnalyzer()
        self.optimizer = ParlayOptimizer(bankroll=bankroll or settings.default_bankroll)
        self._openai = None
        if HAS_OPENAI:
            self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def close(self):
        await self.client.close()

    # ── Core: Get Today's Predictions ────────────────────────────────────

    async def get_todays_predictions(
        self,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
        min_confidence: float = 60,
    ) -> dict[Sport, list[Prediction]]:
        """
        Get predictions for all games today across selected sports.

        Returns a dict of Sport -> list of Predictions, sorted by confidence.
        """
        if sports is None:
            sports = [
                Sport.SOCCER, Sport.BASKETBALL, Sport.TENNIS,
                Sport.BASEBALL, Sport.AMERICAN_FOOTBALL, Sport.VOLLEYBALL,
                Sport.HOCKEY,
            ]

        all_predictions: dict[Sport, list[Prediction]] = {}

        for sport in sports:
            logger.info(f"📊 Analyzing {sport.value} events...")
            try:
                events = await self.client.get_scheduled_events(sport, target_date)

                # Filter to upcoming games only
                events = [
                    e for e in events
                    if e.status.value == "not_started"
                ]

                # Limit to top leagues / prioritize important matches
                events.sort(key=lambda e: e.tournament.priority, reverse=True)
                events = events[:30]  # Cap per sport for performance

                predictions = []
                for event in events:
                    try:
                        # Enrich with full stats
                        enriched = await self.client.enrich_event(event)
                        # Generate statistical predictions
                        preds = self.analyzer.generate_predictions(enriched)
                        predictions.extend(preds)
                    except Exception as e:
                        logger.warning(
                            f"Could not analyze {event.home_team.name} vs "
                            f"{event.away_team.name}: {e}"
                        )

                # Filter by confidence
                predictions = [
                    p for p in predictions if p.confidence >= min_confidence
                ]
                predictions.sort(key=lambda p: p.confidence, reverse=True)

                all_predictions[sport] = predictions
                logger.info(
                    f"✅ {sport.value}: {len(predictions)} predictions "
                    f"above {min_confidence}%"
                )
            except Exception as e:
                logger.error(f"❌ Failed to analyze {sport.value}: {e}")
                all_predictions[sport] = []

        return all_predictions

    # ── Parlay Builder ───────────────────────────────────────────────────

    async def build_parlay(
        self,
        num_legs: int = 6,
        sports: Optional[list[Sport]] = None,
        strategy: str = "balanced",
        target_date: Optional[date] = None,
    ) -> ParlayPrediction:
        """
        Build the best N-leg parlay across all sports.

        This is the premium feature: it finds the highest probability
        combination of bets for a parlay slip.
        """
        all_preds = await self.get_todays_predictions(sports, target_date)

        # Flatten all predictions
        flat = []
        for sport_preds in all_preds.values():
            flat.extend(sport_preds)

        if not flat:
            return ParlayPrediction(
                legs=[], reasoning="No games with sufficient data found for today."
            )

        parlay = self.optimizer.build_parlay(flat, num_legs, strategy)

        # Enhance with AI reasoning if available
        if self._openai and parlay.legs:
            parlay = await self._enhance_with_ai(parlay)

        return parlay

    async def build_multiple_parlays(
        self,
        num_legs: int = 6,
        count: int = 3,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
    ) -> list[ParlayPrediction]:
        """Generate multiple parlay options with different strategies."""
        all_preds = await self.get_todays_predictions(sports, target_date)
        flat = []
        for sport_preds in all_preds.values():
            flat.extend(sport_preds)

        if not flat:
            return []

        parlays = self.optimizer.build_multiple_parlays(flat, num_legs, count)

        if self._openai:
            enhanced = []
            for p in parlays:
                enhanced.append(await self._enhance_with_ai(p))
            return enhanced

        return parlays

    # ── Hard Rock Bet Parlay Types ───────────────────────────────────────

    async def build_sgp(
        self,
        event_id: int,
        num_legs: int = 4,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
    ) -> ParlayPrediction:
        """Build a Same Game Parlay for a specific event."""
        all_preds = await self.get_todays_predictions(sports, target_date, min_confidence=20)
        flat = [p for preds in all_preds.values() for p in preds]
        return self.optimizer.build_sgp(flat, event_id, num_legs)

    async def build_all_sgps(
        self,
        num_legs: int = 4,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
    ) -> list[ParlayPrediction]:
        """Build SGPs for all available games."""
        all_preds = await self.get_todays_predictions(sports, target_date, min_confidence=20)
        flat = [p for preds in all_preds.values() for p in preds]
        return self.optimizer.build_sgp_for_all_events(flat, num_legs)

    async def build_round_robin(
        self,
        num_picks: int = 5,
        combo_size: int = 3,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
    ) -> list[ParlayPrediction]:
        """Build Round Robin parlays — all combos from top picks."""
        all_preds = await self.get_todays_predictions(sports, target_date)
        flat = [p for preds in all_preds.values() for p in preds]
        return self.optimizer.build_round_robin(flat, num_picks, combo_size)

    async def build_teaser(
        self,
        num_legs: int = 3,
        teaser_points: float = 6.0,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
    ) -> ParlayPrediction:
        """Build a Teaser — spreads/totals adjusted in your favor."""
        all_preds = await self.get_todays_predictions(sports, target_date)
        flat = [p for preds in all_preds.values() for p in preds]
        return self.optimizer.build_teaser(flat, num_legs, teaser_points)

    async def build_flex_parlay(
        self,
        num_legs: int = 5,
        miss_allowed: int = 1,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
    ) -> ParlayPrediction:
        """Build a Flex Parlay — still win even if some legs lose."""
        all_preds = await self.get_todays_predictions(sports, target_date)
        flat = [p for preds in all_preds.values() for p in preds]
        return self.optimizer.build_flex_parlay(flat, num_legs, miss_allowed)

    # ── Single Game Analysis ─────────────────────────────────────────────

    async def analyze_single_game(
        self, event_id: int
    ) -> list[Prediction]:
        """Deep analysis of a single game with all bet types."""
        details = await self.client.get_event_details(event_id)
        event_data = details.get("event", details)

        sport_slug = (
            event_data.get("tournament", {})
            .get("uniqueTournament", {})
            .get("category", {})
            .get("sport", {})
            .get("slug", "football")
        )

        sport = Sport.SOCCER
        for s in Sport:
            if s.value == sport_slug:
                sport = s
                break

        event = self.client._parse_event(event_data, sport)
        enriched = await self.client.enrich_event(event)
        predictions = self.analyzer.generate_predictions(enriched)

        if self._openai and predictions:
            predictions = await self._enhance_predictions_with_ai(
                enriched, predictions
            )

        return predictions

    # ── AI Enhancement ───────────────────────────────────────────────────

    async def _enhance_with_ai(
        self, parlay: ParlayPrediction
    ) -> ParlayPrediction:
        """Use LLM to validate and enhance parlay reasoning."""
        if not self._openai:
            return parlay

        try:
            legs_info = []
            for leg in parlay.legs:
                legs_info.append({
                    "match": f"{leg.event.home_team.name} vs {leg.event.away_team.name}",
                    "tournament": leg.event.tournament.name,
                    "sport": leg.event.tournament.sport.value,
                    "pick": leg.pick,
                    "confidence": leg.confidence,
                    "reasoning": leg.reasoning[:500],
                })

            prompt = f"""You are a professional sports betting analyst. Review this parlay and provide:
1. Overall assessment (1-2 sentences)
2. Any concerns or risks you see
3. Which leg you think is the weakest and why
4. A confidence rating adjustment if needed

Parlay ({len(parlay.legs)} legs):
{json.dumps(legs_info, indent=2)}

Combined confidence: {parlay.combined_confidence:.1f}%
Combined odds: {parlay.combined_odds:.2f}x
Strategy used: {parlay.risk_level}

Be concise and analytical. Focus on value and risk."""

            response = await self._openai.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert sports betting analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )

            ai_analysis = response.choices[0].message.content
            parlay.reasoning = f"{parlay.reasoning}\n\n🤖 AI Analysis:\n{ai_analysis}"

        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")

        return parlay

    async def _enhance_predictions_with_ai(
        self, event: MatchEvent, predictions: list[Prediction]
    ) -> list[Prediction]:
        """Use LLM to add deeper insights to predictions."""
        if not self._openai:
            return predictions

        try:
            match_info = {
                "home": event.home_team.name,
                "away": event.away_team.name,
                "tournament": event.tournament.name,
                "country": event.tournament.country,
                "sport": event.tournament.sport.value,
            }

            if event.home_stats:
                match_info["home_form"] = event.home_stats.form_string
                match_info["home_position"] = event.home_stats.league_position
                match_info["home_goals_avg"] = event.home_stats.avg_goals_scored
            if event.away_stats:
                match_info["away_form"] = event.away_stats.form_string
                match_info["away_position"] = event.away_stats.league_position
                match_info["away_goals_avg"] = event.away_stats.avg_goals_scored
            if event.h2h:
                match_info["h2h"] = f"{event.h2h.team1_wins}-{event.h2h.draws}-{event.h2h.team2_wins}"

            preds_info = [
                {"pick": p.pick, "confidence": p.confidence, "type": p.bet_type.value}
                for p in predictions[:5]
            ]

            prompt = f"""Analyze this match as a professional sports bettor:

Match: {json.dumps(match_info, indent=2)}

Our statistical model suggests:
{json.dumps(preds_info, indent=2)}

Provide a brief expert analysis (3-4 sentences) covering:
- Key factors that will decide this match
- Any hidden angles the stats might miss (motivation, derby, schedule, etc.)
- Your confidence in the top pick"""

            response = await self._openai.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert sports analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.3,
            )

            ai_insight = response.choices[0].message.content
            if predictions:
                predictions[0].reasoning += f"\n\n🤖 AI Insight:\n{ai_insight}"

        except Exception as e:
            logger.warning(f"AI prediction enhancement failed: {e}")

        return predictions

    # ── Value Bets ───────────────────────────────────────────────────────

    async def find_value_bets(
        self,
        sports: Optional[list[Sport]] = None,
        min_value: float = 0.05,
        target_date: Optional[date] = None,
    ) -> list[Prediction]:
        """
        Find value bets where our estimated probability exceeds
        the implied probability from bookmaker odds.

        A value bet has positive expected value.
        """
        all_preds = await self.get_todays_predictions(sports, target_date)
        value_bets = []

        for sport_preds in all_preds.values():
            for pred in sport_preds:
                if pred.value_rating >= min_value:
                    value_bets.append(pred)

        value_bets.sort(key=lambda p: p.value_rating, reverse=True)
        return value_bets

    # ── Summary / Report ─────────────────────────────────────────────────

    @staticmethod
    def group_predictions_by_match(
        predictions: dict[Sport, list[Prediction]]
    ) -> list[dict]:
        """
        Group all predictions by match, so each match card shows all
        available markets (moneyline, spread, O/U, BTTS, corners, etc.).
        """
        matches = defaultdict(lambda: {"event": None, "predictions": [], "sport": None})
        for sport, preds in predictions.items():
            for pred in preds:
                key = pred.event.id
                if matches[key]["event"] is None:
                    matches[key]["event"] = pred.event
                    matches[key]["sport"] = sport
                matches[key]["predictions"].append(pred)

        # Sort matches by start time
        result = sorted(matches.values(), key=lambda m: m["event"].start_time)
        return result

    async def generate_daily_report(
        self,
        sports: Optional[list[Sport]] = None,
        target_date: Optional[date] = None,
    ) -> str:
        """Generate a comprehensive daily betting report with all markets."""
        d = target_date or date.today()
        all_preds = await self.get_todays_predictions(sports, d)

        lines = []
        lines.append("=" * 70)
        lines.append(f"  🏆 PREMIUM BET PREDICTION REPORT — {d.strftime('%A, %B %d, %Y')}")
        lines.append(f"  ⏰ Generated: {datetime.now().strftime('%I:%M %p')}")
        lines.append("=" * 70)
        lines.append("")

        total_picks = 0
        total_matches = 0

        for sport, preds in all_preds.items():
            if not preds:
                continue
            emoji = SPORT_EMOJIS.get(sport, "🏆")
            lines.append(f"━━━ {emoji} {sport.value.upper().replace('-', ' ')} ━━━")
            lines.append("")

            # Group by match
            grouped = self.group_predictions_by_match({sport: preds})

            for match in grouped:
                ev = match["event"]
                match_preds = match["predictions"]
                total_matches += 1

                lines.append(f"  🏟️  {ev.home_team.name} vs {ev.away_team.name}")
                lines.append(f"     📍 {ev.tournament.name} ({ev.tournament.country})")
                lines.append(f"     📅 {ev.start_time.strftime('%B %d, %Y • %I:%M %p')}")
                lines.append("")

                # Group predictions by bet type
                by_type = defaultdict(list)
                for p in match_preds:
                    by_type[p.bet_type].append(p)

                for bt, bt_preds in by_type.items():
                    bt_label = bt.value.upper().replace("_", " ")
                    lines.append(f"     📊 {bt_label}:")
                    for pred in bt_preds[:3]:
                        conf_emoji = "🟢" if pred.confidence >= 75 else "🟡" if pred.confidence >= 65 else "🟠"
                        odds_str = f" | Odds: {pred.odds:.2f}" if pred.odds > 0 else ""
                        am_str = f" ({pred.american_odds})" if pred.american_odds else ""
                        val_str = f" | Value: {pred.value_rating:+.3f}" if pred.value_rating != 0 else ""
                        push_str = f" ⚠️ {pred.push_note}" if pred.push_note else ""
                        lines.append(
                            f"        {conf_emoji} {pred.pick}{am_str}{odds_str}{val_str}{push_str}"
                            f" — {pred.confidence:.0f}% conf"
                        )
                        total_picks += 1
                lines.append("")

        # Parlay recommendation
        lines.append("=" * 70)
        lines.append("  🎯 RECOMMENDED PARLAY (MIX ALL SPORTS)")
        lines.append("=" * 70)

        flat = []
        for preds in all_preds.values():
            flat.extend(preds)

        if flat:
            parlay = self.optimizer.build_parlay(flat, min(6, len(flat)), "balanced")
            if parlay.legs:
                for i, leg in enumerate(parlay.legs, 1):
                    sport_e = SPORT_EMOJIS.get(leg.event.tournament.sport, "🏆")
                    am = f" ({leg.american_odds})" if leg.american_odds else ""
                    lines.append(
                        f"  Leg {i}: {sport_e} {leg.event.home_team.name} vs {leg.event.away_team.name}"
                    )
                    lines.append(
                        f"         ➤ {leg.pick}{am} | {leg.confidence:.0f}% conf"
                    )
                    lines.append(
                        f"         📅 {leg.event.start_time.strftime('%b %d • %I:%M %p')}"
                    )
                lines.append("")
                lines.append(f"  📊 Combined Confidence: {parlay.combined_confidence:.1f}%")
                lines.append(f"  💰 Combined Odds: {parlay.combined_odds:.2f}x")
                lines.append(f"  📈 Expected Value: {parlay.expected_value:+.4f}")
                lines.append(f"  ⚠️  Risk Level: {parlay.risk_level.upper()}")
                lines.append(f"  💵 Recommended Stake: ${parlay.recommended_stake:.2f}")
        else:
            lines.append("  No qualifying picks found for parlay.")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  📊 Total: {total_picks} picks across {total_matches} matches")
        lines.append(f"  ⏰ Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M:%S %p')}")
        lines.append("  ⚠️  Bet responsibly. Past performance ≠ future results.")
        lines.append("=" * 70)

        return "\n".join(lines)
