"""
Prediction engine — turns one model of a game into a full board of markets.

How this works, and why it changed:

The old version priced every market with its own improvised formula. Totals
used a "defensive adjustment" that divided one team's goals-conceded by the
other's, which is dimensionally meaningless and could inflate a projected
total by more than 10x — that is where "Expected total: 5.79 goals" in a
Premier League game came from. Spreads subtracted a flat 0.12 per point of
handicap regardless of sport, so an NBA -7.5 came out at the 5% floor. And
every price was quoted as 1/probability, i.e. a zero-margin fair price, which
made the value detector fire on its own arithmetic.

Now a single `GameModel` (see `markets.py`) is built per game, and every
market is read off that one distribution. The moneyline, the handicap, the
total and the correct-score board therefore always agree with each other.

The eight-factor analysis is kept — it is genuinely useful signal — but it now
does what it is suited to: a *bounded* adjustment to each side's projected
scoring, rather than being normalised straight into a probability.

Nothing is emitted from data we do not have. A market whose inputs are
missing is skipped, not guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.markets import (
    GameModel, ScoreGrid, expected_scoring, profile_for, normal_cdf,
    poisson_pmf, decimal_to_american, american_to_decimal, priced_decimal,
)
from src.models import (
    Sport, BetType, MatchEvent, TeamStats, HeadToHead,
    Prediction, PlayerInfo,
)


@dataclass
class AnalysisWeights:
    """Configurable weights per sport for the factor model."""
    form: float = 0.25
    home_advantage: float = 0.10
    h2h: float = 0.10
    league_position: float = 0.15
    scoring: float = 0.15
    injuries: float = 0.10
    consistency: float = 0.10
    momentum: float = 0.05


SPORT_WEIGHTS: dict[Sport, AnalysisWeights] = {
    Sport.SOCCER: AnalysisWeights(
        form=0.22, home_advantage=0.12, h2h=0.12, league_position=0.15,
        scoring=0.13, injuries=0.12, consistency=0.09, momentum=0.05,
    ),
    Sport.BASKETBALL: AnalysisWeights(
        form=0.25, home_advantage=0.08, h2h=0.08, league_position=0.18,
        scoring=0.18, injuries=0.12, consistency=0.06, momentum=0.05,
    ),
    Sport.TENNIS: AnalysisWeights(
        form=0.30, home_advantage=0.03, h2h=0.18, league_position=0.10,
        scoring=0.10, injuries=0.15, consistency=0.07, momentum=0.07,
    ),
    Sport.BASEBALL: AnalysisWeights(
        form=0.20, home_advantage=0.06, h2h=0.10, league_position=0.15,
        scoring=0.20, injuries=0.12, consistency=0.10, momentum=0.07,
    ),
    Sport.AMERICAN_FOOTBALL: AnalysisWeights(
        form=0.22, home_advantage=0.10, h2h=0.10, league_position=0.15,
        scoring=0.15, injuries=0.15, consistency=0.08, momentum=0.05,
    ),
    Sport.VOLLEYBALL: AnalysisWeights(
        form=0.25, home_advantage=0.08, h2h=0.12, league_position=0.15,
        scoring=0.15, injuries=0.10, consistency=0.08, momentum=0.07,
    ),
    Sport.HOCKEY: AnalysisWeights(
        form=0.22, home_advantage=0.08, h2h=0.10, league_position=0.15,
        scoring=0.18, injuries=0.12, consistency=0.08, momentum=0.07,
    ),
}

# Share of a game's scoring that lands in the first half/period.
FIRST_HALF_SHARE: dict[Sport, float] = {
    Sport.SOCCER: 0.45,
    Sport.AMERICAN_FOOTBALL: 0.48,
    Sport.BASKETBALL: 0.50,
    Sport.HOCKEY: 0.33,
}

# Positions that plausibly score, per sport, for player markets.
_SCORING_POSITIONS: dict[Sport, tuple[str, ...]] = {
    Sport.SOCCER: ("forward", "attacker", "striker", "winger", "midfielder"),
    Sport.BASKETBALL: ("guard", "forward", "center"),
    Sport.HOCKEY: ("center", "left wing", "right wing", "forward"),
}


class StatisticalAnalyzer:
    """Multi-sport prediction engine."""

    def __init__(self):
        self.default_weights = AnalysisWeights()

    # ── Model construction ───────────────────────────────────────────────

    def build_model(self, event: MatchEvent) -> Optional[GameModel]:
        """Build the single distribution every market is derived from."""
        home, away = event.home_stats, event.away_stats
        if not home or not away:
            return None

        sport = event.tournament.sport
        factors = self.analyze_event(event)

        # The factor model contributes a bounded nudge, not the probability.
        home_share = sum(f.get("home", 0.0) for f in factors.values())
        away_share = sum(f.get("away", 0.0) for f in factors.values())
        span = home_share + away_share
        tilt = (home_share / span) - 0.5 if span > 0 else 0.0
        home_adj = 1.0 + tilt * 0.30
        away_adj = 1.0 - tilt * 0.30

        # Venue-neutral rates on purpose. Feeding in home-only and away-only
        # splits looks more precise but double-counts home advantage — those
        # splits already contain it, and `expected_scoring` then adds the
        # sport's home edge on top. It also leans on a half-size sample, so
        # one lopsided home result swings the projection. The model applies
        # the venue effect itself.
        home_scored = home.avg_goals_scored
        home_conceded = home.avg_goals_conceded
        away_scored = away.avg_goals_scored
        away_conceded = away.avg_goals_conceded

        baseline = float(event.espn_data.get("league_baseline", 0.0) or 0.0)

        scoring = expected_scoring(
            sport,
            home_scored, home_conceded,
            away_scored, away_conceded,
            baseline, home_adj, away_adj,
            home_games=home.games_played,
            away_games=away.games_played,
        )
        model = GameModel(sport, scoring)
        model.factors = factors  # type: ignore[attr-defined]
        return model

    # ── Factor analysis (retained: adjustment + reasoning) ───────────────

    def analyze_event(self, event: MatchEvent) -> dict:
        sport = event.tournament.sport
        w = SPORT_WEIGHTS.get(sport, self.default_weights)
        return {
            "form": self._analyze_form(event.home_stats, event.away_stats, w.form),
            "home_advantage": self._analyze_home_advantage(
                event.home_stats, event.away_stats, sport, w.home_advantage
            ),
            "h2h": self._analyze_h2h(event.h2h, w.h2h),
            "league_position": self._analyze_league_position(
                event.home_stats, event.away_stats, w.league_position
            ),
            "scoring": self._analyze_scoring(
                event.home_stats, event.away_stats, w.scoring
            ),
            "injuries": self._analyze_injuries(
                event.home_injuries, event.away_injuries, w.injuries
            ),
            "consistency": self._analyze_consistency(
                event.home_stats, event.away_stats, w.consistency
            ),
            "momentum": self._analyze_momentum(
                event.home_stats, event.away_stats, w.momentum
            ),
        }

    def calculate_probabilities(self, event: MatchEvent) -> dict[str, float]:
        """Win / draw / loss probabilities from the game model."""
        model = self.build_model(event)
        if model is None:
            if profile_for(event.tournament.sport).has_draw:
                return {"home": 0.34, "draw": 0.32, "away": 0.34}
            return {"home": 0.5, "draw": 0.0, "away": 0.5}
        return model.result_probabilities()

    def calculate_over_under(
        self, event: MatchEvent, line: float = 2.5
    ) -> dict[str, float]:
        model = self.build_model(event)
        if model is None:
            return {"over": 0.5, "under": 0.5}
        return model.over_under(line)

    def calculate_btts(self, event: MatchEvent) -> dict[str, float]:
        model = self.build_model(event)
        if model is None or model.grid is None:
            return {"yes": 0.5, "no": 0.5}
        yes = model.grid.btts()
        return {"yes": round(yes, 4), "no": round(1 - yes, 4)}

    # ── Prediction emitter ───────────────────────────────────────────────

    def _emit(
        self,
        event: MatchEvent,
        bet_type: BetType,
        pick: str,
        probability: float,
        market_display: str,
        reasoning: str,
        *,
        line: Optional[float] = None,
        team_name: str = "",
        push_note: str = "",
        push_probability: float = 0.0,
        book_odds: float = 0.0,
        factors: Optional[dict] = None,
    ) -> Optional[Prediction]:
        """Build one Prediction, priced honestly.

        When a real bookmaker price exists it is used and the expected value
        is computed against it. Otherwise the price shown is our own
        margin-adjusted fair price, flagged as such, and no value is claimed —
        comparing a model price with itself is how the old build manufactured
        edges that were never there.
        """
        probability = min(0.995, max(0.005, probability))

        if book_odds and book_odds > 1.0:
            odds = round(book_odds, 3)
            price_source = "book"
            value_rating = round(probability * odds - 1.0, 4)

            # A closing price is the single best predictor there is. When we
            # disagree with it by this much, the overwhelmingly likely
            # explanation is that our inputs are thin — early-season samples,
            # a roster change we cannot see — not that a real edge is sitting
            # unclaimed. Say so next to the number instead of letting a big
            # EV read as a recommendation.
            if value_rating > 0.15:
                implied = 1.0 / odds
                reasoning += (
                    f"\n\n⚠️ Model says {probability:.0%}, market implies "
                    f"{implied:.0%}. A gap this wide usually means our inputs "
                    "are incomplete rather than that the market is wrong — "
                    "treat it as a flag to look closer, not as an edge."
                )
        else:
            odds = priced_decimal(probability)
            price_source = "model"
            value_rating = 0.0

        return Prediction(
            event=event,
            bet_type=bet_type,
            pick=pick,
            confidence=round(probability * 100, 1),
            probability=round(probability, 4),
            odds=odds,
            value_rating=value_rating,
            reasoning=reasoning,
            factors=factors or {},
            line=line,
            american_odds=decimal_to_american(odds),
            market_display=market_display,
            team_name=team_name,
            push_note=push_note,
            price_source=price_source,
            push_probability=round(push_probability, 4),
        )

    # ── Market generation ────────────────────────────────────────────────

    def generate_predictions(self, event: MatchEvent) -> list[Prediction]:
        """Generate every market this game supports, from one model."""
        model = self.build_model(event)
        if model is None:
            logger.debug(
                f"No stats for {event.home_team.name} vs "
                f"{event.away_team.name} — skipping"
            )
            return []

        sport = event.tournament.sport
        prof = profile_for(sport)
        preds: list[Prediction] = []
        home, away = event.home_team.name, event.away_team.name
        probs = model.result_probabilities()
        grid = model.grid
        factors = getattr(model, "factors", {})
        unit = prof.unit
        exp_total = model.expected_total()

        basis = (
            f"Model: {home} {model.home_lambda:.2f} vs {away} "
            f"{model.away_lambda:.2f} ({exp_total:.2f} total {unit.lower()})."
        )
        if not model.complete:
            basis += " Limited data — treat with caution."
        # Carry the model's own caveats (thin samples, missing baseline) onto
        # the match so they reach the page rather than staying in the engine.
        for note in model.scoring.notes:
            if note not in event.data_notes:
                event.data_notes.append(note)

        def add(p: Optional[Prediction]) -> None:
            if p is not None:
                preds.append(p)

        # ── 1. Moneyline / 1X2 ──────────────────────────────────────────
        book = {
            "home": event.home_odds,
            "draw": event.draw_odds,
            "away": event.away_odds,
        }
        names = {"home": home, "draw": "Draw", "away": away}
        for side in ("home", "draw", "away"):
            p = probs.get(side, 0.0)
            if p <= 0.0:
                continue
            add(self._emit(
                event, BetType.MONEYLINE,
                f"🏆 {names[side]} Win" if side != "draw" else "🤝 Draw", p,
                market_display=f"Moneyline — {names[side]}",
                reasoning=self._build_reasoning(event, factors, side, basis),
                team_name=names[side] if side != "draw" else "",
                book_odds=book.get(side, 0.0) if event.has_book_odds else 0.0,
                factors=factors,
            ))

        # ── 2. Double chance / draw no bet (draw sports only) ───────────
        if prof.has_draw and probs.get("draw", 0) > 0:
            for key, p, label in (
                ("1X", probs["home"] + probs["draw"], f"🛡️ {home} or Draw"),
                ("X2", probs["draw"] + probs["away"], f"🛡️ {away} or Draw"),
                ("12", probs["home"] + probs["away"], f"🛡️ {home} or {away}"),
            ):
                add(self._emit(
                    event, BetType.DOUBLE_CHANCE, label, p,
                    market_display=f"Double Chance — {key}",
                    reasoning=f"Two of the three outcomes win this bet. {basis}",
                ))

            two_way = probs["home"] + probs["away"]
            if two_way > 0:
                for side, team in (("home", home), ("away", away)):
                    add(self._emit(
                        event, BetType.DRAW_NO_BET,
                        f"🔄 {team} (Draw No Bet)", probs[side] / two_way,
                        market_display=f"Draw No Bet — {team}",
                        reasoning=f"Stake returned if the match is drawn. {basis}",
                        team_name=team,
                        push_note="Stake refunded on a draw",
                        push_probability=probs["draw"],
                    ))

        # ── 3. Handicaps: main line plus alternates ─────────────────────
        # Real spread/total prices, where the feed carried them. These make
        # the value numbers mean something: our probability against a price
        # actually being offered, rather than against our own fair price.
        ed = event.espn_data
        book_spread = float(ed.get("spread") or 0.0)
        book_spread_price = {
            "home": float(ed.get("spread_home_decimal") or 0.0),
            "away": float(ed.get("spread_away_decimal") or 0.0),
        }
        book_total = float(ed.get("total_line") or ed.get("overUnder") or 0.0)
        book_total_price = {
            "over": float(ed.get("total_over_decimal") or 0.0),
            "under": float(ed.get("total_under_decimal") or 0.0),
        }

        main_hc = self._main_handicap(sport, model, book_spread)
        for line in self._handicap_lines(sport, model, book_spread):
            hc = model.handicap(line)
            push_p = hc.get("push", 0.0)
            for side, team in (("home", home), ("away", away)):
                p = hc[side]
                if p < 0.06 or p > 0.94:
                    continue
                shown = line if side == "home" else -line
                is_main = line == main_hc
                bt = BetType.SPREAD if is_main else BetType.ALTERNATE_SPREAD
                add(self._emit(
                    event, bt, f"📊 {team} {shown:+g}", p,
                    market_display=f"Handicap {shown:+g} — {team}",
                    reasoning=(
                        f"{team} with a {shown:+g} {unit.lower()} handicap. "
                        f"{basis}"
                    ),
                    line=shown,
                    team_name=team,
                    push_note=(
                        "Push if the margin lands exactly on the line"
                        if push_p > 0 else ""
                    ),
                    push_probability=push_p,
                    # Only the book's own line carries the book's price.
                    book_odds=(
                        book_spread_price[side] if is_main and book_spread else 0.0
                    ),
                ))

        # Asian (quarter) handicaps, grid sports only.
        if grid is not None:
            for line in (-1.25, -0.75, -0.25, 0.25, 0.75, 1.25):
                hc = model.handicap(line)
                for side, team in (("home", home), ("away", away)):
                    p = hc[side]
                    if p < 0.15 or p > 0.85:
                        continue
                    shown = line if side == "home" else -line
                    add(self._emit(
                        event, BetType.ASIAN_HANDICAP,
                        f"⚖️ {team} {shown:+g} (Asian)", p,
                        market_display=f"Asian Handicap {shown:+g} — {team}",
                        reasoning=(
                            "Quarter line: the stake splits across the two "
                            f"neighbouring handicaps. {basis}"
                        ),
                        line=shown,
                        team_name=team,
                        push_note="Half stake refunded if the margin splits the line",
                    ))

        # ── 4. Totals: main line plus alternates ────────────────────────
        main_total, alt_totals = self._total_lines(sport, exp_total, book_total)
        for line in [main_total] + alt_totals:
            ou = model.over_under(line)
            for side in ("over", "under"):
                p = ou[side]
                if p < 0.06 or p > 0.94:
                    continue
                is_main = line == main_total
                bt = (
                    BetType.OVER_UNDER if is_main else BetType.ALTERNATE_TOTAL
                )
                add(self._emit(
                    event, bt,
                    f"{'⬆️ Over' if side == 'over' else '⬇️ Under'} "
                    f"{line:g} {unit}", p,
                    market_display=f"Total {unit} — {side.title()} {line:g}",
                    reasoning=(
                        f"Projected total {exp_total:.2f} {unit.lower()}. {basis}"
                    ),
                    line=line,
                    push_probability=ou.get("push", 0.0),
                    push_note=(
                        "Push if the total lands exactly on the line"
                        if ou.get("push", 0.0) > 0 else ""
                    ),
                    factors={"over_under": ou},
                    book_odds=(
                        book_total_price[side] if is_main and book_total else 0.0
                    ),
                ))

        # ── 5. Team totals ──────────────────────────────────────────────
        for is_home, team, lam in (
            (True, home, model.home_lambda),
            (False, away, model.away_lambda),
        ):
            for line in self._team_total_lines(sport, lam):
                tt = model.team_total(line, is_home)
                for side in ("over", "under"):
                    p = tt[side]
                    if p < 0.06 or p > 0.94:
                        continue
                    add(self._emit(
                        event, BetType.TEAM_TOTAL,
                        f"{'⬆️' if side == 'over' else '⬇️'} {team} "
                        f"{side.title()} {line:g} {unit}", p,
                        market_display=(
                            f"{team} Total {unit} — {side.title()} {line:g}"
                        ),
                        reasoning=(
                            f"{team} projected for {lam:.2f} {unit.lower()}. "
                            f"{basis}"
                        ),
                        line=line,
                        team_name=team,
                    ))

        # ── 6. Odd / even total ─────────────────────────────────────────
        if grid is not None:
            odd = grid.odd_total()
            for label, p in (("Odd", odd), ("Even", 1 - odd)):
                add(self._emit(
                    event, BetType.ODD_EVEN, f"🔢 Total {unit} — {label}", p,
                    market_display=f"Odd/Even Total — {label}",
                    reasoning=f"Parity of the final total. {basis}",
                ))

        # ── 7-13. Remaining market families ─────────────────────────────
        preds.extend(self._margin_markets(event, model, home, away, basis))
        if grid is not None:
            preds.extend(
                self._grid_markets(
                    event, model, home, away, unit, basis, main_total
                )
            )
        preds.extend(self._period_markets(event, model, home, away, unit, basis))
        preds.extend(self._race_markets(event, model, home, away, unit, basis))
        add(self._overtime_market(event, model, basis))
        preds.extend(self._player_markets(event, model, home, away, basis))
        preds.extend(self._corner_markets(event))

        for p in preds:
            p.confidence = min(99.0, max(0.5, p.confidence))
            p.probability = min(0.995, max(0.005, p.probability))
        preds.sort(key=lambda p: p.confidence, reverse=True)
        return preds

    # ── Line selection ───────────────────────────────────────────────────

    def _main_handicap(
        self, sport: Sport, model: GameModel, book_line: float = 0.0
    ) -> float:
        """The main handicap, applied to the home side.

        The bookmaker's own line wins when we have it — ESPN reports `spread`
        from the home team's perspective, the same convention used here — so
        our number is measured against a line that is genuinely on offer.
        """
        if book_line:
            return book_line
        margin = model.home_lambda - model.away_lambda
        line = -round(margin * 2) / 2
        if line == int(line):
            # Keep a half hook so the main line cannot push.
            line += -0.5 if margin > 0 else 0.5
        return line

    def _handicap_lines(
        self, sport: Sport, model: GameModel, book_line: float = 0.0
    ) -> list[float]:
        main = self._main_handicap(sport, model, book_line)
        if sport in (Sport.SOCCER, Sport.HOCKEY):
            offsets = [-2.0, -1.0, 1.0, 2.0]
        elif sport == Sport.BASEBALL:
            offsets = [-1.0, 1.0, 2.0]
        elif sport in (Sport.BASKETBALL, Sport.AMERICAN_FOOTBALL):
            offsets = [-10.0, -6.0, -3.0, 3.0, 6.0, 10.0]
        else:
            offsets = [-4.0, -2.0, 2.0, 4.0]
        cap = self._max_handicap(sport)
        lines = {main} | {main + o for o in offsets}
        return sorted(l for l in lines if 0 < abs(l) <= cap)

    def _max_handicap(self, sport: Sport) -> float:
        return {
            Sport.SOCCER: 4.5, Sport.HOCKEY: 4.5, Sport.BASEBALL: 4.5,
            Sport.BASKETBALL: 24.5, Sport.AMERICAN_FOOTBALL: 24.5,
        }.get(sport, 12.5)

    def _total_lines(
        self, sport: Sport, expected: float, book_line: float = 0.0
    ) -> tuple[float, list[float]]:
        """A main total plus alternates around it.

        When the bookmaker's own total is known it becomes the main line, so
        our probability is compared against the price actually on offer rather
        than against a line nobody is quoting.

        Steps are whole numbers so every line keeps the half hook. Offsetting
        by 0.5 flips the parity and produces lines like "Under 5", which is
        arithmetically the same bet as "Under 4.5" plus a push — two rows that
        look like different markets but are not.
        """
        if sport in (Sport.SOCCER, Sport.HOCKEY, Sport.BASEBALL):
            main = round(expected * 2) / 2
            if main == int(main):
                main += 0.5
            steps = [-2.0, -1.0, 1.0, 2.0]
        else:
            main = round(expected) + 0.5
            step = 5.0 if sport == Sport.BASKETBALL else 3.0
            steps = [-3 * step, -2 * step, -step, step, 2 * step, 3 * step]

        if book_line > 0:
            main = book_line
        return main, [main + s for s in steps if main + s > 0]

    def _team_total_lines(self, sport: Sport, lam: float) -> list[float]:
        if sport in (Sport.SOCCER, Sport.HOCKEY, Sport.BASEBALL):
            base = round(lam * 2) / 2
            if base == int(base):
                base += 0.5
            return [l for l in (base - 1.0, base, base + 1.0) if l > 0]
        base = round(lam) + 0.5
        step = 5.0 if sport == Sport.BASKETBALL else 3.5
        return [l for l in (base - step, base, base + step) if l > 0]

    # ── Market families ──────────────────────────────────────────────────

    def _margin_markets(
        self, event: MatchEvent, model: GameModel,
        home: str, away: str, basis: str,
    ) -> list[Prediction]:
        """Winning-margin bands."""
        out: list[Optional[Prediction]] = []
        grid = model.grid

        if grid is not None:
            for lo, hi in ((1, 1), (2, 2), (3, 5)):
                for team, sign in ((home, 1), (away, -1)):
                    p = sum(grid.margin(sign * m) for m in range(lo, hi + 1))
                    if p < 0.03:
                        continue
                    label = f"by {lo}" if lo == hi else f"by {lo}-{hi}"
                    out.append(self._emit(
                        event, BetType.WINNING_MARGIN,
                        f"📐 {team} to win {label}", p,
                        market_display=f"Winning Margin — {team} {label}",
                        reasoning=f"Margin band from the score model. {basis}",
                        team_name=team,
                    ))
        elif model.normal is not None:
            for lo, hi in ((1, 5), (6, 10), (11, 20)):
                for team, sign in ((home, 1), (away, -1)):
                    if sign > 0:
                        p = model.normal.margin_between(lo - 0.5, hi + 0.5)
                    else:
                        p = model.normal.margin_between(-hi - 0.5, -lo + 0.5)
                    if p < 0.03:
                        continue
                    out.append(self._emit(
                        event, BetType.WINNING_MARGIN,
                        f"📐 {team} to win by {lo}-{hi}", p,
                        market_display=f"Winning Margin — {team} {lo}-{hi}",
                        reasoning=f"Margin band from the model. {basis}",
                        team_name=team,
                    ))
        return [p for p in out if p is not None]

    def _grid_markets(
        self, event: MatchEvent, model: GameModel,
        home: str, away: str, unit: str, basis: str, main_total: float,
    ) -> list[Prediction]:
        """Markets that only make sense on a discrete score grid."""
        grid = model.grid
        assert grid is not None
        sport = event.tournament.sport
        prof = profile_for(sport)
        out: list[Optional[Prediction]] = []

        # Both teams to score
        btts = grid.btts()
        for label, p in (("Yes", btts), ("No", 1 - btts)):
            out.append(self._emit(
                event, BetType.BOTH_TEAMS_SCORE,
                f"{'✅' if label == 'Yes' else '❌'} Both Teams to Score "
                f"— {label}", p,
                market_display=f"Both Teams to Score — {label}",
                reasoning=f"Each side's chance of scoring at least once. {basis}",
            ))

        # Clean sheet / win to nil
        for team, is_home in ((home, True), (away, False)):
            out.append(self._emit(
                event, BetType.CLEAN_SHEET,
                f"🧤 {team} Clean Sheet", grid.clean_sheet(is_home),
                market_display=f"Clean Sheet — {team}",
                reasoning=f"{team} to concede nothing. {basis}",
                team_name=team,
            ))
            wtn = grid.win_to_nil(is_home)
            if wtn > 0.02:
                out.append(self._emit(
                    event, BetType.WIN_TO_NIL,
                    f"🔒 {team} to Win to Nil", wtn,
                    market_display=f"Win to Nil — {team}",
                    reasoning=f"{team} to win without conceding. {basis}",
                    team_name=team,
                ))

        # Correct score — the most likely boards only
        for hg, ag, p in grid.top_scores(12):
            if p < 0.015:
                continue
            out.append(self._emit(
                event, BetType.CORRECT_SCORE,
                f"🎯 {home} {hg}-{ag} {away}", p,
                market_display=f"Correct Score — {hg}-{ag}",
                reasoning=f"Most likely exact scorelines. {basis}",
            ))

        # Exact total
        for total in range(0, 8):
            p = grid._sum(lambda h, a, t=total: h + a == t)
            if p < 0.04:
                continue
            out.append(self._emit(
                event, BetType.EXACT_TOTAL,
                f"🔟 Exactly {total} {unit}", p,
                market_display=f"Exact Total {unit} — {total}",
                reasoning=f"Exact match total. {basis}",
                line=float(total),
            ))

        # Result + total and result + BTTS combinations
        results: list[tuple[str, str, object]] = [
            ("home", home, lambda h, a: h > a),
            ("away", away, lambda h, a: h < a),
        ]
        if prof.has_draw:
            results.insert(1, ("draw", "Draw", lambda h, a: h == a))

        for _key, res_label, res_pred in results:
            for ou_label, ou_pred in (
                ("Over", lambda h, a: h + a > main_total),
                ("Under", lambda h, a: h + a < main_total),
            ):
                p = grid._sum(
                    lambda h, a, r=res_pred, o=ou_pred: r(h, a) and o(h, a)
                )
                if p < 0.04:
                    continue
                out.append(self._emit(
                    event, BetType.RESULT_TOTAL,
                    f"🎲 {res_label} & {ou_label} {main_total:g}", p,
                    market_display=(
                        f"Result + Total — {res_label} & {ou_label} "
                        f"{main_total:g}"
                    ),
                    reasoning=f"Both legs must land. {basis}",
                    line=main_total,
                ))

            for btts_label, btts_pred in (
                ("BTTS Yes", lambda h, a: h > 0 and a > 0),
                ("BTTS No", lambda h, a: h == 0 or a == 0),
            ):
                p = grid._sum(
                    lambda h, a, r=res_pred, b=btts_pred: r(h, a) and b(h, a)
                )
                if p < 0.04:
                    continue
                out.append(self._emit(
                    event, BetType.RESULT_BTTS,
                    f"🎲 {res_label} & {btts_label}", p,
                    market_display=f"Result + BTTS — {res_label} & {btts_label}",
                    reasoning=f"Both legs must land. {basis}",
                ))

        # Which side scores first, including the goalless case
        total_lam = model.home_lambda + model.away_lambda
        if total_lam > 0:
            no_goal = math.exp(-total_lam)
            singular = unit.rstrip("s")
            for label, p, team in (
                (f"⚡ {home} Scores First",
                 (model.home_lambda / total_lam) * (1 - no_goal), home),
                (f"⚡ {away} Scores First",
                 (model.away_lambda / total_lam) * (1 - no_goal), away),
                (f"⚡ No {singular} Scored", no_goal, ""),
            ):
                if p < 0.02:
                    continue
                out.append(self._emit(
                    event, BetType.FIRST_TO_SCORE, label, p,
                    market_display=f"First to Score — {team or 'No Goal'}",
                    reasoning=(
                        "Split by scoring rate, with the goalless case priced "
                        f"separately. {basis}"
                    ),
                    team_name=team,
                ))

        # Baseball: first five innings
        if sport == Sport.BASEBALL:
            f5 = ScoreGrid(model.home_lambda * 0.56, model.away_lambda * 0.56, 12)
            for label, p, team in (
                (f"⚾ {home} Lead After 5", f5.home_win(), home),
                ("⚾ Tied After 5", f5.draw(), ""),
                (f"⚾ {away} Lead After 5", f5.away_win(), away),
            ):
                if p < 0.05:
                    continue
                out.append(self._emit(
                    event, BetType.FIRST_5_INNINGS, label, p,
                    market_display=f"First 5 Innings — {team or 'Tie'}",
                    reasoning=f"Starters typically cover five innings. {basis}",
                    team_name=team,
                ))
            f5_line = round((model.home_lambda + model.away_lambda) * 0.56 * 2) / 2
            if f5_line == int(f5_line):
                f5_line += 0.5
            for side, p in (
                ("Over", f5.over(f5_line)), ("Under", f5.under(f5_line))
            ):
                out.append(self._emit(
                    event, BetType.FIRST_5_INNINGS,
                    f"⚾ F5 {side} {f5_line:g} Runs", p,
                    market_display=f"First 5 Innings Total — {side} {f5_line:g}",
                    reasoning=f"Runs through five innings. {basis}",
                    line=f5_line,
                ))

        # Hockey: regulation three-way (a tie is live until overtime)
        if sport == Sport.HOCKEY:
            rh, rd, ra = grid.home_win(), grid.draw(), grid.away_win()
            tot = rh + rd + ra
            if tot > 0:
                for label, p, team in (
                    (f"🏒 {home} in Regulation", rh / tot, home),
                    ("🏒 Tied After 60 Minutes", rd / tot, ""),
                    (f"🏒 {away} in Regulation", ra / tot, away),
                ):
                    out.append(self._emit(
                        event, BetType.THREE_WAY, label, p,
                        market_display=f"3-Way (Regulation) — {team or 'Tie'}",
                        reasoning=(
                            "Settled at the end of regulation; overtime does "
                            f"not count. {basis}"
                        ),
                        team_name=team,
                    ))

        return [p for p in out if p is not None]

    def _period_markets(
        self, event: MatchEvent, model: GameModel,
        home: str, away: str, unit: str, basis: str,
    ) -> list[Prediction]:
        """Half, quarter and period markets, off the same scoring model."""
        sport = event.tournament.sport
        share = FIRST_HALF_SHARE.get(sport)
        if share is None:
            return []
        out: list[Optional[Prediction]] = []
        prof = profile_for(sport)

        if prof.uses_grid:
            half_max = max(6, prof.max_goals // 2)
            first = ScoreGrid(
                model.home_lambda * share, model.away_lambda * share, half_max
            )
            second = ScoreGrid(
                model.home_lambda * (1 - share),
                model.away_lambda * (1 - share), half_max,
            )
            seg = "Half" if sport == Sport.SOCCER else "Period"
            seg_plural = "Halves" if seg == "Half" else "Periods"

            for label, p, team in (
                (f"⏱️ {home} Leads at {seg}-Time", first.home_win(), home),
                (f"⏱️ Level at {seg}-Time", first.draw(), ""),
                (f"⏱️ {away} Leads at {seg}-Time", first.away_win(), away),
            ):
                out.append(self._emit(
                    event, BetType.HALFTIME_RESULT, label, p,
                    market_display=f"1st {seg} Result — {team or 'Draw'}",
                    reasoning=(
                        f"About {share:.0%} of scoring lands in the first "
                        f"{seg.lower()}. {basis}"
                    ),
                    team_name=team,
                ))

            ht_exp = (model.home_lambda + model.away_lambda) * share
            for line in (0.5, 1.5, 2.5):
                if line > ht_exp + 2.5:
                    continue
                for side, p in (
                    ("Over", first.over(line)), ("Under", first.under(line))
                ):
                    if p < 0.06 or p > 0.94:
                        continue
                    out.append(self._emit(
                        event, BetType.HALFTIME_OVER_UNDER,
                        f"{'⬆️' if side == 'Over' else '⬇️'} 1st {seg} "
                        f"{side} {line:g} {unit}", p,
                        market_display=f"1st {seg} Total — {side} {line:g}",
                        reasoning=(
                            f"First-{seg.lower()} projection {ht_exp:.2f}. "
                            f"{basis}"
                        ),
                        line=line,
                    ))

            if sport == Sport.SOCCER:
                out.extend(
                    self._ht_ft_markets(event, first, second, home, away, basis)
                )

            both = first.over(0.5) * second.over(0.5)
            out.append(self._emit(
                event, BetType.BOTH_HALVES_OVER,
                f"↕️ Both {seg_plural} Over 0.5 {unit}", both,
                market_display=f"Both {seg_plural} Over 0.5",
                reasoning=f"At least one score in each {seg.lower()}. {basis}",
                line=0.5,
            ))
            out.extend(self._highest_segment(event, first, second, seg, basis))
        else:
            normal = model.normal
            assert normal is not None

            for name in ("1st Half", "2nd Half"):
                frac = 0.5
                mean = normal.total_mean * frac
                sd = normal.total_sd * math.sqrt(frac)
                line = round(mean) + 0.5
                for side in ("Over", "Under"):
                    p = (
                        1 - normal_cdf(line, mean, sd) if side == "Over"
                        else normal_cdf(line, mean, sd)
                    )
                    out.append(self._emit(
                        event, BetType.QUARTER_PROPS,
                        f"{'⬆️' if side == 'Over' else '⬇️'} {name} "
                        f"{side} {line:g} {unit}", p,
                        market_display=f"{name} Total — {side} {line:g}",
                        reasoning=f"{name} projection {mean:.1f}. {basis}",
                        line=line,
                    ))

                m_mean = normal.margin_mean * frac
                m_sd = normal.margin_sd * math.sqrt(frac)
                p_home = 1 - normal_cdf(0.5, m_mean, m_sd)
                p_away = normal_cdf(-0.5, m_mean, m_sd)
                tot = p_home + p_away
                if tot > 0:
                    for team, p in ((home, p_home / tot), (away, p_away / tot)):
                        out.append(self._emit(
                            event, BetType.PERIOD_RESULT,
                            f"🥇 {team} Wins the {name}", p,
                            market_display=f"{name} Winner — {team}",
                            reasoning=(
                                f"Margin over the {name.lower()} only. {basis}"
                            ),
                            team_name=team,
                        ))

            if sport == Sport.BASKETBALL:
                for q in (1, 2, 3, 4):
                    mean = normal.total_mean * 0.25
                    sd = normal.total_sd * 0.5
                    line = round(mean) + 0.5
                    for side in ("Over", "Under"):
                        p = (
                            1 - normal_cdf(line, mean, sd) if side == "Over"
                            else normal_cdf(line, mean, sd)
                        )
                        out.append(self._emit(
                            event, BetType.QUARTER_PROPS,
                            f"{'⬆️' if side == 'Over' else '⬇️'} Q{q} "
                            f"{side} {line:g} {unit}", p,
                            market_display=(
                                f"Quarter {q} Total — {side} {line:g}"
                            ),
                            reasoning=f"Quarter projection {mean:.1f}. {basis}",
                            line=line,
                        ))

        return [p for p in out if p is not None]

    def _ht_ft_markets(
        self, event: MatchEvent, first: ScoreGrid, second: ScoreGrid,
        home: str, away: str, basis: str,
    ) -> list[Optional[Prediction]]:
        """Half-time / full-time doubles, by convolving the two halves.

        The old build had no model for this. Here the first-half grid and the
        second-half grid are combined, so the pairing is exact rather than a
        guess at how often a half-time lead is held.
        """
        combos: dict[tuple[str, str], float] = {}
        n, m = first.max_goals, second.max_goals
        for h1 in range(n + 1):
            for a1 in range(n + 1):
                p1 = first.grid[h1][a1]
                if p1 < 1e-9:
                    continue
                ht = "H" if h1 > a1 else "D" if h1 == a1 else "A"
                for h2 in range(m + 1):
                    for a2 in range(m + 1):
                        p2 = second.grid[h2][a2]
                        if p2 < 1e-9:
                            continue
                        fh, fa = h1 + h2, a1 + a2
                        ft = "H" if fh > fa else "D" if fh == fa else "A"
                        combos[(ht, ft)] = combos.get((ht, ft), 0.0) + p1 * p2

        names = {"H": home, "D": "Draw", "A": away}
        out: list[Optional[Prediction]] = []
        for (ht, ft), p in sorted(combos.items(), key=lambda kv: -kv[1]):
            if p < 0.03:
                continue
            out.append(self._emit(
                event, BetType.HT_FT,
                f"🔀 {names[ht]} / {names[ft]} (HT/FT)", p,
                market_display=(
                    f"Half-Time/Full-Time — {names[ht]} / {names[ft]}"
                ),
                reasoning=(
                    "Leader at the break paired with the final result, from "
                    f"the two-half model. {basis}"
                ),
            ))
        return out

    def _highest_segment(
        self, event: MatchEvent, first: ScoreGrid, second: ScoreGrid,
        seg: str, basis: str,
    ) -> list[Optional[Prediction]]:
        """Which half/period sees more scoring."""

        def total_pmf(g: ScoreGrid) -> list[float]:
            pmf = [0.0] * (2 * g.max_goals + 1)
            for h in range(g.max_goals + 1):
                for a in range(g.max_goals + 1):
                    pmf[h + a] += g.grid[h][a]
            return pmf

        p1, p2 = total_pmf(first), total_pmf(second)
        p_first = p_second = p_equal = 0.0
        for i, a in enumerate(p1):
            for j, b in enumerate(p2):
                w = a * b
                if i > j:
                    p_first += w
                elif i < j:
                    p_second += w
                else:
                    p_equal += w

        out: list[Optional[Prediction]] = []
        for label, p in (
            (f"1st {seg}", p_first), (f"2nd {seg}", p_second), ("Equal", p_equal),
        ):
            out.append(self._emit(
                event, BetType.GAME_PROPS,
                f"📈 Highest-Scoring {seg}: {label}", p,
                market_display=f"Highest-Scoring {seg} — {label}",
                reasoning=(
                    f"Comparing the two {seg.lower()} distributions. {basis}"
                ),
            ))
        return out

    def _race_markets(
        self, event: MatchEvent, model: GameModel,
        home: str, away: str, unit: str, basis: str,
    ) -> list[Prediction]:
        """Race to X points/goals.

        Approximated from relative scoring rate, weighted toward the stronger
        side: a team that both scores faster and defends better reaches a
        target first more often than raw rate alone implies.
        """
        targets = {
            Sport.BASKETBALL: (10, 20),
            Sport.HOCKEY: (2, 3),
            Sport.SOCCER: (2,),
            Sport.AMERICAN_FOOTBALL: (10,),
        }.get(event.tournament.sport)
        if not targets:
            return []

        total = model.home_lambda + model.away_lambda
        if total <= 0:
            return []
        rate_share = model.home_lambda / total
        probs = model.result_probabilities()
        blended = 0.7 * rate_share + 0.3 * probs.get("home", rate_share)

        out: list[Optional[Prediction]] = []
        for target in targets:
            for team, p in ((home, blended), (away, 1 - blended)):
                out.append(self._emit(
                    event, BetType.RACE_TO,
                    f"🏃 {team} Race to {target} {unit}", p,
                    market_display=f"Race to {target} {unit} — {team}",
                    reasoning=(
                        "Scoring-rate split, weighted toward overall "
                        f"strength. {basis}"
                    ),
                    line=float(target),
                    team_name=team,
                ))
        return [p for p in out if p is not None]

    def _overtime_market(
        self, event: MatchEvent, model: GameModel, basis: str
    ) -> Optional[Prediction]:
        """Will the game go to overtime?

        Only offered where overtime is part of the format. League soccer has
        no extra time, so the market is not shown for it — the old build
        advertised "Extra Time" on every league fixture.
        """
        sport = event.tournament.sport
        grid = model.grid

        if sport in (Sport.HOCKEY, Sport.BASEBALL) and grid is not None:
            p = grid.draw()
            label = "Extra Innings" if sport == Sport.BASEBALL else "Overtime"
        elif sport in (Sport.BASKETBALL, Sport.AMERICAN_FOOTBALL) and model.normal:
            p = model.normal.margin_between(-0.5, 0.5)
            label = "Overtime"
        else:
            return None

        return self._emit(
            event, BetType.OVERTIME, f"⏰ {label} — Yes",
            max(0.005, min(0.5, p)),
            market_display=f"Will There Be {label}? — Yes",
            reasoning=f"Chance the sides are level when regulation ends. {basis}",
        )

    def _player_markets(
        self, event: MatchEvent, model: GameModel,
        home: str, away: str, basis: str,
    ) -> list[Prediction]:
        """Player markets built from the real team sheet.

        Names and positions come from the live roster feed. The old build
        carried a hand-typed star-player table that had gone stale — it still
        listed Anthony Davis at the Lakers and Son Heung-min at Tottenham —
        and paired those names with fixed, invented probabilities.

        The lines here are an explicit share-of-team-projection estimate and
        the reasoning says so. Without a per-player feed, that is as far as
        the data honestly reaches.
        """
        sport = event.tournament.sport
        positions = _SCORING_POSITIONS.get(sport)
        if not positions:
            return []

        out: list[Optional[Prediction]] = []
        for key, team, lam in (
            ("home_roster", home, model.home_lambda),
            ("away_roster", away, model.away_lambda),
        ):
            roster = event.espn_data.get(key) or []
            # Rank by how likely the position is to score, so a striker is
            # offered ahead of a holding midfielder. Roster order is squad
            # number, which put defensive players at the top of the list.
            def position_rank(entry: dict) -> int:
                pos = str(entry.get("position", "")).lower()
                for i, candidate in enumerate(positions):
                    if candidate in pos:
                        return i
                return len(positions)

            candidates = sorted(
                (
                    r for r in roster
                    if any(
                        pos in str(r.get("position", "")).lower()
                        for pos in positions
                    )
                ),
                key=position_rank,
            )
            if not candidates:
                continue

            if sport in (Sport.SOCCER, Sport.HOCKEY):
                share = 0.20 if sport == Sport.SOCCER else 0.18
                cap = 0.70 if sport == Sport.SOCCER else 0.60
                for player in candidates[:3]:
                    p = 1 - math.exp(-lam * share)
                    out.append(self._emit(
                        event, BetType.ANYTIME_SCORER,
                        f"🎯 {player['name']} — Anytime Goalscorer",
                        max(0.03, min(cap, p)),
                        market_display=(
                            f"Anytime Goalscorer — {player['name']}"
                        ),
                        reasoning=(
                            f"Estimate: {team} projected for {lam:.2f} goals, "
                            f"with roughly a {share:.0%} share for a "
                            f"{str(player.get('position', 'front-line')).lower()}"
                            ". This is a team-level projection, not a "
                            f"per-player model. {basis}"
                        ),
                        team_name=team,
                    ))
            elif sport == Sport.BASKETBALL:
                share = 0.22
                for player in candidates[:3]:
                    line = round(lam * share) + 0.5
                    out.append(self._emit(
                        event, BetType.PLAYER_PROPS,
                        f"🎯 {player['name']} — Over {line:g} Points", 0.5,
                        market_display=(
                            f"Player Points — {player['name']} O{line:g}"
                        ),
                        reasoning=(
                            f"Line set at roughly a {share:.0%} share of "
                            f"{team}'s projected {lam:.1f} points. The line is "
                            "the estimate; with no per-player feed there is no "
                            f"edge to claim on either side. {basis}"
                        ),
                        line=line,
                        team_name=team,
                    ))
        return [p for p in out if p is not None]

    def _corner_markets(self, event: MatchEvent) -> list[Prediction]:
        """Corner totals — only when corner counts were actually collected.

        The previous build always offered these, filling `corners_avg` with a
        random number between 3.5 and 7.5 when no feed supplied one.
        """
        if event.tournament.sport != Sport.SOCCER:
            return []
        home, away = event.home_stats, event.away_stats
        if not home or not away:
            return []
        if home.corners_avg <= 0 or away.corners_avg <= 0:
            return []

        expected = home.corners_avg + away.corners_avg
        out: list[Optional[Prediction]] = []
        for line in (8.5, 9.5, 10.5, 11.5):
            over = 1.0 - sum(
                poisson_pmf(i, expected) for i in range(int(line) + 1)
            )
            over = max(0.02, min(0.98, over))
            for side, p in (("Over", over), ("Under", 1 - over)):
                out.append(self._emit(
                    event, BetType.CORNERS,
                    f"{'⬆️' if side == 'Over' else '⬇️'} {side} {line:g} Corners",
                    p,
                    market_display=f"Total Corners — {side} {line:g}",
                    reasoning=(
                        f"Corner averages: {event.home_team.name} "
                        f"{home.corners_avg:.1f}, {event.away_team.name} "
                        f"{away.corners_avg:.1f} (expected {expected:.1f})."
                    ),
                    line=line,
                ))
        return [p for p in out if p is not None]

    # ── Factor analysis methods ──────────────────────────────────────────

    def _analyze_form(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        def form_score(form: str) -> float:
            score = 0.0
            for i, ch in enumerate(form):
                recency = max(0.2, 1.0 - (i * 0.08))
                if ch == "W":
                    score += 3 * recency
                elif ch == "D":
                    score += 1 * recency
            return score

        hf, af = form_score(home.form_string), form_score(away.form_string)
        total = hf + af + 0.001
        return {
            "home": round((hf / total) * weight, 4),
            "away": round((af / total) * weight, 4),
            "draw": round(weight * 0.15, 4),
            "detail": (
                f"Form (most recent first) — {home.team_name}: "
                f"{home.form_string or 'n/a'}, {away.team_name}: "
                f"{away.form_string or 'n/a'}"
            ),
        }

    def _analyze_home_advantage(
        self, home: Optional[TeamStats], away: Optional[TeamStats],
        sport: Sport, weight: float,
    ) -> dict:
        base_ha = {
            Sport.SOCCER: 0.60, Sport.BASKETBALL: 0.58, Sport.BASEBALL: 0.54,
            Sport.AMERICAN_FOOTBALL: 0.57, Sport.VOLLEYBALL: 0.58,
            Sport.TENNIS: 0.52,
        }.get(sport, 0.55)

        if home:
            played = home.home_wins + home.home_draws + home.home_losses
            if played > 0:
                base_ha = (base_ha + home.home_wins / played) / 2

        return {
            "home": round(weight * base_ha, 4),
            "away": round(weight * (1 - base_ha) * 0.75, 4),
            "draw": round(weight * (1 - base_ha) * 0.25, 4),
            "detail": f"Home advantage: {base_ha:.0%}",
        }

    def _analyze_h2h(self, h2h: Optional[HeadToHead], weight: float) -> dict:
        if not h2h or h2h.total_matches == 0:
            return {
                "home": weight / 3, "draw": weight / 3, "away": weight / 3,
                "detail": "No head-to-head meetings on record",
            }
        total = h2h.total_matches
        return {
            "home": round((h2h.team1_wins / total) * weight, 4),
            "away": round((h2h.team2_wins / total) * weight, 4),
            "draw": round((h2h.draws / total) * weight, 4),
            "detail": (
                f"H2H: {h2h.team1_wins}W-{h2h.draws}D-{h2h.team2_wins}L from "
                f"{total} recent meeting{'s' if total != 1 else ''}"
            ),
        }

    def _analyze_league_position(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        if (
            not home or not away
            or home.league_position == 0 or away.league_position == 0
        ):
            return {
                "home": weight / 3, "draw": weight / 3, "away": weight / 3,
                "detail": "League positions unavailable",
            }
        max_pos = max(home.league_position, away.league_position) + 1
        hs = (max_pos - home.league_position) / max_pos
        as_ = (max_pos - away.league_position) / max_pos
        total = hs + as_ + 0.001
        return {
            "home": round((hs / total) * weight * 0.85, 4),
            "away": round((as_ / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": (
                f"Table: {home.team_name} #{home.league_position} vs "
                f"{away.team_name} #{away.league_position}"
            ),
        }

    def _analyze_scoring(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}
        home_attack = home.avg_goals_scored * (away.avg_goals_conceded + 0.1)
        away_attack = away.avg_goals_scored * (home.avg_goals_conceded + 0.1)
        total = home_attack + away_attack + 0.001
        return {
            "home": round((home_attack / total) * weight * 0.85, 4),
            "away": round((away_attack / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": (
                f"Scoring: {home.team_name} {home.avg_goals_scored:.2f} for / "
                f"{home.avg_goals_conceded:.2f} against; {away.team_name} "
                f"{away.avg_goals_scored:.2f} / {away.avg_goals_conceded:.2f}"
            ),
        }

    def _analyze_injuries(
        self, home_injuries: list[PlayerInfo],
        away_injuries: list[PlayerInfo], weight: float,
    ) -> dict:
        home_impact = min(len(home_injuries) * 0.05, 0.3)
        away_impact = min(len(away_injuries) * 0.05, 0.3)
        home_score = max(0.0, weight * (0.5 + away_impact - home_impact))
        away_score = max(0.0, weight * (0.5 + home_impact - away_impact))
        if not home_injuries and not away_injuries:
            detail = "No injury or suspension data available"
        else:
            detail = (
                f"Absences — home {len(home_injuries)}, away "
                f"{len(away_injuries)}"
            )
        return {
            "home": round(home_score * 0.85, 4),
            "away": round(away_score * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": detail,
        }

    def _analyze_consistency(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        def win_rate(stats: TeamStats) -> float:
            if stats.games_played == 0:
                return 0.5
            return stats.wins / stats.games_played

        hc, ac = win_rate(home), win_rate(away)
        total = hc + ac + 0.001
        return {
            "home": round((hc / total) * weight * 0.85, 4),
            "away": round((ac / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": f"Win rate: {hc:.0%} vs {ac:.0%}",
        }

    def _analyze_momentum(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        def momentum(form: str) -> float:
            score = 0.0
            for ch in form[:5]:
                if ch == "W":
                    score += 2
                elif ch == "D":
                    score += 0.5
            return score

        hm, am = momentum(home.form_string), momentum(away.form_string)
        total = hm + am + 0.001
        return {
            "home": round((hm / total) * weight * 0.85, 4),
            "away": round((am / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": (
                f"Last five: {home.form_string[:5] or 'n/a'} vs "
                f"{away.form_string[:5] or 'n/a'}"
            ),
        }

    # ── Utilities ────────────────────────────────────────────────────────

    def _calculate_value(self, probability: float, odds: float) -> float:
        """Expected value per unit staked: p * odds - 1."""
        if odds <= 1.0 or probability <= 0:
            return 0.0
        return round(probability * odds - 1, 4)

    def _decimal_to_american(self, decimal_odds: float) -> str:
        return decimal_to_american(decimal_odds)

    def _american_to_decimal(self, american: str) -> float:
        return american_to_decimal(american) or 2.0

    def _poisson_cdf(self, k: int, lam: float) -> float:
        """P(X <= k) for a Poisson with mean lam."""
        if lam <= 0:
            return 1.0
        if lam > 100:
            return normal_cdf(k + 0.5, lam, math.sqrt(lam))
        return min(1.0, sum(poisson_pmf(i, lam) for i in range(max(0, k) + 1)))

    def _build_reasoning(
        self, event: MatchEvent, factors: dict, pick: str, basis: str = ""
    ) -> str:
        side = {"home": "HOME", "away": "AWAY"}.get(pick, "DRAW")
        lines = [
            f"{side} pick — {event.home_team.name} vs {event.away_team.name}",
            event.tournament.name
            + (f" ({event.tournament.country})" if event.tournament.country else ""),
        ]
        if basis:
            lines.append(basis)
        lines.append("")

        for name, data in factors.items():
            detail = data.get("detail", "")
            if detail:
                lines.append(f"• {name.replace('_', ' ').title()}: {detail}")

        if event.home_injuries:
            lines.append(
                "• Home absentees: "
                + ", ".join(p.name for p in event.home_injuries[:5])
            )
        if event.away_injuries:
            lines.append(
                "• Away absentees: "
                + ", ".join(p.name for p in event.away_injuries[:5])
            )

        if event.has_book_odds and event.home_odds > 0:
            lines.append(
                f"• Market: home {event.home_odds:.2f} | draw "
                f"{event.draw_odds:.2f} | away {event.away_odds:.2f}"
            )
        else:
            lines.append(
                "• No bookmaker price retrieved — the odds shown are this "
                "model's own fair price, not an offer."
            )

        if event.data_notes:
            lines.append("")
            lines.extend(f"⚠️ {note}" for note in event.data_notes)

        return "\n".join(lines)
