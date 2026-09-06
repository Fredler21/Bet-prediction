"""
Probability engine — one coherent distribution per game, every market read off it.

The previous version priced each market with its own ad-hoc formula, so the
moneyline, the spread and the totals could (and did) contradict each other.
Here a single model of the game is built once:

  * Low-scoring sports (soccer, hockey, baseball) get a Poisson score grid —
    the joint distribution over every plausible scoreline. Win/draw/loss,
    totals, handicaps, correct score, BTTS and margins are all just sums over
    cells of that grid, so they are guaranteed to agree with each other.

  * High-scoring sports (basketball, American football, volleyball) get a
    normal model of margin and total, which fits those sports far better than
    Poisson does.

Team scoring rates come from the attack/defence strength method: a team's
scoring is measured relative to its league's average, then combined with the
opponent's defensive record and the league baseline.

Nothing here invents data. If scoring inputs are missing the caller is told,
and no market is produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.models import Sport

# ── Sport profiles ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SportProfile:
    """How a sport's scoring behaves — drives which model is used."""

    uses_grid: bool          # Poisson score grid vs. normal approximation
    max_goals: int           # grid size (ignored for normal-model sports)
    has_draw: bool           # is a regulation draw a bettable outcome
    home_edge: float         # goals/points added to the home side
    margin_sd: float         # SD of final margin (normal model)
    total_sd: float          # SD of final total (normal model)
    fallback_total: float    # league total to fall back on
    unit: str                # display noun


# Home edges and spreads-of-margin below are the long-run public figures for
# each league (e.g. NFL margins scatter with an SD near two touchdowns, NBA
# near 12 points). They are priors, used only to shape the distribution around
# whatever scoring rates the live data supplies.
SPORT_PROFILES: dict[Sport, SportProfile] = {
    Sport.SOCCER: SportProfile(
        uses_grid=True, max_goals=10, has_draw=True, home_edge=0.30,
        margin_sd=1.8, total_sd=1.6, fallback_total=2.7, unit="Goals",
    ),
    Sport.HOCKEY: SportProfile(
        uses_grid=True, max_goals=12, has_draw=False, home_edge=0.20,
        margin_sd=2.2, total_sd=2.0, fallback_total=6.1, unit="Goals",
    ),
    Sport.BASEBALL: SportProfile(
        uses_grid=True, max_goals=20, has_draw=False, home_edge=0.15,
        margin_sd=4.0, total_sd=3.4, fallback_total=8.8, unit="Runs",
    ),
    Sport.HANDBALL: SportProfile(
        uses_grid=False, max_goals=60, has_draw=True, home_edge=2.0,
        margin_sd=6.5, total_sd=8.0, fallback_total=56.0, unit="Goals",
    ),
    Sport.BASKETBALL: SportProfile(
        uses_grid=False, max_goals=0, has_draw=False, home_edge=2.5,
        margin_sd=12.0, total_sd=17.0, fallback_total=225.0, unit="Points",
    ),
    Sport.AMERICAN_FOOTBALL: SportProfile(
        uses_grid=False, max_goals=0, has_draw=False, home_edge=1.7,
        margin_sd=13.5, total_sd=13.0, fallback_total=44.0, unit="Points",
    ),
    Sport.VOLLEYBALL: SportProfile(
        uses_grid=False, max_goals=0, has_draw=False, home_edge=2.0,
        margin_sd=9.0, total_sd=14.0, fallback_total=180.0, unit="Points",
    ),
    Sport.RUGBY: SportProfile(
        uses_grid=False, max_goals=0, has_draw=True, home_edge=3.0,
        margin_sd=14.0, total_sd=15.0, fallback_total=48.0, unit="Points",
    ),
    Sport.TENNIS: SportProfile(
        uses_grid=False, max_goals=0, has_draw=False, home_edge=0.0,
        margin_sd=1.2, total_sd=1.0, fallback_total=0.0, unit="Games",
    ),
    Sport.MMA: SportProfile(
        uses_grid=False, max_goals=0, has_draw=False, home_edge=0.0,
        margin_sd=1.0, total_sd=1.0, fallback_total=0.0, unit="Rounds",
    ),
}

DEFAULT_PROFILE = SPORT_PROFILES[Sport.SOCCER]


def profile_for(sport: Sport) -> SportProfile:
    return SPORT_PROFILES.get(sport, DEFAULT_PROFILE)


# ── Small numeric helpers ────────────────────────────────────────────────────


def normal_cdf(x: float, mu: float = 0.0, sd: float = 1.0) -> float:
    """P(X <= x) for X ~ N(mu, sd)."""
    if sd <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k), computed in log space so large means stay stable."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def decimal_to_american(decimal_odds: float) -> str:
    """Format decimal odds as an American price."""
    if decimal_odds <= 1.0:
        return ""
    if decimal_odds >= 2.0:
        return f"+{round((decimal_odds - 1) * 100)}"
    return f"{round(-100 / (decimal_odds - 1))}"


def american_to_decimal(american: str) -> float:
    """Parse an American price into decimal odds. 0.0 when unparseable."""
    try:
        val = int(str(american).replace("+", "").strip())
    except (ValueError, TypeError):
        return 0.0
    if val == 0:
        return 0.0
    if val > 0:
        return round(1 + val / 100, 3)
    return round(1 + 100 / abs(val), 3)


def fair_decimal(probability: float) -> float:
    """The zero-margin (break-even) decimal price for a probability."""
    if probability <= 0:
        return 0.0
    return round(1.0 / min(probability, 0.999), 3)


# Implied probability is capped so no market prints a price shorter than
# about -1900, which is roughly where real books stop quoting. Uncapped, a
# 0.95 model probability plus margin comes out as "-14000" — arithmetically
# correct, but it reads as a bug and nobody could bet it.
MAX_IMPLIED = 0.95


def priced_decimal(probability: float, margin: float = 0.045) -> float:
    """A realistic two-way market price, i.e. fair odds less the book's cut.

    Quoting 1/p as though it were a bookmaker's price is what made every
    market on the old site look like a free +EV bet. Real books hold a margin;
    applying one here keeps displayed prices honest and stops the value
    detector from firing on its own arithmetic.
    """
    if probability <= 0:
        return 0.0
    implied = min(probability * (1.0 + margin), MAX_IMPLIED)
    return round(1.0 / implied, 3)


# ── Expected scoring ─────────────────────────────────────────────────────────


@dataclass
class ScoringInputs:
    """Everything needed to project a scoreline, plus how solid it is."""

    home_lambda: float
    away_lambda: float
    complete: bool                 # were real scoring rates available
    notes: list[str] = field(default_factory=list)


def expected_scoring(
    sport: Sport,
    home_scored: float,
    home_conceded: float,
    away_scored: float,
    away_conceded: float,
    league_baseline: float,
    home_adj: float = 1.0,
    away_adj: float = 1.0,
) -> ScoringInputs:
    """Project each side's scoring with the attack/defence strength method.

    `league_baseline` is average scoring per team per game in this league.
    A team scoring 2.0 in a 1.4-goal league has attack strength 1.43; facing a
    defence conceding 1.75 in that league (strength 1.25) the projection is
    1.4 * 1.43 * 1.25. This is the standard construction, and it is what the
    old `avg_conceded / avg_conceded` ratio was failing to be — that version
    could multiply a total by 15x when one side had a tight defence.

    `home_adj` / `away_adj` are bounded multipliers from the form, injury and
    momentum factors.
    """
    prof = profile_for(sport)
    notes: list[str] = []

    baseline = league_baseline
    if baseline <= 0:
        baseline = prof.fallback_total / 2.0
        notes.append("league scoring baseline unavailable — using sport default")

    have_home = home_scored > 0 or home_conceded > 0
    have_away = away_scored > 0 or away_conceded > 0
    complete = have_home and have_away

    if not complete:
        notes.append("scoring history missing for at least one side")
        half = prof.fallback_total / 2.0
        return ScoringInputs(half, half, False, notes)

    if prof.uses_grid:
        # Low-scoring sports: ratio strengths, the standard Poisson form.
        def strength(value: float, default: float = 1.0) -> float:
            if value <= 0 or baseline <= 0:
                return default
            # Clamp so one freak result early in a season cannot dominate.
            return max(0.4, min(2.5, value / baseline))

        home_lambda = baseline * strength(home_scored) * strength(away_conceded)
        away_lambda = baseline * strength(away_scored) * strength(home_conceded)
    else:
        # High-scoring sports: pair each offence with the defence it faces and
        # average the two.
        #
        # Neither of the other obvious forms works here. Multiplying strength
        # ratios (the soccer form) compounds badly once scoring is in the
        # tens or hundreds. Adding both deviations to a league baseline
        # over-extends in the opposite direction: a team +2 on offence facing
        # a defence that concedes +2 comes out +4, when the honest reading is
        # nearer +2. Averaging lands where the market lands — on a Bengals /
        # Buccaneers line it gives 50.2 against a posted total of 50.5, where
        # the additive form gave 58.5.
        home_lambda = (home_scored + away_conceded) / 2.0
        away_lambda = (away_scored + home_conceded) / 2.0

    # Home edge, split so it lifts the host and trims the visitor.
    edge = prof.home_edge
    home_lambda += edge * 0.6
    away_lambda -= edge * 0.4

    home_lambda *= max(0.8, min(1.2, home_adj))
    away_lambda *= max(0.8, min(1.2, away_adj))

    floor = 0.05 if prof.uses_grid else 1.0
    home_lambda = max(floor, home_lambda)
    away_lambda = max(floor, away_lambda)

    return ScoringInputs(round(home_lambda, 3), round(away_lambda, 3), True, notes)


# ── The score grid (low-scoring sports) ──────────────────────────────────────


class ScoreGrid:
    """Joint distribution over (home score, away score).

    Every market for a grid sport is a sum over cells, so the moneyline,
    handicap, total and correct-score prices are all mutually consistent.
    """

    def __init__(self, home_lambda: float, away_lambda: float, max_goals: int = 10):
        self.home_lambda = home_lambda
        self.away_lambda = away_lambda
        self.max_goals = max_goals

        home_pmf = [poisson_pmf(i, home_lambda) for i in range(max_goals + 1)]
        away_pmf = [poisson_pmf(i, away_lambda) for i in range(max_goals + 1)]

        # Renormalise so the truncated tail does not quietly lose probability.
        h_sum = sum(home_pmf) or 1.0
        a_sum = sum(away_pmf) or 1.0
        home_pmf = [p / h_sum for p in home_pmf]
        away_pmf = [p / a_sum for p in away_pmf]

        self.grid = [
            [home_pmf[h] * away_pmf[a] for a in range(max_goals + 1)]
            for h in range(max_goals + 1)
        ]
        self.home_pmf = home_pmf
        self.away_pmf = away_pmf

    def _sum(self, predicate) -> float:
        total = 0.0
        for h in range(self.max_goals + 1):
            row = self.grid[h]
            for a in range(self.max_goals + 1):
                if predicate(h, a):
                    total += row[a]
        return total

    # Result markets
    def home_win(self) -> float:
        return self._sum(lambda h, a: h > a)

    def draw(self) -> float:
        return self._sum(lambda h, a: h == a)

    def away_win(self) -> float:
        return self._sum(lambda h, a: h < a)

    # Totals
    def over(self, line: float) -> float:
        return self._sum(lambda h, a: h + a > line)

    def under(self, line: float) -> float:
        return self._sum(lambda h, a: h + a < line)

    def total_push(self, line: float) -> float:
        """Probability the total lands exactly on a whole-number line."""
        if line != int(line):
            return 0.0
        return self._sum(lambda h, a: h + a == line)

    # Handicaps — `line` is applied to the home side.
    def home_covers(self, line: float) -> float:
        return self._sum(lambda h, a: (h + line) > a)

    def away_covers(self, line: float) -> float:
        return self._sum(lambda h, a: (a + line) > h)

    def handicap_push(self, line: float) -> float:
        if line != int(line):
            return 0.0
        return self._sum(lambda h, a: (h + line) == a)

    # Team totals
    def team_over(self, line: float, home: bool) -> float:
        pmf = self.home_pmf if home else self.away_pmf
        return sum(p for i, p in enumerate(pmf) if i > line)

    # Soccer-flavoured specials
    def btts(self) -> float:
        return self._sum(lambda h, a: h > 0 and a > 0)

    def clean_sheet(self, home: bool) -> float:
        return self._sum(lambda h, a: (a == 0) if home else (h == 0))

    def win_to_nil(self, home: bool) -> float:
        return self._sum(
            lambda h, a: (h > a and a == 0) if home else (a > h and h == 0)
        )

    def exact_score(self, home_goals: int, away_goals: int) -> float:
        if 0 <= home_goals <= self.max_goals and 0 <= away_goals <= self.max_goals:
            return self.grid[home_goals][away_goals]
        return 0.0

    def top_scores(self, count: int = 12) -> list[tuple[int, int, float]]:
        cells = [
            (h, a, self.grid[h][a])
            for h in range(self.max_goals + 1)
            for a in range(self.max_goals + 1)
        ]
        cells.sort(key=lambda c: c[2], reverse=True)
        return cells[:count]

    def margin(self, value: int) -> float:
        """P(home margin == value); negative means the away side won by |value|."""
        return self._sum(lambda h, a: (h - a) == value)

    def odd_total(self) -> float:
        return self._sum(lambda h, a: (h + a) % 2 == 1)

    def expected_total(self) -> float:
        return round(self.home_lambda + self.away_lambda, 2)


# ── Normal model (high-scoring sports) ───────────────────────────────────────


class NormalModel:
    """Margin/total model for sports where Poisson underestimates spread."""

    def __init__(
        self, home_lambda: float, away_lambda: float, profile: SportProfile
    ):
        self.home_mean = home_lambda
        self.away_mean = away_lambda
        self.profile = profile
        self.margin_mean = home_lambda - away_lambda
        self.total_mean = home_lambda + away_lambda
        self.margin_sd = profile.margin_sd
        self.total_sd = profile.total_sd

    def home_win(self) -> float:
        # P(margin > 0), with a half-point continuity correction.
        return 1.0 - normal_cdf(0.5, self.margin_mean, self.margin_sd)

    def away_win(self) -> float:
        return normal_cdf(-0.5, self.margin_mean, self.margin_sd)

    def draw(self) -> float:
        if not self.profile.has_draw:
            return 0.0
        return max(0.0, 1.0 - self.home_win() - self.away_win())

    def over(self, line: float) -> float:
        return 1.0 - normal_cdf(line, self.total_mean, self.total_sd)

    def under(self, line: float) -> float:
        return normal_cdf(line, self.total_mean, self.total_sd)

    def home_covers(self, line: float) -> float:
        """P(home margin + line > 0)."""
        return 1.0 - normal_cdf(-line, self.margin_mean, self.margin_sd)

    def away_covers(self, line: float) -> float:
        return normal_cdf(line, self.margin_mean, self.margin_sd)

    def team_over(self, line: float, home: bool) -> float:
        mean = self.home_mean if home else self.away_mean
        # A single team's scoring varies less than the game total.
        sd = self.profile.total_sd * 0.72
        return 1.0 - normal_cdf(line, mean, sd)

    def margin_between(self, low: float, high: float) -> float:
        return normal_cdf(high, self.margin_mean, self.margin_sd) - normal_cdf(
            low, self.margin_mean, self.margin_sd
        )

    def expected_total(self) -> float:
        return round(self.total_mean, 2)


# ── Unified front end ────────────────────────────────────────────────────────


class GameModel:
    """Wraps whichever underlying model suits the sport."""

    def __init__(self, sport: Sport, scoring: ScoringInputs):
        self.sport = sport
        self.profile = profile_for(sport)
        self.scoring = scoring
        self.complete = scoring.complete

        if self.profile.uses_grid:
            self.grid: Optional[ScoreGrid] = ScoreGrid(
                scoring.home_lambda, scoring.away_lambda, self.profile.max_goals
            )
            self.normal: Optional[NormalModel] = None
            self.engine = self.grid
        else:
            self.grid = None
            self.normal = NormalModel(
                scoring.home_lambda, scoring.away_lambda, self.profile
            )
            self.engine = self.normal

    @property
    def home_lambda(self) -> float:
        return self.scoring.home_lambda

    @property
    def away_lambda(self) -> float:
        return self.scoring.away_lambda

    def result_probabilities(self) -> dict[str, float]:
        """Normalised 1 / X / 2 probabilities."""
        home = self.engine.home_win()
        away = self.engine.away_win()
        draw = self.engine.draw() if self.profile.has_draw else 0.0

        total = home + draw + away
        if total <= 0:
            return {"home": 0.5, "draw": 0.0, "away": 0.5}

        if not self.profile.has_draw:
            # Draws are not bettable; push that mass onto the two sides.
            two_way = home + away
            if two_way <= 0:
                return {"home": 0.5, "draw": 0.0, "away": 0.5}
            return {
                "home": round(home / two_way, 4),
                "draw": 0.0,
                "away": round(away / two_way, 4),
            }

        return {
            "home": round(home / total, 4),
            "draw": round(draw / total, 4),
            "away": round(away / total, 4),
        }

    def over_under(self, line: float) -> dict[str, float]:
        over = self.engine.over(line)
        under = self.engine.under(line)
        push = (
            self.grid.total_push(line) if self.grid is not None else 0.0
        )
        total = over + under + push
        if total <= 0:
            return {"over": 0.5, "under": 0.5, "push": 0.0}
        return {
            "over": round(over / total, 4),
            "under": round(under / total, 4),
            "push": round(push / total, 4),
            "expected_total": self.engine.expected_total(),
        }

    def handicap(self, line: float) -> dict[str, float]:
        """Probabilities for a handicap `line` applied to the home team."""
        home = self.engine.home_covers(line)
        away = self.engine.away_covers(-line)
        push = self.grid.handicap_push(line) if self.grid is not None else 0.0
        total = home + away + push
        if total <= 0:
            return {"home": 0.5, "away": 0.5, "push": 0.0}
        return {
            "home": round(home / total, 4),
            "away": round(away / total, 4),
            "push": round(push / total, 4),
        }

    def team_total(self, line: float, home: bool) -> dict[str, float]:
        over = self.engine.team_over(line, home)
        over = max(0.001, min(0.999, over))
        return {"over": round(over, 4), "under": round(1 - over, 4)}

    def expected_total(self) -> float:
        return self.engine.expected_total()
