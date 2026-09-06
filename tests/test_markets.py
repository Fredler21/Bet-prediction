"""
Regression tests for the probability engine and parlay maths.

These pin down the specific defects that made the old build's numbers wrong,
so they cannot quietly come back:

  * a projected total inflated more than 10x by a bogus "defensive adjustment"
  * handicaps priced with a flat 0.12-per-point penalty in every sport
  * every market quoted at 1/probability, i.e. a zero-margin fair price
  * flex parlays "boosted" by an arbitrary factor instead of the stated
    Poisson-binomial
  * same-game parlays multiplying correlated legs as if independent

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.markets import (  # noqa: E402
    GameModel, ScoreGrid, expected_scoring, profile_for,
    american_to_decimal, decimal_to_american, fair_decimal, priced_decimal,
    normal_cdf, poisson_pmf,
)
from src.models import (  # noqa: E402
    BetType, MatchEvent, Prediction, Sport, Team, Tournament,
)
from src.parlay_optimizer import ParlayOptimizer  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────


def model_for(sport: Sport, hs, hc, as_, ac, baseline) -> GameModel:
    return GameModel(
        sport, expected_scoring(sport, hs, hc, as_, ac, baseline)
    )


def make_event(eid: int = 1, sport: Sport = Sport.SOCCER) -> MatchEvent:
    return MatchEvent(
        id=eid,
        tournament=Tournament(id=eid, name="Test League", sport=sport),
        home_team=Team(id=eid * 10, name="Home FC"),
        away_team=Team(id=eid * 10 + 1, name="Away FC"),
        start_time=datetime.now(timezone.utc),
    )


def make_pred(event, prob, bet_type=BetType.MONEYLINE, pick="pick", line=None):
    return Prediction(
        event=event, bet_type=bet_type, pick=pick,
        confidence=prob * 100, probability=prob,
        odds=round(1 / prob, 3), line=line,
    )


# ── Numeric primitives ───────────────────────────────────────────────────────


def test_poisson_pmf_sums_to_one():
    for lam in (0.4, 1.4, 3.0, 9.5):
        total = sum(poisson_pmf(k, lam) for k in range(0, 60))
        assert total == pytest.approx(1.0, abs=1e-9)


def test_poisson_pmf_survives_large_lambda():
    """A high mean must not overflow.

    Computed naively as lam**k / k!, a basketball-sized mean overflows a
    float outright; this is done in log space instead.
    """
    assert poisson_pmf(200, 220.0) > 0.0
    assert math.isfinite(poisson_pmf(200, 220.0))


def test_normal_cdf_endpoints():
    assert normal_cdf(0.0, 0.0, 1.0) == pytest.approx(0.5)
    assert normal_cdf(-40.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-9)
    assert normal_cdf(40.0, 0.0, 1.0) == pytest.approx(1.0, abs=1e-9)


def test_odds_conversions_round_trip():
    for american in ("+150", "-200", "+100", "-110", "+2500"):
        dec = american_to_decimal(american)
        # decimal_to_american returns the formatted string, e.g. "+150".
        back = int(decimal_to_american(dec).replace("+", ""))
        assert back == pytest.approx(int(american.replace("+", "")), abs=2)


def test_model_price_is_worse_than_fair():
    """A displayed price must carry a margin, not be the fair price.

    Quoting 1/p was what made every market look like a free +EV bet.
    """
    for p in (0.2, 0.5, 0.75):
        assert priced_decimal(p) < fair_decimal(p)


def test_model_price_never_absurd():
    """Nothing should print a price shorter than about -1900."""
    assert priced_decimal(0.999) >= 1.05


# ── Expected scoring ─────────────────────────────────────────────────────────


def test_totals_are_not_inflated_by_defensive_mismatch():
    """The bug that produced "Expected total: 5.79 goals" in the EPL.

    The old code multiplied the total by the average of
    away_conceded/home_conceded and its inverse. Facing a mean defence with a
    very tight one, that factor ran to 15x. A total must stay in a sane band
    however lopsided the defences are.
    """
    model = model_for(Sport.SOCCER, 1.5, 0.1, 1.5, 3.0, 1.45)
    assert 0.5 < model.expected_total() < 7.0

    balanced = model_for(Sport.SOCCER, 1.45, 1.45, 1.45, 1.45, 1.45)
    assert 2.3 < balanced.expected_total() < 3.4


def test_average_soccer_match_matches_long_run_rates():
    """Two league-average sides should land near the historical 45/25/30."""
    model = model_for(Sport.SOCCER, 1.45, 1.45, 1.45, 1.45, 1.45)
    p = model.result_probabilities()
    assert p["home"] == pytest.approx(0.45, abs=0.05)
    assert p["draw"] == pytest.approx(0.25, abs=0.05)
    assert p["away"] == pytest.approx(0.30, abs=0.05)


def test_result_probabilities_always_normalise():
    for sport, args in (
        (Sport.SOCCER, (1.4, 1.4, 1.4, 1.4, 1.4)),
        (Sport.BASKETBALL, (115.0, 113.0, 112.0, 116.0, 114.0)),
        (Sport.HOCKEY, (3.1, 2.9, 3.0, 3.1, 3.05)),
        (Sport.BASEBALL, (4.5, 4.3, 4.4, 4.5, 4.4)),
        (Sport.AMERICAN_FOOTBALL, (23.0, 21.0, 21.0, 23.0, 22.0)),
    ):
        p = model_for(sport, *args).result_probabilities()
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-3)


def test_no_draw_sports_report_no_draw():
    for sport in (Sport.BASKETBALL, Sport.BASEBALL, Sport.HOCKEY):
        assert not profile_for(sport).has_draw
        args = (
            (115.0, 113.0, 112.0, 116.0, 114.0)
            if sport == Sport.BASKETBALL else (3.5, 3.2, 3.3, 3.4, 3.35)
        )
        assert model_for(sport, *args).result_probabilities()["draw"] == 0.0


def test_high_scoring_sports_use_averaging_not_addition():
    """Adding both deviations to a baseline over-extends the total.

    These inputs sit against a real posted DraftKings total of 50.5. The
    additive form returned 58.5; pairing each offence with the defence it
    faces and averaging lands within a point of the market.
    """
    model = model_for(Sport.AMERICAN_FOOTBALL, 24.35, 28.94, 22.35, 24.18, 22.0)
    assert model.expected_total() == pytest.approx(50.5, abs=2.0)


def test_missing_scoring_data_is_reported_not_invented():
    scoring = expected_scoring(Sport.SOCCER, 0, 0, 0, 0, 0)
    assert scoring.complete is False
    assert scoring.notes


# ── Score grid coherence ─────────────────────────────────────────────────────


def test_grid_probabilities_sum_to_one():
    grid = ScoreGrid(1.6, 1.3, 10)
    assert grid.home_win() + grid.draw() + grid.away_win() == pytest.approx(
        1.0, abs=1e-6
    )


def test_grid_over_under_partition_is_exact():
    grid = ScoreGrid(1.6, 1.3, 10)
    for line in (1.5, 2.5, 3.5):
        assert grid.over(line) + grid.under(line) == pytest.approx(1.0, abs=1e-6)
    # A whole line leaves room for a push.
    assert grid.over(3) + grid.under(3) + grid.total_push(3) == pytest.approx(
        1.0, abs=1e-6
    )


def test_handicap_and_moneyline_agree():
    """A -0.5 handicap is the same bet as the moneyline, so must price alike.

    Each market having its own formula is exactly how the old build ended up
    with a spread that contradicted its own moneyline.
    """
    model = model_for(Sport.SOCCER, 1.8, 1.1, 1.2, 1.6, 1.45)
    # handicap() rounds its output to 4 decimal places.
    assert model.handicap(-0.5)["home"] == pytest.approx(
        model.grid.home_win(), abs=1e-4
    )


def test_handicap_is_monotonic_in_the_line():
    """Giving away more goals can only ever lower the cover probability."""
    model = model_for(Sport.SOCCER, 2.0, 1.0, 1.1, 1.7, 1.45)
    probs = [model.handicap(l)["home"] for l in (-2.5, -1.5, -0.5, 0.5, 1.5)]
    assert probs == sorted(probs)


def test_nba_spread_is_not_floored_at_five_percent():
    """The old flat 0.12-per-point penalty crushed NBA spreads to the floor.

    A four-point favourite covering -7.5 is a normal 35-45% proposition, not
    the 5% the linear model produced.
    """
    model = model_for(Sport.BASKETBALL, 117.0, 113.0, 113.0, 116.0, 115.0)
    assert 0.25 < model.handicap(-7.5)["home"] < 0.55


def test_btts_is_a_probability():
    """The old formula mixed a raw goal count into a probability.

    `avg_goals_scored * 0.6 + ...` exceeds 1.0 for any decent attack, so it
    was only ever a probability by virtue of being clamped.
    """
    for hs, ac in ((0.5, 0.5), (1.5, 1.5), (3.0, 3.0)):
        grid = ScoreGrid(hs, ac, 10)
        assert 0.0 <= grid.btts() <= 1.0
    # More scoring on both sides can only make BTTS likelier.
    assert ScoreGrid(2.2, 2.2, 10).btts() > ScoreGrid(0.7, 0.7, 10).btts()


def test_win_to_nil_implies_clean_sheet_and_win():
    grid = ScoreGrid(1.9, 1.0, 10)
    assert grid.win_to_nil(True) <= grid.clean_sheet(True)
    assert grid.win_to_nil(True) <= grid.home_win()


def test_correct_scores_are_ordered_and_bounded():
    grid = ScoreGrid(1.6, 1.2, 10)
    tops = grid.top_scores(6)
    assert [p for _, _, p in tops] == sorted(
        [p for _, _, p in tops], reverse=True
    )
    assert sum(p for _, _, p in tops) < 1.0


def test_team_total_is_monotonic():
    model = model_for(Sport.SOCCER, 1.8, 1.1, 1.2, 1.6, 1.45)
    probs = [model.team_total(l, True)["over"] for l in (0.5, 1.5, 2.5, 3.5)]
    assert probs == sorted(probs, reverse=True)


# ── Parlay maths ─────────────────────────────────────────────────────────────


def test_flex_parlay_uses_exact_poisson_binomial():
    """The old code multiplied by (1 + 0.3 * misses) and ignored the legs.

    With mixed leg probabilities the answer is a Poisson-binomial sum, and it
    must match a direct calculation.
    """
    probs = [0.8, 0.7, 0.6, 0.55, 0.5]
    opt = ParlayOptimizer(min_confidence=1, bankroll=1000)
    legs = [make_pred(make_event(i), p) for i, p in enumerate(probs, 1)]
    parlay = opt.build_flex_parlay(legs, num_legs=5, miss_allowed=1)

    all_win = math.prod(probs)
    one_miss = sum(
        (1 - probs[i]) * math.prod(probs[j] for j in range(5) if j != i)
        for i in range(5)
    )
    assert parlay.combined_confidence == pytest.approx(
        100 * (all_win + one_miss), abs=0.05
    )
    # Allowing a miss must be strictly better than needing every leg.
    assert parlay.combined_confidence > 100 * all_win


def test_flex_with_no_misses_allowed_equals_straight_parlay():
    probs = [0.7, 0.6, 0.5]
    opt = ParlayOptimizer(min_confidence=1, bankroll=1000)
    legs = [make_pred(make_event(i), p) for i, p in enumerate(probs, 1)]
    parlay = opt.build_flex_parlay(legs, num_legs=3, miss_allowed=0)
    assert parlay.combined_confidence == pytest.approx(
        100 * math.prod(probs), abs=0.05
    )


def test_sgp_probability_sits_between_independent_and_weakest_leg():
    """Same-game legs move together, so multiplying them understates the ticket.

    Independence is the floor; a perfectly correlated ticket can be no more
    likely than its least likely leg.
    """
    event = make_event(42)
    legs = [
        make_pred(event, 0.70, BetType.OVER_UNDER, "Over 2.5 Goals", 2.5),
        make_pred(event, 0.60, BetType.MONEYLINE, "Home FC Win"),
        make_pred(event, 0.55, BetType.BOTH_TEAMS_SCORE, "BTTS Yes"),
    ]
    opt = ParlayOptimizer(min_confidence=1, bankroll=1000)
    sgp = opt.build_sgp(legs, event.id, num_legs=3)

    independent = math.prod(l.probability for l in sgp.legs)
    weakest = min(l.probability for l in sgp.legs)
    assert 100 * independent <= sgp.combined_confidence <= 100 * weakest


def test_sgp_drops_contradictory_legs():
    """Best-per-bet-type can pair Over 2.5 with Under 2.5 — a dead ticket."""
    event = make_event(43)
    legs = [
        make_pred(event, 0.70, BetType.OVER_UNDER, "Over 2.5 Goals", 2.5),
        make_pred(event, 0.65, BetType.ALTERNATE_TOTAL, "Under 2.5 Goals", 2.5),
        make_pred(event, 0.60, BetType.MONEYLINE, "Home FC Win"),
    ]
    opt = ParlayOptimizer(min_confidence=1, bankroll=1000)
    kept = opt._drop_contradictions(legs)
    picks = {p.pick for p in kept}
    assert not ({"Over 2.5 Goals", "Under 2.5 Goals"} <= picks)
    assert "Home FC Win" in picks


def test_round_robin_splits_the_stake_across_tickets():
    """A round robin is every combination bet together.

    Sizing each ticket as if it were the only bet on the slip multiplied the
    real outlay by the number of combinations.
    """
    opt = ParlayOptimizer(min_confidence=1, bankroll=1000)
    legs = [make_pred(make_event(i), 0.7) for i in range(1, 5)]
    parlays = opt.build_round_robin(legs, num_picks=4, combo_size=2)
    assert len(parlays) == 6  # C(4, 2)
    assert all(p.recommended_stake <= 1000 * 0.05 / 6 + 0.01 for p in parlays)


def test_zero_min_confidence_is_respected():
    """`min_confidence=0` meant "no floor", but was replaced by the default.

    The old constructor did `min_confidence or settings.parlay_min_confidence`,
    so a legitimate 0 became 70 and every leg was filtered away.
    """
    assert ParlayOptimizer(min_confidence=0).min_confidence == 0


def test_kelly_never_stakes_on_a_negative_edge():
    opt = ParlayOptimizer(min_confidence=1, bankroll=1000)
    legs = [make_pred(make_event(i), 0.5) for i in range(1, 4)]
    parlay = opt.build_parlay(legs, num_legs=3)
    advice = opt.calculate_bankroll_advice(parlay)
    assert advice.recommended_stake >= 0
    assert advice.recommended_stake <= 1000 * 0.05
