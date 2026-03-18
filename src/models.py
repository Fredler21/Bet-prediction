"""Data models for sports, events, statistics and predictions."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ── Sport Types ──────────────────────────────────────────────────────────────

class Sport(str, Enum):
    SOCCER = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"
    BASEBALL = "baseball"
    AMERICAN_FOOTBALL = "american-football"
    VOLLEYBALL = "volleyball"
    HOCKEY = "ice-hockey"
    MMA = "mma"
    HANDBALL = "handball"
    RUGBY = "rugby"


SPORT_EMOJIS = {
    Sport.SOCCER: "⚽",
    Sport.BASKETBALL: "🏀",
    Sport.TENNIS: "🎾",
    Sport.BASEBALL: "⚾",
    Sport.AMERICAN_FOOTBALL: "🏈",
    Sport.VOLLEYBALL: "🏐",
    Sport.HOCKEY: "🏒",
    Sport.MMA: "🥊",
    Sport.HANDBALL: "🤾",
    Sport.RUGBY: "🏉",
}


class BetType(str, Enum):
    MONEYLINE = "moneyline"                     # Who wins (1X2 for soccer)
    SPREAD = "spread"                           # Point spread / handicap
    OVER_UNDER = "over_under"                   # Total points/goals
    BOTH_TEAMS_SCORE = "btts"                   # Both teams to score (soccer)
    DOUBLE_CHANCE = "double_chance"              # 1X, X2, 12
    DRAW_NO_BET = "draw_no_bet"                 # Refund on draw
    CORRECT_SCORE = "correct_score"             # Exact final score
    FIRST_HALF = "first_half"
    PLAYER_PROPS = "player_props"               # Individual player stats
    TEAM_TOTAL = "team_total"                   # Team-specific total goals/points
    CORNERS = "corners"                         # Corners over/under
    GAME_RESULT_90 = "game_result_90"           # Game result (90 min + stoppage)
    HALFTIME_OVER_UNDER = "halftime_over_under" # 1st half over/under
    HALFTIME_RESULT = "halftime_result"         # 1st half result
    # ── Hard Rock Bet Expansion ──
    GAME_PROPS = "game_props"                   # First to score, OT, odd/even, etc.
    ALTERNATE_SPREAD = "alternate_spread"       # Alternate spread lines
    ALTERNATE_TOTAL = "alternate_total"         # Alternate O/U lines
    FIRST_TO_SCORE = "first_to_score"           # Which team scores first
    OVERTIME = "overtime"                       # Will there be OT / extra innings
    ODD_EVEN = "odd_even"                       # Odd or even total
    QUARTER_PROPS = "quarter_props"             # Quarter/period totals & results
    RACE_TO = "race_to"                         # Race to X points/goals
    FUTURES = "futures"                         # Season-long: championship, MVP, etc.


class MatchStatus(str, Enum):
    NOT_STARTED = "not_started"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


# ── Core Data Models ─────────────────────────────────────────────────────────

@dataclass
class Team:
    id: int
    name: str
    short_name: str = ""
    logo_url: str = ""
    sport: Sport = Sport.SOCCER


@dataclass
class Tournament:
    id: int
    name: str
    sport: Sport
    country: str = ""
    slug: str = ""
    priority: int = 0  # Higher = more important league


@dataclass
class TeamStats:
    """Comprehensive team statistics for analysis."""
    team_id: int
    team_name: str

    # Form (last N games)
    wins: int = 0
    draws: int = 0
    losses: int = 0
    form_string: str = ""  # e.g., "WWDLW"

    # Scoring
    goals_scored: int = 0
    goals_conceded: int = 0
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0

    # Home/Away splits
    home_wins: int = 0
    home_draws: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_draws: int = 0
    away_losses: int = 0
    home_goals_scored: float = 0.0
    home_goals_conceded: float = 0.0
    away_goals_scored: float = 0.0
    away_goals_conceded: float = 0.0

    # Advanced
    possession_avg: float = 0.0
    shots_on_target_avg: float = 0.0
    corners_avg: float = 0.0
    cards_avg: float = 0.0
    clean_sheets: int = 0
    btts_percentage: float = 0.0
    over_2_5_percentage: float = 0.0

    # Standings
    league_position: int = 0
    points: int = 0
    games_played: int = 0

    # Sport-specific
    extra: dict = field(default_factory=dict)


@dataclass
class HeadToHead:
    """Head-to-head record between two teams."""
    team1_id: int
    team2_id: int
    total_matches: int = 0
    team1_wins: int = 0
    team2_wins: int = 0
    draws: int = 0
    team1_goals: int = 0
    team2_goals: int = 0
    recent_matches: list = field(default_factory=list)


@dataclass
class PlayerInfo:
    """Key player information."""
    id: int
    name: str
    team_id: int
    position: str = ""
    is_injured: bool = False
    is_suspended: bool = False
    injury_description: str = ""
    rating: float = 0.0
    goals: int = 0
    assists: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class MatchEvent:
    """A single match/event to analyze."""
    id: int
    tournament: Tournament
    home_team: Team
    away_team: Team
    start_time: datetime
    status: MatchStatus = MatchStatus.NOT_STARTED

    # Stats (populated by analysis)
    home_stats: Optional[TeamStats] = None
    away_stats: Optional[TeamStats] = None
    h2h: Optional[HeadToHead] = None
    home_injuries: list[PlayerInfo] = field(default_factory=list)
    away_injuries: list[PlayerInfo] = field(default_factory=list)

    # Scores (populated for finished/live games)
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    # Odds from SofaScore / ESPN
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0

    # ESPN enrichment data (spread, O/U, records)
    espn_data: dict = field(default_factory=dict)


# ── Prediction Models ────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """A single bet prediction."""
    event: MatchEvent
    bet_type: BetType
    pick: str  # e.g., "Home Win", "Over 2.5", "Team A -3.5"
    confidence: float  # 0-100
    probability: float  # 0.0-1.0 estimated probability
    odds: float = 0.0
    value_rating: float = 0.0  # Expected value
    reasoning: str = ""
    factors: dict = field(default_factory=dict)
    # Explicit display fields
    line: Optional[float] = None          # e.g., 2.5 for O/U 2.5, -0.5 for spread
    american_odds: str = ""               # e.g., "+400", "-150"
    market_display: str = ""              # Full formatted market string
    team_name: str = ""                   # Specific team name for team markets
    push_note: str = ""                   # e.g., "Push if tied"


@dataclass
class ParlayPrediction:
    """A multi-leg parlay recommendation."""
    legs: list[Prediction]
    combined_confidence: float = 0.0
    combined_odds: float = 0.0
    expected_value: float = 0.0
    recommended_stake: float = 0.0
    risk_level: str = "medium"  # low, medium, high
    reasoning: str = ""
    # ── Hard Rock Bet parlay types ──
    parlay_type: str = "standard"  # standard, sgp, round_robin, teaser, flex
    teaser_points: float = 0.0     # Points bought for teaser (e.g., 6.0)
    flex_miss_allowed: int = 0     # Legs allowed to lose in flex parlay


@dataclass
class BankrollAdvice:
    """Bankroll management recommendation."""
    recommended_stake: float
    kelly_stake: float
    risk_percentage: float
    bankroll: float
    reasoning: str = ""
