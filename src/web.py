"""
FastAPI Web Server — Premium REST API + Web Dashboard for the Prediction Agent.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agent import PredictionAgent
from src.models import Sport, SPORT_EMOJIS


# ── Lifespan ─────────────────────────────────────────────────────────────────

agent: Optional[PredictionAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = PredictionAgent()
    yield
    if agent:
        await agent.close()


app = FastAPI(
    title="🏆 Bet Prediction AI Agent",
    description="Premium sports betting predictions powered by API-Football + ESPN data + AI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────

class ParlayRequest(BaseModel):
    num_legs: int = 6
    strategy: str = "balanced"
    sports: Optional[list[str]] = None
    target_date: Optional[str] = None  # YYYY-MM-DD


class SGPRequest(BaseModel):
    event_id: int
    num_legs: int = 4
    sports: Optional[list[str]] = None
    target_date: Optional[str] = None


class RoundRobinRequest(BaseModel):
    num_picks: int = 5
    combo_size: int = 3
    sports: Optional[list[str]] = None
    target_date: Optional[str] = None


class TeaserRequest(BaseModel):
    num_legs: int = 3
    teaser_points: float = 6.0
    sports: Optional[list[str]] = None
    target_date: Optional[str] = None


class FlexParlayRequest(BaseModel):
    num_legs: int = 5
    miss_allowed: int = 1
    sports: Optional[list[str]] = None
    target_date: Optional[str] = None


class PredictionResponse(BaseModel):
    match: str
    home_team: str
    away_team: str
    tournament: str
    country: str
    sport: str
    sport_emoji: str
    pick: str
    bet_type: str
    confidence: float
    probability: float
    odds: float
    american_odds: str
    value_rating: float
    reasoning: str
    start_time: str
    start_date: str
    market_display: str
    line: Optional[float] = None
    team_name: str
    push_note: str


class MatchGroupResponse(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    tournament: str
    country: str
    sport: str
    sport_emoji: str
    start_time: str
    start_date: str
    status: str = "not_started"
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    predictions: list[PredictionResponse]


class ParlayLegResponse(BaseModel):
    match: str
    home_team: str
    away_team: str
    sport: str
    sport_emoji: str
    tournament: str
    pick: str
    bet_type: str
    confidence: float
    odds: float
    american_odds: str
    start_time: str
    start_date: str
    market_display: str
    push_note: str


class ParlayResponse(BaseModel):
    legs: list[ParlayLegResponse]
    combined_confidence: float
    combined_odds: float
    expected_value: float
    risk_level: str
    recommended_stake: float
    reasoning: str
    parlay_type: str = "standard"
    teaser_points: float = 0.0
    flex_miss_allowed: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_sports(sports: Optional[list[str]]) -> Optional[list[Sport]]:
    if not sports:
        return None
    result = []
    for s in sports:
        for sport in Sport:
            if sport.value == s or sport.name.lower() == s.lower():
                result.append(sport)
                break
    return result or None


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def _to_utc_iso(dt: datetime) -> str:
    """Return ISO 8601 UTC string for a datetime (aware or naive)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pred_to_response(p) -> PredictionResponse:
    sport = p.event.tournament.sport
    return PredictionResponse(
        match=f"{p.event.home_team.name} vs {p.event.away_team.name}",
        home_team=p.event.home_team.name,
        away_team=p.event.away_team.name,
        tournament=p.event.tournament.name,
        country=p.event.tournament.country,
        sport=sport.value,
        sport_emoji=SPORT_EMOJIS.get(sport, "🏆"),
        pick=p.pick,
        bet_type=p.bet_type.value,
        confidence=p.confidence,
        probability=p.probability,
        odds=p.odds,
        american_odds=p.american_odds or "",
        value_rating=p.value_rating,
        reasoning=p.reasoning,
        start_time=_to_utc_iso(p.event.start_time),
        start_date=_to_utc_iso(p.event.start_time),
        market_display=p.market_display or p.bet_type.value,
        line=p.line,
        team_name=p.team_name or "",
        push_note=p.push_note or "",
    )


def _leg_to_response(leg) -> ParlayLegResponse:
    sport = leg.event.tournament.sport
    return ParlayLegResponse(
        match=f"{leg.event.home_team.name} vs {leg.event.away_team.name}",
        home_team=leg.event.home_team.name,
        away_team=leg.event.away_team.name,
        sport=sport.value,
        sport_emoji=SPORT_EMOJIS.get(sport, "🏆"),
        tournament=leg.event.tournament.name,
        pick=leg.pick,
        bet_type=leg.bet_type.value,
        confidence=leg.confidence,
        odds=leg.odds,
        american_odds=leg.american_odds or "",
        start_time=_to_utc_iso(leg.event.start_time),
        start_date=_to_utc_iso(leg.event.start_time),
        market_display=leg.market_display or leg.bet_type.value,
        push_note=leg.push_note or "",
    )


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/predictions", response_model=list[PredictionResponse])
async def get_predictions(
    sport: Optional[str] = Query(None, description="Sport slug"),
    min_confidence: float = Query(60, ge=0, le=100),
    target_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    sports = _parse_sports([sport] if sport else None)
    d = _parse_date(target_date)
    preds = await agent.get_todays_predictions(sports=sports, target_date=d, min_confidence=min_confidence)

    results = []
    for sport_preds in preds.values():
        for p in sport_preds:
            results.append(_pred_to_response(p))

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


@app.get("/api/matches", response_model=list[MatchGroupResponse])
async def get_matches_grouped(
    sport: Optional[str] = Query(None),
    min_confidence: float = Query(50, ge=0, le=100),
    target_date: Optional[str] = Query(None),
):
    """Get predictions grouped by match — each match shows all available markets."""
    sports = _parse_sports([sport] if sport else None)
    d = _parse_date(target_date)
    preds = await agent.get_todays_predictions(sports=sports, target_date=d, min_confidence=min_confidence)

    grouped = agent.group_predictions_by_match(preds)
    results = []
    for match in grouped:
        ev = match["event"]
        sport_obj = match["sport"]
        results.append(MatchGroupResponse(
            match_id=ev.id,
            home_team=ev.home_team.name,
            away_team=ev.away_team.name,
            tournament=ev.tournament.name,
            country=ev.tournament.country,
            sport=sport_obj.value,
            sport_emoji=SPORT_EMOJIS.get(sport_obj, "🏆"),
            start_time=_to_utc_iso(ev.start_time),
            start_date=_to_utc_iso(ev.start_time),
            status=ev.status.value,
            home_score=ev.home_score,
            away_score=ev.away_score,
            predictions=[_pred_to_response(p) for p in match["predictions"]],
        ))
    return results


@app.get("/api/past-games", response_model=list[MatchGroupResponse])
async def get_past_games(
    target_date: Optional[str] = Query(None, description="YYYY-MM-DD (defaults to yesterday)"),
    sport: Optional[str] = Query(None),
    min_confidence: float = Query(50, ge=0, le=100),
):
    """Get games for a past date with predictions and final scores."""
    from datetime import timedelta
    d = _parse_date(target_date) or (date.today() - timedelta(days=1))
    sports = _parse_sports([sport] if sport else None)
    preds = await agent.get_todays_predictions(sports=sports, target_date=d, min_confidence=min_confidence)
    grouped = agent.group_predictions_by_match(preds)
    results = []
    for match in grouped:
        ev = match["event"]
        sport_obj = match["sport"]
        results.append(MatchGroupResponse(
            match_id=ev.id,
            home_team=ev.home_team.name,
            away_team=ev.away_team.name,
            tournament=ev.tournament.name,
            country=ev.tournament.country,
            sport=sport_obj.value,
            sport_emoji=SPORT_EMOJIS.get(sport_obj, "🏆"),
            start_time=_to_utc_iso(ev.start_time),
            start_date=_to_utc_iso(ev.start_time),
            status=ev.status.value,
            home_score=ev.home_score,
            away_score=ev.away_score,
            predictions=[_pred_to_response(p) for p in match["predictions"]],
        ))
    return results


@app.post("/api/parlay", response_model=ParlayResponse)
async def build_parlay(req: ParlayRequest):
    sports = _parse_sports(req.sports)
    d = _parse_date(req.target_date)

    parlay = await agent.build_parlay(
        num_legs=req.num_legs,
        sports=sports,
        strategy=req.strategy,
        target_date=d,
    )

    if not parlay.legs:
        raise HTTPException(status_code=404, detail="No qualifying picks for parlay.")

    return _parlay_to_response(parlay)


def _parlay_to_response(parlay) -> ParlayResponse:
    return ParlayResponse(
        legs=[_leg_to_response(leg) for leg in parlay.legs],
        combined_confidence=parlay.combined_confidence,
        combined_odds=parlay.combined_odds,
        expected_value=parlay.expected_value,
        risk_level=parlay.risk_level,
        recommended_stake=parlay.recommended_stake,
        reasoning=parlay.reasoning,
        parlay_type=getattr(parlay, 'parlay_type', 'standard'),
        teaser_points=getattr(parlay, 'teaser_points', 0.0),
        flex_miss_allowed=getattr(parlay, 'flex_miss_allowed', 0),
    )


@app.get("/api/parlays", response_model=list[ParlayResponse])
async def build_multiple_parlays(
    num_legs: int = Query(6, ge=2, le=15),
    count: int = Query(3, ge=1, le=5),
):
    parlays = await agent.build_multiple_parlays(num_legs=num_legs, count=count)
    return [_parlay_to_response(p) for p in parlays]


# ── Hard Rock Bet Parlay Types ───────────────────────────────────────────

@app.post("/api/sgp", response_model=ParlayResponse)
async def build_sgp(req: SGPRequest):
    """Build a Same Game Parlay for a specific event."""
    sports = _parse_sports(req.sports)
    d = _parse_date(req.target_date)
    parlay = await agent.build_sgp(req.event_id, req.num_legs, sports, d)
    if not parlay.legs:
        raise HTTPException(status_code=404, detail="Not enough bet types for SGP on this event.")
    return _parlay_to_response(parlay)


@app.get("/api/sgps", response_model=list[ParlayResponse])
async def build_all_sgps(
    num_legs: int = Query(4, ge=2, le=10),
    sport: Optional[str] = Query(None),
    target_date: Optional[str] = Query(None),
):
    """Build Same Game Parlays for ALL available games."""
    sports = _parse_sports([sport] if sport else None)
    d = _parse_date(target_date)
    sgps = await agent.build_all_sgps(num_legs, sports, d)
    return [_parlay_to_response(s) for s in sgps[:20]]


@app.post("/api/round-robin", response_model=list[ParlayResponse])
async def build_round_robin(req: RoundRobinRequest):
    """Build Round Robin — all parlay combos from top picks."""
    sports = _parse_sports(req.sports)
    d = _parse_date(req.target_date)
    parlays = await agent.build_round_robin(req.num_picks, req.combo_size, sports, d)
    if not parlays:
        raise HTTPException(status_code=404, detail="Not enough qualifying picks for round robin.")
    return [_parlay_to_response(p) for p in parlays[:15]]


@app.post("/api/teaser", response_model=ParlayResponse)
async def build_teaser(req: TeaserRequest):
    """Build a Teaser — buy points on spreads/totals."""
    sports = _parse_sports(req.sports)
    d = _parse_date(req.target_date)
    parlay = await agent.build_teaser(req.num_legs, req.teaser_points, sports, d)
    if not parlay.legs:
        raise HTTPException(status_code=404, detail="Not enough spread/total picks for a teaser.")
    return _parlay_to_response(parlay)


@app.post("/api/flex-parlay", response_model=ParlayResponse)
async def build_flex_parlay(req: FlexParlayRequest):
    """Build a Flex Parlay — still win even if some legs lose."""
    sports = _parse_sports(req.sports)
    d = _parse_date(req.target_date)
    parlay = await agent.build_flex_parlay(req.num_legs, req.miss_allowed, sports, d)
    if not parlay.legs:
        raise HTTPException(status_code=404, detail="Not enough qualifying picks for flex parlay.")
    return _parlay_to_response(parlay)


@app.get("/api/value-bets", response_model=list[PredictionResponse])
async def get_value_bets(
    min_value: float = Query(0.05, description="Minimum expected value"),
    target_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    d = _parse_date(target_date)
    bets = await agent.find_value_bets(min_value=min_value, target_date=d)
    return [_pred_to_response(p) for p in bets]


@app.get("/api/report")
async def get_daily_report():
    report = await agent.generate_daily_report()
    return {"report": report, "generated_at": datetime.now().isoformat()}


@app.get("/api/sports")
async def list_sports():
    return [
        {"name": s.name, "slug": s.value, "emoji": SPORT_EMOJIS.get(s, "🏆")}
        for s in Sport
    ]


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Web Dashboard ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the premium web dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🏆 Premium Bet Prediction AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #080c14;
  --card: #0f1520;
  --card-hover: #141d2e;
  --border: #1a2540;
  --accent: #00e5a0;
  --accent-glow: rgba(0,229,160,0.15);
  --accent2: #7c5cfc;
  --accent2-glow: rgba(124,92,252,0.15);
  --gold: #ffd700;
  --text: #e8edf5;
  --dim: #5a6b85;
  --green: #00e5a0;
  --yellow: #ffbe0b;
  --orange: #ff8c42;
  --red: #ff4d6a;
  --blue: #3b82f6;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  background-image: radial-gradient(ellipse at 50% 0%, rgba(0,229,160,0.03) 0%, transparent 60%);
}
.container { max-width: 1500px; margin: 0 auto; padding: 16px 20px; }

/* ── Header ── */
header {
  text-align: center;
  padding: 25px 0 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}
header h1 { font-size: 2em; font-weight: 800; letter-spacing: -0.5px; }
header h1 .glow { color: var(--accent); text-shadow: 0 0 20px var(--accent-glow); }
header .sub { color: var(--dim); font-size: 0.95em; margin-top: 6px; }
header .date-badge {
  display: inline-block;
  background: var(--accent-glow);
  color: var(--accent);
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 0.8em;
  font-weight: 600;
  margin-top: 8px;
}

/* ── Date Navigator ── */
.date-nav {
  display: flex; align-items: center; gap: 10px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 8px 14px;
  margin-bottom: 14px; flex-wrap: wrap;
}
.date-nav-label {
  font-weight: 700; font-size: 0.95em; color: var(--text); flex: 1;
  text-align: center; min-width: 140px;
}
.date-nav-btn {
  padding: 5px 12px; border-radius: 8px; border: 1px solid var(--border);
  background: transparent; color: var(--text); cursor: pointer;
  font-size: 13px; font-weight: 600; transition: all 0.15s;
  font-family: inherit;
}
.date-nav-btn:hover { border-color: var(--accent); color: var(--accent); }
.date-nav-btn.today-btn {
  background: var(--accent-glow); color: var(--accent); border-color: var(--accent);
}
.date-nav input[type="date"] {
  padding: 5px 10px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--card); color: var(--text); font-family: inherit; font-size: 13px;
}
.date-nav input[type="date"]:focus { border-color: var(--accent); outline: none; }

/* ── Controls ── */
.controls {
  display: flex; gap: 10px; flex-wrap: wrap;
  margin-bottom: 16px; align-items: center;
}
.controls select, .controls input {
  padding: 9px 14px; border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--card); color: var(--text);
  font-size: 13px; font-family: inherit;
}
.controls select:focus, .controls input:focus { border-color: var(--accent); outline: none; }
.btn {
  padding: 9px 18px; border-radius: 8px; border: none;
  font-weight: 700; cursor: pointer; font-size: 13px;
  font-family: inherit; transition: all 0.15s;
}
.btn-primary {
  background: linear-gradient(135deg, var(--accent), #00c48c);
  color: #000;
}
.btn-accent2 {
  background: linear-gradient(135deg, var(--accent2), #6841e0);
  color: #fff;
}
.btn-gold {
  background: linear-gradient(135deg, var(--gold), #f0a800);
  color: #000;
}
.btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }

/* ── Sport Chips (multi-select) ── */
.sport-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
.sport-chip {
  padding: 6px 14px; border-radius: 20px; font-size: 12px;
  font-weight: 600; cursor: pointer; user-select: none;
  border: 1px solid var(--border); background: var(--card);
  color: var(--dim); transition: all 0.15s;
}
.sport-chip:hover { border-color: var(--accent); color: var(--text); }
.sport-chip.active {
  background: var(--accent-glow); border-color: var(--accent);
  color: var(--accent);
}

/* ── Tabs ── */
.tabs { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid var(--border); }
.tab {
  padding: 10px 22px; cursor: pointer; color: var(--dim);
  border-bottom: 2px solid transparent; margin-bottom: -2px;
  font-weight: 600; font-size: 14px; transition: all 0.2s;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ── Match Cards ── */
.match-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(480px, 100%), 1fr)); gap: 16px; }

.match-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; overflow: hidden;
  transition: all 0.2s;
}
.match-card:hover { border-color: var(--accent); background: var(--card-hover); }

.match-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 16px; border-bottom: 1px solid var(--border);
  background: rgba(0,229,160,0.02);
}
.match-teams { font-weight: 700; font-size: 1.05em; }
.match-teams .vs { color: var(--dim); font-weight: 400; margin: 0 6px; }
.match-meta { color: var(--dim); font-size: 0.8em; margin-top: 4px; }
.match-meta span { margin-right: 12px; }
.match-time-badge {
  text-align: right; white-space: nowrap;
}
.match-time-badge .time { font-weight: 700; font-size: 1.1em; color: var(--accent); }
.match-time-badge .date { font-size: 0.75em; color: var(--dim); }

.match-markets { padding: 12px 16px; }
.market-group { margin-bottom: 12px; }
.market-group:last-child { margin-bottom: 0; }
.market-label {
  font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px;
  color: var(--dim); font-weight: 700; margin-bottom: 6px;
  display: flex; align-items: center; gap: 6px;
}
.market-label .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }

.market-picks { display: flex; gap: 6px; flex-wrap: wrap; }
.pick-chip {
  padding: 6px 12px; border-radius: 8px; font-size: 0.8em;
  font-weight: 600; border: 1px solid var(--border);
  background: rgba(255,255,255,0.02); cursor: default;
  display: flex; align-items: center; gap: 6px;
  transition: all 0.15s;
}
.pick-chip:hover { border-color: var(--accent); }
.pick-chip .odds-tag {
  background: rgba(0,229,160,0.1); color: var(--accent);
  padding: 1px 6px; border-radius: 4px; font-size: 0.85em;
}
.pick-chip .conf-tag { font-size: 0.85em; }
.conf-high { color: var(--green); }
.conf-mid { color: var(--yellow); }
.conf-low { color: var(--orange); }
.push-tag {
  font-size: 0.7em; color: var(--yellow); background: rgba(255,190,11,0.1);
  padding: 1px 6px; border-radius: 4px;
}

/* ── Parlay Section ── */
.parlay-section {
  background: var(--card); border: 2px solid var(--accent2);
  border-radius: 16px; padding: 20px; margin-bottom: 20px;
  box-shadow: 0 0 30px var(--accent2-glow);
}
.parlay-title {
  font-size: 1.3em; font-weight: 800; margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
}
.parlay-title .badge {
  background: var(--accent2); color: #fff; padding: 3px 10px;
  border-radius: 12px; font-size: 0.6em; text-transform: uppercase;
  letter-spacing: 1px;
}

.parlay-leg {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 14px; border-radius: 10px; margin-bottom: 6px;
  background: rgba(255,255,255,0.02); border: 1px solid var(--border);
  transition: border-color 0.15s;
}
.parlay-leg:hover { border-color: var(--accent); }
.leg-num {
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--accent2); color: #fff; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.8em; flex-shrink: 0;
}
.leg-info { flex: 1; margin-left: 12px; }
.leg-match { font-weight: 600; font-size: 0.95em; }
.leg-detail { color: var(--dim); font-size: 0.8em; margin-top: 2px; }
.leg-pick {
  text-align: right; font-weight: 700; color: var(--accent);
  font-size: 0.95em; white-space: nowrap; margin-left: 12px;
}
.leg-pick .odds-sm { color: var(--dim); font-weight: 400; font-size: 0.85em; }

.parlay-summary {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px; margin-top: 16px; padding-top: 16px;
  border-top: 1px solid var(--border);
}
.summary-item {
  text-align: center; padding: 10px;
  background: rgba(0,229,160,0.03); border-radius: 8px;
}
.summary-item .value { font-size: 1.3em; font-weight: 700; color: var(--accent); }
.summary-item .label { font-size: 0.7em; color: var(--dim); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Value Bets ── */
.value-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(380px, 100%), 1fr)); gap: 14px; }
.value-card {
  background: var(--card); border-left: 3px solid var(--gold);
  border-radius: 10px; padding: 16px;
  border-right: 1px solid var(--border);
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.value-card .ev-badge {
  background: rgba(255,215,0,0.12); color: var(--gold);
  padding: 3px 10px; border-radius: 12px; font-weight: 700;
  font-size: 0.85em;
}

/* ── Loading/Empty ── */
.loading { text-align: center; padding: 60px; color: var(--dim); }
.loading .spinner {
  width: 40px; height: 40px; border: 3px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.8s linear infinite; margin: 0 auto 15px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.empty { text-align: center; padding: 60px; color: var(--dim); }

#content { min-height: 350px; }

footer {
  text-align: center; padding: 20px; color: var(--dim);
  font-size: 0.8em; border-top: 1px solid var(--border); margin-top: 30px;
}

/* ── Mobile Responsive ─────────────────────────────────────────── */
@media (max-width: 768px) {
  .container { padding: 8px 10px; }
  header { padding: 14px 0 12px; }
  header h1 { font-size: 1.25em; letter-spacing: -0.3px; }
  header .sub { font-size: 0.75em; }
  header .date-badge { font-size: 0.72em; padding: 3px 10px; }

  /* Date navigator: wrap and center on mobile */
  .date-nav { justify-content: center; gap: 6px; }
  .date-nav-label { width: 100%; order: -1; text-align: center; font-size: 0.9em; }
  .date-nav-btn { padding: 4px 10px; font-size: 12px; }
  .date-nav input[type="date"] { font-size: 12px; padding: 4px 8px; }

  /* Tabs: horizontal scroll */
  .tabs {
    overflow-x: auto;
    flex-wrap: nowrap;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
    padding-bottom: 1px;
  }
  .tabs::-webkit-scrollbar { display: none; }
  .tab { padding: 8px 12px; font-size: 12px; white-space: nowrap; }

  /* Sport chips */
  .sport-chip { font-size: 11px; padding: 5px 10px; }

  /* Controls: stack vertically */
  .controls { flex-direction: column; gap: 8px; }
  .controls input, .controls select { width: 100%; box-sizing: border-box; }
  .btn { width: 100%; padding: 12px; font-size: 14px; }

  /* Grids: single column */
  .match-grid { grid-template-columns: 1fr !important; gap: 10px; }
  .value-grid { grid-template-columns: 1fr !important; gap: 10px; }

  /* Match card header: stack */
  .match-header { flex-direction: column; align-items: flex-start; gap: 6px; }
  .match-time-badge { text-align: left; display: flex; align-items: center; gap: 10px; }
  .match-time-badge .time { font-size: 1em; }
  .match-time-badge .date { font-size: 0.75em; }
  .match-teams { font-size: 0.95em; }

  /* Pick chips: wrap tightly */
  .pick-chip { font-size: 0.75em; padding: 5px 9px; }

  /* Parlay legs: stack */
  .parlay-section { padding: 14px; }
  .parlay-title { font-size: 1.1em; flex-wrap: wrap; gap: 6px; }
  .parlay-leg { flex-wrap: wrap; gap: 6px; }
  .leg-pick { text-align: left; margin-left: 40px; font-size: 0.9em; }
  .parlay-summary { grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .summary-item .value { font-size: 1.1em; }
  .summary-item { padding: 8px; }

  /* Value cards */
  .value-card { padding: 12px; }
}

@media (max-width: 400px) {
  header h1 { font-size: 1.05em; }
  .tab { padding: 7px 10px; font-size: 11px; }
  .parlay-summary { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🏆 Premium Bet Prediction <span class="glow">AI Agent</span></h1>
    <div class="sub">Powered by SofaScore Data • Statistical Analysis • AI Reasoning</div>
    <div class="date-badge" id="todayDate"></div>
  </header>

  <!-- Sport multi-select chips -->
  <div class="sport-chips" id="sportChips">
    <div class="sport-chip active" data-sport="" onclick="toggleSport(this)">🌐 All Sports</div>
    <div class="sport-chip" data-sport="football" onclick="toggleSport(this)">⚽ Soccer</div>
    <div class="sport-chip" data-sport="basketball" onclick="toggleSport(this)">🏀 Basketball</div>
    <div class="sport-chip" data-sport="tennis" onclick="toggleSport(this)">🎾 Tennis</div>
    <div class="sport-chip" data-sport="baseball" onclick="toggleSport(this)">⚾ Baseball</div>
    <div class="sport-chip" data-sport="american-football" onclick="toggleSport(this)">🏈 Football</div>
    <div class="sport-chip" data-sport="volleyball" onclick="toggleSport(this)">🏐 Volleyball</div>
    <div class="sport-chip" data-sport="ice-hockey" onclick="toggleSport(this)">🏒 Hockey</div>
  </div>

  <!-- Filter Mode Chips -->
  <div class="sport-chips" id="filterChips" style="margin-bottom:8px">
    <span style="color:var(--dim);font-size:12px;font-weight:600;margin-right:4px">FILTER:</span>
    <div class="sport-chip active" data-filter="balanced" onclick="setFilter(this)">⚖️ Balanced</div>
    <div class="sport-chip" data-filter="safe" onclick="setFilter(this)">🛡️ Safe (75%+)</div>
    <div class="sport-chip" data-filter="value" onclick="setFilter(this)">💎 Value (+EV)</div>
  </div>

  <div class="controls">
    <input type="number" id="minConf" value="50" min="0" max="100" placeholder="Min confidence %">
    <input type="number" id="parlayLegs" value="6" min="2" max="15" placeholder="Parlay legs">
    <select id="parlayStrategy">
      <option value="balanced">⚖️ Balanced</option>
      <option value="safe">🛡️ Safe</option>
      <option value="value">💎 Value</option>
    </select>
    <button class="btn btn-primary" onclick="loadMatches()">📊 All Markets</button>
    <button class="btn btn-accent2" onclick="loadParlay()">🎯 Build Parlay</button>
    <button class="btn btn-gold" onclick="loadValueBets()">💰 Value Bets</button>
  </div>

  <!-- Date Navigator -->
  <div class="date-nav">
    <button class="date-nav-btn" onclick="shiftDate(-1)">&#8592; Prev</button>
    <button class="date-nav-btn" onclick="shiftDate(-2)">&#8676; -2d</button>
    <div class="date-nav-label" id="dateNavLabel">Today</div>
    <input type="date" id="dateNavPicker" onchange="setNavDate(this.value)">
    <button class="date-nav-btn today-btn" onclick="setNavDate(localDateStr())">Today</button>
    <button class="date-nav-btn" onclick="shiftDate(1)">Tomorrow &#8594;</button>
    <button class="date-nav-btn" onclick="shiftDate(2)">+2d &#8677;</button>
  </div>

  <div class="tabs">
    <div class="tab active" data-tab="matches" onclick="loadMatches()">🏟️ Matches & Markets</div>
    <div class="tab" data-tab="parlay" onclick="loadParlay()">🎯 Parlay Builder</div>
    <div class="tab" data-tab="sgp" onclick="loadSGPs()">🎰 Same Game Parlays</div>
    <div class="tab" data-tab="roundrobin" onclick="loadRoundRobin()">🔄 Round Robin</div>
    <div class="tab" data-tab="teaser" onclick="loadTeaser()">🎲 Teaser</div>
    <div class="tab" data-tab="flex" onclick="loadFlexParlay()">💪 Flex Parlay</div>
    <div class="tab" data-tab="value" onclick="loadValueBets()">💰 Value Bets</div>
    <div class="tab" data-tab="past" onclick="loadPastGames()">📅 Past Games</div>
  </div>

  <div id="content">
    <div class="empty">
      <p style="font-size:3em;margin-bottom:12px;">🏆</p>
      <p style="font-size:1.1em;font-weight:600;">Premium AI Bet Predictions — Hard Rock Bet Style</p>
      <p style="margin-top:6px;">Click <b style="color:var(--accent)">All Markets</b> to see every match with all available bet types</p>
      <p style="margin-top:4px;font-size:0.85em;">SGP • Round Robin • Teaser • Flex Parlay • 23+ Bet Types • Mix Any Sports!</p>
    </div>
  </div>

  <footer>
    <p>⚠️ Bet responsibly. Predictions are for entertainment and informational purposes only.</p>
    <p>Data sourced from API-Football & ESPN. Past performance does not guarantee future results.</p>
  </footer>
</div>

<script>
const API = '';
let selectedSports = [];
let activeFilter = 'balanced';
let activeTab = 'matches';
let activeDateStr = '';  // YYYY-MM-DD, set on boot

// Returns today's date in the user's LOCAL timezone as YYYY-MM-DD
function localDateStr() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

// Format a YYYY-MM-DD into a human-readable label
function dateLabelOf(ymd) {
  const today = localDateStr();
  const d = new Date(ymd + 'T12:00:00');
  const base = d.toLocaleDateString('en-US', {weekday:'long', month:'long', day:'numeric'});
  if (ymd === today) return `📅 Today — ${base}`;
  const tomorrow = offsetDate(today, 1);
  const yesterday = offsetDate(today, -1);
  if (ymd === tomorrow) return `🔮 Tomorrow — ${base}`;
  if (ymd === yesterday) return `📜 Yesterday — ${base}`;
  const delta = Math.round((new Date(ymd) - new Date(today)) / 86400000);
  const tag = delta > 0 ? `+${delta}d` : `${delta}d`;
  return `📅 ${base} (${tag})`;
}

function offsetDate(ymd, days) {
  const d = new Date(ymd + 'T12:00:00');
  d.setDate(d.getDate() + days);
  return d.toISOString().split('T')[0];
}

function setNavDate(ymd, reload = true) {
  activeDateStr = ymd;
  document.getElementById('dateNavLabel').textContent = dateLabelOf(ymd);
  document.getElementById('dateNavPicker').value = ymd;
  if (reload) reloadActiveTab();
}

function shiftDate(delta) {
  setNavDate(offsetDate(activeDateStr, delta));
}

// ── Time formatting: convert UTC ISO strings to user's local timezone ──
function fmtTime(utcStr) {
  if (!utcStr) return '';
  const d = new Date(utcStr);
  if (isNaN(d)) return utcStr;
  return d.toLocaleTimeString('en-US', {hour:'numeric', minute:'2-digit', hour12:true});
}
function fmtDate(utcStr) {
  if (!utcStr) return '';
  const d = new Date(utcStr);
  if (isNaN(d)) return utcStr;
  return d.toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'});
}

// Boot: init date navigator to today
setNavDate(localDateStr(), false);

// Set today's date in header
document.getElementById('todayDate').textContent =
  '📅 ' + new Date().toLocaleDateString('en-US', {weekday:'long', year:'numeric', month:'long', day:'numeric'})
  + ' • ' + new Date().toLocaleTimeString('en-US', {hour:'2-digit', minute:'2-digit'});

function toggleSport(el) {
  const sport = el.dataset.sport;
  if (sport === '') {
    document.querySelectorAll('#sportChips .sport-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
    selectedSports = [];
  } else {
    document.querySelector('#sportChips .sport-chip[data-sport=""]').classList.remove('active');
    el.classList.toggle('active');
    selectedSports = [...document.querySelectorAll('#sportChips .sport-chip.active')]
      .map(c => c.dataset.sport).filter(Boolean);
    if (selectedSports.length === 0) {
      document.querySelector('#sportChips .sport-chip[data-sport=""]').classList.add('active');
    }
  }
  // Auto-reload current view
  reloadActiveTab();
}

function setFilter(el) {
  document.querySelectorAll('#filterChips .sport-chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  activeFilter = el.dataset.filter;
  // Update confidence input to match filter
  if (activeFilter === 'safe') {
    document.getElementById('minConf').value = 75;
  } else if (activeFilter === 'value') {
    document.getElementById('minConf').value = 40;
  } else {
    document.getElementById('minConf').value = 50;
  }
  // Also sync parlay strategy
  document.getElementById('parlayStrategy').value = activeFilter;
  reloadActiveTab();
}

function reloadActiveTab() {
  if (activeTab === 'matches') loadMatches();
  else if (activeTab === 'parlay') loadParlay();
  else if (activeTab === 'sgp') loadSGPs();
  else if (activeTab === 'roundrobin') loadRoundRobin();
  else if (activeTab === 'teaser') loadTeaser();
  else if (activeTab === 'flex') loadFlexParlay();
  else if (activeTab === 'value') loadValueBets();
  else if (activeTab === 'past') loadPastGames();
}

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
}

function showLoading() {
  document.getElementById('content').innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <p>🔄 Analyzing games & crunching statistics across all sports...</p>
    </div>`;
}

function confClass(conf) {
  if (conf >= 75) return 'conf-high';
  if (conf >= 60) return 'conf-mid';
  return 'conf-low';
}

function confEmoji(conf) {
  if (conf >= 75) return '🟢';
  if (conf >= 65) return '🟡';
  return '🟠';
}

// Market type labels with emojis
const marketLabels = {
  'moneyline': '🏆 Winner',
  'game_result_90': '⚽ Game Result (90 min + Stoppage Time)',
  'spread': '📊 Spread / Handicap',
  'over_under': '⬆️⬇️ Total Goals/Points Over/Under',
  'team_total': '🎯 Team-Specific Totals',
  'btts': '⚽ Both Teams to Score',
  'double_chance': '🛡️ Double Chance',
  'draw_no_bet': '🔄 Draw No Bet',
  'correct_score': '🎯 Correct Score',
  'corners': '🔲 Corners',
  'halftime_result': '⏱️ Half-Time Result',
  'halftime_over_under': '⏱️ 1st Half Over/Under',
  'player_props': '🎯 Player Props',
  'first_half': '⏱️ First Half',
  'game_props': '🎲 Game Props',
  'alternate_spread': '📊 Alternate Spreads',
  'alternate_total': '⬆️⬇️ Alternate Totals',
  'first_to_score': '⚡ First to Score',
  'overtime': '⏰ Overtime / Extra Time',
  'odd_even': '🔢 Odd/Even Total',
  'quarter_props': '📊 Quarter / Period Props',
  'race_to': '🏃 Race to X',
  'futures': '🏅 Futures',
};

async function loadMatches() {
  activeTab = 'matches';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="matches"]').classList.add('active');
  showLoading();
  const minConf = document.getElementById('minConf').value;
  const tdate = activeDateStr;

  try {
    // If multiple sports selected, fetch each; else fetch all
    let matchData = [];
    if (selectedSports.length > 0) {
      const promises = selectedSports.map(s =>
        fetch(`${API}/api/matches?min_confidence=${minConf}&sport=${s}&target_date=${tdate}`).then(r => r.json())
      );
      const results = await Promise.all(promises);
      results.forEach(r => { if (Array.isArray(r)) matchData.push(...r); });
    } else {
      const res = await fetch(`${API}/api/matches?min_confidence=${minConf}&target_date=${tdate}`);
      const d = await res.json();
      if (Array.isArray(d)) matchData = d;
    }

    // Apply filter mode
    if (activeFilter === 'safe') {
      matchData = matchData.map(m => ({
        ...m,
        predictions: m.predictions.filter(p => p.confidence >= 75)
      })).filter(m => m.predictions.length > 0);
    } else if (activeFilter === 'value') {
      matchData = matchData.map(m => ({
        ...m,
        predictions: m.predictions.filter(p => p.value_rating > 0)
      })).filter(m => m.predictions.length > 0);
    }

    if (!matchData.length) {
      document.getElementById('content').innerHTML = '<div class="empty"><p style="font-size:2em">🔍</p><p>No matches found for this filter. Try switching to <b>Balanced</b> or selecting different sports.</p></div>';
      return;
    }

    let html = '<div style="margin-bottom:12px;color:var(--dim);font-size:0.85em">'
             + `📊 ${matchData.length} matches • ${matchData.reduce((s,m) => s + m.predictions.length, 0)} predictions`
             + (activeFilter === 'safe' ? ' • 🛡️ Showing only <b style="color:var(--green)">75%+ confidence</b> picks' : '')
             + (activeFilter === 'value' ? ' • 💎 Showing only <b style="color:var(--gold)">+EV value</b> picks' : '')
             + '</div>';
    html += '<div class="match-grid">';

    for (const match of matchData) {
      // Group predictions by bet type
      const byType = {};
      for (const p of match.predictions) {
        if (!byType[p.bet_type]) byType[p.bet_type] = [];
        byType[p.bet_type].push(p);
      }

    html += `<div class="match-card">
      <div class="match-header">
        <div>
          <div class="match-teams">
            ${match.sport_emoji} <strong>${match.home_team}</strong>
            <span class="vs">vs</span>
            <strong>${match.away_team}</strong>
          </div>
          <div class="match-meta">
            <span>🏟️ ${match.tournament}</span>
            <span>🌍 ${match.country}</span>
          </div>
        </div>
        <div class="match-time-badge">
          <div class="time">${fmtTime(match.start_time)}</div>
          <div class="date">📅 ${fmtDate(match.start_date)}</div>
        </div>
      </div>
      <div class="match-markets">`;

    for (const [betType, preds] of Object.entries(byType)) {
      const label = marketLabels[betType] || betType.replace(/_/g, ' ').toUpperCase();
      html += `<div class="market-group">
        <div class="market-label"><div class="dot"></div>${label}</div>
        <div class="market-picks">`;

      for (const p of preds.slice(0, 5)) {
        const oddsTag = p.odds > 0
          ? `<span class="odds-tag">${p.american_odds || p.odds.toFixed(2)}</span>`
          : '';
        const pushTag = p.push_note
          ? `<span class="push-tag">⚠️ ${p.push_note}</span>`
          : '';
        html += `<div class="pick-chip">
          <span>${p.pick}</span>
          ${oddsTag}
          <span class="conf-tag ${confClass(p.confidence)}">${p.confidence.toFixed(0)}%</span>
          ${pushTag}
        </div>`;
      }
      html += '</div></div>';
    }

    html += '</div></div>';
  }
  html += '</div>';
  document.getElementById('content').innerHTML = html;

  } catch(e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>❌ Error loading matches: ${e.message}</p></div>`;
  }
}

async function loadParlay() {
  activeTab = 'parlay';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="parlay"]').classList.add('active');
  showLoading();
  const legs = document.getElementById('parlayLegs').value;
  const strategy = document.getElementById('parlayStrategy').value;

  const body = { num_legs: parseInt(legs), strategy: strategy, target_date: localDateStr() };
  if (selectedSports.length > 0) body.sports = selectedSports;

  try {
    const res = await fetch(`${API}/api/parlay`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.detail) {
      document.getElementById('content').innerHTML = `<div class="empty"><p>⚠️ ${data.detail}</p></div>`;
      return;
    }

    // Count unique sports
    const parlaSports = new Set(data.legs.map(l => l.sport));
    const mixLabel = parlaSports.size > 1
      ? `🌐 Mixed Sports (${[...parlaSports].map(s => data.legs.find(l => l.sport === s).sport_emoji).join(' ')})`
      : data.legs[0].sport_emoji + ' ' + data.legs[0].sport;

    let html = `<div class="parlay-section">
      <div class="parlay-title">
        🎯 ${data.legs.length}-Leg Parlay
        <span class="badge">${strategy}</span>
        <span style="font-size:0.65em;color:var(--dim);font-weight:400;margin-left:8px">${mixLabel}</span>
      </div>`;

    for (let i = 0; i < data.legs.length; i++) {
      const leg = data.legs[i];
      const pushHtml = leg.push_note ? `<span class="push-tag" style="margin-left:6px">⚠️ ${leg.push_note}</span>` : '';
      html += `<div class="parlay-leg">
        <div class="leg-num">${i + 1}</div>
        <div class="leg-info">
          <div class="leg-match">${leg.sport_emoji} ${leg.home_team} vs ${leg.away_team}</div>
          <div class="leg-detail">
            🏟️ ${leg.tournament} • 📅 ${fmtDate(leg.start_date)} • ⏰ ${fmtTime(leg.start_time)}
          </div>
        </div>
        <div class="leg-pick">
          ${leg.pick} ${pushHtml}
          <span class="odds-sm">${leg.american_odds || (leg.odds > 0 ? leg.odds.toFixed(2) : '')}</span>
          <span class="conf-tag ${confClass(leg.confidence)}" style="margin-left:6px">${leg.confidence.toFixed(0)}%</span>
        </div>
      </div>`;
    }

    html += `<div class="parlay-summary">
      <div class="summary-item"><div class="value">${data.combined_confidence.toFixed(1)}%</div><div class="label">Combined Conf</div></div>
      <div class="summary-item"><div class="value">${data.combined_odds.toFixed(2)}x</div><div class="label">Combined Odds</div></div>
      <div class="summary-item"><div class="value">${data.expected_value > 0 ? '+' : ''}${data.expected_value.toFixed(4)}</div><div class="label">Expected Value</div></div>
      <div class="summary-item"><div class="value">${data.risk_level.toUpperCase()}</div><div class="label">Risk Level</div></div>
      <div class="summary-item"><div class="value">$${data.recommended_stake.toFixed(2)}</div><div class="label">Rec. Stake</div></div>
    </div>`;

    html += '</div>';
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>❌ Error: ${e.message}</p></div>`;
  }
}

async function loadValueBets() {
  activeTab = 'value';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="value"]').classList.add('active');
  showLoading();
  try {
    const res = await fetch(`${API}/api/value-bets?target_date=${activeDateStr}`);
    const data = await res.json();
    if (!data.length) {
      document.getElementById('content').innerHTML = '<div class="empty"><p style="font-size:2em">💰</p><p>No value bets found currently. Check back later!</p></div>';
      return;
    }
    let html = '<div class="value-grid">';
    for (const p of data) {
      html += `<div class="value-card">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
          <div>
            <div style="font-weight:700">${p.sport_emoji} ${p.home_team} vs ${p.away_team}</div>
            <div style="color:var(--dim);font-size:0.8em;margin-top:3px">
              🏟️ ${p.tournament} • 📅 ${fmtDate(p.start_date)} • ⏰ ${fmtTime(p.start_time)}
            </div>
          </div>
          <span class="ev-badge">EV: +${p.value_rating.toFixed(3)}</span>
        </div>
        <div style="background:rgba(0,229,160,0.05);border-left:3px solid var(--accent);padding:8px 12px;border-radius:0 8px 8px 0;font-weight:600;margin-bottom:8px">
          ➤ ${p.pick}
          ${p.american_odds ? '<span style="color:var(--accent);margin-left:8px">' + p.american_odds + '</span>' : ''}
        </div>
        <div style="display:flex;gap:12px;color:var(--dim);font-size:0.8em;flex-wrap:wrap">
          <span>📊 ${p.market_display || p.bet_type}</span>
          <span>🎯 Conf: <span class="${confClass(p.confidence)}">${p.confidence.toFixed(0)}%</span></span>
          <span>📈 Our prob: ${(p.probability*100).toFixed(0)}%</span>
          <span>💰 Implied: ${p.odds > 0 ? (100/p.odds).toFixed(0) : '-'}%</span>
        </div>
      </div>`;
    }
    html += '</div>';
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>❌ Error: ${e.message}</p></div>`;
  }
}

// Auto-load on page ready
window.addEventListener('DOMContentLoaded', () => { loadMatches(); });

// ── Render a generic parlay section ──
function renderParlaySection(data, title, badge, borderColor) {
  const parlaSports = new Set(data.legs.map(l => l.sport));
  const mixLabel = parlaSports.size > 1
    ? '🌐 Mixed (' + [...parlaSports].map(s => data.legs.find(l => l.sport === s).sport_emoji).join(' ') + ')'
    : data.legs[0].sport_emoji + ' ' + data.legs[0].sport;

  let html = `<div class="parlay-section" style="border-color:${borderColor}">
    <div class="parlay-title">
      ${title}
      <span class="badge" style="background:${borderColor}">${badge}</span>
      <span style="font-size:0.65em;color:var(--dim);font-weight:400;margin-left:8px">${mixLabel}</span>
    </div>`;

  for (let i = 0; i < data.legs.length; i++) {
    const leg = data.legs[i];
    const pushHtml = leg.push_note ? `<span class="push-tag" style="margin-left:6px">⚠️ ${leg.push_note}</span>` : '';
    html += `<div class="parlay-leg">
      <div class="leg-num" style="background:${borderColor}">${i + 1}</div>
      <div class="leg-info">
        <div class="leg-match">${leg.sport_emoji} ${leg.home_team} vs ${leg.away_team}</div>
        <div class="leg-detail">🏟️ ${leg.tournament} • 📅 ${fmtDate(leg.start_date)} • ⏰ ${fmtTime(leg.start_time)}</div>
      </div>
      <div class="leg-pick">
        ${leg.pick} ${pushHtml}
        <span class="odds-sm">${leg.american_odds || (leg.odds > 0 ? leg.odds.toFixed(2) : '')}</span>
        <span class="conf-tag ${confClass(leg.confidence)}" style="margin-left:6px">${leg.confidence.toFixed(0)}%</span>
      </div>
    </div>`;
  }

  let extras = '';
  if (data.teaser_points > 0) extras += `<div class="summary-item"><div class="value">+${data.teaser_points}</div><div class="label">Pts Bought</div></div>`;
  if (data.flex_miss_allowed > 0) extras += `<div class="summary-item"><div class="value">${data.flex_miss_allowed}</div><div class="label">Misses OK</div></div>`;

  html += `<div class="parlay-summary">
    <div class="summary-item"><div class="value">${data.combined_confidence.toFixed(1)}%</div><div class="label">Combined Conf</div></div>
    <div class="summary-item"><div class="value">${data.combined_odds.toFixed(2)}x</div><div class="label">Combined Odds</div></div>
    <div class="summary-item"><div class="value">${data.expected_value > 0 ? '+' : ''}${data.expected_value.toFixed(4)}</div><div class="label">Expected Value</div></div>
    <div class="summary-item"><div class="value">${data.risk_level.toUpperCase()}</div><div class="label">Risk Level</div></div>
    <div class="summary-item"><div class="value">$${data.recommended_stake.toFixed(2)}</div><div class="label">Rec. Stake</div></div>
    ${extras}
  </div></div>`;
  return html;
}

// ── Same Game Parlays ──
async function loadSGPs() {
  activeTab = 'sgp';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="sgp"]').classList.add('active');
  showLoading();
  try {
    let url = `${API}/api/sgps?num_legs=4&target_date=${activeDateStr}`;
    if (selectedSports.length > 0) url += `&sport=${selectedSports[0]}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!data.length) {
      document.getElementById('content').innerHTML = '<div class="empty"><p style="font-size:2em">🎰</p><p>No SGPs available. Need at least 2 different bet types per game.</p></div>';
      return;
    }
    let html = `<div style="margin-bottom:14px;color:var(--dim);font-size:0.85em">🎰 ${data.length} Same Game Parlays found — combine bets from the SAME game</div>`;
    for (const sgp of data) {
      const match = sgp.legs[0] ? `${sgp.legs[0].home_team} vs ${sgp.legs[0].away_team}` : 'Unknown';
      html += renderParlaySection(sgp, `🎰 SGP — ${match}`, 'SGP', '#ff6b35');
    }
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>❌ Error: ${e.message}</p></div>`;
  }
}

// ── Round Robin ──
async function loadRoundRobin() {
  activeTab = 'roundrobin';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="roundrobin"]').classList.add('active');
  showLoading();
  try {
    const body = { num_picks: 5, combo_size: 3, target_date: activeDateStr };
    if (selectedSports.length > 0) body.sports = selectedSports;
    const res = await fetch(`${API}/api/round-robin`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.detail) {
      document.getElementById('content').innerHTML = `<div class="empty"><p>⚠️ ${data.detail}</p></div>`;
      return;
    }
    let html = `<div style="margin-bottom:14px;color:var(--dim);font-size:0.85em">🔄 ${data.length} Round Robin combos — every possible ${body.combo_size}-leg parlay from your top ${body.num_picks} picks</div>`;
    for (let i = 0; i < data.length; i++) {
      html += renderParlaySection(data[i], `🔄 Round Robin #${i+1}`, `${body.combo_size} of ${body.num_picks}`, '#3b82f6');
    }
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>❌ Error: ${e.message}</p></div>`;
  }
}

// ── Teaser ──
async function loadTeaser() {
  activeTab = 'teaser';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="teaser"]').classList.add('active');
  showLoading();
  try {
    const body = { num_legs: 3, teaser_points: 6.0, target_date: activeDateStr };
    if (selectedSports.length > 0) body.sports = selectedSports;
    const res = await fetch(`${API}/api/teaser`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.detail) {
      document.getElementById('content').innerHTML = `<div class="empty"><p>⚠️ ${data.detail}</p></div>`;
      return;
    }
    let html = `<div style="margin-bottom:14px;color:var(--dim);font-size:0.85em">🎲 Teaser — Buy +${body.teaser_points} points on spreads & totals. Lower payout, higher win rate!</div>`;
    html += renderParlaySection(data, `🎲 ${data.legs.length}-Leg Teaser (+${body.teaser_points} pts)`, 'TEASER', '#f59e0b');
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>❌ Error: ${e.message}</p></div>`;
  }
}

// ── Flex Parlay ──
async function loadFlexParlay() {
  activeTab = 'flex';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="flex"]').classList.add('active');
  showLoading();
  try {
    const body = { num_legs: 5, miss_allowed: 1, target_date: activeDateStr };
    if (selectedSports.length > 0) body.sports = selectedSports;
    const res = await fetch(`${API}/api/flex-parlay`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.detail) {
      document.getElementById('content').innerHTML = `<div class="empty"><p>⚠️ ${data.detail}</p></div>`;
      return;
    }
    let html = `<div style="margin-bottom:14px;color:var(--dim);font-size:0.85em">💪 Flex Parlay — Miss up to ${body.miss_allowed} leg(s) and STILL win! Reduced payout for the insurance.</div>`;
    html += renderParlaySection(data, `💪 Flex Parlay (${body.miss_allowed} miss allowed)`, 'FLEX', '#10b981');
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>❌ Error: ${e.message}</p></div>`;
  }
}

// ── Past Games ───────────────────────────────────────────────────────────────
async function loadPastGames() {
  activeTab = 'past';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="past"]').classList.add('active');
  const targetDate = activeDateStr;
  showLoading();
  try {
    let matchData = [];
    if (selectedSports.length > 0) {
      const promises = selectedSports.map(s =>
        fetch(`${API}/api/past-games?target_date=${targetDate}&sport=${s}&min_confidence=50`).then(r => r.json())
      );
      const resolved = await Promise.all(promises);
      resolved.forEach(r => { if (Array.isArray(r)) matchData.push(...r); });
    } else {
      const res = await fetch(`${API}/api/past-games?target_date=${targetDate}&min_confidence=50`);
      const d = await res.json();
      if (Array.isArray(d)) matchData = d;
    }
    const fmtDateLabel = new Date(targetDate + 'T12:00:00Z').toLocaleDateString('en-US', {weekday:'long', month:'long', day:'numeric', year:'numeric'});
    let html = `<div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;flex-wrap:wrap">
      <div style="color:var(--dim);font-size:0.9em">📅 <strong style="color:var(--text)">${fmtDateLabel}</strong></div>
      <div style="color:var(--dim);font-size:0.8em">${matchData.length} match${matchData.length !== 1 ? 'es' : ''} found — use the date navigator above to change date</div>
    </div>`;
    if (!matchData.length) {
      html += '<div class="empty"><p style="font-size:2em">📅</p><p>No games found for this date. Try a different date or sport filter.</p></div>';
      document.getElementById('content').innerHTML = html;
      return;
    }
    html += '<div class="match-grid">';
    for (const match of matchData) {
      const isFinished = match.status === 'finished';
      const isLive = match.status === 'live';
      const scoreHtml = isFinished && match.home_score != null
        ? `<div style="font-size:1.4em;font-weight:700;color:var(--text);letter-spacing:2px">${match.home_score} \u2013 ${match.away_score}</div>`
        : `<div style="color:var(--dim);font-size:0.8em">${fmtTime(match.start_time)}</div>`;
      const statusColor = isFinished ? 'var(--dim)' : isLive ? '#ef4444' : 'var(--accent)';
      const statusLabel = isFinished ? '\u2705 Final' : isLive ? '\ud83d\udd34 LIVE' : '\u23f0 Scheduled';
      const topPicks = match.predictions.slice(0, 3);
      html += `<div class="match-card">
        <div class="match-header">
          <div>
            <div class="match-teams">
              ${match.sport_emoji} <strong>${match.home_team}</strong>
              <span class="vs">vs</span>
              <strong>${match.away_team}</strong>
            </div>
            <div class="match-meta">
              <span>\ud83c\udfd9\ufe0f ${match.tournament}</span>
              <span style="color:${statusColor}">${statusLabel}</span>
            </div>
          </div>
          <div style="text-align:right">
            ${scoreHtml}
            <div class="date" style="margin-top:4px">\ud83d\udcc5 ${fmtDate(match.start_date)}</div>
          </div>
        </div>
        <div class="match-markets">
          <div class="market-group">
            <div class="market-label"><div class="dot"></div>\ud83c\udfc6 Top Predictions</div>
            <div class="market-picks">`;
      for (const p of topPicks) {
        html += `<div class="pick-chip">
          <span>${p.pick}</span>
          <span class="odds-tag">${p.american_odds || p.odds.toFixed(2)}</span>
          <span class="conf-tag ${confClass(p.confidence)}">${p.confidence.toFixed(0)}%</span>
        </div>`;
      }
      html += `</div></div></div></div>`;
    }
    html += '</div>';
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('content').innerHTML = `<div class="empty"><p>\u274c Error: ${e.message}</p></div>`;
  }
}
</script>
</body>
</html>"""
