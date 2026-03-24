"""
FastAPI Web Server — Premium REST API + Web Dashboard for the Prediction Agent.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
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
    title="🏆 Bet Prediction Abibi",
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
    now_utc = datetime.now(timezone.utc)
    for match in grouped:
        ev = match["event"]
        sport_obj = match["sport"]

        # Skip finished / cancelled matches
        if ev.status.value in ("finished", "cancelled"):
            continue
        # Skip matches that kicked off more than 105 min ago (game is likely over)
        if ev.status.value == "not_started":
            game_start = ev.start_time if ev.start_time.tzinfo else ev.start_time.replace(tzinfo=timezone.utc)
            if game_start < now_utc - timedelta(minutes=105):
                continue

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
    """Get all matches for any date — finished, live, or scheduled — with scores."""
    from datetime import timedelta
    d = _parse_date(target_date) or (date.today() - timedelta(days=1))
    sports = _parse_sports([sport] if sport else None)
    matches = await agent.get_past_results(sports=sports, target_date=d)
    results = []
    for match in matches:
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
            predictions=[],
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
<title>Bet Prediction Abibi</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
/* ─────────────────────────────────────────────
   PREMIUM DARK THEME — Bet Prediction Abibi
───────────────────────────────────────────── */
:root {
  --bg:           #080810;
  --surface:      #0d0d1a;
  --card:         #111120;
  --card2:        #16162a;
  --card-hov:     #18183a;
  --border:       rgba(255,255,255,0.06);
  --border2:      rgba(255,255,255,0.10);
  --accent:       #7c3aed;
  --accent2:      #a855f7;
  --accent-glow:  rgba(124,58,237,0.25);
  --accent-dim:   rgba(124,58,237,0.12);
  --gold:         #fbbf24;
  --gold-dim:     rgba(251,191,36,0.12);
  --text:         #e8e8f4;
  --dim:          #4a4a6a;
  --dim2:         #6a6a8a;
  --green:        #10b981;
  --green-dim:    rgba(16,185,129,0.12);
  --yellow:       #f59e0b;
  --yellow-dim:   rgba(245,158,11,0.12);
  --orange:       #f97316;
  --orange-dim:   rgba(249,115,22,0.12);
  --red:          #ef4444;
  --blue:         #3b82f6;
  --odds-bg:      rgba(255,255,255,0.05);
  --odds-hov:     rgba(124,58,237,0.15);
}

* { margin:0; padding:0; box-sizing:border-box; }

html { scroll-behavior:smooth; }

body {
  font-family:'Inter',sans-serif;
  background:var(--bg);
  color:var(--text);
  min-height:100vh;
  -webkit-font-smoothing:antialiased;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--accent); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent2); }

.container { max-width:1200px; margin:0 auto; padding:0 0 60px; }

/* ─────────────────────────────────────────────
   TOP NAV  (sticky, glass)
───────────────────────────────────────────── */
.top-nav {
  position:sticky; top:0; z-index:100;
  background:rgba(8,8,16,0.85);
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border2);
  padding:0 20px;
  display:flex; align-items:center; gap:16px; height:52px;
}
.logo {
  display:flex; align-items:center; gap:8px;
}
.logo-icon {
  width:28px; height:28px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:7px;
  display:flex; align-items:center; justify-content:center;
  font-size:14px; flex-shrink:0;
  box-shadow:0 0 12px var(--accent-glow);
}
.logo-text {
  font-weight:800; font-size:1.05em; letter-spacing:-0.3px;
  background:linear-gradient(135deg,#c4b5fd,#fbbf24);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text;
}
.nav-pill {
  margin-left:8px;
  background:rgba(16,185,129,0.15);
  color:var(--green);
  font-size:0.65em; font-weight:700;
  padding:3px 8px; border-radius:20px;
  border:1px solid rgba(16,185,129,0.3);
  letter-spacing:0.5px;
  animation:pulse-green 2s ease-in-out infinite;
}
@keyframes pulse-green {
  0%,100% { opacity:1; }
  50% { opacity:0.6; }
}
.live-time {
  margin-left:auto;
  font-size:0.8em; color:var(--dim2);
  font-variant-numeric:tabular-nums;
  letter-spacing:0.5px;
}

/* ─────────────────────────────────────────────
   DATE NAVIGATOR
───────────────────────────────────────────── */
.date-nav {
  display:flex; align-items:center; gap:6px; flex-wrap:wrap;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  padding:8px 20px;
}
.date-nav-label {
  font-weight:700; font-size:0.88em; flex:1; min-width:160px;
  color:var(--text);
}
.date-nav-btn {
  padding:5px 12px; border-radius:6px;
  border:1px solid var(--border2);
  background:transparent; color:var(--dim2);
  cursor:pointer; font-size:11px; font-weight:600;
  font-family:inherit; transition:all 0.15s;
}
.date-nav-btn:hover { border-color:var(--accent); color:var(--accent); background:var(--accent-dim); }
.date-nav-btn.today { background:var(--accent-dim); color:var(--accent2); border-color:var(--accent); }
.date-nav input[type="date"] {
  padding:4px 8px; border-radius:6px;
  border:1px solid var(--border2);
  background:var(--card); color:var(--text);
  font-family:inherit; font-size:11px;
  transition:border-color 0.15s;
}
.date-nav input[type="date"]:focus { border-color:var(--accent); outline:none; }

/* ─────────────────────────────────────────────
   SPORT FILTER BAR
───────────────────────────────────────────── */
.filter-bar {
  background:var(--surface);
  border-bottom:1px solid var(--border);
  padding:10px 20px;
  display:flex; gap:6px; flex-wrap:wrap; align-items:center;
}
.filter-label { font-size:11px; color:var(--dim2); font-weight:700; letter-spacing:0.5px; margin-right:4px; }
.sport-chip {
  padding:5px 12px; border-radius:20px;
  font-size:11px; font-weight:600;
  cursor:pointer;
  border:1px solid var(--border2);
  color:var(--dim2); background:transparent;
  transition:all 0.15s; white-space:nowrap;
}
.sport-chip:hover { border-color:var(--accent2); color:var(--text); }
.sport-chip.active {
  background:var(--accent-dim); border-color:var(--accent2);
  color:var(--accent2);
  box-shadow:0 0 8px var(--accent-glow);
}

/* ─────────────────────────────────────────────
   LEAGUE FILTER BAR
───────────────────────────────────────────── */
.league-bar {
  background:var(--surface);
  border-bottom:1px solid var(--border);
  padding:8px 20px;
  display:flex; gap:6px; flex-wrap:wrap; align-items:center;
  min-height:40px;
}
.league-bar.hidden { display:none; }
.league-chip {
  padding:4px 11px; border-radius:20px;
  font-size:11px; font-weight:600;
  cursor:pointer;
  border:1px solid var(--border2);
  color:var(--dim2); background:transparent;
  transition:all 0.15s; white-space:nowrap; max-width:200px;
  overflow:hidden; text-overflow:ellipsis;
}
.league-chip:hover { border-color:var(--gold); color:var(--text); }
.league-chip.active {
  background:var(--gold-dim); border-color:var(--gold);
  color:var(--gold);
  box-shadow:0 0 8px rgba(251,191,36,0.2);
}
.league-search {
  padding:4px 10px; border-radius:20px;
  border:1px solid var(--border2);
  background:var(--card); color:var(--text);
  font-size:11px; font-family:inherit; width:140px;
  transition:border-color 0.15s;
}
.league-search:focus { border-color:var(--gold); outline:none; }
.league-search::placeholder { color:var(--dim); }

/* ─────────────────────────────────────────────
   MAIN TABS
───────────────────────────────────────────── */
.main-tabs {
  display:flex; gap:0;
  background:var(--surface);
  border-bottom:1px solid var(--border2);
  padding:0 20px; overflow-x:auto; flex-wrap:nowrap;
}
.main-tab {
  padding:13px 16px; cursor:pointer;
  color:var(--dim2); white-space:nowrap;
  border-bottom:2px solid transparent; margin-bottom:-1px;
  font-weight:600; font-size:13px;
  background:none; border-top:none; border-left:none; border-right:none;
  font-family:inherit; transition:all 0.2s; letter-spacing:0.2px;
}
.main-tab:hover { color:var(--text); }
.main-tab.active { color:#fff; border-bottom-color:var(--accent2); }

/* ─────────────────────────────────────────────
   CONTROLS BAR
───────────────────────────────────────────── */
.controls-bar {
  padding:10px 20px; display:flex; gap:8px; flex-wrap:wrap;
  background:var(--card); border-bottom:1px solid var(--border);
  align-items:center;
}
.ctrl-input, .ctrl-select {
  padding:7px 12px; border-radius:7px;
  border:1px solid var(--border2);
  background:var(--surface); color:var(--text);
  font-size:12px; font-family:inherit;
  transition:border-color 0.15s;
}
.ctrl-input { width:120px; }
.ctrl-input:focus, .ctrl-select:focus { border-color:var(--accent); outline:none; box-shadow:0 0 0 3px var(--accent-dim); }
.btn {
  padding:8px 18px; border-radius:7px; border:none;
  font-weight:700; cursor:pointer; font-size:12px;
  font-family:inherit; transition:all 0.2s;
  letter-spacing:0.3px;
}
.btn-primary {
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;
  box-shadow:0 2px 12px var(--accent-glow);
}
.btn-primary:hover { transform:translateY(-1px); box-shadow:0 4px 20px var(--accent-glow); }
.btn-gold {
  background:linear-gradient(135deg,#92400e,#b45309);
  color:#fef3c7;
  box-shadow:0 2px 10px rgba(180,83,9,0.3);
}
.btn-gold:hover { transform:translateY(-1px); }
.btn:active { transform:translateY(0); }

/* ─────────────────────────────────────────────
   CONFIDENCE PILLS
───────────────────────────────────────────── */
.conf-pill {
  display:inline-flex; align-items:center; gap:5px;
  padding:3px 9px; border-radius:20px;
  font-size:0.75em; font-weight:700; white-space:nowrap;
}
.conf-pill-hi { background:var(--green-dim); color:var(--green); border:1px solid rgba(16,185,129,0.25); }
.conf-pill-md { background:var(--yellow-dim); color:var(--yellow); border:1px solid rgba(245,158,11,0.25); }
.conf-pill-lo { background:var(--orange-dim); color:var(--orange); border:1px solid rgba(249,115,22,0.25); }
.conf-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.conf-dot-hi { background:var(--green); box-shadow:0 0 6px var(--green); }
.conf-dot-md { background:var(--yellow); box-shadow:0 0 6px var(--yellow); }
.conf-dot-lo { background:var(--orange); box-shadow:0 0 6px var(--orange); }

/* Legacy colour helpers for JS */
.conf-hi { color:var(--green); }
.conf-md { color:var(--yellow); }
.conf-lo { color:var(--orange); }

/* ─────────────────────────────────────────────
   CONFIDENCE LEGEND (inline banner)
───────────────────────────────────────────── */
.conf-legend {
  background:var(--card2);
  border-bottom:1px solid var(--border);
  padding:8px 20px;
  display:flex; gap:16px; flex-wrap:wrap; align-items:center;
}
.conf-legend-label { font-size:11px; color:var(--dim2); font-weight:700; letter-spacing:0.5px; margin-right:4px; }
.conf-legend-item { display:flex; align-items:center; gap:5px; font-size:11px; color:var(--dim2); }

/* ─────────────────────────────────────────────
   MATCH GRID
───────────────────────────────────────────── */
.match-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(min(580px,100%),1fr));
  gap:1px; background:var(--border);
}

/* ─────────────────────────────────────────────
   MATCH CARD
───────────────────────────────────────────── */
.hrb-card {
  background:var(--card);
  overflow:hidden;
  transition:background 0.2s, box-shadow 0.2s;
}
.hrb-card:hover {
  background:var(--card-hov);
  box-shadow:inset 0 0 0 1px rgba(168,85,247,0.15),0 4px 24px rgba(0,0,0,0.4);
}

.hrb-card-header {
  padding:14px 16px 12px;
  border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(22,22,42,0.8),transparent);
}
.hrb-breadcrumb {
  font-size:0.7em; color:var(--dim2); margin-bottom:9px;
  display:flex; align-items:center; gap:3px; flex-wrap:wrap;
}
.hrb-breadcrumb b { color:var(--text); font-weight:600; }
.hrb-breadcrumb .sep { color:var(--dim); font-size:0.85em; }

.hrb-match-teams {
  display:flex; align-items:center; gap:12px; margin-bottom:8px;
}
.hrb-team { font-weight:800; font-size:1.05em; flex:1; letter-spacing:-0.3px; }
.hrb-team.away { text-align:right; }
.hrb-vs {
  font-size:0.7em; color:var(--dim2); font-weight:500;
  padding:3px 10px;
  background:rgba(255,255,255,0.04);
  border:1px solid var(--border);
  border-radius:20px; white-space:nowrap;
}
.hrb-match-time {
  font-size:0.78em; color:var(--dim2); text-align:center;
  display:flex; align-items:center; gap:6px; justify-content:center;
}
.live-badge {
  background:rgba(239,68,68,0.15); color:#ef4444;
  font-size:0.7em; font-weight:800; padding:2px 7px; border-radius:4px;
  border:1px solid rgba(239,68,68,0.3);
  animation:pulse-red 1.2s ease-in-out infinite;
}
@keyframes pulse-red { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ─────────────────────────────────────────────
   INNER TABS
───────────────────────────────────────────── */
.hrb-inner-tabs {
  display:flex; overflow-x:auto;
  background:rgba(0,0,0,0.2);
  border-bottom:1px solid var(--border);
}
.hrb-inner-tab {
  padding:9px 14px; cursor:pointer;
  font-size:12px; font-weight:600; color:var(--dim2);
  border-bottom:2px solid transparent; margin-bottom:-1px;
  white-space:nowrap; background:none;
  border-top:none; border-left:none; border-right:none;
  font-family:inherit; transition:all 0.15s;
}
.hrb-inner-tab:hover { color:var(--text); }
.hrb-inner-tab.active { color:#fff; border-bottom-color:var(--accent2); }

/* ─────────────────────────────────────────────
   MONEYLINE GRID
───────────────────────────────────────────── */
.hrb-ml-header {
  display:grid; grid-template-columns:1fr 90px 90px 90px;
  padding:7px 16px; border-bottom:1px solid var(--border);
  font-size:0.68em; color:var(--dim); font-weight:700;
  text-transform:uppercase; letter-spacing:0.6px;
  background:rgba(0,0,0,0.15);
}
.hrb-ml-header span { text-align:center; }
.hrb-ml-header span:first-child { text-align:left; }

.hrb-ml-row {
  display:grid; grid-template-columns:1fr 90px 90px 90px;
  padding:10px 16px; border-bottom:1px solid var(--border);
  align-items:center; gap:6px;
}
.hrb-ml-teams { font-size:0.8em; color:var(--dim2); line-height:1.7; }

.hrb-odds-btn {
  position:relative; overflow:hidden;
  text-align:center;
  background:var(--odds-bg);
  border:1px solid var(--border2);
  border-radius:7px; padding:9px 4px;
  font-weight:700; font-size:0.88em;
  cursor:pointer; transition:all 0.2s;
}
.hrb-odds-btn::before {
  content:'';
  position:absolute; inset:0;
  background:radial-gradient(circle at 50% 50%, var(--accent-glow), transparent 70%);
  opacity:0; transition:opacity 0.2s;
}
.hrb-odds-btn:hover { background:var(--odds-hov); border-color:var(--accent); transform:scale(1.03); }
.hrb-odds-btn:hover::before { opacity:1; }
.hrb-odds-btn.positive { color:var(--green); }
.hrb-odds-btn.negative { color:var(--text); }
.hrb-odds-btn.neutral { color:var(--dim2); }
.hrb-odds-btn:active { transform:scale(0.98); }

/* ─────────────────────────────────────────────
   MARKET ACCORDION ROWS
───────────────────────────────────────────── */
.hrb-market-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:13px 16px; border-bottom:1px solid var(--border);
  cursor:pointer; transition:background 0.15s; gap:8px;
}
.hrb-market-row:hover { background:rgba(124,58,237,0.06); }
.hrb-market-name { font-size:0.88em; font-weight:500; flex:1; }
.hrb-market-right { display:flex; align-items:center; gap:8px; flex-shrink:0; }

.sgp-badge {
  font-size:0.62em; font-weight:800; color:#fff;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  padding:3px 8px; border-radius:4px; letter-spacing:0.5px;
  box-shadow:0 1px 6px var(--accent-glow);
}
.expand-arrow { color:var(--dim); font-size:11px; transition:transform 0.25s; }
.expand-arrow.open { transform:rotate(180deg); }

/* ─────────────────────────────────────────────
   EXPANDED PICKS PANEL
───────────────────────────────────────────── */
.hrb-picks-panel {
  display:none;
  border-bottom:1px solid var(--border);
  background:rgba(0,0,0,0.3);
}
.hrb-picks-panel.open {
  display:block;
  animation:slideDown 0.2s ease-out;
}
@keyframes slideDown {
  from { opacity:0; transform:translateY(-6px); }
  to   { opacity:1; transform:translateY(0); }
}
.hrb-pick-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:9px 16px 9px 24px;
  border-bottom:1px solid rgba(255,255,255,0.03); gap:8px;
}
.hrb-pick-row:last-child { border-bottom:none; }
.hrb-pick-name { font-size:0.85em; font-weight:500; flex:1; color:var(--text); }
.hrb-pick-meta { display:flex; align-items:center; gap:8px; flex-shrink:0; }
.hrb-pick-odds {
  background:var(--odds-bg); padding:4px 10px;
  border-radius:5px; border:1px solid var(--border2);
  font-size:0.82em; font-weight:700;
}

/* ─────────────────────────────────────────────
   LOADING / EMPTY
───────────────────────────────────────────── */
.loading-box { text-align:center; padding:80px 20px; color:var(--dim2); }
.spinner {
  width:40px; height:40px;
  border:3px solid var(--border2);
  border-top-color:var(--accent);
  border-radius:50%;
  animation:spin 0.75s linear infinite;
  margin:0 auto 16px;
  box-shadow:0 0 12px var(--accent-glow);
}
@keyframes spin { to { transform:rotate(360deg); } }
.loading-box p { font-size:0.9em; }
.empty-box { text-align:center; padding:80px 20px; color:var(--dim2); }
.empty-box p { font-size:0.9em; margin-top:10px; }

/* ─────────────────────────────────────────────
   PARLAY CARDS
───────────────────────────────────────────── */
.parlay-wrap { padding:16px; }
.parlay-card {
  background:var(--card);
  border:1px solid var(--border2);
  border-radius:12px; overflow:hidden;
  margin-bottom:16px;
  transition:box-shadow 0.2s;
}
.parlay-card:hover { box-shadow:0 8px 32px rgba(0,0,0,0.4); }
.parlay-card-header {
  padding:15px 18px;
  display:flex; align-items:center; gap:12px;
  border-bottom:1px solid var(--border);
  background:linear-gradient(135deg,rgba(124,58,237,0.1),transparent);
}
.parlay-card-title { font-weight:700; font-size:1em; flex:1; }
.parlay-type-badge {
  font-size:0.62em; font-weight:800; color:#fff;
  padding:4px 10px; border-radius:5px;
  text-transform:uppercase; letter-spacing:0.7px;
}

.parlay-leg-row {
  display:flex; align-items:center; gap:12px;
  padding:12px 18px; border-bottom:1px solid var(--border);
}
.leg-num-circle {
  width:28px; height:28px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:0.75em; color:#fff; flex-shrink:0;
  box-shadow:0 0 10px rgba(124,58,237,0.4);
}
.leg-info { flex:1; min-width:0; }
.leg-match-name { font-weight:600; font-size:0.88em; }
.leg-detail { font-size:0.73em; color:var(--dim2); margin-top:2px; }
.leg-pick-right { text-align:right; flex-shrink:0; }
.leg-pick-name { font-weight:700; color:#fff; font-size:0.88em; }
.leg-pick-odds { color:var(--dim2); font-size:0.75em; margin-top:2px; }

.parlay-stats {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(90px,1fr)); gap:0;
  border-top:1px solid var(--border);
  background:rgba(0,0,0,0.2);
}
.stat-item {
  padding:12px 10px; text-align:center;
  border-right:1px solid var(--border);
}
.stat-item:last-child { border-right:none; }
.stat-value { font-weight:800; font-size:1.05em; color:var(--accent2); }
.stat-label { font-size:0.65em; color:var(--dim); text-transform:uppercase; letter-spacing:0.5px; margin-top:3px; }

/* ─────────────────────────────────────────────
   VALUE BETS GRID
───────────────────────────────────────────── */
.value-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(min(380px,100%),1fr));
  gap:12px; padding:16px;
}
.value-card {
  background:var(--card); border:1px solid var(--border2);
  border-radius:10px; padding:16px;
  border-left:3px solid var(--gold);
  transition:box-shadow 0.2s, transform 0.2s;
}
.value-card:hover { box-shadow:0 4px 20px rgba(251,191,36,0.1); transform:translateY(-2px); }

/* ─────────────────────────────────────────────
   FOOTER
───────────────────────────────────────────── */
footer {
  padding:24px 20px; text-align:center;
  color:var(--dim); font-size:0.75em;
  border-top:1px solid var(--border);
  margin-top:20px; line-height:1.8;
}

/* ─────────────────────────────────────────────
   RESPONSIVE
───────────────────────────────────────────── */
@media (max-width:768px) {
  .match-grid { grid-template-columns:1fr; }
  .hrb-ml-header, .hrb-ml-row { grid-template-columns:1fr 72px 72px 72px; }
  .parlay-leg-row { flex-wrap:wrap; }
  .leg-pick-right { text-align:left; width:100%; margin-left:40px; }
  .parlay-stats { grid-template-columns:repeat(2,1fr); }
  .controls-bar { gap:6px; }
  .ctrl-input, .ctrl-select { width:100%; }
  .btn { width:100%; padding:11px; }
  .main-tabs { overflow-x:auto; }
  .main-tab { font-size:11px; padding:11px 10px; }
}
@media (max-width:400px) {
  .hrb-ml-header, .hrb-ml-row { grid-template-columns:1fr 60px 60px 60px; }
  .hrb-odds-btn { font-size:0.78em; padding:7px 2px; }
}
</style>
</head>
<body>
<div class="container">

<!-- ── Top Nav ── -->
<div class="top-nav">
  <div class="logo">
    <div class="logo-icon">🏆</div>
    <span class="logo-text">Bet Prediction Abibi</span>
    <span class="nav-pill">LIVE</span>
  </div>
  <span class="live-time" id="liveTime"></span>
</div>

<!-- ── Date Navigator ── -->
<div class="date-nav">
  <button class="date-nav-btn" onclick="shiftDate(-2)">&#171; &#8209;2d</button>
  <button class="date-nav-btn" onclick="shiftDate(-1)">&#9664; Prev</button>
  <div class="date-nav-label" id="dateNavLabel">Today</div>
  <input type="date" id="dateNavPicker" onchange="setNavDate(this.value)">
  <button class="date-nav-btn today" onclick="setNavDate(localDateStr())">Today</button>
  <button class="date-nav-btn" onclick="shiftDate(1)">Tomorrow &#9654;</button>
  <button class="date-nav-btn" onclick="shiftDate(2)">+2d &#187;</button>
</div>

<!-- ── Sport chips ── -->
<div class="filter-bar">
  <span class="filter-label">SPORT</span>
  <div class="sport-chip active" data-sport="" onclick="toggleSport(this)">&#127758; All</div>
  <div class="sport-chip" data-sport="football" onclick="toggleSport(this)">&#9917; Soccer</div>
  <div class="sport-chip" data-sport="basketball" onclick="toggleSport(this)">&#127936; Basketball</div>
  <div class="sport-chip" data-sport="tennis" onclick="toggleSport(this)">&#127934; Tennis</div>
  <div class="sport-chip" data-sport="baseball" onclick="toggleSport(this)">&#9918; Baseball</div>
  <div class="sport-chip" data-sport="american-football" onclick="toggleSport(this)">&#127944; Football</div>
  <div class="sport-chip" data-sport="volleyball" onclick="toggleSport(this)">&#127952; Volleyball</div>
  <div class="sport-chip" data-sport="ice-hockey" onclick="toggleSport(this)">&#127954; Hockey</div>
</div>

<!-- ── League filter ── -->
<div class="league-bar hidden" id="leagueBar">
  <span class="filter-label">LEAGUE</span>
  <input class="league-search" id="leagueSearch" type="text" placeholder="&#128269; Search league..." oninput="filterLeagueChips(this.value)">
  <div id="leagueChips" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
</div>

<!-- ── Main tabs ── -->
<div class="main-tabs">
  <button class="main-tab active" data-tab="matches" onclick="switchMain(this,'matches')">Matches</button>
  <button class="main-tab" data-tab="parlay" onclick="switchMain(this,'parlay')">Parlay</button>
  <button class="main-tab" data-tab="sgp" onclick="switchMain(this,'sgp')">SGP</button>
  <button class="main-tab" data-tab="roundrobin" onclick="switchMain(this,'roundrobin')">Round Robin</button>
  <button class="main-tab" data-tab="teaser" onclick="switchMain(this,'teaser')">Teaser</button>
  <button class="main-tab" data-tab="flex" onclick="switchMain(this,'flex')">Flex Parlay</button>
  <button class="main-tab" data-tab="value" onclick="switchMain(this,'value')">Value Bets</button>
</div>

<!-- ── Controls bar ── -->
<div class="controls-bar">
  <input class="ctrl-input" type="number" id="minConf" value="50" min="0" max="100" placeholder="Min Conf %">
  <input class="ctrl-input" type="number" id="parlayLegs" value="6" min="2" max="15" placeholder="Legs">
  <select class="ctrl-select" id="parlayStrategy">
    <option value="balanced">Balanced</option>
    <option value="safe">Safe</option>
    <option value="value">Value</option>
  </select>
  <button class="btn btn-primary" onclick="reloadActive()">&#8635; Refresh</button>
  <button class="btn btn-gold" onclick="loadValueBets()">&#128176; Value Bets</button>
</div>

<!-- ── Confidence legend ── -->
<div class="conf-legend">
  <span class="conf-legend-label">CONFIDENCE</span>
  <div class="conf-legend-item"><span class="conf-dot conf-dot-hi" style="width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 5px var(--green);flex-shrink:0;display:inline-block"></span> <span>75%+ Strong</span></div>
  <div class="conf-legend-item"><span class="conf-dot conf-dot-md" style="width:8px;height:8px;border-radius:50%;background:var(--yellow);box-shadow:0 0 5px var(--yellow);flex-shrink:0;display:inline-block"></span> <span>60–74% Moderate</span></div>
  <div class="conf-legend-item"><span class="conf-dot conf-dot-lo" style="width:8px;height:8px;border-radius:50%;background:var(--orange);box-shadow:0 0 5px var(--orange);flex-shrink:0;display:inline-block"></span> <span>&lt;60% Caution</span></div>
  <span style="margin-left:auto;font-size:11px;color:var(--dim)">AI-weighted: form · position · H2H · injuries · home advantage</span>
</div>

<div id="content">
  <div class="loading-box"><div class="spinner"></div><p>Analyzing games&hellip;</p></div>
</div>

<footer>
  <p>&#9888;&#65039; Bet responsibly. For informational and entertainment purposes only.</p>
  <p style="margin-top:4px">Data from API-Football &amp; ESPN. Past performance does not guarantee future results.</p>
</footer>

</div><!-- /container -->

<script>
const API = '';
let selectedSports = [];
let activeTab = 'matches';
let activeDateStr = '';
let innerTabState = {};
let selectedLeague = ''; // '' = all leagues
let _allMatchData = [];  // cache for client-side league filtering

function localDateStr() {
  const n = new Date();
  return `${n.getFullYear()}-${String(n.getMonth()+1).padStart(2,'0')}-${String(n.getDate()).padStart(2,'0')}`;
}
function offsetDate(ymd, days) {
  const d = new Date(ymd+'T12:00:00'); d.setDate(d.getDate()+days);
  return d.toISOString().split('T')[0];
}
function dateLabelOf(ymd) {
  const today = localDateStr();
  const d = new Date(ymd+'T12:00:00');
  const base = d.toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'});
  if (ymd===today) return `Today — ${base}`;
  if (ymd===offsetDate(today,1)) return `Tomorrow — ${base}`;
  if (ymd===offsetDate(today,-1)) return `Yesterday — ${base}`;
  const delta = Math.round((new Date(ymd)-new Date(today))/86400000);
  return `${base} (${delta>0?'+':''}${delta}d)`;
}
function setNavDate(ymd, reload=true) {
  activeDateStr = ymd;
  document.getElementById('dateNavLabel').textContent = dateLabelOf(ymd);
  document.getElementById('dateNavPicker').value = ymd;
  if (reload) reloadActive();
}
function shiftDate(d) { setNavDate(offsetDate(activeDateStr, d)); }
function fmtTime(s) {
  if (!s) return '';
  const d = new Date(s); if (isNaN(d)) return s;
  return d.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',hour12:true,timeZone:'America/New_York'});
}
function fmtDate(s) {
  if (!s) return '';
  const d = new Date(s); if (isNaN(d)) return s;
  return d.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'America/New_York'});
}

// Live clock
setInterval(() => {
  const el = document.getElementById('liveTime');
  if (el) el.textContent = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
}, 1000);

setNavDate(localDateStr(), false);

function toggleSport(el) {
  const s = el.dataset.sport;
  if (s === '') {
    document.querySelectorAll('.sport-chip').forEach(c=>c.classList.remove('active'));
    el.classList.add('active'); selectedSports = [];
  } else {
    document.querySelector('.sport-chip[data-sport=""]').classList.remove('active');
    el.classList.toggle('active');
    selectedSports = [...document.querySelectorAll('.sport-chip.active')].map(c=>c.dataset.sport).filter(Boolean);
    if (!selectedSports.length) document.querySelector('.sport-chip[data-sport=""]').classList.add('active');
  }
  selectedLeague = ''; // reset league filter when sport changes
  reloadActive();
}

// ── League filter helpers ────────────────────────────────────────────────
function buildLeagueBar(matchData) {
  const bar = document.getElementById('leagueBar');
  const chips = document.getElementById('leagueChips');
  if (!bar || !chips) return;

  // Collect unique tournaments sorted by count
  const counts = {};
  matchData.forEach(m => { counts[m.tournament] = (counts[m.tournament]||0) + 1; });
  const leagues = Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([t])=>t);

  if (leagues.length <= 1) { bar.classList.add('hidden'); return; }
  bar.classList.remove('hidden');

  chips.innerHTML = leagues.map(l => {
    const active = selectedLeague === l ? ' active' : '';
    return `<div class="league-chip${active}" data-league="${l.replace(/"/g,'&quot;')}" onclick="toggleLeague(this)" title="${l}">${l}</div>`;
  }).join('');
}

function toggleLeague(el) {
  const l = el.dataset.league;
  if (selectedLeague === l) {
    selectedLeague = '';
    document.querySelectorAll('.league-chip').forEach(c => c.classList.remove('active'));
  } else {
    selectedLeague = l;
    document.querySelectorAll('.league-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
  }
  renderFilteredMatches();
}

function filterLeagueChips(q) {
  q = q.toLowerCase();
  document.querySelectorAll('.league-chip').forEach(c => {
    c.style.display = c.dataset.league.toLowerCase().includes(q) ? '' : 'none';
  });
}

function renderFilteredMatches() {
  const data = selectedLeague
    ? _allMatchData.filter(m => m.tournament === selectedLeague)
    : _allMatchData;
  const tdate = activeDateStr;
  const isPast = tdate < localDateStr();
  const totalPreds = data.reduce((s,m)=>s+m.predictions.length,0);
  const leagueLabel = selectedLeague ? ` &bull; <span style="color:var(--gold)">${selectedLeague}</span>` : '';
  const info = `<div style="padding:10px 20px;font-size:0.8em;color:var(--dim2);border-bottom:1px solid var(--border);background:var(--card2)">
    ${data.length} matches${!isPast?' &bull; '+totalPreds+' total predictions':''}${leagueLabel} &mdash; ${dateLabelOf(tdate)}
  </div>`;
  if (!data.length) {
    document.getElementById('content').innerHTML = info + '<div class="empty-box"><p>No matches in this league for the selected date.</p></div>';
    return;
  }
  const cards = data.map(m => renderMatchCard(m)).join('');
  document.getElementById('content').innerHTML = info + '<div class="match-grid">' + cards + '</div>';
}
function switchMain(el, tab) {
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active'); activeTab = tab; reloadActive();
}
function reloadActive() {
  if (activeTab==='matches') loadMatches();
  else if (activeTab==='parlay') loadParlay();
  else if (activeTab==='sgp') loadSGPs();
  else if (activeTab==='roundrobin') loadRoundRobin();
  else if (activeTab==='teaser') loadTeaser();
  else if (activeTab==='flex') loadFlexParlay();
  else if (activeTab==='value') loadValueBets();
  else loadMatches();
}
function showLoading() {
  document.getElementById('content').innerHTML = '<div class="loading-box"><div class="spinner"></div><p>Analyzing games&hellip;</p></div>';
}

function confClass(c) { return c>=75?'conf-hi':c>=60?'conf-md':'conf-lo'; }
function confPill(c) {
  const cls = c>=75?'conf-pill-hi':c>=60?'conf-pill-md':'conf-pill-lo';
  const dcls = c>=75?'conf-dot-hi':c>=60?'conf-dot-md':'conf-dot-lo';
  return `<span class="conf-pill ${cls}"><span class="conf-dot ${dcls}"></span>${c.toFixed(0)}%</span>`;
}

// Bet type categorisation
const GAME_LINE_TYPES = new Set(['moneyline','game_result_90','spread','alternate_spread','over_under','alternate_total','btts','double_chance','draw_no_bet','correct_score','game_props','first_to_score','overtime','race_to','futures','winning_margin']);
const PLAYER_PROP_TYPES = new Set(['player_props']);
const HALVES_TYPES = new Set(['first_half','halftime_result','halftime_over_under']);
const TEAM_TYPES = new Set(['team_total','quarter_props']);
const GOALS_TYPES = new Set(['odd_even','corners']);
const SGP_TYPES = new Set(['over_under','btts','double_chance','game_result_90','spread','correct_score','player_props','first_half','halftime_result','halftime_over_under','team_total','odd_even','alternate_spread','alternate_total','game_props','winning_margin']);

const marketNames = {
  moneyline:'Moneyline', game_result_90:'Game Result (90 Min + Stoppage Time)',
  spread:'Spread', alternate_spread:'Alternate Spread',
  over_under:'Total Goals', alternate_total:'Alternate Total',
  btts:'Both Teams to Score', double_chance:'Double Chance',
  draw_no_bet:'Winner (Push if Tied)', correct_score:'Correct Score',
  game_props:'Game Props', first_to_score:'First to Score',
  overtime:'Overtime / Extra Time', odd_even:'Odd vs Even Total Goals',
  quarter_props:'Quarter / Period Props', race_to:'Race to X',
  futures:'Futures', first_half:'1st Half Markets',
  halftime_result:'Half-Time Result', halftime_over_under:'1st Half Over/Under',
  player_props:'Anytime Goalscorer', team_total:'Team Total Goals',
  corners:'Corners Over/Under', winning_margin:'Winning Margin',
};

function oddsClass(ao) {
  if (!ao) return 'neutral';
  return ao.startsWith('+') ? 'positive' : 'negative';
}
function togglePanel(pid, aid) {
  const p = document.getElementById(pid);
  const a = document.getElementById(aid);
  if (!p) return;
  p.classList.toggle('open');
  if (a) a.classList.toggle('open');
}
function switchInnerTab(cardId, tabKey, btn) {
  innerTabState[cardId] = tabKey;
  document.querySelectorAll(`[id^="ipane-${cardId}-"]`).forEach(el => el.style.display='none');
  const pane = document.getElementById(`ipane-${cardId}-${tabKey}`);
  if (pane) pane.style.display = 'block';
  const container = btn.closest('.hrb-inner-tabs');
  if (container) {
    container.querySelectorAll('.hrb-inner-tab').forEach(t=>t.classList.remove('active'));
    btn.classList.add('active');
  }
}

function renderMatchCard(match) {
  const id = 'm' + (match.match_id || (match.home_team+match.away_team).replace(/\W/g,''));
  const isFinished = match.status === 'finished';
  const isLive = match.status === 'live';
  const hasScore = match.home_score != null && match.away_score != null;

  const byCategory = {
    gl: match.predictions.filter(p => GAME_LINE_TYPES.has(p.bet_type)),
    pp: match.predictions.filter(p => PLAYER_PROP_TYPES.has(p.bet_type)),
    hv: match.predictions.filter(p => HALVES_TYPES.has(p.bet_type)),
    tm: match.predictions.filter(p => TEAM_TYPES.has(p.bet_type)),
    go: match.predictions.filter(p => GOALS_TYPES.has(p.bet_type)),
  };
  const tabDefs = [
    { key:'gl', label:'Game Lines' },
    { key:'pp', label:'Player Props' },
    { key:'hv', label:'Halves' },
    { key:'tm', label:'Teams' },
    { key:'go', label:'Goals' },
  ].filter(t => t.key==='gl' || byCategory[t.key].length > 0);

  const defaultKey = innerTabState[id] || 'gl';

  let timeLine = '';
  if (isFinished && hasScore) {
    timeLine = `<strong style="font-size:1.2em;letter-spacing:3px">${match.home_score} &ndash; ${match.away_score}</strong>&nbsp;<span style="color:var(--dim2);font-size:0.72em">&#10003; Final</span>`;
  } else if (isLive && hasScore) {
    timeLine = `<strong style="font-size:1.2em;color:var(--red);letter-spacing:3px">${match.home_score} &ndash; ${match.away_score}</strong>&nbsp;<span class="live-badge">&#9679; LIVE</span>`;
  } else {
    timeLine = `&#128197; ${fmtDate(match.start_date)} &nbsp;&bull;&nbsp; &#9200; ${fmtTime(match.start_time)} EDT`;
  }

  function renderCatContent(key, preds) {
    if (!preds.length) {
      return '<div style="padding:14px 16px;color:var(--dim2);font-size:0.83em">No predictions available for this category.</div>';
    }
    const grouped = {};
    for (const p of preds) {
      if (!grouped[p.bet_type]) grouped[p.bet_type] = [];
      grouped[p.bet_type].push(p);
    }
    let html = '';

    if (key === 'gl' && grouped['moneyline']) {
      const ml = grouped['moneyline'];
      const hp = ml.find(p => p.pick && (p.pick.toLowerCase().includes(match.home_team.toLowerCase().split(' ')[0]) || p.pick.toLowerCase().includes('home')));
      const ap = ml.find(p => p.pick && (p.pick.toLowerCase().includes(match.away_team.toLowerCase().split(' ')[0]) || p.pick.toLowerCase().includes('away')));
      const tp = ml.find(p => p.pick && (p.pick.toLowerCase().includes('draw') || p.pick.toLowerCase().includes('tie') || p.pick.toLowerCase() === 'x'));
      const p0 = hp || ml[0]; const p1 = tp || ml[1]; const p2 = ap || ml[2];
      html += `<div class="hrb-ml-header"><span></span><span>HOME</span><span>TIE</span><span>AWAY</span></div>
        <div class="hrb-ml-row">
          <div class="hrb-ml-teams"><div>${match.home_team}</div><div>${match.away_team}</div></div>
          <div class="hrb-odds-btn ${p0?oddsClass(p0.american_odds):'neutral'}">${p0?(p0.american_odds||p0.odds.toFixed(2)):'-'}</div>
          <div class="hrb-odds-btn ${p1?oddsClass(p1.american_odds):'neutral'}">${p1?(p1.american_odds||p1.odds.toFixed(2)):'-'}</div>
          <div class="hrb-odds-btn ${p2?oddsClass(p2.american_odds):'neutral'}">${p2?(p2.american_odds||p2.odds.toFixed(2)):'-'}</div>
        </div>`;
      delete grouped['moneyline'];
    }

    for (const [bt, bpreds] of Object.entries(grouped)) {
      const pid = `pp-${id}-${key}-${bt.replace(/[^a-z0-9]/g,'')}`;
      const aid = `pa-${id}-${key}-${bt.replace(/[^a-z0-9]/g,'')}`;
      const hasSgp = SGP_TYPES.has(bt);
      let label = marketNames[bt] || bt.replace(/_/g,' ');
      if (bt==='team_total' && bpreds[0]?.team_name) label = `${bpreds[0].team_name} Total Goals`;
      html += `<div class="hrb-market-row" onclick="togglePanel('${pid}','${aid}')">
        <span class="hrb-market-name">${label}</span>
        <div class="hrb-market-right">
          ${hasSgp ? '<span class="sgp-badge">SGP</span>' : ''}
          <span class="expand-arrow" id="${aid}">&#9660;</span>
        </div>
      </div>
      <div class="hrb-picks-panel" id="${pid}">`;
      for (const p of bpreds.slice(0,8)) {
        const od = p.american_odds || (p.odds>0?p.odds.toFixed(2):'—');
        html += `<div class="hrb-pick-row">
          <span class="hrb-pick-name">${p.pick}</span>
          <div class="hrb-pick-meta">
            <span class="hrb-pick-odds ${oddsClass(p.american_odds)}">${od}</span>
            ${confPill(p.confidence)}
          </div>
        </div>`;
      }
      html += '</div>';
    }
    return html;
  }

  let tabNav = '';
  let tabPanes = '';
  for (const t of tabDefs) {
    const active = t.key === defaultKey;
    tabNav += `<button class="hrb-inner-tab${active?' active':''}" onclick="switchInnerTab('${id}','${t.key}',this)">${t.label}</button>`;
    tabPanes += `<div id="ipane-${id}-${t.key}" style="display:${active?'block':'none'}">${renderCatContent(t.key, byCategory[t.key])}</div>`;
  }

  return `<div class="hrb-card">
    <div class="hrb-card-header">
      <div class="hrb-breadcrumb">
        <span>${match.sport_emoji}</span><span class="sep">/</span>
        <span>${match.country}</span><span class="sep">/</span>
        <span>${match.tournament}</span><span class="sep">/</span>
        <b>${match.home_team} vs ${match.away_team}</b>
      </div>
      <div class="hrb-match-teams">
        <div class="hrb-team">${match.home_team}</div>
        <div class="hrb-vs">vs</div>
        <div class="hrb-team away">${match.away_team}</div>
      </div>
      <div class="hrb-match-time">${timeLine}</div>
    </div>
    <div class="hrb-inner-tabs">${tabNav}</div>
    <div>${tabPanes}</div>
  </div>`;
}

async function loadMatches() {
  activeTab = 'matches';
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  const mt = document.querySelector('[data-tab="matches"]'); if (mt) mt.classList.add('active');
  showLoading();
  const minConf = document.getElementById('minConf').value;
  const tdate = activeDateStr;
  const isPast = tdate < localDateStr();
  try {
    let matchData = [];
    if (isPast) {
      const qsp = selectedSports.length ? `&sport=${selectedSports[0]}` : '';
      const res = await fetch(`${API}/api/past-games?target_date=${tdate}&min_confidence=0${qsp}`);
      const d = await res.json(); if (Array.isArray(d)) matchData = d;
    } else {
      if (selectedSports.length > 0) {
        const results = await Promise.all(selectedSports.map(s=>
          fetch(`${API}/api/matches?min_confidence=${minConf}&sport=${s}&target_date=${tdate}`).then(r=>r.json())
        ));
        results.forEach(r=>{ if(Array.isArray(r)) matchData.push(...r); });
      } else {
        const res = await fetch(`${API}/api/matches?min_confidence=${minConf}&target_date=${tdate}`);
        const d = await res.json(); if (Array.isArray(d)) matchData = d;
      }
    }
    if (!matchData.length) {
      document.getElementById('content').innerHTML = '<div class="empty-box"><p style="font-size:2em">&#128269;</p><p>No matches found. Try a different date or sport.</p></div>';
      document.getElementById('leagueBar').classList.add('hidden');
      return;
    }
    // Sort by start_time ascending
    matchData.sort((a,b) => new Date(a.start_time) - new Date(b.start_time));
    _allMatchData = matchData;
    // Reset league selection if it no longer exists in new data
    if (selectedLeague && !matchData.some(m => m.tournament === selectedLeague)) selectedLeague = '';
    buildLeagueBar(matchData);
    renderFilteredMatches();
  } catch(e) {
    document.getElementById('content').innerHTML = `<div class="empty-box"><p>&#10060; ${e.message}</p></div>`;
  }
}

function renderParlayCard(data, title, badge, color) {
  color = color || 'var(--accent)';
  let html = `<div class="parlay-card">
    <div class="parlay-card-header" style="border-left:3px solid ${color}">
      <span class="parlay-card-title">${title}</span>
      <span class="parlay-type-badge" style="background:${color}">${badge}</span>
    </div>`;
  data.legs.forEach((leg,i) => {
    const push = leg.push_note ? `<span style="font-size:0.7em;color:var(--yellow);margin-left:4px">&#9888; ${leg.push_note}</span>` : '';
    html += `<div class="parlay-leg-row">
      <div class="leg-num-circle" style="background:${color}">${i+1}</div>
      <div class="leg-info">
        <div class="leg-match-name">${leg.sport_emoji} ${leg.home_team} vs ${leg.away_team}</div>
        <div class="leg-detail">${leg.tournament} &bull; ${fmtTime(leg.start_time)}</div>
      </div>
      <div class="leg-pick-right">
        <div class="leg-pick-name">${leg.pick}${push}</div>
        <div class="leg-pick-odds">${leg.american_odds||''} ${confPill(leg.confidence)}</div>
      </div>
    </div>`;
  });
  let extras = '';
  if (data.teaser_points>0) extras += `<div class="stat-item"><div class="stat-value">+${data.teaser_points}</div><div class="stat-label">Pts Bought</div></div>`;
  if (data.flex_miss_allowed>0) extras += `<div class="stat-item"><div class="stat-value">${data.flex_miss_allowed}</div><div class="stat-label">Misses OK</div></div>`;
  html += `<div class="parlay-stats">
    <div class="stat-item"><div class="stat-value">${data.combined_confidence.toFixed(1)}%</div><div class="stat-label">Confidence</div></div>
    <div class="stat-item"><div class="stat-value">${data.combined_odds.toFixed(2)}x</div><div class="stat-label">Odds</div></div>
    <div class="stat-item"><div class="stat-value">${data.expected_value>0?'+':''}${data.expected_value.toFixed(3)}</div><div class="stat-label">Exp. Value</div></div>
    <div class="stat-item"><div class="stat-value">${data.risk_level.toUpperCase()}</div><div class="stat-label">Risk</div></div>
    <div class="stat-item"><div class="stat-value">$${data.recommended_stake.toFixed(2)}</div><div class="stat-label">Stake</div></div>
    ${extras}
  </div></div>`;
  return html;
}

async function loadParlay() {
  activeTab='parlay';
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  document.querySelector('[data-tab="parlay"]').classList.add('active');
  showLoading();
  const legs = document.getElementById('parlayLegs').value;
  const strategy = document.getElementById('parlayStrategy').value;
  const body = {num_legs:parseInt(legs), strategy, target_date:activeDateStr};
  if (selectedSports.length) body.sports = selectedSports;
  try {
    const res = await fetch(`${API}/api/parlay`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    if (data.detail) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#9888; ${data.detail}</p></div>`; return; }
    document.getElementById('content').innerHTML = `<div class="parlay-wrap">${renderParlayCard(data, `${data.legs.length}-Leg Parlay`, strategy.toUpperCase())}</div>`;
  } catch(e) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#10060; ${e.message}</p></div>`; }
}

async function loadSGPs() {
  activeTab='sgp';
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  document.querySelector('[data-tab="sgp"]').classList.add('active');
  showLoading();
  try {
    let url = `${API}/api/sgps?num_legs=4&target_date=${activeDateStr}`;
    if (selectedSports.length) url += `&sport=${selectedSports[0]}`;
    const res = await fetch(url); const data = await res.json();
    if (!data.length) { document.getElementById('content').innerHTML='<div class="empty-box"><p style="font-size:2em">&#127920;</p><p>No SGPs available.</p></div>'; return; }
    let html = `<div class="parlay-wrap"><div style="margin-bottom:12px;color:var(--dim2);font-size:0.83em">${data.length} Same Game Parlays</div>`;
    for (const sgp of data) {
      const m = sgp.legs[0] ? `${sgp.legs[0].home_team} vs ${sgp.legs[0].away_team}` : '';
      html += renderParlayCard(sgp, `SGP &mdash; ${m}`, 'SGP', '#ff6b35');
    }
    html += '</div>';
    document.getElementById('content').innerHTML = html;
  } catch(e) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#10060; ${e.message}</p></div>`; }
}

async function loadRoundRobin() {
  activeTab='roundrobin';
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  document.querySelector('[data-tab="roundrobin"]').classList.add('active');
  showLoading();
  try {
    const body = {num_picks:5, combo_size:3, target_date:activeDateStr};
    if (selectedSports.length) body.sports = selectedSports;
    const res = await fetch(`${API}/api/round-robin`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    if (!data.length||data.detail) { document.getElementById('content').innerHTML='<div class="empty-box"><p>Not enough picks for round robin.</p></div>'; return; }
    let html = `<div class="parlay-wrap"><div style="margin-bottom:12px;color:var(--dim2);font-size:0.83em">${data.length} Round Robin combinations</div>`;
    data.forEach((p,i) => { html += renderParlayCard(p, `Round Robin #${i+1}`, 'RR', '#0ea5e9'); });
    html += '</div>';
    document.getElementById('content').innerHTML = html;
  } catch(e) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#10060; ${e.message}</p></div>`; }
}

async function loadTeaser() {
  activeTab='teaser';
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  document.querySelector('[data-tab="teaser"]').classList.add('active');
  showLoading();
  try {
    const body = {num_legs:3, teaser_points:6, target_date:activeDateStr};
    if (selectedSports.length) body.sports = selectedSports;
    const res = await fetch(`${API}/api/teaser`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    if (data.detail) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#9888; ${data.detail}</p></div>`; return; }
    document.getElementById('content').innerHTML = `<div class="parlay-wrap">${renderParlayCard(data, `Teaser (+${body.teaser_points} pts)`, 'TEASER', '#f59e0b')}</div>`;
  } catch(e) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#10060; ${e.message}</p></div>`; }
}

async function loadFlexParlay() {
  activeTab='flex';
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  document.querySelector('[data-tab="flex"]').classList.add('active');
  showLoading();
  try {
    const body = {num_legs:5, miss_allowed:1, target_date:activeDateStr};
    if (selectedSports.length) body.sports = selectedSports;
    const res = await fetch(`${API}/api/flex-parlay`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    if (data.detail) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#9888; ${data.detail}</p></div>`; return; }
    document.getElementById('content').innerHTML = `<div class="parlay-wrap">${renderParlayCard(data, `Flex Parlay (miss ${body.miss_allowed})`, 'FLEX', '#22c55e')}</div>`;
  } catch(e) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#10060; ${e.message}</p></div>`; }
}

async function loadValueBets() {
  activeTab='value';
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));
  document.querySelector('[data-tab="value"]').classList.add('active');
  showLoading();
  try {
    const res = await fetch(`${API}/api/value-bets?target_date=${activeDateStr}`);
    const data = await res.json();
    if (!data.length) { document.getElementById('content').innerHTML='<div class="empty-box"><p style="font-size:2em">&#128176;</p><p>No value bets found.</p></div>'; return; }
    let html = '<div class="value-grid">';
    for (const p of data) {
      html += `<div class="value-card">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
          <div>
            <div style="font-weight:700;font-size:0.95em">${p.sport_emoji} ${p.home_team} vs ${p.away_team}</div>
            <div style="color:var(--dim2);font-size:0.75em;margin-top:3px">${p.tournament} &bull; ${fmtTime(p.start_time)}</div>
          </div>
          <span style="background:var(--gold-dim);color:var(--gold);padding:4px 10px;border-radius:6px;font-weight:700;font-size:0.78em;border:1px solid rgba(251,191,36,0.25);white-space:nowrap;margin-left:8px">EV +${p.value_rating.toFixed(3)}</span>
        </div>
        <div style="background:var(--accent-dim);border-left:3px solid var(--accent);padding:8px 12px;border-radius:0 7px 7px 0;font-weight:600;font-size:0.9em;margin-bottom:10px">
          ${p.pick} ${p.american_odds?`<span style="color:var(--accent2);margin-left:8px">${p.american_odds}</span>`:''}
        </div>
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
          <span style="font-size:0.75em;color:var(--dim2)">${p.market_display||p.bet_type}</span>
          ${confPill(p.confidence)}
          <span style="font-size:0.75em;color:var(--dim2)">Prob: ${(p.probability*100).toFixed(0)}%</span>
        </div>
      </div>`;
    }
    html += '</div>';
    document.getElementById('content').innerHTML = html;
  } catch(e) { document.getElementById('content').innerHTML=`<div class="empty-box"><p>&#10060; ${e.message}</p></div>`; }
}

window.addEventListener('DOMContentLoaded', () => loadMatches());
</script>
</body>
</html>"""
