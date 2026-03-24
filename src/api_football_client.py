"""
API-Football Client — Primary data source for the prediction agent.

Data priority:
  1. API-Football v3 (soccer — all tracked leagues)
  2. ESPN public API (basketball, baseball, NFL, NHL — free, no key)
  3. Demo data (generated fallback when APIs are unavailable)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, date, timezone
from typing import Optional

import httpx
from cachetools import TTLCache
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models import Sport, Team, Tournament, TeamStats, HeadToHead, MatchEvent, MatchStatus, PlayerInfo

# Reuse ESPN constants + demo provider from sofascore_client (no duplication)
from src.sofascore_client import (
    DemoDataProvider,
    _ESPN_LEAGUES,
    _HEADERS,
    _ESPN_BASE,
    SofaScoreClient,
)

# ── API-Football config ───────────────────────────────────────────────────────
_AF_BASE = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
_AF_KEY = os.getenv("API_FOOTBALL_KEY", "")

# Map API-Football league IDs → (display name, country, our tid, priority)
_AF_LEAGUES: dict[int, tuple[str, str, int, int]] = {
    # ── Top 5 + UEFA ──────────────────────────────────────────────────────
    2:   ("Champions League",        "Europe",      7,   520),
    3:   ("Europa League",           "Europe",      6,   510),
    848: ("Conference League",       "Europe",      848, 490),
    39:  ("Premier League",          "England",     17,  500),
    40:  ("Championship",            "England",     40,  300),
    45:  ("FA Cup",                  "England",     45,  280),
    140: ("La Liga",                 "Spain",       8,   480),
    143: ("Copa del Rey",            "Spain",       143, 270),
    135: ("Serie A",                 "Italy",       23,  460),
    136: ("Serie B",                 "Italy",       136, 240),
    78:  ("Bundesliga",              "Germany",     35,  450),
    79:  ("2. Bundesliga",           "Germany",     79,  260),
    61:  ("Ligue 1",                 "France",      34,  440),
    # ── Other top European ────────────────────────────────────────────────
    88:  ("Eredivisie",              "Netherlands", 88,  360),
    94:  ("Primeira Liga",           "Portugal",    94,  350),
    203: ("Süper Lig",               "Turkey",      203, 330),
    179: ("Scottish Premiership",    "Scotland",    179, 290),
    207: ("Swiss Super League",      "Switzerland", 207, 250),
    # ── Americas ──────────────────────────────────────────────────────────
    253: ("MLS",                     "USA",         253, 430),
    262: ("Liga MX",                 "Mexico",      262, 420),
    71:  ("Serie A",                 "Brazil",      71,  400),
    128: ("Primera Division",        "Argentina",   128, 390),
    13:  ("Copa Libertadores",       "South America", 13, 410),
    # ── International / Qualifiers ────────────────────────────────────────
    5:   ("UEFA Nations League",     "Europe",      5,   380),
    4:   ("Euro Qualification",      "Europe",      4,   370),
    32:  ("World Cup - Qual. UEFA",  "Europe",      32,  365),
    9:   ("World Cup - Qual. CONMEBOL", "South America", 9, 355),
    29:  ("World Cup - Qual. CONCACAF", "North America", 29, 345),
}

_STATUS_MAP: dict[str, MatchStatus] = {
    "NS":   MatchStatus.NOT_STARTED,
    "1H":   MatchStatus.LIVE,
    "HT":   MatchStatus.LIVE,
    "2H":   MatchStatus.LIVE,
    "ET":   MatchStatus.LIVE,
    "P":    MatchStatus.LIVE,
    "FT":   MatchStatus.FINISHED,
    "AET":  MatchStatus.FINISHED,
    "PEN":  MatchStatus.FINISHED,
    "PST":  MatchStatus.POSTPONED,
    "CANC": MatchStatus.CANCELLED,
    "ABD":  MatchStatus.CANCELLED,
}

# Aggressive caching — standings/fixtures change at most daily
_cache: TTLCache = TTLCache(maxsize=500, ttl=300)
# Standings are valid for a whole day; use a separate long-TTL cache
_standings_cache: TTLCache = TTLCache(maxsize=50, ttl=3600)


class APIFootballClient:
    """
    Unified sports data client.

    Soccer → API-Football v3 (real fixtures, standings, H2H)
    Other sports → ESPN public API (real data, no key required)
    Fallback → DemoDataProvider (generated data)
    """

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self._demo = DemoDataProvider()
        # SofaScoreClient(demo_mode=True) acts as ESPN+demo gateway
        # (SofaScore API calls are never made; it goes straight to ESPN)
        self._fallback = SofaScoreClient(demo_mode=True)

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                headers={
                    "x-apisports-key": _AF_KEY,
                    "Accept": "application/json",
                },
                timeout=httpx.Timeout(20.0),
                follow_redirects=True,
            )
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()
        await self._fallback.close()

    # ── Low-level API-Football GET ────────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _af_get(self, endpoint: str, params: dict | None = None) -> dict:
        """Cached GET against API-Football v3. Returns {} on any error."""
        if not _AF_KEY:
            return {}

        cache_key = f"{endpoint}:{sorted((params or {}).items())}"
        if cache_key in _cache:
            return _cache[cache_key]

        http = await self._get_http()
        try:
            resp = await http.get(f"{_AF_BASE}/{endpoint}", params=params or {})
            resp.raise_for_status()
            data = resp.json()
            if data.get("errors"):
                logger.warning(f"API-Football {endpoint} errors: {data['errors']}")
                return {}
            _cache[cache_key] = data
            return data
        except httpx.HTTPStatusError as e:
            logger.warning(f"API-Football {endpoint} → HTTP {e.response.status_code}")
            return {}
        except Exception as e:
            logger.warning(f"API-Football {endpoint} failed: {e}")
            return {}

    # ── Scheduled Events ──────────────────────────────────────────────────

    async def get_scheduled_events(
        self,
        sport: Sport,
        target_date: Optional[date] = None,
    ) -> list[MatchEvent]:
        d = target_date or date.today()

        if sport == Sport.SOCCER and _AF_KEY:
            events = await self._fetch_af_fixtures(d)
            if events:
                logger.info(f"Found {len(events)} soccer fixtures via API-Football")
                return events

        # ESPN (NBA, NFL, MLB, NHL, …) + demo fallback
        return await self._fallback.get_scheduled_events(sport, d)

    async def _fetch_af_fixtures(self, d: date) -> list[MatchEvent]:
        """Fetch all real fixtures for a date.

        Priority: tracked top-league games first. If none are found (e.g.
        during international breaks) we fall back to ALL leagues returned
        by the API so the app always shows real games instead of demo data.
        """
        data = await self._af_get("fixtures", {"date": d.isoformat(), "timezone": "America/New_York"})
        events_top: list[MatchEvent] = []
        events_all: list[MatchEvent] = []
        for f in data.get("response", []):
            try:
                events_all.append(self._parse_af_fixture(f))
                lid = f.get("league", {}).get("id", 0)
                if lid in _AF_LEAGUES:
                    events_top.append(events_all[-1])
            except Exception as e:
                logger.warning(f"Skipping AF fixture: {e}")
        if events_top:
            return events_top
        # No top-league games today (e.g. international break) — use all real games
        if events_all:
            logger.info(f"No top-league fixtures today, using all {len(events_all)} real API fixtures")
        return events_all

    def _parse_af_fixture(self, f: dict) -> MatchEvent:
        fix    = f.get("fixture", {})
        league = f.get("league", {})
        teams  = f.get("teams", {})
        goals  = f.get("goals", {})

        lid = league.get("id", 0)
        name, country, tid, priority = _AF_LEAGUES.get(lid, (
            league.get("name", "Unknown League"),
            league.get("country", ""),
            lid,
            50,  # low priority for untracked leagues
        ))

        status = _STATUS_MAP.get(fix.get("status", {}).get("short", "NS"), MatchStatus.NOT_STARTED)
        ts = fix.get("timestamp", 0)
        # Store as UTC so the frontend can convert to any local timezone
        start_time = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)

        is_active = status in (MatchStatus.FINISHED, MatchStatus.LIVE)
        home_score = goals.get("home") if is_active else None
        away_score = goals.get("away") if is_active else None

        home_t = teams.get("home", {})
        away_t = teams.get("away", {})

        return MatchEvent(
            id=fix.get("id", 0),
            tournament=Tournament(
                id=tid,
                name=name,
                sport=Sport.SOCCER,
                country=country,
                slug=name.lower().replace(" ", "-"),
                priority=priority,
            ),
            home_team=Team(
                id=home_t.get("id", 0),
                name=home_t.get("name", "Unknown"),
                short_name=(home_t.get("name", "") or "")[:3].upper(),
                sport=Sport.SOCCER,
            ),
            away_team=Team(
                id=away_t.get("id", 0),
                name=away_t.get("name", "Unknown"),
                short_name=(away_t.get("name", "") or "")[:3].upper(),
                sport=Sport.SOCCER,
            ),
            start_time=start_time,
            status=status,
            home_score=home_score,
            away_score=away_score,
        )

    # ── Live Events ───────────────────────────────────────────────────────

    async def get_live_events(self, sport: Sport) -> list[MatchEvent]:
        if sport == Sport.SOCCER and _AF_KEY:
            data = await self._af_get("fixtures", {"live": "all"})
            events: list[MatchEvent] = []
            for f in data.get("response", []):
                if f.get("league", {}).get("id", 0) not in _AF_LEAGUES:
                    continue
                try:
                    events.append(self._parse_af_fixture(f))
                except Exception:
                    pass
            return events
        return []

    # ── Enrichment ────────────────────────────────────────────────────────

    async def enrich_event(self, event: MatchEvent) -> MatchEvent:
        if event.tournament.sport == Sport.SOCCER and _AF_KEY:
            league_id = self._tid_to_league_id(event.tournament.id)
            if league_id:
                return await self._enrich_af_event(event, league_id)

        # Non-soccer or no key: delegate to SofaScoreClient's demo/ESPN enricher
        return await self._fallback.enrich_event(event)

    async def _enrich_af_event(self, event: MatchEvent, league_id: int) -> MatchEvent:
        season = self._current_season()

        # Standings are expensive to re-fetch per match: use long TTL cache key
        standings_key = f"standings:{league_id}:{season}"
        if standings_key in _standings_cache:
            standings_data = _standings_cache[standings_key]
        else:
            standings_data = await self._af_get("standings", {"league": league_id, "season": season})
            if standings_data:
                _standings_cache[standings_key] = standings_data

        h2h_data, odds_data = await asyncio.gather(
            self._af_get("fixtures/headtohead", {
                "h2h": f"{event.home_team.id}-{event.away_team.id}",
                "last": 10,
            }),
            self._af_get("odds", {"fixture": event.id, "bookmaker": 1}),
            return_exceptions=True,
        )

        if standings_data and not isinstance(standings_data, Exception):
            self._apply_af_standings(event, standings_data)
        if not isinstance(h2h_data, Exception) and h2h_data:
            self._apply_af_h2h(event, h2h_data)
        if not isinstance(odds_data, Exception) and odds_data:
            self._apply_af_odds(event, odds_data)

        # If standings call had no data for these teams, fill in with demo stats
        if event.home_stats is None or event.away_stats is None:
            enriched = self._demo.enrich_event(event)
            if event.home_stats is None:
                event.home_stats = enriched.home_stats
            if event.away_stats is None:
                event.away_stats = enriched.away_stats
            if event.h2h is None:
                event.h2h = enriched.h2h

        return event

    # ── AF Response Parsers ───────────────────────────────────────────────

    def _apply_af_standings(self, event: MatchEvent, data: dict) -> None:
        rows: list[dict] = []
        for entry in data.get("response", []):
            for group in entry.get("league", {}).get("standings", []):
                rows.extend(group)

        for row in rows:
            team_data = row.get("team", {})
            team_id   = team_data.get("id")
            if team_id not in (event.home_team.id, event.away_team.id):
                continue

            all_s  = row.get("all", {})
            goals  = all_s.get("goals", {})
            gp     = all_s.get("played", 0)
            wins   = all_s.get("win", 0)
            draws  = all_s.get("draw", 0)
            losses = all_s.get("lose", 0)
            gf     = goals.get("for") or 0
            ga     = goals.get("against") or 0

            stats = TeamStats(
                team_id=team_id,
                team_name=team_data.get("name", ""),
                wins=wins,
                draws=draws,
                losses=losses,
                form_string=row.get("form", "") or "",
                goals_scored=gf,
                goals_conceded=ga,
                avg_goals_scored=round(gf / max(gp, 1), 2),
                avg_goals_conceded=round(ga / max(gp, 1), 2),
                games_played=gp,
                league_position=row.get("rank", 0),
                points=row.get("points", 0),
            )
            if team_id == event.home_team.id:
                stats.team_name = event.home_team.name
                event.home_stats = stats
            else:
                stats.team_name = event.away_team.name
                event.away_stats = stats

    def _apply_af_h2h(self, event: MatchEvent, data: dict) -> None:
        fixtures = data.get("response", [])
        h2h = HeadToHead(
            team1_id=event.home_team.id,
            team2_id=event.away_team.id,
            total_matches=len(fixtures),
        )
        recent: list[dict] = []
        for f in fixtures:
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            hg    = goals.get("home") or 0
            ag    = goals.get("away") or 0
            is_home = teams.get("home", {}).get("id") == event.home_team.id
            if is_home:
                h2h.team1_goals += hg
                h2h.team2_goals += ag
                if hg > ag:   h2h.team1_wins += 1
                elif ag > hg: h2h.team2_wins += 1
                else:         h2h.draws += 1
            else:
                h2h.team1_goals += ag
                h2h.team2_goals += hg
                if ag > hg:   h2h.team1_wins += 1
                elif hg > ag: h2h.team2_wins += 1
                else:         h2h.draws += 1
            recent.append(f)
            if len(recent) >= 10:
                break
        h2h.recent_matches = recent
        event.h2h = h2h

    def _apply_af_odds(self, event: MatchEvent, data: dict) -> None:
        for item in data.get("response", []):
            for bookmaker in item.get("bookmakers", []):
                for bet in bookmaker.get("bets", []):
                    if bet.get("name") in ("Match Winner", "Home/Draw/Away"):
                        for val in bet.get("values", []):
                            try:
                                odd = float(val.get("odd", 0))
                            except (TypeError, ValueError):
                                continue
                            v = val.get("value", "")
                            if v == "Home":   event.home_odds = odd
                            elif v == "Draw": event.draw_odds = odd
                            elif v == "Away": event.away_odds = odd
                        return  # first matching market is enough

    # ── Helpers ───────────────────────────────────────────────────────────

    def _tid_to_league_id(self, tid: int) -> Optional[int]:
        """Convert our internal tournament id back to an API-Football league id."""
        for lid, (_, _, t, _) in _AF_LEAGUES.items():
            if t == tid:
                return lid
        return None

    @staticmethod
    def _current_season() -> int:
        """Return the current football season year (Aug–Jul cycle)."""
        today = date.today()
        return today.year if today.month >= 8 else today.year - 1
