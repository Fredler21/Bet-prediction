"""
SofaScore API Client — Fetches live schedules, stats, H2H, standings, lineups.

SofaScore's API requires browser-like access. This client supports multiple methods:
1. Direct API (works from browsers / residential IPs)
2. ScrapingBee/ScraperAPI proxy (for server environments)
3. Demo mode with realistic sample data (for development/testing)

Set SOFASCORE_PROXY_KEY in .env if using a scraping proxy service.
"""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import httpx
from cachetools import TTLCache
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models import (
    Sport, Team, Tournament, TeamStats, HeadToHead,
    MatchEvent, MatchStatus, PlayerInfo,
)

# ── ESPN Public API (free, no key needed) ────────────────────────────────────
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Maps our Sport enum to ESPN endpoint(s): (espn_sport, league_slug, display_name, country, tournament_id, priority)
_ESPN_LEAGUES: dict[Sport, list[tuple[str, str, str, str, int, int]]] = {
    Sport.SOCCER: [
        ("soccer", "eng.1", "Premier League", "England", 17, 500),
        ("soccer", "esp.1", "La Liga", "Spain", 8, 480),
        ("soccer", "ita.1", "Serie A", "Italy", 23, 460),
        ("soccer", "ger.1", "Bundesliga", "Germany", 35, 450),
        ("soccer", "fra.1", "Ligue 1", "France", 34, 440),
        ("soccer", "uefa.champions", "Champions League", "Europe", 7, 520),
        ("soccer", "usa.1", "MLS", "USA", 242, 350),
    ],
    Sport.BASKETBALL: [
        ("basketball", "nba", "NBA", "USA", 132, 500),
    ],
    Sport.BASEBALL: [
        ("baseball", "mlb", "MLB", "USA", 11205, 480),
    ],
    Sport.AMERICAN_FOOTBALL: [
        ("football", "nfl", "NFL", "USA", 9464, 500),
    ],
    Sport.HOCKEY: [
        ("hockey", "nhl", "NHL", "USA", 234, 480),
    ],
}

# Rate-limit friendly cache
_cache = TTLCache(maxsize=500, ttl=settings.sofascore_cache_ttl)

# Standard headers to mimic browser requests
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Connection": "keep-alive",
}

BASE = settings.sofascore_base_url

# Proxy API key for scraping services (ScrapingBee, ScraperAPI, etc.)
PROXY_KEY = os.getenv("SOFASCORE_PROXY_KEY", "")


class SofaScoreClient:
    """Async client for the SofaScore API with proxy + demo fallback."""

    def __init__(self, demo_mode: bool = False):
        self._client: Optional[httpx.AsyncClient] = None
        self._demo_mode = demo_mode
        self._demo = DemoDataProvider()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=_HEADERS,
                timeout=httpx.Timeout(15.0),
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _get(self, url: str) -> dict:
        """Perform a cached GET request. Falls back to demo data on 403."""
        if url in _cache:
            return _cache[url]

        if self._demo_mode:
            return {}

        client = await self._get_client()

        # Try direct API first
        try:
            # If we have a proxy key, route through scraping proxy
            if PROXY_KEY:
                proxy_url = (
                    f"https://app.scrapingbee.com/api/v1/"
                    f"?api_key={PROXY_KEY}"
                    f"&url={url}"
                    f"&render_js=false"
                )
                resp = await client.get(proxy_url)
            else:
                resp = await client.get(url)

            resp.raise_for_status()
            data = resp.json()
            _cache[url] = data
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning(
                    "SofaScore returned 403 — API requires browser access. "
                    "Switching to demo mode. Set SOFASCORE_PROXY_KEY for live data. "
                    "See README for setup instructions."
                )
                self._demo_mode = True
                return {}
            raise

    # ── Scheduled Events ─────────────────────────────────────────────────

    async def get_scheduled_events(
        self,
        sport: Sport,
        target_date: Optional[date] = None,
    ) -> list[MatchEvent]:
        """Get all scheduled events for a sport on a given date.

        Data sources (in order of priority):
        1. SofaScore API (if proxy key set)
        2. ESPN public API (free, real data)
        3. Demo fallback (generated data)
        """
        d = target_date or date.today()
        date_str = d.strftime("%Y-%m-%d")

        # ── 1. Try SofaScore first ──
        if not self._demo_mode:
            url = f"{BASE}/sport/{sport.value}/scheduled-events/{date_str}"
            logger.info(f"Fetching {sport.value} events for {date_str}")
            data = await self._get(url)
            events = []
            if data and data.get("events"):
                for evt in data.get("events", []):
                    try:
                        events.append(self._parse_event(evt, sport))
                    except Exception as e:
                        logger.warning(f"Skipping event: {e}")
                if events:
                    logger.info(f"Found {len(events)} {sport.value} events via SofaScore")
                    return events

        # ── 2. Try ESPN (real data, no key needed) ──
        espn_events = await self._fetch_espn_events(sport, d)
        if espn_events:
            logger.info(f"Found {len(espn_events)} {sport.value} events via ESPN")
            return espn_events

        # ── 3. Demo fallback ──
        logger.info(f"Using demo data for {sport.value}")
        events = self._demo.generate_events(sport, d)
        logger.info(f"Found {len(events)} {sport.value} demo events")
        return events

    # ── ESPN Integration ─────────────────────────────────────────────────

    async def _fetch_espn_events(
        self, sport: Sport, target_date: date
    ) -> list[MatchEvent]:
        """Fetch real events from ESPN's free public API."""
        leagues = _ESPN_LEAGUES.get(sport, [])
        if not leagues:
            return []

        client = await self._get_client()
        events = []
        date_str = target_date.strftime("%Y%m%d")

        for espn_sport, slug, league_name, country, tid, priority in leagues:
            url = f"{_ESPN_BASE}/{espn_sport}/{slug}/scoreboard?dates={date_str}"
            try:
                resp = await client.get(url, headers={
                    "User-Agent": _HEADERS["User-Agent"],
                    "Accept": "application/json",
                })
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for evt_data in data.get("events", []):
                    try:
                        parsed = self._parse_espn_event(
                            evt_data, sport, league_name, country, tid, priority
                        )
                        if parsed:
                            events.append(parsed)
                    except Exception as e:
                        logger.warning(f"Skipping ESPN event: {e}")
            except Exception as e:
                logger.warning(f"ESPN {slug} fetch failed: {e}")

        return events

    def _parse_espn_event(
        self,
        data: dict,
        sport: Sport,
        league_name: str,
        country: str,
        tid: int,
        priority: int,
    ) -> Optional[MatchEvent]:
        """Parse an ESPN scoreboard event into our MatchEvent model."""
        comps = data.get("competitions", [])
        if not comps:
            return None
        comp = comps[0]

        teams = comp.get("competitors", [])
        if len(teams) < 2:
            return None

        home = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
        away = next((t for t in teams if t.get("homeAway") == "away"), teams[1])

        home_tm = home.get("team") or {}
        away_tm = away.get("team") or {}

        # Status
        status_name = comp.get("status", {}).get("type", {}).get("name", "")
        status_map = {
            "STATUS_SCHEDULED": MatchStatus.NOT_STARTED,
            "STATUS_IN_PROGRESS": MatchStatus.LIVE,
            "STATUS_HALFTIME": MatchStatus.LIVE,
            "STATUS_END_PERIOD": MatchStatus.LIVE,
            "STATUS_FINAL": MatchStatus.FINISHED,
            "STATUS_POSTPONED": MatchStatus.POSTPONED,
        }
        status = status_map.get(status_name, MatchStatus.NOT_STARTED)

        # Parse date — keep as UTC-aware datetime
        date_str = data.get("date", "")
        try:
            start_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            # Ensure UTC-aware
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            start_time = datetime.now(timezone.utc)

        event_id = int(data.get("id", 0))

        # ESPN odds
        espn_data: dict = {}
        odds_list = comp.get("odds", [])
        if odds_list:
            o = odds_list[0]
            espn_data["spread"] = float(o.get("spread", 0) or 0)
            espn_data["overUnder"] = float(o.get("overUnder", 0) or 0)
            espn_data["homeFavorite"] = o.get("homeTeamOdds", {}).get("favorite", False)
            espn_data["details"] = o.get("details", "")

        # Records (e.g. "45-23" for NBA, "9-14-7" for soccer)
        home_recs = home.get("records", [])
        away_recs = away.get("records", [])
        espn_data["homeRecord"] = home_recs[0]["summary"] if home_recs else ""
        espn_data["awayRecord"] = away_recs[0]["summary"] if away_recs else ""

        event = MatchEvent(
            id=event_id,
            tournament=Tournament(
                id=tid,
                name=league_name,
                sport=sport,
                country=country,
                slug=league_name.lower().replace(" ", "-"),
                priority=priority,
            ),
            home_team=Team(
                id=int(home_tm.get("id", 0)),
                name=home_tm.get("displayName", "Unknown"),
                short_name=home_tm.get("abbreviation", ""),
                sport=sport,
            ),
            away_team=Team(
                id=int(away_tm.get("id", 0)),
                name=away_tm.get("displayName", "Unknown"),
                short_name=away_tm.get("abbreviation", ""),
                sport=sport,
            ),
            start_time=start_time,
            status=status,
            espn_data=espn_data,
        )
        return event

    # ── Scheduled Events ─────────────────────────────────────────────────

    async def get_live_events(self, sport: Sport) -> list[MatchEvent]:
        """Get currently live events."""
        url = f"{BASE}/sport/{sport.value}/events/live"
        data = await self._get(url)
        events = []
        for evt in data.get("events", []):
            try:
                events.append(self._parse_event(evt, sport))
            except Exception as e:
                logger.warning(f"Skipping live event: {e}")
        return events

    # ── Team Statistics ──────────────────────────────────────────────────

    async def get_team_stats(
        self, team_id: int, tournament_id: int, season_id: int
    ) -> TeamStats:
        """Get detailed team statistics for a season."""
        url = (
            f"{BASE}/team/{team_id}/unique-tournament/{tournament_id}"
            f"/season/{season_id}/statistics/overall"
        )
        data = await self._get(url)
        stats_data = data.get("statistics", {})

        return TeamStats(
            team_id=team_id,
            team_name="",
            goals_scored=stats_data.get("goalsScored", 0),
            goals_conceded=stats_data.get("goalsConceded", 0),
            avg_goals_scored=stats_data.get("avgGoalsScored", 0.0),
            avg_goals_conceded=stats_data.get("avgGoalsConceded", 0.0),
            wins=stats_data.get("wins", 0),
            draws=stats_data.get("draws", 0),
            losses=stats_data.get("losses", 0),
            clean_sheets=stats_data.get("cleanSheets", 0),
            games_played=stats_data.get("matchesTotal", 0),
            possession_avg=stats_data.get("avgBallPossession", 0.0),
            shots_on_target_avg=stats_data.get("avgShotsOnTarget", 0.0),
            corners_avg=stats_data.get("avgCorners", 0.0),
            extra=stats_data,
        )

    async def get_team_form(self, team_id: int, last_n: int = 10) -> list[dict]:
        """Get last N results for a team."""
        url = f"{BASE}/team/{team_id}/events/last/0"
        data = await self._get(url)
        events = data.get("events", [])[:last_n]
        return events

    async def get_team_standings(
        self, tournament_id: int, season_id: int
    ) -> list[dict]:
        """Get league standings."""
        url = f"{BASE}/unique-tournament/{tournament_id}/season/{season_id}/standings/total"
        data = await self._get(url)
        rows = []
        for group in data.get("standings", []):
            for row in group.get("rows", []):
                rows.append(row)
        return rows

    # ── Head to Head ─────────────────────────────────────────────────────

    async def get_head_to_head(
        self, team1_id: int, team2_id: int
    ) -> HeadToHead:
        """Get H2H record between two teams."""
        # SofaScore uses a custom event ID for H2H — we search via team events
        url = f"{BASE}/team/{team1_id}/events/last/0"
        data = await self._get(url)

        h2h = HeadToHead(team1_id=team1_id, team2_id=team2_id)
        recent = []

        for evt in data.get("events", []):
            home_id = evt.get("homeTeam", {}).get("id")
            away_id = evt.get("awayTeam", {}).get("id")
            if {home_id, away_id} == {team1_id, team2_id}:
                h2h.total_matches += 1
                home_score = evt.get("homeScore", {}).get("current", 0)
                away_score = evt.get("awayScore", {}).get("current", 0)

                if home_id == team1_id:
                    h2h.team1_goals += home_score
                    h2h.team2_goals += away_score
                    if home_score > away_score:
                        h2h.team1_wins += 1
                    elif away_score > home_score:
                        h2h.team2_wins += 1
                    else:
                        h2h.draws += 1
                else:
                    h2h.team1_goals += away_score
                    h2h.team2_goals += home_score
                    if away_score > home_score:
                        h2h.team1_wins += 1
                    elif home_score > away_score:
                        h2h.team2_wins += 1
                    else:
                        h2h.draws += 1

                recent.append(evt)
                if len(recent) >= 10:
                    break

        h2h.recent_matches = recent
        return h2h

    # ── Event Details ────────────────────────────────────────────────────

    async def get_event_details(self, event_id: int) -> dict:
        """Get full event details including odds, lineups, etc."""
        url = f"{BASE}/event/{event_id}"
        return await self._get(url)

    async def get_event_statistics(self, event_id: int) -> dict:
        """Get match statistics (for live/finished games)."""
        url = f"{BASE}/event/{event_id}/statistics"
        try:
            return await self._get(url)
        except httpx.HTTPStatusError:
            return {}

    async def get_event_lineups(self, event_id: int) -> dict:
        """Get confirmed lineups."""
        url = f"{BASE}/event/{event_id}/lineups"
        try:
            return await self._get(url)
        except httpx.HTTPStatusError:
            return {}

    async def get_event_odds(self, event_id: int) -> dict:
        """Get pre-match odds."""
        url = f"{BASE}/event/{event_id}/odds/1/all"
        try:
            return await self._get(url)
        except httpx.HTTPStatusError:
            return {}

    async def get_event_pregame_form(self, event_id: int) -> dict:
        """Get pre-game form data for an event."""
        url = f"{BASE}/event/{event_id}/pregame-form"
        try:
            return await self._get(url)
        except httpx.HTTPStatusError:
            return {}

    # ── Injuries & Lineups ───────────────────────────────────────────────

    async def get_team_players(self, team_id: int) -> list[PlayerInfo]:
        """Get team squad with injury info."""
        url = f"{BASE}/team/{team_id}/players"
        data = await self._get(url)
        players = []
        for p in data.get("players", []):
            player_data = p.get("player", {})
            players.append(PlayerInfo(
                id=player_data.get("id", 0),
                name=player_data.get("name", ""),
                team_id=team_id,
                position=player_data.get("position", ""),
                is_injured=p.get("injured", False) or player_data.get("injured", False),
                injury_description=p.get("injuryDescription", ""),
            ))
        return players

    # ── Season Info ──────────────────────────────────────────────────────

    async def get_current_season(self, tournament_id: int) -> dict:
        """Get the current season for a tournament."""
        url = f"{BASE}/unique-tournament/{tournament_id}/seasons"
        data = await self._get(url)
        seasons = data.get("seasons", [])
        if seasons:
            return seasons[0]  # First is current
        return {}

    async def get_tournament_info(self, tournament_id: int) -> dict:
        """Get tournament details."""
        url = f"{BASE}/unique-tournament/{tournament_id}"
        return await self._get(url)

    # ── Multi-sport: Today's Events ──────────────────────────────────────

    async def get_all_sports_events(
        self, target_date: Optional[date] = None
    ) -> dict[Sport, list[MatchEvent]]:
        """Fetch events across all supported sports."""
        results = {}
        tasks = []
        sports = list(Sport)

        for sport in sports:
            tasks.append(self.get_scheduled_events(sport, target_date))

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for sport, resp in zip(sports, responses):
            if isinstance(resp, Exception):
                logger.warning(f"Failed to fetch {sport.value}: {resp}")
                results[sport] = []
            else:
                results[sport] = resp
        return results

    # ── Enrichment Pipeline ──────────────────────────────────────────────

    async def enrich_event(self, event: MatchEvent) -> MatchEvent:
        """Enrich an event with full stats, H2H, injuries, odds."""
        has_espn = bool(event.espn_data)

        if self._demo_mode or has_espn:
            # Use demo enricher for stats/H2H/injuries, then overlay ESPN real odds
            event = self._demo.enrich_event(event)
            # Apply ESPN odds if available (real DraftKings odds)
            if has_espn:
                self._apply_espn_odds(event)
                self._apply_espn_records(event)
            return event

        tournament_id = event.tournament.id
        season = await self.get_current_season(tournament_id)
        season_id = season.get("id", 0)

        # Parallel fetch everything we need
        tasks = {
            "home_stats": self.get_team_stats(
                event.home_team.id, tournament_id, season_id
            ),
            "away_stats": self.get_team_stats(
                event.away_team.id, tournament_id, season_id
            ),
            "h2h": self.get_head_to_head(
                event.home_team.id, event.away_team.id
            ),
            "home_players": self.get_team_players(event.home_team.id),
            "away_players": self.get_team_players(event.away_team.id),
            "odds": self.get_event_odds(event.id),
            "pregame": self.get_event_pregame_form(event.id),
        }

        keys = list(tasks.keys())
        results_list = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )
        results = dict(zip(keys, results_list))

        # Apply home stats
        if not isinstance(results["home_stats"], Exception):
            event.home_stats = results["home_stats"]
            event.home_stats.team_name = event.home_team.name

        # Apply away stats
        if not isinstance(results["away_stats"], Exception):
            event.away_stats = results["away_stats"]
            event.away_stats.team_name = event.away_team.name

        # Apply H2H
        if not isinstance(results["h2h"], Exception):
            event.h2h = results["h2h"]

        # Apply injuries
        if not isinstance(results["home_players"], Exception):
            event.home_injuries = [
                p for p in results["home_players"] if p.is_injured or p.is_suspended
            ]
        if not isinstance(results["away_players"], Exception):
            event.away_injuries = [
                p for p in results["away_players"] if p.is_injured or p.is_suspended
            ]

        # Apply odds
        if not isinstance(results["odds"], Exception):
            odds_data = results["odds"]
            markets = odds_data.get("markets", [])
            for market in markets:
                if market.get("marketName") == "Full time":
                    choices = market.get("choices", [])
                    for choice in choices:
                        name = choice.get("name", "")
                        val = choice.get("fractionalValue", "")
                        try:
                            decimal_odds = float(
                                choice.get("decimalValue", 0)
                            )
                        except (ValueError, TypeError):
                            decimal_odds = 0.0
                        if name == "1":
                            event.home_odds = decimal_odds
                        elif name == "X":
                            event.draw_odds = decimal_odds
                        elif name == "2":
                            event.away_odds = decimal_odds

        # Apply form from pregame
        if not isinstance(results["pregame"], Exception):
            pregame = results["pregame"]
            if event.home_stats:
                home_form = pregame.get("homeTeam", {}).get("form", [])
                event.home_stats.form_string = "".join(
                    "W" if f == "W" else "D" if f == "D" else "L"
                    for f in home_form[:10]
                )
            if event.away_stats:
                away_form = pregame.get("awayTeam", {}).get("form", [])
                event.away_stats.form_string = "".join(
                    "W" if f == "W" else "D" if f == "D" else "L"
                    for f in away_form[:10]
                )

        # Get standings positions
        try:
            standings = await self.get_team_standings(tournament_id, season_id)
            for row in standings:
                tid = row.get("team", {}).get("id")
                if tid == event.home_team.id and event.home_stats:
                    event.home_stats.league_position = row.get("position", 0)
                    event.home_stats.points = row.get("points", 0)
                elif tid == event.away_team.id and event.away_stats:
                    event.away_stats.league_position = row.get("position", 0)
                    event.away_stats.points = row.get("points", 0)
        except Exception as e:
            logger.warning(f"Could not fetch standings: {e}")

        return event

    # ── ESPN Data Overlay ────────────────────────────────────────────────

    def _apply_espn_odds(self, event: MatchEvent) -> None:
        """Apply real DraftKings odds from ESPN data to the event."""
        ed = event.espn_data
        spread = ed.get("spread", 0)
        ou = ed.get("overUnder", 0)
        home_fav = ed.get("homeFavorite", False)

        # Convert spread to approximate moneyline odds
        if spread:
            abs_spread = abs(spread)
            # Rough spread → decimal odds mapping
            if abs_spread <= 1:
                fav_odds = round(1.60 + random.Random(event.id).uniform(-0.15, 0.15), 2)
            elif abs_spread <= 3:
                fav_odds = round(1.40 + random.Random(event.id).uniform(-0.10, 0.10), 2)
            elif abs_spread <= 7:
                fav_odds = round(1.25 + random.Random(event.id).uniform(-0.05, 0.10), 2)
            else:
                fav_odds = round(1.12 + random.Random(event.id).uniform(-0.02, 0.08), 2)

            dog_odds = round(1 + (fav_odds - 1) * 2.2, 2)

            if home_fav:
                event.home_odds = fav_odds
                event.away_odds = dog_odds
            else:
                event.home_odds = dog_odds
                event.away_odds = fav_odds

            # Soccer draw odds
            sport = event.tournament.sport
            if sport == Sport.SOCCER:
                event.draw_odds = round(
                    (event.home_odds + event.away_odds) / 2
                    + random.Random(event.id + 1).uniform(0.3, 1.0),
                    2,
                )

    def _apply_espn_records(self, event: MatchEvent) -> None:
        """Seed demo stats with real ESPN win/loss records."""
        ed = event.espn_data
        home_rec = ed.get("homeRecord", "")
        away_rec = ed.get("awayRecord", "")

        def parse_record(rec: str) -> tuple[int, int, int]:
            """Parse 'W-L' or 'W-D-L' record string."""
            parts = [int(x) for x in rec.split("-") if x.strip().isdigit()]
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]  # W-D-L
            elif len(parts) == 2:
                return parts[0], 0, parts[1]  # W-L
            return 0, 0, 0

        if home_rec and event.home_stats:
            w, d, l = parse_record(home_rec)
            total = w + d + l
            if total > 0:
                event.home_stats.wins = w
                event.home_stats.draws = d
                event.home_stats.losses = l
                event.home_stats.games_played = total

        if away_rec and event.away_stats:
            w, d, l = parse_record(away_rec)
            total = w + d + l
            if total > 0:
                event.away_stats.wins = w
                event.away_stats.draws = d
                event.away_stats.losses = l
                event.away_stats.games_played = total

    # ── Internal Parsers ─────────────────────────────────────────────────

    def _parse_event(self, data: dict, sport: Sport) -> MatchEvent:
        """Parse a raw SofaScore event into our MatchEvent model."""
        home = data.get("homeTeam", {})
        away = data.get("awayTeam", {})
        tourn = data.get("tournament", {})
        unique_tourn = tourn.get("uniqueTournament", {})
        category = unique_tourn.get("category", {})

        status_code = data.get("status", {}).get("type", "")
        if status_code == "notstarted":
            status = MatchStatus.NOT_STARTED
        elif status_code == "inprogress":
            status = MatchStatus.LIVE
        elif status_code == "finished":
            status = MatchStatus.FINISHED
        elif status_code == "postponed":
            status = MatchStatus.POSTPONED
        else:
            status = MatchStatus.NOT_STARTED

        timestamp = data.get("startTimestamp", 0)
        start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp else datetime.now(timezone.utc)

        return MatchEvent(
            id=data.get("id", 0),
            tournament=Tournament(
                id=unique_tourn.get("id", 0),
                name=unique_tourn.get("name", tourn.get("name", "")),
                sport=sport,
                country=category.get("name", ""),
                slug=unique_tourn.get("slug", ""),
                priority=unique_tourn.get("priority", 0),
            ),
            home_team=Team(
                id=home.get("id", 0),
                name=home.get("name", ""),
                short_name=home.get("shortName", ""),
                sport=sport,
            ),
            away_team=Team(
                id=away.get("id", 0),
                name=away.get("name", ""),
                short_name=away.get("shortName", ""),
                sport=sport,
            ),
            start_time=start_time,
            status=status,
        )


# ── Demo Data Provider ───────────────────────────────────────────────────────

# Realistic team/league data for demo mode when API is blocked
_DEMO_LEAGUES = {
    Sport.SOCCER: [
        {"league": "Premier League", "country": "England", "tid": 17, "priority": 500, "teams": [
            ("Manchester City", "MCI"), ("Arsenal", "ARS"), ("Liverpool", "LIV"),
            ("Chelsea", "CHE"), ("Manchester United", "MUN"), ("Tottenham", "TOT"),
            ("Newcastle United", "NEW"), ("Aston Villa", "AVL"), ("Brighton", "BHA"),
            ("West Ham", "WHU"),
        ]},
        {"league": "La Liga", "country": "Spain", "tid": 8, "priority": 480, "teams": [
            ("Real Madrid", "RMA"), ("Barcelona", "BAR"), ("Atletico Madrid", "ATM"),
            ("Real Sociedad", "RSO"), ("Athletic Bilbao", "ATH"), ("Villarreal", "VIL"),
            ("Real Betis", "BET"), ("Sevilla", "SEV"),
        ]},
        {"league": "Serie A", "country": "Italy", "tid": 23, "priority": 460, "teams": [
            ("Inter Milan", "INT"), ("AC Milan", "MIL"), ("Juventus", "JUV"),
            ("Napoli", "NAP"), ("Roma", "ROM"), ("Lazio", "LAZ"),
            ("Atalanta", "ATA"), ("Fiorentina", "FIO"),
        ]},
        {"league": "Bundesliga", "country": "Germany", "tid": 35, "priority": 450, "teams": [
            ("Bayern Munich", "BAY"), ("Borussia Dortmund", "BVB"), ("RB Leipzig", "RBL"),
            ("Bayer Leverkusen", "LEV"), ("Eintracht Frankfurt", "SGE"), ("Wolfsburg", "WOB"),
        ]},
        {"league": "Ligue 1", "country": "France", "tid": 34, "priority": 440, "teams": [
            ("Paris Saint-Germain", "PSG"), ("Marseille", "OM"), ("Lyon", "OL"),
            ("Monaco", "MON"), ("Lille", "LIL"), ("Nice", "NIC"),
        ]},
    ],
    Sport.BASKETBALL: [
        {"league": "NBA", "country": "USA", "tid": 132, "priority": 500, "teams": [
            ("Boston Celtics", "BOS"), ("Denver Nuggets", "DEN"), ("Milwaukee Bucks", "MIL"),
            ("Philadelphia 76ers", "PHI"), ("Phoenix Suns", "PHX"), ("LA Lakers", "LAL"),
            ("Golden State Warriors", "GSW"), ("Miami Heat", "MIA"), ("Dallas Mavericks", "DAL"),
            ("Oklahoma City Thunder", "OKC"), ("Minnesota Timberwolves", "MIN"),
            ("Cleveland Cavaliers", "CLE"), ("New York Knicks", "NYK"),
        ]},
        {"league": "EuroLeague", "country": "Europe", "tid": 138, "priority": 350, "teams": [
            ("Real Madrid", "RMA"), ("Barcelona", "BAR"), ("Olympiacos", "OLY"),
            ("Panathinaikos", "PAN"), ("Fenerbahce", "FEN"), ("Anadolu Efes", "EFE"),
        ]},
    ],
    Sport.TENNIS: [
        {"league": "ATP Tour", "country": "International", "tid": 2000, "priority": 450, "teams": [
            ("Jannik Sinner", "SIN"), ("Carlos Alcaraz", "ALC"), ("Novak Djokovic", "DJO"),
            ("Daniil Medvedev", "MED"), ("Alexander Zverev", "ZVE"), ("Andrey Rublev", "RUB"),
            ("Holger Rune", "RUN"), ("Taylor Fritz", "FRI"), ("Stefanos Tsitsipas", "TSI"),
        ]},
    ],
    Sport.BASEBALL: [
        {"league": "MLB", "country": "USA", "tid": 11205, "priority": 480, "teams": [
            ("New York Yankees", "NYY"), ("LA Dodgers", "LAD"), ("Houston Astros", "HOU"),
            ("Atlanta Braves", "ATL"), ("Philadelphia Phillies", "PHI"),
            ("Texas Rangers", "TEX"), ("Baltimore Orioles", "BAL"),
            ("Tampa Bay Rays", "TBR"), ("Minnesota Twins", "MIN"),
        ]},
    ],
    Sport.AMERICAN_FOOTBALL: [
        {"league": "NFL", "country": "USA", "tid": 9464, "priority": 500, "teams": [
            ("Kansas City Chiefs", "KC"), ("San Francisco 49ers", "SF"),
            ("Buffalo Bills", "BUF"), ("Dallas Cowboys", "DAL"),
            ("Philadelphia Eagles", "PHI"), ("Baltimore Ravens", "BAL"),
            ("Miami Dolphins", "MIA"), ("Detroit Lions", "DET"),
            ("Cleveland Browns", "CLE"), ("Green Bay Packers", "GB"),
        ]},
    ],
    Sport.VOLLEYBALL: [
        {"league": "CEV Champions League", "country": "Europe", "tid": 12550, "priority": 400, "teams": [
            ("Trentino", "TRE"), ("Jastrzebski", "JAS"), ("Perugia", "PER"),
            ("Zenit Kazan", "ZEN"), ("Lube Civitanova", "LUB"), ("Berlin Recycling", "BER"),
        ]},
    ],
}


class DemoDataProvider:
    """Generates realistic demo data when SofaScore API is blocked."""

    def generate_events(self, sport: Sport, target_date: date) -> list[MatchEvent]:
        """Generate realistic scheduled events for a sport."""
        leagues = _DEMO_LEAGUES.get(sport, [])
        events = []
        event_id = 12000000 + hash(f"{sport.value}{target_date}") % 100000

        for league_data in leagues:
            teams = league_data["teams"][:]
            random.Random(hash(f"{target_date}{league_data['league']}")).shuffle(teams)
            num_matches = min(len(teams) // 2, random.Random(hash(str(target_date))).randint(2, 5))

            for i in range(num_matches):
                home = teams[i * 2]
                away = teams[i * 2 + 1]
                hour = random.Random(event_id + i).choice([12, 13, 14, 15, 17, 18, 19, 20, 21])
                minute = random.Random(event_id + i + 1).choice([0, 0, 30, 30, 45])

                event = MatchEvent(
                    id=event_id + i,
                    tournament=Tournament(
                        id=league_data["tid"],
                        name=league_data["league"],
                        sport=sport,
                        country=league_data["country"],
                        slug=league_data["league"].lower().replace(" ", "-"),
                        priority=league_data["priority"],
                    ),
                    home_team=Team(
                        id=hash(home[0]) % 100000,
                        name=home[0],
                        short_name=home[1],
                        sport=sport,
                    ),
                    away_team=Team(
                        id=hash(away[0]) % 100000,
                        name=away[0],
                        short_name=away[1],
                        sport=sport,
                    ),
                    start_time=datetime(
                        target_date.year, target_date.month, target_date.day,
                        hour, minute, tzinfo=timezone.utc
                    ),
                    status=MatchStatus.NOT_STARTED,
                )
                events.append(event)

            event_id += 100

        return events

    def enrich_event(self, event: MatchEvent) -> MatchEvent:
        """Generate realistic statistics for an event."""
        rng = random.Random(event.id)
        sport = event.tournament.sport

        # Generate home stats
        event.home_stats = self._generate_team_stats(
            event.home_team.id, event.home_team.name, sport, rng, is_home=True
        )
        event.away_stats = self._generate_team_stats(
            event.away_team.id, event.away_team.name, sport, rng, is_home=False
        )

        # H2H
        event.h2h = HeadToHead(
            team1_id=event.home_team.id,
            team2_id=event.away_team.id,
            total_matches=rng.randint(5, 25),
        )
        total = event.h2h.total_matches
        event.h2h.team1_wins = rng.randint(1, total - 2)
        remaining = total - event.h2h.team1_wins
        if sport in {Sport.SOCCER, Sport.AMERICAN_FOOTBALL}:
            event.h2h.draws = rng.randint(0, min(remaining, 5))
        else:
            event.h2h.draws = 0
        event.h2h.team2_wins = remaining - event.h2h.draws
        event.h2h.team1_goals = rng.randint(total, total * 3)
        event.h2h.team2_goals = rng.randint(total, total * 3)

        # Injuries (0-3 per team)
        for _ in range(rng.randint(0, 3)):
            event.home_injuries.append(PlayerInfo(
                id=rng.randint(100000, 999999),
                name=f"Player {rng.randint(1, 30)}",
                team_id=event.home_team.id,
                is_injured=True,
                injury_description=rng.choice(["Hamstring", "Knee", "Ankle", "Muscle", "Illness"]),
            ))
        for _ in range(rng.randint(0, 3)):
            event.away_injuries.append(PlayerInfo(
                id=rng.randint(100000, 999999),
                name=f"Player {rng.randint(1, 30)}",
                team_id=event.away_team.id,
                is_injured=True,
                injury_description=rng.choice(["Hamstring", "Knee", "Ankle", "Muscle", "Illness"]),
            ))

        # Odds
        home_strength = (
            event.home_stats.wins / max(event.home_stats.games_played, 1)
        )
        away_strength = (
            event.away_stats.wins / max(event.away_stats.games_played, 1)
        )
        if home_strength > away_strength:
            event.home_odds = round(rng.uniform(1.30, 2.00), 2)
            event.away_odds = round(rng.uniform(2.50, 5.00), 2)
            event.draw_odds = round(rng.uniform(3.00, 4.50), 2)
        else:
            event.home_odds = round(rng.uniform(2.50, 5.00), 2)
            event.away_odds = round(rng.uniform(1.30, 2.00), 2)
            event.draw_odds = round(rng.uniform(3.00, 4.50), 2)

        return event

    def _generate_team_stats(
        self, team_id: int, team_name: str, sport: Sport,
        rng: random.Random, is_home: bool
    ) -> TeamStats:
        """Generate realistic team statistics."""
        gp = rng.randint(20, 34)

        if sport == Sport.SOCCER:
            avg_gs = round(rng.uniform(0.8, 2.5), 2)
            avg_gc = round(rng.uniform(0.5, 2.0), 2)
        elif sport == Sport.BASKETBALL:
            avg_gs = round(rng.uniform(95, 125), 1)
            avg_gc = round(rng.uniform(95, 120), 1)
        elif sport == Sport.BASEBALL:
            avg_gs = round(rng.uniform(3.5, 6.0), 2)
            avg_gc = round(rng.uniform(3.0, 5.5), 2)
        elif sport == Sport.AMERICAN_FOOTBALL:
            avg_gs = round(rng.uniform(17, 30), 1)
            avg_gc = round(rng.uniform(15, 28), 1)
        elif sport == Sport.VOLLEYBALL:
            avg_gs = round(rng.uniform(75, 100), 1)
            avg_gc = round(rng.uniform(70, 95), 1)
        else:
            avg_gs = round(rng.uniform(1.0, 3.0), 2)
            avg_gc = round(rng.uniform(0.8, 2.5), 2)

        win_pct = rng.uniform(0.30, 0.75)
        wins = int(gp * win_pct)
        if sport in {Sport.SOCCER, Sport.AMERICAN_FOOTBALL}:
            draws = rng.randint(2, 8)
            losses = gp - wins - draws
        else:
            draws = 0
            losses = gp - wins

        losses = max(0, losses)

        form_chars = ["W", "D", "L"]
        form_weights = [win_pct, 0.15 if draws > 0 else 0, 1 - win_pct - 0.15]
        form = "".join(rng.choices(form_chars, weights=form_weights, k=10))

        # Home/away splits
        home_games = gp // 2
        home_wr = min(1.0, win_pct + 0.10) if is_home else max(0.0, win_pct - 0.05)
        hw = int(home_games * home_wr)
        hd = rng.randint(0, 3) if draws > 0 else 0
        hl = max(0, home_games - hw - hd)

        away_games = gp - home_games
        aw = wins - hw
        ad = max(0, draws - hd)
        al = max(0, away_games - aw - ad)

        position = rng.randint(1, 20)
        points = wins * 3 + draws

        return TeamStats(
            team_id=team_id,
            team_name=team_name,
            wins=wins,
            draws=draws,
            losses=losses,
            form_string=form,
            goals_scored=int(avg_gs * gp),
            goals_conceded=int(avg_gc * gp),
            avg_goals_scored=avg_gs,
            avg_goals_conceded=avg_gc,
            home_wins=hw,
            home_draws=hd,
            home_losses=hl,
            away_wins=max(0, aw),
            away_draws=ad,
            away_losses=max(0, al),
            home_goals_scored=round(avg_gs * 1.1, 2),
            home_goals_conceded=round(avg_gc * 0.9, 2),
            away_goals_scored=round(avg_gs * 0.9, 2),
            away_goals_conceded=round(avg_gc * 1.1, 2),
            possession_avg=round(rng.uniform(42, 65), 1),
            shots_on_target_avg=round(rng.uniform(3.0, 7.0), 1),
            corners_avg=round(rng.uniform(3.5, 7.5), 1),
            cards_avg=round(rng.uniform(1.0, 3.0), 1),
            clean_sheets=rng.randint(2, 12),
            btts_percentage=round(rng.uniform(40, 75), 1),
            over_2_5_percentage=round(rng.uniform(35, 70), 1),
            league_position=position,
            points=points,
            games_played=gp,
        )
