"""
Sports data client — fixtures, statistics, rosters and bookmaker prices.

ESPN's public API is the working data source, and needs no key. It supplies
35 league feeds, standings, completed results, team rosters, and DraftKings
moneyline/spread/total prices where it carries them. Real statistics are
assembled in `src.espn_stats`.

SofaScore is kept as an optional source but its API rejects server-side
requests; set SOFASCORE_PROXY_KEY to route through a scraping proxy if you
want it.

Two things to know before changing this file:

1. ESPN must be called with `_ESPN_HEADERS`, not `_HEADERS`. It answers 403 to
   anything claiming to be a desktop browser without a browser's TLS
   fingerprint. Sending the spoofed Chrome User-Agent made every ESPN call
   fail in production, and the code then fell back to generating fixtures —
   so the live site served invented matches between real teams.

2. Generated data is off unless ENABLE_SAMPLE_DATA=1, and everything it
   produces is marked `data_source="sample"`. A missing feed returns an empty
   list. Nothing here fills a gap with a plausible-looking number.
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
from src.espn_stats import ESPNStatsProvider
from src.models import (
    Sport, Team, Tournament, TeamStats, HeadToHead,
    MatchEvent, MatchStatus, PlayerInfo,
)

# ── ESPN Public API (free, no key needed) ────────────────────────────────────
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Maps our Sport enum to ESPN endpoint(s):
# (espn_sport, league_slug, display_name, country, tournament_id, priority)
# Every slug below was checked against the live scoreboard endpoint.
_ESPN_LEAGUES: dict[Sport, list[tuple[str, str, str, str, int, int]]] = {
    Sport.SOCCER: [
        ("soccer", "uefa.champions", "Champions League", "Europe", 7, 520),
        ("soccer", "uefa.europa", "Europa League", "Europe", 6, 510),
        ("soccer", "uefa.europa.conf", "Conference League", "Europe", 848, 490),
        ("soccer", "eng.1", "Premier League", "England", 17, 500),
        ("soccer", "esp.1", "La Liga", "Spain", 8, 480),
        ("soccer", "ita.1", "Serie A", "Italy", 23, 460),
        ("soccer", "ger.1", "Bundesliga", "Germany", 35, 450),
        ("soccer", "fra.1", "Ligue 1", "France", 34, 440),
        ("soccer", "usa.1", "MLS", "USA", 242, 430),
        ("soccer", "mex.1", "Liga MX", "Mexico", 262, 420),
        ("soccer", "conmebol.libertadores", "Copa Libertadores",
         "South America", 13, 410),
        ("soccer", "bra.1", "Brasileirão Série A", "Brazil", 71, 400),
        ("soccer", "arg.1", "Liga Profesional", "Argentina", 128, 390),
        ("soccer", "uefa.nations", "UEFA Nations League", "Europe", 5, 380),
        ("soccer", "fifa.worldq.uefa", "World Cup Qualifying (UEFA)",
         "Europe", 32, 370),
        ("soccer", "ned.1", "Eredivisie", "Netherlands", 88, 360),
        ("soccer", "por.1", "Primeira Liga", "Portugal", 94, 350),
        ("soccer", "ksa.1", "Saudi Pro League", "Saudi Arabia", 350, 340),
        ("soccer", "tur.1", "Süper Lig", "Turkey", 203, 330),
        ("soccer", "eng.2", "Championship", "England", 40, 300),
        ("soccer", "sco.1", "Scottish Premiership", "Scotland", 179, 290),
        ("soccer", "eng.fa", "FA Cup", "England", 45, 280),
        ("soccer", "eng.league_cup", "Carabao Cup", "England", 41, 275),
        ("soccer", "esp.copa_del_rey", "Copa del Rey", "Spain", 143, 270),
    ],
    Sport.BASKETBALL: [
        ("basketball", "nba", "NBA", "USA", 132, 500),
        ("basketball", "wnba", "WNBA", "USA", 133, 420),
        ("basketball", "mens-college-basketball", "NCAA Men's Basketball",
         "USA", 1340, 360),
        ("basketball", "womens-college-basketball", "NCAA Women's Basketball",
         "USA", 1341, 300),
        ("basketball", "nba-development", "NBA G League", "USA", 1342, 240),
    ],
    Sport.BASEBALL: [
        ("baseball", "mlb", "MLB", "USA", 11205, 480),
        ("baseball", "college-baseball", "NCAA Baseball", "USA", 11206, 260),
    ],
    Sport.AMERICAN_FOOTBALL: [
        ("football", "nfl", "NFL", "USA", 9464, 500),
        ("football", "college-football", "NCAA Football", "USA", 9465, 400),
    ],
    Sport.HOCKEY: [
        ("hockey", "nhl", "NHL", "USA", 234, 480),
        ("hockey", "mens-college-hockey", "NCAA Men's Ice Hockey",
         "USA", 235, 250),
    ],
}

# Reverse lookup: our tournament id -> (espn_sport, slug). Needed so the
# enrichment step knows which league feed to pull standings from.
_TID_TO_ESPN: dict[int, tuple[str, str]] = {
    entry[4]: (entry[0], entry[1])
    for entries in _ESPN_LEAGUES.values()
    for entry in entries
}


def _american_str_to_decimal(raw) -> float:
    """Convert an American price like '-205' or '+170' to decimal odds."""
    if raw is None:
        return 0.0
    try:
        val = int(str(raw).replace("+", "").strip())
    except (ValueError, TypeError):
        return 0.0
    if val == 0:
        return 0.0
    if val > 0:
        return round(1 + val / 100, 3)
    return round(1 + 100 / abs(val), 3)


def _side_price(node: dict) -> tuple[float, Optional[float]]:
    """Pull (decimal odds, line) from an ESPN odds side.

    Each side looks like {"close": {"odds": "-115", "line": "-3.5"},
    "open": {...}}. The closing price is preferred; the opening price is the
    fallback when a game has not been re-priced yet.
    """
    if not isinstance(node, dict):
        return 0.0, None
    for window in ("close", "open", "current"):
        block = node.get(window)
        if not isinstance(block, dict):
            continue
        dec = _american_str_to_decimal(block.get("odds"))
        line_raw = str(block.get("line", "") or "").lstrip("ou")
        try:
            line = float(line_raw) if line_raw else None
        except ValueError:
            line = None
        if dec > 1.0:
            return dec, line
    # Some feeds put the price straight on the node.
    return _american_str_to_decimal(node.get("odds")), None


def _parse_espn_market(o: dict) -> dict:
    """Extract the real market (moneyline, spread, total) from ESPN odds.

    All three markets carry both a line and a price per side, which is what
    makes genuine expected-value comparison possible — our own model price is
    not evidence of anything on its own.
    """
    out: dict = {}
    provider = (o.get("provider") or {}).get("displayName") or (
        o.get("provider") or {}
    ).get("name")
    if provider:
        out["odds_provider"] = provider
    out["details"] = o.get("details", "")

    try:
        out["spread"] = float(o.get("spread") or 0)
    except (TypeError, ValueError):
        out["spread"] = 0.0
    try:
        out["overUnder"] = float(o.get("overUnder") or 0)
    except (TypeError, ValueError):
        out["overUnder"] = 0.0
    out["homeFavorite"] = bool(
        (o.get("homeTeamOdds") or {}).get("favorite", False)
    )

    ml = o.get("moneyline") or {}
    for key, label in (("home", "home"), ("away", "away"), ("draw", "draw")):
        dec, _ = _side_price(ml.get(key) or {})
        if dec > 1.0:
            out[f"moneyline_{label}_decimal"] = dec

    ps = o.get("pointSpread") or {}
    for key in ("home", "away"):
        dec, line = _side_price(ps.get(key) or {})
        if dec > 1.0:
            out[f"spread_{key}_decimal"] = dec
        if line is not None:
            out[f"spread_{key}_line"] = line

    tot = o.get("total") or {}
    for key in ("over", "under"):
        dec, line = _side_price(tot.get(key) or {})
        if dec > 1.0:
            out[f"total_{key}_decimal"] = dec
        if line is not None:
            out[f"total_line"] = line

    return out

# Rate-limit friendly cache
_cache = TTLCache(maxsize=500, ttl=settings.sofascore_cache_ttl)

# Headers for the SofaScore endpoints, which do want browser-like requests.
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

# ESPN must NOT get the headers above. It answers 403 to anything claiming to
# be a desktop browser without a browser's TLS fingerprint, and a plain client
# identifier is served normally. Sending the spoofed Chrome User-Agent is what
# made every ESPN call fail in production, which silently dropped the whole
# site onto generated fixtures — real teams, invented matches. Keep these
# headers plain.
_ESPN_HEADERS = {
    "User-Agent": "bet-prediction/1.0 (+https://github.com/Fredler21/Bet-prediction)",
    "Accept": "application/json",
}

BASE = settings.sofascore_base_url

# Proxy API key for scraping services (ScrapingBee, ScraperAPI, etc.)
PROXY_KEY = os.getenv("SOFASCORE_PROXY_KEY", "")

# Generated fixtures are OFF unless explicitly switched on for local work.
#
# This used to be the silent fallback whenever a feed failed, and because the
# ESPN calls were being rejected (see _ESPN_HEADERS) it was what production
# actually served: invented fixtures between real teams. On 6 September 2026
# the live site was offering "Manchester United vs West Ham" when the real
# fixture that day was Manchester United at Everton. A missing feed must show
# nothing rather than something made up.
SAMPLE_DATA_ENABLED = os.getenv("ENABLE_SAMPLE_DATA", "").lower() in {
    "1", "true", "yes",
}


class SofaScoreClient:
    """Async client for the SofaScore API with proxy + demo fallback."""

    def __init__(self, demo_mode: bool = False):
        self._client: Optional[httpx.AsyncClient] = None
        self._demo_mode = demo_mode
        self._demo = DemoDataProvider()
        # Real statistics come from here; see src/espn_stats.py.
        self._stats = ESPNStatsProvider()

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
        await self._stats.close()

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

        # ── 2. ESPN (real data, no key needed) ──
        espn_events = await self._fetch_espn_events(sport, d)
        if espn_events:
            logger.info(f"Found {len(espn_events)} {sport.value} events via ESPN")
            return espn_events

        # ── 3. Nothing real for this sport and date ──
        # Returning an empty list is the correct answer: most of these sports
        # are simply out of season on any given day. Generating stand-in
        # fixtures here is what put invented matches on the live site.
        if not SAMPLE_DATA_ENABLED:
            logger.info(f"No real {sport.value} fixtures for {date_str}")
            return []

        logger.warning(
            f"ENABLE_SAMPLE_DATA is set — generating stand-in "
            f"{sport.value} fixtures for {date_str}. These are not real."
        )
        events = self._demo.generate_events(sport, d)
        for evt in events:
            evt.data_source = "sample"
            evt.data_notes.append(
                "SAMPLE DATA — this fixture is generated, not a real match."
            )
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
        events: list[MatchEvent] = []
        date_str = target_date.strftime("%Y%m%d")

        async def fetch_league(
            espn_sport: str, slug: str, league_name: str,
            country: str, tid: int, priority: int,
        ) -> list[MatchEvent]:
            url = f"{_ESPN_BASE}/{espn_sport}/{slug}/scoreboard?dates={date_str}"
            out: list[MatchEvent] = []
            try:
                # _ESPN_HEADERS, not _HEADERS — see the note by their definition.
                resp = await client.get(url, headers=_ESPN_HEADERS)
                if resp.status_code != 200:
                    logger.warning(f"ESPN {slug} → HTTP {resp.status_code}")
                    return out
                data = resp.json()
                for evt_data in data.get("events", []):
                    try:
                        parsed = self._parse_espn_event(
                            evt_data, sport, league_name, country, tid, priority
                        )
                        if parsed:
                            parsed.espn_data["espn_sport"] = espn_sport
                            parsed.espn_data["espn_slug"] = slug
                            out.append(parsed)
                    except Exception as e:
                        logger.warning(f"Skipping ESPN event: {e}")
            except Exception as e:
                logger.warning(f"ESPN {slug} fetch failed: {e}")
            return out

        # Fetch every league for this sport concurrently — there are now two
        # dozen soccer competitions, and doing them in series was slow enough
        # to risk the serverless request timeout.
        results = await asyncio.gather(
            *(fetch_league(*league) for league in leagues),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):
                logger.warning(f"ESPN league fetch error: {res}")
                continue
            events.extend(res)

        # The same fixture can appear in both a league and a cup feed.
        unique: dict[int, MatchEvent] = {}
        for evt in events:
            if evt.id not in unique or evt.tournament.priority > unique[evt.id].tournament.priority:
                unique[evt.id] = evt
        return list(unique.values())

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

        # Status. Match on `state`/`completed` rather than the status name:
        # soccer reports STATUS_FULL_TIME, which the old name-only map did not
        # know, so finished matches were treated as not started and had fresh
        # predictions generated for them.
        status_type = comp.get("status", {}).get("type", {}) or {}
        status_name = status_type.get("name", "")
        state = str(status_type.get("state", "")).lower()

        if status_name in ("STATUS_POSTPONED", "STATUS_DELAYED"):
            status = MatchStatus.POSTPONED
        elif status_name in ("STATUS_CANCELED", "STATUS_CANCELLED"):
            status = MatchStatus.CANCELLED
        elif status_type.get("completed") or state == "post":
            status = MatchStatus.FINISHED
        elif state == "in":
            status = MatchStatus.LIVE
        else:
            status = MatchStatus.NOT_STARTED

        # Parse scores for finished/live games
        try:
            home_score_val = int(home.get("score", 0) or 0) if status in (MatchStatus.FINISHED, MatchStatus.LIVE) else None
            away_score_val = int(away.get("score", 0) or 0) if status in (MatchStatus.FINISHED, MatchStatus.LIVE) else None
        except (ValueError, TypeError):
            home_score_val, away_score_val = None, None

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

        # Real bookmaker market, when ESPN carries one for this game.
        espn_data: dict = {}
        odds_list = [o for o in (comp.get("odds") or []) if isinstance(o, dict)]
        if odds_list:
            espn_data.update(_parse_espn_market(odds_list[0]))

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
            home_score=home_score_val,
            away_score=away_score_val,
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
        """Attach real statistics, head-to-head and market prices.

        This method used to call the demo enricher whenever ESPN data was
        present, which meant a genuine fixture was analysed with
        `random.uniform` scoring rates, a random form string, a random
        head-to-head record and injuries named "Player 17". Only the win/loss
        record was real. Every confidence figure the site published rested on
        those numbers.

        Real ESPN standings and completed results are used instead. Anything
        that cannot be sourced is left empty and noted on the event, so the
        model can skip it rather than fill it in.
        """
        if event.espn_data.get("espn_slug"):
            return await self._enrich_from_espn(event)

        if self._demo_mode:
            if not SAMPLE_DATA_ENABLED:
                event.data_source = "partial"
                event.data_notes.append(
                    "No statistics feed reached for this fixture."
                )
                return event
            event = self._demo.enrich_event(event)
            event.data_source = "sample"
            event.data_notes.append(
                "SAMPLE DATA — figures are generated for local development "
                "and are not real."
            )
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

    # ── Real ESPN enrichment ─────────────────────────────────────────────

    async def _enrich_from_espn(self, event: MatchEvent) -> MatchEvent:
        """Build the event's statistics from real ESPN data."""
        espn_sport = event.espn_data.get("espn_sport", "")
        slug = event.espn_data.get("espn_slug", "")
        if not espn_sport or not slug:
            event.data_source = "partial"
            return event

        provider = self._stats
        sport = event.tournament.sport
        season = event.start_time.year

        standings = await provider.get_standings(espn_sport, slug, season)
        if not standings:
            # US leagues answer the season-less URL with the current season.
            standings = await provider.get_standings(espn_sport, slug)

        (home_stats, home_ok), (away_stats, away_ok), h2h, rosters = (
            await asyncio.gather(
                provider.build_team_stats(
                    sport, espn_sport, slug, event.home_team.id,
                    event.home_team.name, standings.get(event.home_team.id),
                ),
                provider.build_team_stats(
                    sport, espn_sport, slug, event.away_team.id,
                    event.away_team.name, standings.get(event.away_team.id),
                ),
                provider.get_head_to_head(
                    espn_sport, slug, event.home_team.id, event.away_team.id
                ),
                self._fetch_rosters(espn_sport, slug, event),
                return_exceptions=False,
            )
        )

        event.home_stats = home_stats
        event.away_stats = away_stats
        event.h2h = h2h  # None when the sides have not met — no invented record

        # The league scoring baseline anchors the attack/defence model. Before
        # a season's first game the table is all zeroes, so fall back to last
        # season — which is also where the team form came from, keeping the
        # baseline and the team rates on the same footing.
        vals = [r["avg_for"] for r in standings.values() if r["avg_for"] > 0]
        if not vals:
            prior = await provider.get_standings(espn_sport, slug, season - 1)
            vals = [r["avg_for"] for r in prior.values() if r["avg_for"] > 0]
            if vals:
                event.data_notes.append(
                    "League scoring baseline taken from last season — the new "
                    "one has not started."
                )
        if vals:
            event.espn_data["league_baseline"] = round(sum(vals) / len(vals), 3)

        if rosters:
            event.espn_data.update(rosters)

        self._apply_espn_odds(event)

        notes: list[str] = []
        if not (home_ok and away_ok):
            missing = []
            if not home_ok:
                missing.append(event.home_team.name)
            if not away_ok:
                missing.append(event.away_team.name)
            notes.append(
                "No season statistics available for "
                + " and ".join(missing)
                + " — this projection falls back to league averages."
            )
        if h2h is None:
            notes.append("No recent head-to-head meetings on record.")
        for stats in (home_stats, away_stats):
            if stats.extra.get("form_from_prior_season"):
                notes.append(
                    f"{stats.team_name} form is from last season — the new "
                    "campaign has not started."
                )

        event.data_notes.extend(notes)
        event.data_source = "live" if (home_ok and away_ok) else "partial"
        return event

    async def _fetch_rosters(
        self, espn_sport: str, slug: str, event: MatchEvent
    ) -> dict:
        """Fetch both squads so player markets can use real names.

        Replaces the hand-typed star-player table, which had drifted out of
        date (Anthony Davis was still listed at the Lakers).
        """
        client = await self._get_client()

        async def one(team_id: int) -> list[dict]:
            url = f"{_ESPN_BASE}/{espn_sport}/{slug}/teams/{team_id}/roster"
            try:
                resp = await client.get(url, headers=_ESPN_HEADERS)
                if resp.status_code != 200:
                    return []
                data = resp.json()
            except Exception as e:
                logger.debug(f"Roster fetch failed for {team_id}: {e}")
                return []

            athletes = data.get("athletes") or []
            # Some sports group athletes by position bucket.
            if athletes and isinstance(athletes[0], dict) and "items" in athletes[0]:
                flat: list[dict] = []
                for group in athletes:
                    flat.extend(group.get("items") or [])
                athletes = flat

            out: list[dict] = []
            for a in athletes:
                name = a.get("fullName") or a.get("displayName")
                if not name:
                    continue
                pos = (a.get("position") or {})
                out.append({
                    "name": name,
                    "position": pos.get("displayName") or pos.get("name") or "",
                })
            return out

        try:
            home_roster, away_roster = await asyncio.gather(
                one(event.home_team.id), one(event.away_team.id)
            )
        except Exception:
            return {}
        result = {}
        if home_roster:
            result["home_roster"] = home_roster
        if away_roster:
            result["away_roster"] = away_roster
        return result

    # ── ESPN Data Overlay ────────────────────────────────────────────────

    def _apply_espn_odds(self, event: MatchEvent) -> None:
        """Apply the real bookmaker moneyline, when ESPN supplied one.

        The previous version of this method derived odds from the spread and
        then jittered them with `random.Random(event.id).uniform(...)`, while
        its docstring claimed they were real DraftKings prices. Those invented
        numbers were then fed to the expected-value calculation, so the "value
        bets" list was ranking noise.

        Now a price is set only when the feed actually carried one, and
        `has_book_odds` records whether that happened.
        """
        ed = event.espn_data
        home_ml = ed.get("moneyline_home_decimal", 0.0)
        away_ml = ed.get("moneyline_away_decimal", 0.0)
        draw_ml = ed.get("moneyline_draw_decimal", 0.0)

        if home_ml > 1.0 and away_ml > 1.0:
            event.home_odds = home_ml
            event.away_odds = away_ml
            event.draw_odds = draw_ml if draw_ml > 1.0 else 0.0
            event.has_book_odds = True
            event.data_notes.append(
                f"Bookmaker prices from {ed.get('odds_provider', 'the feed')}."
            )
        else:
            event.home_odds = 0.0
            event.away_odds = 0.0
            event.draw_odds = 0.0
            event.has_book_odds = False

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


# Sport season months — avoid generating out-of-season demo games
_SPORT_SEASONS: dict[Sport, list[int]] = {
    Sport.SOCCER:            list(range(1, 13)),           # Year-round (multiple leagues)
    Sport.BASKETBALL:        [1, 2, 3, 4, 5, 6, 10, 11, 12],  # NBA Oct–Jun
    Sport.BASEBALL:          list(range(3, 11)),            # MLB Mar–Oct
    Sport.AMERICAN_FOOTBALL: [1, 2, 9, 10, 11, 12],        # NFL Sep–Feb
    Sport.HOCKEY:            [1, 2, 3, 4, 5, 6, 10, 11, 12],  # NHL Oct–Jun
    Sport.TENNIS:            list(range(1, 13)),            # ATP/WTA year-round
    Sport.VOLLEYBALL:        list(range(1, 13)),            # CEV year-round
}


class DemoDataProvider:
    """Generates realistic demo data when SofaScore API is blocked."""

    def generate_events(self, sport: Sport, target_date: date) -> list[MatchEvent]:
        """Generate realistic scheduled events for a sport."""
        # Don't generate games when the sport is out of season
        if target_date.month not in _SPORT_SEASONS.get(sport, list(range(1, 13))):
            return []

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
