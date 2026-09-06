"""
ESPN Statistics Provider — builds REAL TeamStats from ESPN's public API.

This module replaces the randomly-generated "demo" statistics that previously
fed the prediction model. Everything here is derived from actual ESPN data:

  * Standings   -> games played, W/D/L, goals/points for & against, table rank
  * Team schedule -> real form string (last N results), home/away splits,
                     real head-to-head record between two teams

No key is required. Responses are cached because standings and completed
results only change once a day.

If a figure cannot be sourced, it is left at zero and the caller is told the
data is incomplete — nothing is invented.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from cachetools import TTLCache
from loguru import logger

from src.models import Sport, TeamStats, HeadToHead

_ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports"
_ESPN_V2 = "https://site.api.espn.com/apis/v2/sports"

# Standings and finished results change at most once a day.
_standings_cache: TTLCache = TTLCache(maxsize=64, ttl=1800)
_schedule_cache: TTLCache = TTLCache(maxsize=512, ttl=1800)

# ESPN rejects (403) requests that claim to be a desktop browser but do not
# carry a browser's TLS fingerprint. A plain client identifier is accepted, so
# do NOT put a spoofed Chrome User-Agent here — that is what broke the live
# data feed and silently pushed the site onto generated numbers.
_UA = "bet-prediction/1.0 (+https://github.com/Fredler21/Bet-prediction)"

# Sports where a regulation draw is a real, bettable outcome.
DRAW_SPORTS = {Sport.SOCCER, Sport.HANDBALL, Sport.RUGBY}


def _stat_map(entry: dict) -> dict[str, float]:
    """Flatten an ESPN standings entry's stat list into {name: value}."""
    out: dict[str, float] = {}
    for s in entry.get("stats", []) or []:
        name = s.get("name") or s.get("type") or ""
        if not name:
            continue
        val = s.get("value")
        if val is None:
            # Some splits (Home/Road) carry only a displayValue like "12-4".
            disp = s.get("displayValue") or ""
            out[f"{name}__display"] = disp  # type: ignore[assignment]
            continue
        try:
            out[name] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _walk_entries(node: Any, found: list[dict]) -> None:
    """Recursively collect every standings `entries` list in a payload.

    Soccer nests them one level deep; NBA/NFL/MLB/NHL nest them per
    conference or division, so a plain lookup misses most teams.
    """
    if isinstance(node, dict):
        standings = node.get("standings")
        if isinstance(standings, dict) and isinstance(standings.get("entries"), list):
            found.extend(standings["entries"])
        for key in ("children", "groups"):
            child = node.get(key)
            if isinstance(child, list):
                for c in child:
                    _walk_entries(c, found)
        if isinstance(node.get("entries"), list) and "standings" not in node:
            # Some payloads expose entries directly.
            found.extend(node["entries"])
    elif isinstance(node, list):
        for c in node:
            _walk_entries(c, found)


def _parse_split(display: str) -> tuple[int, int, int]:
    """Parse a 'W-L' or 'W-D-L' split string into (w, d, l)."""
    parts = [p for p in display.split("-") if p.strip().lstrip("+").isdigit()]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return 0, 0, 0
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    if len(nums) == 2:
        return nums[0], 0, nums[1]
    return 0, 0, 0


class ESPNStatsProvider:
    """Fetches and assembles real team statistics from ESPN."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(12.0, connect=6.0),
                headers={"User-Agent": _UA, "Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _fetch(self, url: str, cache: TTLCache) -> dict:
        if url in cache:
            return cache[url]
        client = await self._get_client()
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(f"ESPN {resp.status_code} for {url}")
                return {}
            data = resp.json()
        except Exception as e:  # network, JSON, timeout
            logger.warning(f"ESPN fetch failed for {url}: {e}")
            return {}
        cache[url] = data
        return data

    # ── Standings ────────────────────────────────────────────────────────

    async def get_standings(
        self, espn_sport: str, slug: str, season: Optional[int] = None
    ) -> dict[int, dict]:
        """Return {team_id: parsed standings row} for a league."""
        url = f"{_ESPN_V2}/{espn_sport}/{slug}/standings"
        if season:
            url += f"?season={season}"

        data = await self._fetch(url, _standings_cache)
        if not data:
            return {}

        entries: list[dict] = []
        _walk_entries(data, entries)

        rows: dict[int, dict] = {}
        for entry in entries:
            team = entry.get("team") or {}
            try:
                tid = int(team.get("id", 0))
            except (TypeError, ValueError):
                continue
            if not tid or tid in rows:
                continue

            st = _stat_map(entry)
            gp = int(st.get("gamesPlayed", 0))
            wins = int(st.get("wins", 0))
            losses = int(st.get("losses", 0))
            draws = int(st.get("ties", 0))
            # NHL counts overtime losses separately from regulation losses.
            ot_losses = int(st.get("OTLosses", 0))

            points_for = st.get("pointsFor", 0.0)
            points_against = st.get("pointsAgainst", 0.0)
            # US leagues publish per-game averages directly; soccer gives totals.
            avg_for = st.get("avgPointsFor", 0.0)
            avg_against = st.get("avgPointsAgainst", 0.0)
            if not avg_for and gp:
                avg_for = points_for / gp
            if not avg_against and gp:
                avg_against = points_against / gp

            if not gp:
                gp = wins + draws + losses + ot_losses

            home_w, home_d, home_l = _parse_split(str(st.get("Home__display", "")))
            away_w, away_d, away_l = _parse_split(str(st.get("Road__display", "")))

            rows[tid] = {
                "games_played": gp,
                "wins": wins,
                "draws": draws,
                "losses": losses + ot_losses,
                "points_for": points_for,
                "points_against": points_against,
                "avg_for": round(float(avg_for), 3),
                "avg_against": round(float(avg_against), 3),
                "rank": int(st.get("rank", 0)),
                "points": int(st.get("points", 0)),
                "home_split": (home_w, home_d, home_l),
                "away_split": (away_w, away_d, away_l),
            }

        if rows:
            logger.info(f"ESPN standings: {len(rows)} teams for {espn_sport}/{slug}")
        return rows

    # ── Team schedule / form ─────────────────────────────────────────────

    async def get_team_schedule(
        self, espn_sport: str, slug: str, team_id: int,
        season: Optional[int] = None,
    ) -> list[dict]:
        """Return that team's completed games, most recent first.

        Each item: {opponent_id, opponent_name, is_home, scored, conceded,
                    result ('W'/'D'/'L'), has_score, date}

        Note the two payload shapes ESPN uses. US leagues inline the score as
        {"value": 110, ...}; soccer returns a `$ref` URL instead, so for those
        the result is taken from the `winner` flags and `scored`/`conceded`
        stay None rather than being guessed at.
        """
        url = f"{_ESPN_SITE}/{espn_sport}/{slug}/teams/{team_id}/schedule"
        if season:
            url += f"?season={season}"
        data = await self._fetch(url, _schedule_cache)
        if not data:
            return []

        games: list[dict] = []
        for evt in data.get("events", []) or []:
            comps = evt.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]

            # `completed` covers every sport; matching on status names misses
            # soccer, which reports STATUS_FULL_TIME rather than STATUS_FINAL.
            status_type = (
                comp.get("status", {}).get("type")
                or evt.get("status", {}).get("type")
                or {}
            )
            if not status_type.get("completed"):
                continue

            competitors = comp.get("competitors") or []
            if len(competitors) < 2:
                continue

            me = next(
                (c for c in competitors if str((c.get("team") or {}).get("id")) == str(team_id)),
                None,
            )
            opp = next(
                (c for c in competitors if str((c.get("team") or {}).get("id")) != str(team_id)),
                None,
            )
            if not me or not opp:
                continue

            def _score(c: dict) -> Optional[int]:
                raw = c.get("score")
                if isinstance(raw, dict):
                    # A {"$ref": ...} payload carries no value — treat as absent.
                    raw = raw.get("value", raw.get("displayValue"))
                try:
                    return int(float(raw))
                except (TypeError, ValueError):
                    return None

            scored = _score(me)
            conceded = _score(opp)
            has_score = scored is not None and conceded is not None

            if has_score:
                if scored > conceded:
                    result = "W"
                elif scored < conceded:
                    result = "L"
                else:
                    result = "D"
            elif me.get("winner") is True:
                result = "W"
            elif opp.get("winner") is True:
                result = "L"
            elif me.get("winner") is False and opp.get("winner") is False:
                result = "D"
            else:
                # Completed but no usable outcome — skip rather than guess.
                continue

            try:
                dt = datetime.fromisoformat(
                    str(evt.get("date", "")).replace("Z", "+00:00")
                )
            except ValueError:
                dt = datetime.now(timezone.utc)

            games.append({
                "opponent_id": int((opp.get("team") or {}).get("id", 0) or 0),
                "opponent_name": (opp.get("team") or {}).get("displayName", ""),
                "is_home": me.get("homeAway") == "home",
                "scored": scored,
                "conceded": conceded,
                "has_score": has_score,
                "result": result,
                "date": dt,
            })

        games.sort(key=lambda g: g["date"], reverse=True)

        # Between seasons the current schedule holds only fixtures, so form
        # would come back blank. Fall back to the previous season once, and
        # label it, rather than showing a team with no history.
        if not games and season is None:
            prior = datetime.now(timezone.utc).year - 1
            games = await self.get_team_schedule(espn_sport, slug, team_id, prior)
            for g in games:
                g["prior_season"] = True

        return games

    # ── Assembling TeamStats ─────────────────────────────────────────────

    async def build_team_stats(
        self,
        sport: Sport,
        espn_sport: str,
        slug: str,
        team_id: int,
        team_name: str,
        standings_row: Optional[dict] = None,
    ) -> tuple[TeamStats, bool]:
        """Build a TeamStats from real ESPN data.

        Returns (stats, is_complete). `is_complete` is False when neither
        standings nor completed games were available, which tells the caller
        this matchup should not be presented as a confident read.
        """
        stats = TeamStats(team_id=team_id, team_name=team_name)
        games = await self.get_team_schedule(espn_sport, slug, team_id)

        got_standings = bool(standings_row)
        if standings_row:
            stats.games_played = standings_row["games_played"]
            stats.wins = standings_row["wins"]
            stats.draws = standings_row["draws"]
            stats.losses = standings_row["losses"]
            stats.avg_goals_scored = standings_row["avg_for"]
            stats.avg_goals_conceded = standings_row["avg_against"]
            stats.goals_scored = int(standings_row["points_for"])
            stats.goals_conceded = int(standings_row["points_against"])
            stats.league_position = standings_row["rank"]
            stats.points = standings_row["points"]

            hw, hd, hl = standings_row["home_split"]
            aw, ad, al = standings_row["away_split"]
            stats.home_wins, stats.home_draws, stats.home_losses = hw, hd, hl
            stats.away_wins, stats.away_draws, stats.away_losses = aw, ad, al

        if games:
            # Real form string, most recent first.
            stats.form_string = "".join(g["result"] for g in games[:10])

            # Fill gaps that standings did not cover, straight from results.
            if not stats.games_played:
                stats.games_played = len(games)
                stats.wins = sum(1 for g in games if g["result"] == "W")
                stats.draws = sum(1 for g in games if g["result"] == "D")
                stats.losses = sum(1 for g in games if g["result"] == "L")

            def _wdl(subset: list[dict]) -> tuple[int, int, int]:
                return (
                    sum(1 for g in subset if g["result"] == "W"),
                    sum(1 for g in subset if g["result"] == "D"),
                    sum(1 for g in subset if g["result"] == "L"),
                )

            home_games = [g for g in games if g["is_home"]]
            away_games = [g for g in games if not g["is_home"]]

            if home_games and not any(
                (stats.home_wins, stats.home_draws, stats.home_losses)
            ):
                stats.home_wins, stats.home_draws, stats.home_losses = _wdl(home_games)
            if away_games and not any(
                (stats.away_wins, stats.away_draws, stats.away_losses)
            ):
                stats.away_wins, stats.away_draws, stats.away_losses = _wdl(away_games)

            # Everything below needs real scorelines. Soccer team schedules
            # return the score as a $ref, so those games are excluded here and
            # the scoring rates come from standings totals instead.
            scored_games = [g for g in games if g["has_score"]]
            if scored_games:
                if not stats.avg_goals_scored:
                    stats.avg_goals_scored = round(
                        sum(g["scored"] for g in scored_games) / len(scored_games), 3
                    )
                if not stats.avg_goals_conceded:
                    stats.avg_goals_conceded = round(
                        sum(g["conceded"] for g in scored_games) / len(scored_games), 3
                    )

                home_scored = [g for g in scored_games if g["is_home"]]
                away_scored = [g for g in scored_games if not g["is_home"]]
                if home_scored:
                    stats.home_goals_scored = round(
                        sum(g["scored"] for g in home_scored) / len(home_scored), 3
                    )
                    stats.home_goals_conceded = round(
                        sum(g["conceded"] for g in home_scored) / len(home_scored), 3
                    )
                if away_scored:
                    stats.away_goals_scored = round(
                        sum(g["scored"] for g in away_scored) / len(away_scored), 3
                    )
                    stats.away_goals_conceded = round(
                        sum(g["conceded"] for g in away_scored) / len(away_scored), 3
                    )

                stats.clean_sheets = sum(1 for g in scored_games if g["conceded"] == 0)
                btts = sum(
                    1 for g in scored_games if g["scored"] > 0 and g["conceded"] > 0
                )
                stats.btts_percentage = round(100 * btts / len(scored_games), 1)
                over_25 = sum(
                    1 for g in scored_games if g["scored"] + g["conceded"] > 2.5
                )
                stats.over_2_5_percentage = round(
                    100 * over_25 / len(scored_games), 1
                )
                stats.extra["recent_scored"] = [g["scored"] for g in scored_games[:10]]
                stats.extra["recent_conceded"] = [
                    g["conceded"] for g in scored_games[:10]
                ]

            stats.extra["recent_games"] = len(games)
            stats.extra["games_with_scores"] = len(scored_games)
            if any(g.get("prior_season") for g in games):
                stats.extra["form_from_prior_season"] = True

        is_complete = bool(games) or got_standings
        if not is_complete:
            logger.warning(
                f"No ESPN stats available for {team_name} ({espn_sport}/{slug})"
            )
        return stats, is_complete

    # ── Head to head ─────────────────────────────────────────────────────

    async def get_head_to_head(
        self,
        espn_sport: str,
        slug: str,
        home_id: int,
        away_id: int,
    ) -> Optional[HeadToHead]:
        """Real H2H, derived from the home team's completed fixtures.

        ESPN's team schedule only covers the current season, so this is an
        honest partial record rather than an all-time one. Returns None when
        the two sides have not met, so the model can skip the factor instead
        of inventing a record.
        """
        games = await self.get_team_schedule(espn_sport, slug, home_id)
        meetings = [g for g in games if g["opponent_id"] == away_id]
        if not meetings:
            return None

        h2h = HeadToHead(team1_id=home_id, team2_id=away_id)
        h2h.total_matches = len(meetings)
        h2h.team1_wins = sum(1 for g in meetings if g["result"] == "W")
        h2h.team2_wins = sum(1 for g in meetings if g["result"] == "L")
        h2h.draws = sum(1 for g in meetings if g["result"] == "D")
        scored_meetings = [g for g in meetings if g["has_score"]]
        h2h.team1_goals = sum(g["scored"] for g in scored_meetings)
        h2h.team2_goals = sum(g["conceded"] for g in scored_meetings)
        h2h.recent_matches = [
            {
                "date": g["date"].isoformat(),
                "home": g["is_home"],
                "score": (
                    f"{g['scored']}-{g['conceded']}" if g["has_score"] else ""
                ),
                "result": g["result"],
            }
            for g in meetings[:5]
        ]
        return h2h

    # ── League scoring baseline ──────────────────────────────────────────

    async def league_baseline(
        self, espn_sport: str, slug: str, season: Optional[int] = None
    ) -> float:
        """Average goals/points scored per team per game across the league.

        This is the anchor for the attack/defence strength model — without it
        there is no way to say whether 1.8 goals a game is good or bad.
        """
        rows = await self.get_standings(espn_sport, slug, season)
        vals = [r["avg_for"] for r in rows.values() if r["avg_for"] > 0]
        if not vals:
            return 0.0
        return round(sum(vals) / len(vals), 3)
