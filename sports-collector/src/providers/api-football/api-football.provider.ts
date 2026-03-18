import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios, { AxiosInstance, AxiosError } from 'axios';
import axiosRetry from 'axios-retry';
import {
  SportsProviderInterface,
  NormalizedLeague,
  NormalizedTeam,
  NormalizedMatch,
  NormalizedMatchEvent,
  NormalizedStanding,
  NormalizedPlayer,
} from '../interfaces/sports-provider.interface';
import {
  ApiFootballResponse,
  ApiFootballLeague,
  ApiFootballTeam,
  ApiFootballFixture,
  ApiFootballStandingWrapper,
  ApiFootballPlayerWrapper,
  ApiFootballEvent,
  ApiFootballStanding,
} from './api-football.types';

// https://www.api-football.com/documentation-v3#section/Status-Codes
const STATUS_MAP: Record<string, string> = {
  TBD: 'SCHEDULED',
  NS: 'SCHEDULED',
  '1H': 'LIVE',
  HT: 'HALFTIME',
  '2H': 'LIVE',
  ET: 'LIVE',
  BT: 'LIVE',
  P: 'LIVE',
  SUSP: 'LIVE',
  INT: 'LIVE',
  LIVE: 'LIVE',
  FT: 'FINISHED',
  AET: 'FINISHED',
  PEN: 'FINISHED',
  AWD: 'FINISHED',
  WO: 'FINISHED',
  PST: 'POSTPONED',
  CANC: 'CANCELLED',
  ABD: 'CANCELLED',
};

const EVENT_TYPE_MAP: Record<string, string> = {
  subst: 'SUBSTITUTION',
  Var: 'VAR',
};

@Injectable()
export class ApiFootballProvider implements SportsProviderInterface {
  readonly providerName = 'api-football';
  private readonly logger = new Logger(ApiFootballProvider.name);
  private readonly client: AxiosInstance;

  constructor(private readonly config: ConfigService) {
    this.client = axios.create({
      baseURL: config.get<string>(
        'API_FOOTBALL_BASE_URL',
        'https://v3.football.api-sports.io',
      ),
      headers: {
        'x-apisports-key': config.get<string>('API_FOOTBALL_KEY', ''),
        'x-rapidapi-key': config.get<string>('API_FOOTBALL_KEY', ''),
      },
      timeout: 15_000,
    });

    axiosRetry(this.client, {
      retries: 3,
      retryDelay: axiosRetry.exponentialDelay,
      retryCondition: (error: AxiosError) =>
        axiosRetry.isNetworkOrIdempotentRequestError(error) ||
        error.response?.status === 429 ||
        error.response?.status === 503,
      onRetry: (retryCount, error) =>
        this.logger.warn(
          `API-Football retry ${retryCount}: ${error.message}`,
        ),
    });

    // Monitor rate-limit headers
    this.client.interceptors.response.use((response) => {
      const remaining = response.headers['x-ratelimit-requests-remaining'];
      const limit = response.headers['x-ratelimit-requests-limit'];
      if (remaining !== undefined && parseInt(remaining) < 20) {
        this.logger.warn(
          `API-Football rate limit: ${remaining}/${limit} remaining`,
        );
      }
      return response;
    });
  }

  // ── Public API ────────────────────────────────────────────────────────

  async fetchLeagues(season: number): Promise<NormalizedLeague[]> {
    this.logger.log(`Fetching all current leagues for season ${season}`);
    const data = await this.get<ApiFootballLeague>(
      `/leagues?season=${season}&current=true`,
    );
    return data.map((item) => this.normalizeLeague(item, season));
  }

  async fetchLeaguesByIds(
    ids: number[],
    season: number,
  ): Promise<NormalizedLeague[]> {
    const results: NormalizedLeague[] = [];
    for (const id of ids) {
      const data = await this.get<ApiFootballLeague>(
        `/leagues?id=${id}&season=${season}`,
      );
      results.push(...data.map((item) => this.normalizeLeague(item, season)));
    }
    return results;
  }

  async fetchTeams(
    leagueExternalId: string,
    season: number,
  ): Promise<NormalizedTeam[]> {
    this.logger.log(`Fetching teams for league ${leagueExternalId} s${season}`);
    const data = await this.get<ApiFootballTeam>(
      `/teams?league=${leagueExternalId}&season=${season}`,
    );
    return data.map((item) => this.normalizeTeam(item));
  }

  async fetchFixtures(
    leagueExternalId: string,
    season: number,
    date?: string,
  ): Promise<NormalizedMatch[]> {
    let url = `/fixtures?league=${leagueExternalId}&season=${season}`;
    if (date) url += `&date=${date}`;
    this.logger.log(`Fetching fixtures: ${url}`);
    const data = await this.get<ApiFootballFixture>(url);
    return data.map((item) => this.normalizeFixture(item));
  }

  async fetchLiveFixtures(leagueIds?: string[]): Promise<NormalizedMatch[]> {
    const live = leagueIds?.length ? leagueIds.join('-') : 'all';
    const data = await this.get<ApiFootballFixture>(`/fixtures?live=${live}`);
    return data.map((item) => this.normalizeFixture(item));
  }

  async fetchStandings(
    leagueExternalId: string,
    season: number,
  ): Promise<NormalizedStanding[]> {
    this.logger.log(`Fetching standings league=${leagueExternalId} s${season}`);
    const data = await this.get<ApiFootballStandingWrapper>(
      `/standings?league=${leagueExternalId}&season=${season}`,
    );
    const standings: NormalizedStanding[] = [];
    for (const item of data) {
      for (const group of item.league.standings) {
        for (const s of group) {
          standings.push(this.normalizeStanding(s));
        }
      }
    }
    return standings;
  }

  async fetchPlayers(
    teamExternalId: string,
    season: number,
  ): Promise<NormalizedPlayer[]> {
    this.logger.log(`Fetching players team=${teamExternalId} s${season}`);
    const players: NormalizedPlayer[] = [];
    let page = 1;

    // API-Football paginates players — fetch all pages
    while (true) {
      const raw = await this.getRaw<ApiFootballPlayerWrapper>(
        `/players?team=${teamExternalId}&season=${season}&page=${page}`,
      );
      players.push(
        ...raw.response.map((item) =>
          this.normalizePlayer(item, teamExternalId),
        ),
      );
      if (raw.paging.current >= raw.paging.total) break;
      page++;
    }
    return players;
  }

  // ── Normalizers ───────────────────────────────────────────────────────

  private normalizeLeague(
    item: ApiFootballLeague,
    season: number,
  ): NormalizedLeague {
    const currentSeason = item.seasons.find((s) => s.current) ?? item.seasons[0];
    return {
      externalId: String(item.league.id),
      name: item.league.name,
      slug: item.league.name.toLowerCase().replace(/[\s_]+/g, '-'),
      country: item.country.name,
      logo: item.league.logo || undefined,
      sport: 'football',
      currentSeason: currentSeason ? String(currentSeason.year) : String(season),
      seasonStart: currentSeason?.start ? new Date(currentSeason.start) : undefined,
      seasonEnd: currentSeason?.end ? new Date(currentSeason.end) : undefined,
    };
  }

  private normalizeTeam(item: ApiFootballTeam): NormalizedTeam {
    return {
      externalId: String(item.team.id),
      name: item.team.name,
      shortName: item.team.code ?? undefined,
      logo: item.team.logo || undefined,
      country: item.team.country ?? undefined,
      founded: item.team.founded ?? undefined,
      venue: item.venue?.name ?? undefined,
    };
  }

  private normalizeFixture(item: ApiFootballFixture): NormalizedMatch {
    const status = STATUS_MAP[item.fixture.status.short] ?? 'SCHEDULED';
    return {
      externalId: String(item.fixture.id),
      homeTeamExternalId: String(item.teams.home.id),
      awayTeamExternalId: String(item.teams.away.id),
      leagueExternalId: String(item.league.id),
      season: String(item.league.season),
      scheduledAt: new Date(item.fixture.date),
      status,
      minute: item.fixture.status.elapsed ?? undefined,
      homeScore: item.goals.home ?? undefined,
      awayScore: item.goals.away ?? undefined,
      homeHalfScore: item.score.halftime.home ?? undefined,
      awayHalfScore: item.score.halftime.away ?? undefined,
      venue: item.fixture.venue.name ?? undefined,
      referee: item.fixture.referee ?? undefined,
      events: (item.events ?? []).map((e) => this.normalizeEvent(e)),
    };
  }

  private normalizeEvent(e: ApiFootballEvent): NormalizedMatchEvent {
    let type: string;
    if (e.type === 'Goal') {
      type = e.detail === 'Own Goal' ? 'OWN_GOAL' : 'GOAL';
    } else if (e.type === 'Card') {
      if (e.detail.includes('Yellow Red')) {
        type = 'YELLOW_RED_CARD';
      } else if (e.detail.includes('Red')) {
        type = 'RED_CARD';
      } else {
        type = 'YELLOW_CARD';
      }
    } else {
      type = EVENT_TYPE_MAP[e.type] ?? e.type.toUpperCase();
    }
    return {
      type,
      minute: e.time.elapsed,
      playerExternalId: e.player.id ? String(e.player.id) : undefined,
      playerName: e.player.name || undefined,
      teamExternalId: String(e.team.id),
      detail: e.detail || undefined,
      comments: e.comments ?? undefined,
    };
  }

  private normalizeStanding(s: ApiFootballStanding): NormalizedStanding {
    return {
      teamExternalId: String(s.team.id),
      rank: s.rank,
      points: s.points,
      played: s.all.played,
      won: s.all.win,
      drawn: s.all.draw,
      lost: s.all.lose,
      goalsFor: s.all.goals.for,
      goalsAgainst: s.all.goals.against,
      goalDiff: s.goalsDiff,
      form: s.form || undefined,
      description: s.description ?? undefined,
    };
  }

  private normalizePlayer(
    item: ApiFootballPlayerWrapper,
    teamExternalId: string,
  ): NormalizedPlayer {
    const stats = item.statistics[0];
    return {
      externalId: String(item.player.id),
      name: item.player.name,
      firstName: item.player.firstname || undefined,
      lastName: item.player.lastname || undefined,
      nationality: item.player.nationality ?? undefined,
      birthDate:
        item.player.birth?.date ? new Date(item.player.birth.date) : undefined,
      position: stats?.games.position ?? undefined,
      number: stats?.games.number ?? undefined,
      photo: item.player.photo || undefined,
      teamExternalId,
    };
  }

  // ── HTTP helpers ──────────────────────────────────────────────────────

  private async get<T>(path: string): Promise<T[]> {
    const raw = await this.getRaw<T>(path);
    return raw.response;
  }

  private async getRaw<T>(path: string): Promise<ApiFootballResponse<T>> {
    const { data } = await this.client.get<ApiFootballResponse<T>>(path);

    if (Array.isArray(data.errors) && data.errors.length > 0) {
      throw new Error(`API-Football error: ${JSON.stringify(data.errors)}`);
    }
    if (
      !Array.isArray(data.errors) &&
      Object.keys(data.errors ?? {}).length > 0
    ) {
      throw new Error(`API-Football error: ${JSON.stringify(data.errors)}`);
    }

    return data;
  }
}
