/**
 * Normalized internal types — provider-agnostic.
 * All external API data is converted to these shapes before being stored.
 */

export interface NormalizedLeague {
  externalId: string;
  name: string;
  slug: string;
  country: string;
  logo?: string;
  sport: string;
  currentSeason?: string;
  seasonStart?: Date;
  seasonEnd?: Date;
}

export interface NormalizedTeam {
  externalId: string;
  name: string;
  shortName?: string;
  logo?: string;
  country?: string;
  founded?: number;
  venue?: string;
}

export interface NormalizedMatchEvent {
  type: string;          // GOAL | YELLOW_CARD | RED_CARD | SUBSTITUTION | VAR …
  minute: number;
  playerExternalId?: string;
  playerName?: string;
  teamExternalId?: string;
  detail?: string;
  comments?: string;
}

export interface NormalizedMatch {
  externalId: string;
  homeTeamExternalId: string;
  awayTeamExternalId: string;
  leagueExternalId: string;
  season: string;        // e.g. "2024"
  scheduledAt: Date;
  status: string;        // one of MatchStatus enum values
  minute?: number;
  homeScore?: number;
  awayScore?: number;
  homeHalfScore?: number;
  awayHalfScore?: number;
  venue?: string;
  referee?: string;
  events?: NormalizedMatchEvent[];
}

export interface NormalizedStanding {
  teamExternalId: string;
  rank: number;
  points: number;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goalsFor: number;
  goalsAgainst: number;
  goalDiff: number;
  form?: string;
  description?: string;
}

export interface NormalizedPlayer {
  externalId: string;
  name: string;
  firstName?: string;
  lastName?: string;
  nationality?: string;
  birthDate?: Date;
  position?: string;
  number?: number;
  photo?: string;
  teamExternalId?: string;
}

/**
 * Common interface every sports data provider must implement.
 * Add a new provider by implementing this interface and registering it
 * in ProviderFactory.
 */
export interface SportsProviderInterface {
  readonly providerName: string;

  fetchLeagues(season: number): Promise<NormalizedLeague[]>;

  fetchLeaguesByIds(ids: number[], season: number): Promise<NormalizedLeague[]>;

  fetchTeams(leagueExternalId: string, season: number): Promise<NormalizedTeam[]>;

  fetchFixtures(
    leagueExternalId: string,
    season: number,
    date?: string,
  ): Promise<NormalizedMatch[]>;

  fetchLiveFixtures(leagueIds?: string[]): Promise<NormalizedMatch[]>;

  fetchStandings(
    leagueExternalId: string,
    season: number,
  ): Promise<NormalizedStanding[]>;

  fetchPlayers(
    teamExternalId: string,
    season: number,
  ): Promise<NormalizedPlayer[]>;
}

export const SPORTS_PROVIDER_TOKEN = Symbol('SPORTS_PROVIDER');
