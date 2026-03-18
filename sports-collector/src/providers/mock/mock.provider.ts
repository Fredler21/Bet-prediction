import { Injectable } from '@nestjs/common';
import {
  SportsProviderInterface,
  NormalizedLeague,
  NormalizedTeam,
  NormalizedMatch,
  NormalizedStanding,
  NormalizedPlayer,
} from '../interfaces/sports-provider.interface';

/**
 * Mock provider — returns realistic static data for local development
 * and CI environments where no API key is available.
 * Activate via: SPORTS_PROVIDER=mock in .env
 */
@Injectable()
export class MockSportsProvider implements SportsProviderInterface {
  readonly providerName = 'mock';

  private readonly leagues: NormalizedLeague[] = [
    {
      externalId: '39',
      name: 'Premier League',
      slug: 'premier-league',
      country: 'England',
      sport: 'football',
      currentSeason: '2024',
    },
    {
      externalId: '140',
      name: 'La Liga',
      slug: 'la-liga',
      country: 'Spain',
      sport: 'football',
      currentSeason: '2024',
    },
  ];

  private readonly teams: NormalizedTeam[] = [
    { externalId: '42',  name: 'Arsenal',          shortName: 'ARS', country: 'England' },
    { externalId: '33',  name: 'Manchester United', shortName: 'MUN', country: 'England' },
    { externalId: '40',  name: 'Liverpool',         shortName: 'LIV', country: 'England' },
    { externalId: '50',  name: 'Manchester City',   shortName: 'MCI', country: 'England' },
    { externalId: '47',  name: 'Tottenham',         shortName: 'TOT', country: 'England' },
    { externalId: '49',  name: 'Chelsea',           shortName: 'CHE', country: 'England' },
    { externalId: '529', name: 'Barcelona',         shortName: 'BAR', country: 'Spain'   },
    { externalId: '541', name: 'Real Madrid',       shortName: 'RMA', country: 'Spain'   },
  ];

  async fetchLeagues(_season: number): Promise<NormalizedLeague[]> {
    return this.leagues;
  }

  async fetchLeaguesByIds(ids: number[], _season: number): Promise<NormalizedLeague[]> {
    return this.leagues.filter((l) => ids.map(String).includes(l.externalId));
  }

  async fetchTeams(_leagueExternalId: string, _season: number): Promise<NormalizedTeam[]> {
    return this.teams;
  }

  async fetchFixtures(
    leagueExternalId: string,
    season: number,
    _date?: string,
  ): Promise<NormalizedMatch[]> {
    const now = new Date();
    return [
      {
        externalId: `mock-${leagueExternalId}-${season}-1`,
        homeTeamExternalId: '42',
        awayTeamExternalId: '33',
        leagueExternalId,
        season: String(season),
        scheduledAt: new Date(now.getTime() + 2 * 3600_000),
        status: 'SCHEDULED',
      },
      {
        externalId: `mock-${leagueExternalId}-${season}-2`,
        homeTeamExternalId: '40',
        awayTeamExternalId: '50',
        leagueExternalId,
        season: String(season),
        scheduledAt: new Date(now.getTime() + 5 * 3600_000),
        status: 'SCHEDULED',
      },
    ];
  }

  async fetchLiveFixtures(_leagueIds?: string[]): Promise<NormalizedMatch[]> {
    return [
      {
        externalId: 'mock-live-1001',
        homeTeamExternalId: '42',
        awayTeamExternalId: '40',
        leagueExternalId: '39',
        season: '2024',
        scheduledAt: new Date(Date.now() - 60 * 60_000),
        status: 'LIVE',
        minute: 67,
        homeScore: 2,
        awayScore: 1,
        events: [
          {
            type: 'GOAL',
            minute: 23,
            playerExternalId: 'mock-p-7',
            playerName: 'Bukayo Saka',
            teamExternalId: '42',
            detail: 'Normal Goal',
          },
          {
            type: 'GOAL',
            minute: 55,
            playerExternalId: 'mock-p-11',
            playerName: 'Mohamed Salah',
            teamExternalId: '40',
            detail: 'Normal Goal',
          },
          {
            type: 'GOAL',
            minute: 61,
            playerExternalId: 'mock-p-9',
            playerName: 'Gabriel Jesus',
            teamExternalId: '42',
            detail: 'Normal Goal',
          },
        ],
      },
    ];
  }

  async fetchStandings(
    leagueExternalId: string,
    _season: number,
  ): Promise<NormalizedStanding[]> {
    if (leagueExternalId !== '39') return [];
    return [
      {
        teamExternalId: '42',
        rank: 1, points: 62, played: 28, won: 19, drawn: 5, lost: 4,
        goalsFor: 68, goalsAgainst: 30, goalDiff: 38, form: 'WWWDW',
        description: 'Champions League',
      },
      {
        teamExternalId: '50',
        rank: 2, points: 59, played: 28, won: 18, drawn: 5, lost: 5,
        goalsFor: 71, goalsAgainst: 35, goalDiff: 36, form: 'WDWWW',
        description: 'Champions League',
      },
      {
        teamExternalId: '40',
        rank: 3, points: 55, played: 28, won: 17, drawn: 4, lost: 7,
        goalsFor: 63, goalsAgainst: 34, goalDiff: 29, form: 'WWLWW',
        description: 'Champions League',
      },
      {
        teamExternalId: '49',
        rank: 4, points: 51, played: 28, won: 15, drawn: 6, lost: 7,
        goalsFor: 55, goalsAgainst: 40, goalDiff: 15, form: 'WLWDW',
        description: 'Champions League',
      },
    ];
  }

  async fetchPlayers(
    teamExternalId: string,
    _season: number,
  ): Promise<NormalizedPlayer[]> {
    return [
      {
        externalId: 'mock-p-7',
        name: 'Bukayo Saka',
        position: 'Attacker',
        number: 7,
        nationality: 'England',
        teamExternalId,
      },
      {
        externalId: 'mock-p-29',
        name: 'Kai Havertz',
        position: 'Attacker',
        number: 29,
        nationality: 'Germany',
        teamExternalId,
      },
    ];
  }
}
