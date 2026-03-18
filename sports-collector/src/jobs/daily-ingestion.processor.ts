import { Injectable, Logger } from '@nestjs/common';
import { Processor, WorkerHost } from '@nestjs/bullmq';
import { Job } from 'bullmq';
import { ConfigService } from '@nestjs/config';
import { SPORTS_PROVIDER_TOKEN, SportsProviderInterface } from '../providers/interfaces/sports-provider.interface';
import { Inject } from '@nestjs/common';
import { LeaguesService } from '../modules/leagues/leagues.service';
import { TeamsService } from '../modules/teams/teams.service';
import { MatchesService } from '../modules/matches/matches.service';
import { StandingsService } from '../modules/standings/standings.service';
import { PlayersService } from '../modules/players/players.service';

export interface DailyIngestionPayload {
  leagueIds: string[];
  season: number;
  includePlayerFetch?: boolean;
}

/**
 * Processes the 'daily-ingestion' BullMQ queue.
 *
 * Job lifecycle:
 *   1. ingest-leagues   → upsert all tracked leagues + seasons
 *   2. ingest-teams     → upsert all teams for each league
 *   3. ingest-fixtures  → upsert all upcoming & recent fixtures
 *   4. ingest-standings → upsert current standings table
 *   5. ingest-players   → (optional) upsert squad for each team
 */
@Processor('daily-ingestion', {
  // Concurrency: 1 to respect API rate limits
  concurrency: 1,
})
export class DailyIngestionProcessor extends WorkerHost {
  private readonly logger = new Logger(DailyIngestionProcessor.name);

  constructor(
    @Inject(SPORTS_PROVIDER_TOKEN)
    private readonly provider: SportsProviderInterface,
    private readonly config: ConfigService,
    private readonly leaguesService: LeaguesService,
    private readonly teamsService: TeamsService,
    private readonly matchesService: MatchesService,
    private readonly standingsService: StandingsService,
    private readonly playersService: PlayersService,
  ) {
    super();
  }

  async process(job: Job<DailyIngestionPayload>): Promise<void> {
    this.logger.log(`Processing job [${job.name}] id=${job.id}`);
    const { leagueIds, season, includePlayerFetch = false } = job.data;

    try {
      switch (job.name) {
        case 'ingest-leagues':
          await this.ingestLeagues(leagueIds, season);
          break;
        case 'ingest-teams':
          await this.ingestTeams(leagueIds, season);
          break;
        case 'ingest-fixtures':
          await this.ingestFixtures(leagueIds, season);
          break;
        case 'ingest-standings':
          await this.ingestStandings(leagueIds, season);
          break;
        case 'ingest-players':
          if (includePlayerFetch) await this.ingestPlayers(leagueIds, season);
          break;
        case 'ingest-all':
          await this.ingestLeagues(leagueIds, season);
          await this.ingestTeams(leagueIds, season);
          await this.ingestFixtures(leagueIds, season);
          await this.ingestStandings(leagueIds, season);
          if (includePlayerFetch) await this.ingestPlayers(leagueIds, season);
          break;
        default:
          this.logger.warn(`Unknown job name: ${job.name}`);
      }
    } catch (err) {
      this.logger.error(
        `Job [${job.name}] failed: ${(err as Error).message}`,
        (err as Error).stack,
      );
      throw err; // Re-throw so BullMQ marks it as failed (enables retries)
    }
  }

  // ── Pipeline steps ────────────────────────────────────────────────────

  private async ingestLeagues(leagueIds: string[], season: number): Promise<void> {
    this.logger.log(`Ingesting ${leagueIds.length} leagues for season ${season}`);
    const leagues = await this.provider.fetchLeaguesByIds(
      leagueIds.map(Number),
      season,
    );
    for (const league of leagues) {
      await this.leaguesService.upsertLeague(league);
    }
    this.logger.log(`Ingested ${leagues.length} leagues`);
  }

  private async ingestTeams(leagueIds: string[], season: number): Promise<void> {
    for (const leagueExternalId of leagueIds) {
      try {
        const teams = await this.provider.fetchTeams(leagueExternalId, season);
        await this.teamsService.upsertMany(teams);
        this.logger.log(`Ingested ${teams.length} teams for league ${leagueExternalId}`);
      } catch (err) {
        this.logger.warn(
          `Failed to ingest teams for league ${leagueExternalId}: ${(err as Error).message}`,
        );
      }
    }
  }

  private async ingestFixtures(leagueIds: string[], season: number): Promise<void> {
    for (const leagueExternalId of leagueIds) {
      try {
        const fixtures = await this.provider.fetchFixtures(leagueExternalId, season);

        // Resolve the season record
        const seasonRecord = await this.leaguesService.getSeasonByYearAndLeague(
          leagueExternalId,
          String(season),
        );
        if (!seasonRecord) {
          this.logger.warn(
            `Season record not found for league ${leagueExternalId} season ${season} — run ingest-leagues first`,
          );
          continue;
        }

        await this.matchesService.upsertMany(fixtures, seasonRecord.id);
        this.logger.log(
          `Ingested ${fixtures.length} fixtures for league ${leagueExternalId}`,
        );
      } catch (err) {
        this.logger.warn(
          `Failed to ingest fixtures for league ${leagueExternalId}: ${(err as Error).message}`,
        );
      }
    }
  }

  private async ingestStandings(leagueIds: string[], season: number): Promise<void> {
    for (const leagueExternalId of leagueIds) {
      try {
        const standings = await this.provider.fetchStandings(leagueExternalId, season);
        if (!standings.length) continue;

        const seasonRecord = await this.leaguesService.getSeasonByYearAndLeague(
          leagueExternalId,
          String(season),
        );
        if (!seasonRecord) continue;

        // Resolve the internal League id
        const league = await this.leaguesService.findAll().then((leagues) =>
          leagues.find((l) => l.externalId === leagueExternalId),
        );
        if (!league) continue;

        await this.standingsService.upsertStandings(
          league.id,
          seasonRecord.id,
          standings,
        );
      } catch (err) {
        this.logger.warn(
          `Failed to ingest standings for league ${leagueExternalId}: ${(err as Error).message}`,
        );
      }
    }
  }

  private async ingestPlayers(leagueIds: string[], season: number): Promise<void> {
    for (const leagueExternalId of leagueIds) {
      try {
        const teams = await this.provider.fetchTeams(leagueExternalId, season);
        for (const team of teams) {
          const players = await this.provider.fetchPlayers(team.externalId, season);
          await this.playersService.upsertMany(players);
        }
      } catch (err) {
        this.logger.warn(
          `Failed to ingest players for league ${leagueExternalId}: ${(err as Error).message}`,
        );
      }
    }
  }
}
