import { Injectable, Logger, Inject } from '@nestjs/common';
import { Processor, WorkerHost } from '@nestjs/bullmq';
import { Job } from 'bullmq';
import { SPORTS_PROVIDER_TOKEN, SportsProviderInterface } from '../providers/interfaces/sports-provider.interface';
import { LeaguesService } from '../modules/leagues/leagues.service';
import { MatchesService } from '../modules/matches/matches.service';
import { LiveScoresGateway } from '../gateways/live-scores.gateway';
import { ConfigService } from '@nestjs/config';

/**
 * Processes the 'live-updates' BullMQ queue.
 *
 * Runs every 30 seconds (triggered by JobSchedulerService).
 * Fetches live fixtures from the provider, updates the DB and
 * Redis cache, then broadcasts changes via WebSocket.
 */
@Processor('live-updates', { concurrency: 1 })
export class LiveUpdatesProcessor extends WorkerHost {
  private readonly logger = new Logger(LiveUpdatesProcessor.name);

  constructor(
    @Inject(SPORTS_PROVIDER_TOKEN)
    private readonly provider: SportsProviderInterface,
    private readonly config: ConfigService,
    private readonly leaguesService: LeaguesService,
    private readonly matchesService: MatchesService,
    private readonly liveGateway: LiveScoresGateway,
  ) {
    super();
  }

  async process(job: Job): Promise<void> {
    if (job.name !== 'update-live') return;

    const leagueIds = this.config
      .get<string>('LEAGUES_TO_TRACK', '')
      .split(',')
      .filter(Boolean);

    const liveFixtures = await this.provider.fetchLiveFixtures(leagueIds);

    if (!liveFixtures.length) {
      // No live games: invalidate stale cache so clients see the latest DB state
      await this.matchesService.invalidateLiveCache();
      return;
    }

    this.logger.log(`Processing ${liveFixtures.length} live fixtures`);

    const season = this.config.get<number>('CURRENT_SEASON', 2024);
    let updatedCount = 0;

    for (const fixture of liveFixtures) {
      try {
        const seasonRecord = await this.leaguesService.getSeasonByYearAndLeague(
          fixture.leagueExternalId,
          fixture.season,
        );
        if (!seasonRecord) continue;

        await this.matchesService.upsertMatch(fixture, seasonRecord.id);
        updatedCount++;
      } catch (err) {
        this.logger.warn(
          `Failed to update live match ${fixture.externalId}: ${(err as Error).message}`,
        );
      }
    }

    // Invalidate live cache so next read fetches fresh DB data
    await this.matchesService.invalidateLiveCache();

    // Fetch the freshly updated live matches for broadcasting
    const updatedLiveMatches = await this.matchesService.getLiveMatches();

    // Broadcast to all connected WebSocket clients
    this.liveGateway.broadcastLiveUpdate(updatedLiveMatches);

    this.logger.log(
      `Live update complete: ${updatedCount}/${liveFixtures.length} matches updated`,
    );
  }
}
