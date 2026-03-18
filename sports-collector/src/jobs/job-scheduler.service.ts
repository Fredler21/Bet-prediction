import { Injectable, Logger } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import { InjectQueue } from '@nestjs/bullmq';
import { Queue } from 'bullmq';
import { ConfigService } from '@nestjs/config';

/**
 * Schedules background jobs using cron expressions.
 *
 * Daily ingestion — runs at 2:00 AM every day and enqueues
 * a full data ingest (leagues → teams → fixtures → standings).
 *
 * Live updates — runs every 30 seconds and enqueues a live-match
 * fetch. Skips enqueueing if a job is already waiting, to avoid
 * pile-up when processing is slow.
 */
@Injectable()
export class JobSchedulerService {
  private readonly logger = new Logger(JobSchedulerService.name);

  constructor(
    @InjectQueue('daily-ingestion') private readonly ingestionQueue: Queue,
    @InjectQueue('live-updates') private readonly liveQueue: Queue,
    private readonly config: ConfigService,
  ) {}

  // ── Daily full ingestion — 2:00 AM ────────────────────────────────────

  @Cron('0 0 2 * * *')
  async scheduleDailyIngestion(): Promise<void> {
    const leagueIds = this.config
      .get<string>('LEAGUES_TO_TRACK', '39,140,135,78,61,2')
      .split(',')
      .filter(Boolean);

    const season = this.config.get<number>('CURRENT_SEASON', 2024);

    this.logger.log(
      `Scheduling daily ingestion for leagues [${leagueIds.join(',')}] season=${season}`,
    );

    await this.ingestionQueue.add(
      'ingest-all',
      { leagueIds, season, includePlayerFetch: false },
      {
        removeOnComplete: { count: 50 },
        removeOnFail: { count: 100 },
        attempts: 3,
        backoff: { type: 'exponential', delay: 60_000 },
      },
    );
  }

  // ── Live score updates — every 30 seconds ─────────────────────────────

  @Cron('*/30 * * * * *')
  async scheduleLiveUpdates(): Promise<void> {
    // Prevent queue pile-up: only add if no job is currently waiting
    const waitingJobs = await this.liveQueue.getWaitingCount();
    if (waitingJobs > 0) {
      this.logger.debug('Live update job still in queue — skipping');
      return;
    }

    await this.liveQueue.add(
      'update-live',
      {},
      {
        removeOnComplete: true,
        removeOnFail: { count: 10 },
        attempts: 2,
        backoff: { type: 'fixed', delay: 5_000 },
      },
    );
  }

  // ── Manual trigger (useful for initial seed) ──────────────────────────

  async triggerIngestionNow(includePlayerFetch = false): Promise<void> {
    const leagueIds = this.config
      .get<string>('LEAGUES_TO_TRACK', '39,140,135,78,61,2')
      .split(',')
      .filter(Boolean);

    const season = this.config.get<number>('CURRENT_SEASON', 2024);

    this.logger.log('Triggering immediate full ingestion');
    await this.ingestionQueue.add(
      'ingest-all',
      { leagueIds, season, includePlayerFetch },
      { removeOnComplete: true, removeOnFail: { count: 5 } },
    );
  }
}
