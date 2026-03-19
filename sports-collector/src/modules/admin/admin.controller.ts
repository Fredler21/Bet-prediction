import { Controller, Post, HttpCode, Logger } from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import { JobSchedulerService } from '../../jobs/job-scheduler.service';

/**
 * Admin endpoints for manually triggering data ingestion.
 * Protect these with an API key / IP allowlist in production.
 */
@ApiTags('admin')
@Controller('admin')
export class AdminController {
  private readonly logger = new Logger(AdminController.name);

  constructor(private readonly scheduler: JobSchedulerService) {}

  @Post('ingest')
  @HttpCode(202)
  @ApiOperation({
    summary: 'Trigger a full data ingestion immediately',
    description: 'Enqueues an ingest-all BullMQ job. Responds 202 Accepted.',
  })
  async triggerIngestion() {
    await this.scheduler.triggerIngestionNow();
    this.logger.log('Manual ingestion triggered via admin endpoint');
    return { queued: true, message: 'Full ingestion job enqueued' };
  }

  @Post('ingest/players')
  @HttpCode(202)
  @ApiOperation({ summary: 'Trigger full ingestion INCLUDING player squads' })
  async triggerIngestionWithPlayers() {
    await this.scheduler.triggerIngestionNow(true);
    return { queued: true, message: 'Full ingestion (with players) job enqueued' };
  }
}
