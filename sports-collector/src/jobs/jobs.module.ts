import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bullmq';
import { LeaguesModule } from '../modules/leagues/leagues.module';
import { TeamsModule } from '../modules/teams/teams.module';
import { MatchesModule } from '../modules/matches/matches.module';
import { PlayersModule } from '../modules/players/players.module';
import { StandingsModule } from '../modules/standings/standings.module';
import { ProviderModule } from '../providers/provider.module';
import { GatewaysModule } from '../gateways/gateways.module';
import { DailyIngestionProcessor } from './daily-ingestion.processor';
import { LiveUpdatesProcessor } from './live-updates.processor';
import { JobSchedulerService } from './job-scheduler.service';

@Module({
  imports: [
    BullModule.registerQueue(
      { name: 'daily-ingestion' },
      { name: 'live-updates' },
    ),
    ProviderModule,
    LeaguesModule,
    TeamsModule,
    MatchesModule,
    PlayersModule,
    StandingsModule,
    GatewaysModule,
  ],
  providers: [
    DailyIngestionProcessor,
    LiveUpdatesProcessor,
    JobSchedulerService,
  ],
  exports: [JobSchedulerService],
})
export class JobsModule {}
