import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { ScheduleModule } from '@nestjs/schedule';
import { BullModule } from '@nestjs/bullmq';
import { PrismaModule } from './common/prisma/prisma.module';
import { RedisModule } from './common/redis/redis.module';
import { LeaguesModule } from './modules/leagues/leagues.module';
import { TeamsModule } from './modules/teams/teams.module';
import { MatchesModule } from './modules/matches/matches.module';
import { PlayersModule } from './modules/players/players.module';
import { StandingsModule } from './modules/standings/standings.module';
import { AdminModule } from './modules/admin/admin.module';
import { JobsModule } from './jobs/jobs.module';

@Module({
  imports: [
    // Config — available globally throughout all modules
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),

    // Cron scheduler
    ScheduleModule.forRoot(),

    // BullMQ — shared Redis connection for all queues
    BullModule.forRootAsync({
      inject: [ConfigService],
      useFactory: (config: ConfigService) => ({
        connection: {
          host: config.get<string>('REDIS_HOST', 'localhost'),
          port: config.get<number>('REDIS_PORT', 6379),
          password: config.get<string>('REDIS_PASSWORD') || undefined,
        },
      }),
    }),

    // Infrastructure (global)
    PrismaModule,
    RedisModule,

    // Feature modules
    LeaguesModule,
    TeamsModule,
    MatchesModule,
    PlayersModule,
    StandingsModule,
    AdminModule,

    // Background jobs
    JobsModule,
  ],
})
export class AppModule {}
