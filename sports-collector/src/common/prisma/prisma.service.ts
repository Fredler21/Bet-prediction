import { Injectable, OnModuleInit, OnModuleDestroy, Logger } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(PrismaService.name);

  constructor() {
    super({
      log: [
        { level: 'warn', emit: 'event' },
        { level: 'error', emit: 'event' },
        // Enable query logging only in development
        ...(process.env.NODE_ENV === 'development'
          ? [{ level: 'query' as const, emit: 'event' as const }]
          : []),
      ],
    });
  }

  async onModuleInit() {
    await this.$connect();

    // Log slow queries in development
    if (process.env.NODE_ENV === 'development') {
      (this.$on as any)('query', (e: { query: string; duration: number }) => {
        if (e.duration > 500) {
          this.logger.warn(`Slow query (${e.duration}ms): ${e.query}`);
        }
      });
    }

    (this.$on as any)('error', (e: { message: string }) => {
      this.logger.error(`Prisma error: ${e.message}`);
    });

    this.logger.log('Database connected');
  }

  async onModuleDestroy() {
    await this.$disconnect();
    this.logger.log('Database disconnected');
  }
}
