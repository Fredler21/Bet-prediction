import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { ApiFootballProvider } from './api-football/api-football.provider';
import { MockSportsProvider } from './mock/mock.provider';
import {
  SportsProviderInterface,
  SPORTS_PROVIDER_TOKEN,
} from './interfaces/sports-provider.interface';

/**
 * ProviderFactory resolves the active sports data provider based on
 * the SPORTS_PROVIDER environment variable.
 *
 * Adding a new provider:
 *   1. Implement SportsProviderInterface
 *   2. Add it to the providers array below
 *   3. Add a case to the useFactory switch statement
 */
@Module({
  imports: [ConfigModule],
  providers: [
    ApiFootballProvider,
    MockSportsProvider,
    {
      provide: SPORTS_PROVIDER_TOKEN,
      inject: [ConfigService, ApiFootballProvider, MockSportsProvider],
      useFactory: (
        config: ConfigService,
        apiFootball: ApiFootballProvider,
        mock: MockSportsProvider,
      ): SportsProviderInterface => {
        const provider = config.get<string>('SPORTS_PROVIDER', 'mock');
        switch (provider) {
          case 'api-football':
            return apiFootball;
          case 'mock':
          default:
            return mock;
        }
      },
    },
  ],
  exports: [SPORTS_PROVIDER_TOKEN],
})
export class ProviderModule {}
