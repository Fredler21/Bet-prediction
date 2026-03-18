import { Module } from '@nestjs/common';
import { LiveScoresGateway } from './live-scores.gateway';

@Module({
  providers: [LiveScoresGateway],
  exports: [LiveScoresGateway],
})
export class GatewaysModule {}
