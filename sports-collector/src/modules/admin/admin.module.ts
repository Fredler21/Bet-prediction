import { Module } from '@nestjs/common';
import { AdminController } from './admin.controller';
import { JobsModule } from '../../jobs/jobs.module';

@Module({
  imports: [JobsModule],
  controllers: [AdminController],
})
export class AdminModule {}
