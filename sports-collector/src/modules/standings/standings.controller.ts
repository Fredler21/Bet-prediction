import { Controller, Get, Param, ParseIntPipe } from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import { StandingsService } from './standings.service';

@ApiTags('standings')
@Controller('standings')
export class StandingsController {
  constructor(private readonly standingsService: StandingsService) {}

  @Get('league/:leagueId')
  @ApiOperation({ summary: 'Get current season standings for a league' })
  getByLeague(@Param('leagueId', ParseIntPipe) leagueId: number) {
    return this.standingsService.getByLeague(leagueId);
  }
}
