import {
  Controller,
  Get,
  Param,
  ParseIntPipe,
  Query,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiQuery, ApiResponse } from '@nestjs/swagger';
import { MatchesService } from './matches.service';

@ApiTags('matches')
@Controller('matches')
export class MatchesController {
  constructor(private readonly matchesService: MatchesService) {}

  @Get('live')
  @ApiOperation({
    summary: 'Get all currently live matches',
    description: 'Results are cached in Redis for 30 seconds.',
  })
  @ApiResponse({ status: 200, description: 'Live matches with events and team details' })
  getLive() {
    return this.matchesService.getLiveMatches();
  }

  @Get('date/:date')
  @ApiOperation({ summary: 'Get matches scheduled on a specific date (YYYY-MM-DD)' })
  getByDate(@Param('date') date: string) {
    return this.matchesService.getMatchesByDate(new Date(date));
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get full match details by internal ID' })
  @ApiResponse({ status: 404, description: 'Match not found' })
  getOne(@Param('id', ParseIntPipe) id: number) {
    return this.matchesService.getMatchById(id);
  }

  @Get('season/:seasonId')
  @ApiOperation({ summary: 'Get all matches for a season' })
  getBySeason(@Param('seasonId', ParseIntPipe) seasonId: number) {
    return this.matchesService.getMatchesBySeason(seasonId);
  }
}
