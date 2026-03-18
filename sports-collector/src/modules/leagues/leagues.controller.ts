import {
  Controller,
  Get,
  Param,
  ParseIntPipe,
  HttpCode,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { LeaguesService } from './leagues.service';

@ApiTags('leagues')
@Controller('leagues')
export class LeaguesController {
  constructor(private readonly leaguesService: LeaguesService) {}

  @Get()
  @ApiOperation({ summary: 'Get all tracked leagues' })
  @ApiResponse({ status: 200, description: 'List of all leagues' })
  findAll() {
    return this.leaguesService.findAll();
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a league by internal ID' })
  @ApiResponse({ status: 200, description: 'League details with seasons' })
  @ApiResponse({ status: 404, description: 'League not found' })
  findOne(@Param('id', ParseIntPipe) id: number) {
    return this.leaguesService.findById(id);
  }

  @Get(':id/standings')
  @ApiOperation({ summary: 'Get current standings for a league' })
  @HttpCode(200)
  getStandings(@Param('id', ParseIntPipe) id: number) {
    // Delegates to StandingsService — exposed here for convenience
    return { leagueId: id, message: 'Use /api/standings/league/:leagueId' };
  }
}
