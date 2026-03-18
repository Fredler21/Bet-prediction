import { Controller, Get, Param, ParseIntPipe, Query } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiQuery } from '@nestjs/swagger';
import { TeamsService } from './teams.service';

@ApiTags('teams')
@Controller('teams')
export class TeamsController {
  constructor(private readonly teamsService: TeamsService) {}

  @Get()
  @ApiOperation({ summary: 'List teams' })
  @ApiQuery({ name: 'limit', required: false, type: Number })
  @ApiQuery({ name: 'offset', required: false, type: Number })
  findAll(
    @Query('limit') limit = 100,
    @Query('offset') offset = 0,
  ) {
    return this.teamsService.findAll(+limit, +offset);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get team by internal ID (includes squad)' })
  findOne(@Param('id', ParseIntPipe) id: number) {
    return this.teamsService.findById(id);
  }
}
