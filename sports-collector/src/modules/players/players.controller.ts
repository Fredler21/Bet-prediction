import { Controller, Get, Param, ParseIntPipe, Query } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiQuery } from '@nestjs/swagger';
import { PlayersService } from './players.service';

@ApiTags('players')
@Controller('players')
export class PlayersController {
  constructor(private readonly playersService: PlayersService) {}

  @Get('search')
  @ApiOperation({ summary: 'Search players by name' })
  @ApiQuery({ name: 'q', description: 'Search query' })
  @ApiQuery({ name: 'limit', required: false, type: Number })
  search(@Query('q') q: string, @Query('limit') limit = 20) {
    return this.playersService.search(q, +limit);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get player by internal ID' })
  findOne(@Param('id', ParseIntPipe) id: number) {
    return this.playersService.findById(id);
  }
}
