import { Injectable, Logger } from '@nestjs/common';
import { PrismaService } from '../../common/prisma/prisma.service';
import { RedisService } from '../../common/redis/redis.service';
import { NormalizedStanding } from '../../providers/interfaces/sports-provider.interface';
import { Standing } from '@prisma/client';

@Injectable()
export class StandingsService {
  private readonly logger = new Logger(StandingsService.name);
  private readonly CACHE_TTL = 1800; // 30 min

  constructor(
    private readonly prisma: PrismaService,
    private readonly redis: RedisService,
  ) {}

  async upsertStandings(
    leagueId: number,
    seasonId: number,
    standings: NormalizedStanding[],
  ): Promise<void> {
    for (const s of standings) {
      const team = await this.prisma.team.findUnique({
        where: { externalId: s.teamExternalId },
      });
      if (!team) {
        this.logger.warn(`Team not found for standing: ${s.teamExternalId}`);
        continue;
      }

      await this.prisma.standing.upsert({
        where: { leagueId_seasonId_teamId: { leagueId, seasonId, teamId: team.id } },
        create: {
          leagueId,
          seasonId,
          teamId: team.id,
          rank: s.rank,
          points: s.points,
          played: s.played,
          won: s.won,
          drawn: s.drawn,
          lost: s.lost,
          goalsFor: s.goalsFor,
          goalsAgainst: s.goalsAgainst,
          goalDiff: s.goalDiff,
          form: s.form,
          description: s.description,
        },
        update: {
          rank: s.rank,
          points: s.points,
          played: s.played,
          won: s.won,
          drawn: s.drawn,
          lost: s.lost,
          goalsFor: s.goalsFor,
          goalsAgainst: s.goalsAgainst,
          goalDiff: s.goalDiff,
          form: s.form,
          description: s.description,
        },
      });
    }

    await this.redis.del(`standings:league:${leagueId}`);
    this.logger.log(`Upserted ${standings.length} standings for league #${leagueId}`);
  }

  async getByLeague(leagueId: number): Promise<Standing[]> {
    const cacheKey = `standings:league:${leagueId}`;
    const cached = await this.redis.getJson<Standing[]>(cacheKey);
    if (cached) return cached;

    const standings = await this.prisma.standing.findMany({
      where: {
        leagueId,
        season: { isCurrent: true },
      },
      include: {
        team: true,
        league: true,
      },
      orderBy: { rank: 'asc' },
    });

    await this.redis.setJson(cacheKey, standings, this.CACHE_TTL);
    return standings;
  }
}
