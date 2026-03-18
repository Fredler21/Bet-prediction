import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../../common/prisma/prisma.service';
import { RedisService } from '../../common/redis/redis.service';
import { NormalizedLeague } from '../../providers/interfaces/sports-provider.interface';
import { League, Season } from '@prisma/client';

@Injectable()
export class LeaguesService {
  private readonly logger = new Logger(LeaguesService.name);
  private readonly CACHE_TTL = 3600; // 1 hour

  constructor(
    private readonly prisma: PrismaService,
    private readonly redis: RedisService,
  ) {}

  /**
   * Upsert a league and its current season from normalized provider data.
   * Returns both the league and season records.
   */
  async upsertLeague(
    data: NormalizedLeague,
  ): Promise<{ league: League; season: Season }> {
    // Ensure the Sport record exists
    const sport = await this.prisma.sport.upsert({
      where: { slug: data.sport },
      create: {
        name: data.sport.charAt(0).toUpperCase() + data.sport.slice(1),
        slug: data.sport,
      },
      update: {},
    });

    const league = await this.prisma.league.upsert({
      where: { externalId: data.externalId },
      create: {
        externalId: data.externalId,
        name: data.name,
        slug: data.slug,
        country: data.country,
        logo: data.logo,
        sportId: sport.id,
      },
      update: {
        name: data.name,
        slug: data.slug,
        country: data.country,
        logo: data.logo,
      },
    });

    const seasonYear = data.currentSeason ?? String(new Date().getFullYear());
    const seasonExternalId = `${data.externalId}-${seasonYear}`;

    const season = await this.prisma.season.upsert({
      where: { externalId: seasonExternalId },
      create: {
        externalId: seasonExternalId,
        year: seasonYear,
        leagueId: league.id,
        isCurrent: true,
        startDate: data.seasonStart,
        endDate: data.seasonEnd,
      },
      update: {
        isCurrent: true,
        startDate: data.seasonStart,
        endDate: data.seasonEnd,
      },
    });

    await this.redis.del(`leagues:all`, `league:${league.id}`);
    this.logger.log(`Upserted league: ${league.name} (${seasonYear})`);

    return { league, season };
  }

  async findAll(): Promise<League[]> {
    const cached = await this.redis.getJson<League[]>('leagues:all');
    if (cached) return cached;

    const leagues = await this.prisma.league.findMany({
      include: {
        sport: true,
        seasons: { where: { isCurrent: true }, take: 1 },
      },
      orderBy: [{ sport: { name: 'asc' } }, { country: 'asc' }, { name: 'asc' }],
    });

    await this.redis.setJson('leagues:all', leagues, this.CACHE_TTL);
    return leagues;
  }

  async findById(id: number): Promise<League> {
    const cached = await this.redis.getJson<League>(`league:${id}`);
    if (cached) return cached;

    const league = await this.prisma.league.findUnique({
      where: { id },
      include: { sport: true, seasons: { orderBy: { year: 'desc' } } },
    });
    if (!league) throw new NotFoundException(`League #${id} not found`);

    await this.redis.setJson(`league:${id}`, league, this.CACHE_TTL);
    return league;
  }

  async getCurrentSeason(leagueId: number): Promise<Season | null> {
    return this.prisma.season.findFirst({
      where: { leagueId, isCurrent: true },
    });
  }

  async getSeasonByYearAndLeague(
    leagueExternalId: string,
    year: string,
  ): Promise<Season | null> {
    const externalId = `${leagueExternalId}-${year}`;
    return this.prisma.season.findUnique({ where: { externalId } });
  }
}
