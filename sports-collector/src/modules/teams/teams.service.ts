import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../../common/prisma/prisma.service';
import { RedisService } from '../../common/redis/redis.service';
import { NormalizedTeam } from '../../providers/interfaces/sports-provider.interface';
import { Team } from '@prisma/client';

@Injectable()
export class TeamsService {
  private readonly logger = new Logger(TeamsService.name);
  private readonly CACHE_TTL = 1800; // 30 min

  constructor(
    private readonly prisma: PrismaService,
    private readonly redis: RedisService,
  ) {}

  async upsertTeam(data: NormalizedTeam): Promise<Team> {
    const team = await this.prisma.team.upsert({
      where: { externalId: data.externalId },
      create: {
        externalId: data.externalId,
        name: data.name,
        shortName: data.shortName,
        logo: data.logo,
        country: data.country,
        founded: data.founded,
        venue: data.venue,
      },
      update: {
        name: data.name,
        shortName: data.shortName,
        logo: data.logo,
        country: data.country,
        founded: data.founded,
        venue: data.venue,
      },
    });

    await this.redis.del(`team:${team.id}`, `team:ext:${team.externalId}`);
    return team;
  }

  async upsertMany(teams: NormalizedTeam[]): Promise<void> {
    await Promise.all(teams.map((t) => this.upsertTeam(t)));
    this.logger.log(`Upserted ${teams.length} teams`);
  }

  async findById(id: number): Promise<Team> {
    const cached = await this.redis.getJson<Team>(`team:${id}`);
    if (cached) return cached;

    const team = await this.prisma.team.findUnique({
      where: { id },
      include: { players: true },
    });
    if (!team) throw new NotFoundException(`Team #${id} not found`);

    await this.redis.setJson(`team:${id}`, team, this.CACHE_TTL);
    return team;
  }

  async findByExternalId(externalId: string): Promise<Team | null> {
    const cached = await this.redis.getJson<Team>(`team:ext:${externalId}`);
    if (cached) return cached;

    const team = await this.prisma.team.findUnique({ where: { externalId } });
    if (team) {
      await this.redis.setJson(`team:ext:${externalId}`, team, this.CACHE_TTL);
    }
    return team;
  }

  async findAll(limit = 100, offset = 0): Promise<Team[]> {
    return this.prisma.team.findMany({
      take: limit,
      skip: offset,
      orderBy: [{ country: 'asc' }, { name: 'asc' }],
    });
  }
}
