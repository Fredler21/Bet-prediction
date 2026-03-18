import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../../common/prisma/prisma.service';
import { RedisService } from '../../common/redis/redis.service';
import { NormalizedPlayer } from '../../providers/interfaces/sports-provider.interface';
import { Player } from '@prisma/client';

@Injectable()
export class PlayersService {
  private readonly logger = new Logger(PlayersService.name);
  private readonly CACHE_TTL = 1800; // 30 min

  constructor(
    private readonly prisma: PrismaService,
    private readonly redis: RedisService,
  ) {}

  async upsertPlayer(data: NormalizedPlayer): Promise<Player> {
    let teamId: number | undefined;
    if (data.teamExternalId) {
      const team = await this.prisma.team.findUnique({
        where: { externalId: data.teamExternalId },
      });
      teamId = team?.id;
    }

    const player = await this.prisma.player.upsert({
      where: { externalId: data.externalId },
      create: {
        externalId: data.externalId,
        name: data.name,
        firstName: data.firstName,
        lastName: data.lastName,
        nationality: data.nationality,
        birthDate: data.birthDate,
        position: data.position,
        number: data.number,
        photo: data.photo,
        teamId,
      },
      update: {
        name: data.name,
        position: data.position,
        number: data.number,
        photo: data.photo,
        teamId,
      },
    });

    await this.redis.del(`player:${player.id}`);
    return player;
  }

  async upsertMany(players: NormalizedPlayer[]): Promise<void> {
    let saved = 0;
    for (const p of players) {
      try {
        await this.upsertPlayer(p);
        saved++;
      } catch (err) {
        this.logger.warn(`Skipped player ${p.externalId}: ${(err as Error).message}`);
      }
    }
    this.logger.log(`Upserted ${saved}/${players.length} players`);
  }

  async findById(id: number): Promise<Player> {
    const cached = await this.redis.getJson<Player>(`player:${id}`);
    if (cached) return cached;

    const player = await this.prisma.player.findUnique({
      where: { id },
      include: { team: true },
    });
    if (!player) throw new NotFoundException(`Player #${id} not found`);

    await this.redis.setJson(`player:${id}`, player, this.CACHE_TTL);
    return player;
  }

  async findByTeam(teamId: number): Promise<Player[]> {
    return this.prisma.player.findMany({
      where: { teamId },
      orderBy: [{ position: 'asc' }, { number: 'asc' }, { name: 'asc' }],
    });
  }

  async search(query: string, limit = 20): Promise<Player[]> {
    return this.prisma.player.findMany({
      where: {
        name: { contains: query, mode: 'insensitive' },
      },
      take: limit,
      include: { team: true },
    });
  }
}
