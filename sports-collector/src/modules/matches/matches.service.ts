import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../../common/prisma/prisma.service';
import { RedisService } from '../../common/redis/redis.service';
import { NormalizedMatch } from '../../providers/interfaces/sports-provider.interface';
import { Match, MatchStatus, MatchEventType } from '@prisma/client';

const LIVE_CACHE_KEY = 'matches:live';
const LIVE_CACHE_TTL = 30;    // seconds — matches live-update interval
const MATCH_CACHE_TTL = 300;  // 5 minutes per individual match

@Injectable()
export class MatchesService {
  private readonly logger = new Logger(MatchesService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly redis: RedisService,
  ) {}

  // ── Write ─────────────────────────────────────────────────────────────

  /**
   * Upsert a match. Requires home/away teams to already exist in the DB.
   * @param data Normalized match from provider
   * @param seasonId Internal Season.id (resolved by caller)
   */
  async upsertMatch(data: NormalizedMatch, seasonId: number): Promise<Match> {
    const [homeTeam, awayTeam] = await Promise.all([
      this.prisma.team.findUnique({ where: { externalId: data.homeTeamExternalId } }),
      this.prisma.team.findUnique({ where: { externalId: data.awayTeamExternalId } }),
    ]);

    if (!homeTeam) {
      throw new Error(`Home team not found: externalId=${data.homeTeamExternalId}`);
    }
    if (!awayTeam) {
      throw new Error(`Away team not found: externalId=${data.awayTeamExternalId}`);
    }

    const status = this.parseStatus(data.status);

    const match = await this.prisma.match.upsert({
      where: { externalId: data.externalId },
      create: {
        externalId: data.externalId,
        seasonId,
        homeTeamId: homeTeam.id,
        awayTeamId: awayTeam.id,
        scheduledAt: data.scheduledAt,
        status,
        minute: data.minute,
        homeScore: data.homeScore,
        awayScore: data.awayScore,
        homeHalfScore: data.homeHalfScore,
        awayHalfScore: data.awayHalfScore,
        venue: data.venue,
        referee: data.referee,
      },
      update: {
        status,
        minute: data.minute,
        homeScore: data.homeScore,
        awayScore: data.awayScore,
        homeHalfScore: data.homeHalfScore,
        awayHalfScore: data.awayHalfScore,
      },
    });

    // Upsert match events (goals, cards, etc.) when provided
    if (data.events?.length) {
      await this.upsertMatchEvents(match.id, data);
    }

    // Invalidate per-match and live caches
    await this.redis.del(`match:${match.id}`, LIVE_CACHE_KEY);
    return match;
  }

  async upsertMany(matches: NormalizedMatch[], seasonId: number): Promise<void> {
    let saved = 0;
    for (const match of matches) {
      try {
        await this.upsertMatch(match, seasonId);
        saved++;
      } catch (err) {
        this.logger.warn(
          `Skipped match ${match.externalId}: ${(err as Error).message}`,
        );
      }
    }
    this.logger.log(`Upserted ${saved}/${matches.length} matches`);
    await this.redis.del(LIVE_CACHE_KEY);
  }

  // ── Read ──────────────────────────────────────────────────────────────

  async getLiveMatches(): Promise<Match[]> {
    const cached = await this.redis.getJson<Match[]>(LIVE_CACHE_KEY);
    if (cached) return cached;

    const matches = await this.prisma.match.findMany({
      where: { status: { in: [MatchStatus.LIVE, MatchStatus.HALFTIME] } },
      include: {
        homeTeam: true,
        awayTeam: true,
        season: { include: { league: { include: { sport: true } } } },
        events: { orderBy: { minute: 'asc' } },
      },
      orderBy: { scheduledAt: 'asc' },
    });

    await this.redis.setJson(LIVE_CACHE_KEY, matches, LIVE_CACHE_TTL);
    return matches;
  }

  async getMatchById(id: number): Promise<Match> {
    const cached = await this.redis.getJson<Match>(`match:${id}`);
    if (cached) return cached;

    const match = await this.prisma.match.findUnique({
      where: { id },
      include: {
        homeTeam: true,
        awayTeam: true,
        events: {
          include: { player: true },
          orderBy: { minute: 'asc' },
        },
        season: { include: { league: { include: { sport: true } } } },
      },
    });
    if (!match) throw new NotFoundException(`Match #${id} not found`);

    const ttl = match.status === MatchStatus.LIVE ? LIVE_CACHE_TTL : MATCH_CACHE_TTL;
    await this.redis.setJson(`match:${id}`, match, ttl);
    return match;
  }

  async getMatchesByDate(date: Date): Promise<Match[]> {
    const start = new Date(date);
    start.setHours(0, 0, 0, 0);
    const end = new Date(date);
    end.setHours(23, 59, 59, 999);

    return this.prisma.match.findMany({
      where: { scheduledAt: { gte: start, lte: end } },
      include: {
        homeTeam: true,
        awayTeam: true,
        season: { include: { league: true } },
      },
      orderBy: { scheduledAt: 'asc' },
    });
  }

  async getMatchesBySeason(seasonId: number): Promise<Match[]> {
    return this.prisma.match.findMany({
      where: { seasonId },
      include: { homeTeam: true, awayTeam: true },
      orderBy: { scheduledAt: 'asc' },
    });
  }

  async invalidateLiveCache(): Promise<void> {
    await this.redis.del(LIVE_CACHE_KEY);
  }

  // ── Private helpers ───────────────────────────────────────────────────

  private parseStatus(raw: string): MatchStatus {
    const map: Record<string, MatchStatus> = {
      SCHEDULED: MatchStatus.SCHEDULED,
      LIVE:      MatchStatus.LIVE,
      HALFTIME:  MatchStatus.HALFTIME,
      FINISHED:  MatchStatus.FINISHED,
      POSTPONED: MatchStatus.POSTPONED,
      CANCELLED: MatchStatus.CANCELLED,
    };
    return map[raw] ?? MatchStatus.SCHEDULED;
  }

  private async upsertMatchEvents(
    matchId: number,
    data: NormalizedMatch,
  ): Promise<void> {
    const typeMap: Record<string, MatchEventType> = {
      GOAL:           MatchEventType.GOAL,
      OWN_GOAL:       MatchEventType.OWN_GOAL,
      PENALTY:        MatchEventType.PENALTY,
      YELLOW_CARD:    MatchEventType.YELLOW_CARD,
      RED_CARD:       MatchEventType.RED_CARD,
      YELLOW_RED_CARD:MatchEventType.YELLOW_RED_CARD,
      SUBSTITUTION:   MatchEventType.SUBSTITUTION,
      VAR:            MatchEventType.VAR,
    };

    // Remove stale events and re-insert (provider sends full event list)
    await this.prisma.matchEvent.deleteMany({ where: { matchId } });

    const eventData = (data.events ?? []).map((e) => ({
      matchId,
      type: typeMap[e.type] ?? MatchEventType.GOAL,
      minute: e.minute,
      detail: e.detail,
      comments: e.comments,
      teamId: e.teamExternalId ? parseInt(e.teamExternalId, 10) : undefined,
    }));

    if (eventData.length > 0) {
      await this.prisma.matchEvent.createMany({ data: eventData });
    }
  }
}
