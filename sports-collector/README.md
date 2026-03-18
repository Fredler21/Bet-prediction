# Sports Collector

A production-ready sports data collection service built with NestJS, Prisma (PostgreSQL), Redis, BullMQ, and WebSockets. Designed to run as a standalone backend — think of it as your own lightweight Sofascore-style data pipeline.

---

## Features

- **Multi-provider data ingestion** — supports API-Football v3 or a built-in mock provider for zero-cost local development
- **Scheduled ingestion** — daily cron at 2 AM pulls leagues, teams, fixtures, standings, and players into Postgres
- **Live score updates** — every 30 s BullMQ worker polls live fixtures and broadcasts changes via Socket.IO
- **REST API** — leagues, matches, teams, players, standings, admin trigger
- **WebSocket gateway** — `/live` namespace; clients can subscribe to individual match rooms (`match:{id}`)
- **Redis caching** — hot data cached with short TTLs to avoid hammering the DB on every request

---

## Tech Stack

| Layer | Tech |
|---|---|
| Framework | NestJS 10 |
| Database | PostgreSQL 16 + Prisma ORM |
| Cache / Queue broker | Redis 7 + BullMQ |
| Real-time | Socket.IO (NestJS gateway) |
| API provider | API-Football v3 (or mock) |
| Container | Docker (multi-stage, non-root) |

---

## Quick Start (local, no API key needed)

```bash
# 1. Copy env file
cp .env.example .env

# 2. Start Postgres + Redis
docker compose up postgres redis -d

# 3. Install deps and run migrations
npm install
npm run db:migrate    # applies prisma/migrations
npm run db:generate   # generates Prisma client

# 4. Start dev server (uses mock provider by default)
npm run start:dev
```

Open <http://localhost:3001/api/docs> for the Swagger UI.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3001` | HTTP port |
| `DATABASE_URL` | see `.env.example` | Postgres connection string |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | _(empty)_ | Redis password (leave empty if none) |
| `SPORTS_PROVIDER` | `mock` | `mock` or `api-football` |
| `API_FOOTBALL_KEY` | _(required for api-football)_ | API-Football v3 key |
| `LEAGUES_TO_TRACK` | `39,140,135,78,61,2` | Comma-separated league IDs |
| `CURRENT_SEASON` | `2024` | Season year (2024 = 2024/25) |
| `DAILY_INGESTION_CRON` | `0 0 2 * * *` | 6-field cron for daily run |
| `LIVE_UPDATE_INTERVAL_SECONDS` | `30` | Live polling interval |

---

## REST Endpoints

All routes are prefixed with `/api`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check (used by Railway/Render) |
| `GET` | `/api/leagues` | List tracked leagues |
| `GET` | `/api/matches` | List fixtures (filter: `?leagueId=&date=&status=`) |
| `GET` | `/api/matches/:id` | Single match + events |
| `GET` | `/api/teams` | List teams |
| `GET` | `/api/teams/:id` | Single team |
| `GET` | `/api/players` | List players (filter: `?teamId=`) |
| `GET` | `/api/standings` | Standings table (filter: `?leagueId=&season=`) |
| `POST` | `/api/admin/ingest` | Manually trigger full data ingestion |

Full docs at `/api/docs` (Swagger, disabled in production).

---

## WebSocket

Connect to the `/live` namespace:

```js
import { io } from 'socket.io-client';

const socket = io('http://localhost:3001/live');

// Subscribe to all live updates
socket.on('live:update', (matches) => console.log(matches));

// Subscribe to a specific match room
socket.emit('join:match', { matchId: 'abc123' });
socket.on('match:update', (match) => console.log(match));
```

---

## Deployment

### Railway (recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

1. Create a new Railway project → **Deploy from GitHub repo**
2. Set the **Root Directory** to `sports-collector`
3. Add a **Postgres** plugin and a **Redis** plugin from the Railway dashboard
4. Set env vars: `API_FOOTBALL_KEY`, `SPORTS_PROVIDER=api-football`, `LEAGUES_TO_TRACK`, `CURRENT_SEASON`
5. Railway reads `railway.toml` and builds via the Dockerfile automatically

### Render

1. Go to [render.com](https://render.com) → **New Blueprint**
2. Connect your GitHub repo — Render reads `render.yaml` and provisions:
   - Web service (Docker build)
   - Managed Postgres
   - Managed Redis
3. Set `API_FOOTBALL_KEY` in Render's environment secrets

### Docker Compose (self-hosted)

```bash
export API_FOOTBALL_KEY=your_key_here
docker compose up --build -d
```

This spins up Postgres, Redis, and the app together. Migrations run automatically on startup.

---

## Database Schema

```
Sport → League → Season → Match → MatchEvent
                        └→ Team (home/away)
 Team → Player
League → Standing
```

Full schema in [prisma/schema.prisma](prisma/schema.prisma).

---

## Getting an API-Football Key

1. Sign up at <https://www.api-football.com> (free tier: 100 calls/day)
2. Copy your key from the dashboard
3. Set `API_FOOTBALL_KEY=your_key` and `SPORTS_PROVIDER=api-football` in your env

The free tier is enough to track 2–3 leagues in real-time. For more leagues, upgrade or use multiple keys.
