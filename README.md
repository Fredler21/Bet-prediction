# 🏆 Bet Prediction

Multi-sport betting model. Pulls real fixtures, standings and results from
ESPN's public API, builds one probability model per game, and reads a full
board of markets off it.

**Informational only. Bet responsibly.**

---

## What it does

For every fixture it can source real data for, it builds a single
distribution over the final scoreline and derives ~120 markets across 26 bet
types from that one model — so the moneyline, the handicap, the total and the
correct-score board always agree with each other.

Where a real bookmaker price is available it is shown and used to compute
expected value. Where one is not, the price shown is the model's own fair
price, **labelled `MODEL`**, and no value is claimed against it.

## Data sources

| What | Source | Notes |
|------|--------|-------|
| Fixtures | ESPN public API | 35 league feeds, no key required |
| Standings (W/D/L, goals for & against, table position) | ESPN | anchors the scoring model |
| Form, home/away splits, head-to-head | ESPN team schedules | real completed results |
| Rosters (player markets) | ESPN team rosters | real names and positions |
| Moneyline / spread / total prices | DraftKings via ESPN | when the feed carries them |
| Soccer (optional) | API-Football v3 | set `API_FOOTBALL_KEY` |

SofaScore support remains in the code, but its API rejects server-side
requests; it is used only if you supply `SOFASCORE_PROXY_KEY`.

### Leagues covered

**Soccer (24)** — Champions League, Europa League, Conference League, Premier
League, La Liga, Serie A, Bundesliga, Ligue 1, MLS, Liga MX, Copa
Libertadores, Brasileirão, Liga Profesional, Nations League, World Cup
Qualifying (UEFA), Eredivisie, Primeira Liga, Saudi Pro League, Süper Lig,
Championship, Scottish Premiership, FA Cup, Carabao Cup, Copa del Rey

**Basketball (5)** — NBA, WNBA, NCAA M, NCAA W, G League
**Football (2)** — NFL, NCAA
**Baseball (2)** — MLB, NCAA
**Hockey (2)** — NHL, NCAA M

Sports with no feed wired up (tennis, volleyball, MMA, handball, rugby) are
not advertised in the UI, rather than shown as permanently empty tabs.

---

## The model

Scoring rates come from the attack/defence method: a team's scoring measured
against its league's average, combined with the opponent's defensive record.
Two different forms, because the sports behave differently:

- **Low-scoring (soccer, hockey, baseball)** — ratio strengths feeding a
  **Poisson score grid**: the joint distribution over every plausible
  scoreline. Every market is a sum over cells of that grid.
- **High-scoring (basketball, American football)** — each offence paired with
  the defence it faces and averaged, feeding a **normal model** of margin and
  total. Poisson fits these badly, and multiplying strength ratios compounds
  badly once scoring is in the tens or hundreds.

An eight-factor analysis (form, home advantage, H2H, table position, scoring,
injuries, consistency, momentum) applies a **bounded ±15% adjustment** to each
side's projected scoring. It is not normalised into a probability directly.

Rates are **shrunk toward the league average** by sample size — a team's own
record gets weight `n / (n + 6)`. Early in a season the projection sits near
the league average and says so on the match; by mid-season it is almost
entirely the team's own record.

Sanity checks against known long-run rates:

| Check | Model | Reality |
|-------|-------|---------|
| League-average soccer match, 1X2 | 44.5 / 24.2 / 31.3 | ≈ 45 / 25 / 30 |
| NHL game total | 6.14 | ≈ 6.1 |
| MLB game total | 8.84 | ≈ 8.8 |
| NFL total vs a posted DraftKings line | 50.2 | 50.5 |

## Bet types

**Result** — moneyline / 1X2, double chance, draw no bet, 3-way regulation
(hockey), half & period winner
**Handicap** — main spread, alternate spreads, Asian (quarter) handicaps
**Totals** — main total, alternate totals, team totals, exact total, odd/even
**Soccer specials** — both teams to score, clean sheet, win to nil, correct
score, winning margin, first to score (including the goalless case)
**Halves & periods** — half-time result, half-time totals, half-time /
full-time double, both halves over, highest-scoring half, quarter totals
**Combinations** — result + total, result + both teams to score
**Baseball** — first five innings result and total
**Player** — anytime goalscorer, player points
**Game props** — overtime / extra innings, race to X

**Parlays** — standard, same-game (correlation-adjusted, contradictory legs
removed), round robin (per-ticket staking), teaser (lines actually moved and
re-priced from the model), flex (exact Poisson-binomial for at-most-k misses).

---

## Data integrity

The model reports only what it can source.

- **No generated fixtures.** If no real fixture exists for a sport and date,
  the answer is an empty list. Sample data is available for local development
  behind `ENABLE_SAMPLE_DATA=1`, and anything it produces is labelled
  `SAMPLE DATA` in the API and on the page.
- **Every price is labelled.** `BOOK` means a real bookmaker price. `MODEL`
  means our own fair price, and expected value against it is reported as
  zero, because scoring the model against its own number proves nothing.
- **Missing inputs are stated, not filled in.** No head-to-head record, no
  league table entry, form carried over from last season — each is surfaced
  as a note on the match instead of being substituted with a
  plausible-looking number.
- **Implausible edges are flagged, not sold.** A closing line is the best
  public predictor of a result there is. Where the model disagrees with one
  by more than 15 percentage points, the reasoning says the likely
  explanation is thin inputs on our side; disagreements above 30% are kept
  out of the value list entirely.
- **Markets nobody could bet are not shown.** Anything outside a 6–94%
  probability band is skipped, and prices are capped near -1900.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # optional: OPENAI_API_KEY, API_FOOTBALL_KEY

python main.py              # web dashboard on http://localhost:8000
python main.py cli          # terminal UI
python main.py parlay 6     # 6-leg parlay
python main.py report       # daily text report
```

No API key is needed — ESPN's public endpoints require none.

### One deployment note

ESPN answers **403** to requests carrying a spoofed desktop-browser
`User-Agent`. Requests must go out with a plain client identifier
(`_ESPN_HEADERS` in `src/sofascore_client.py`). This is not cosmetic: it was
the cause of the live site serving invented fixtures, because every ESPN call
failed and the code then silently fell back to a fixture generator.

## Architecture

```
main.py                    entry point (web/cli/report/parlay)
api/index.py               Vercel handler
src/
  markets.py               probability engine: score grid, normal model, pricing
  analyzer.py              market generation + eight-factor analysis
  espn_stats.py            real standings, form, H2H, splits from ESPN
  sofascore_client.py      fixtures, rosters, bookmaker prices
  api_football_client.py   optional API-Football soccer path
  parlay_optimizer.py      parlays, correlation, Kelly staking
  agent.py                 orchestrator (+ optional LLM review)
  web.py                   FastAPI API + dashboard
  database.py              SQLite prediction log
  cli.py                   terminal UI
tests/
  test_markets.py          calibration and pricing regression tests
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/matches` | markets grouped by match (+ `data_source`, `data_notes`) |
| GET | `/api/predictions` | flat prediction list |
| GET | `/api/past-games` | any date, with final scores |
| GET | `/api/value-bets` | +EV markets, book-priced only |
| POST | `/api/parlay` | build a parlay |
| GET | `/api/parlays` | parlays across strategies |
| POST | `/api/sgp` · `/api/round-robin` · `/api/teaser` · `/api/flex-parlay` | parlay variants |
| GET | `/api/sports` | sports with a real feed |
| GET | `/api/report` · `/api/health` · `/` | report, health, dashboard |

## Optional AI layer

With `OPENAI_API_KEY` set, an LLM reviews picks for angles the statistics miss
(motivation, derbies, scheduling). Without it the system runs on the
statistical model alone — the model is the product; the LLM is commentary.

## Known limitations

- **Early season is thin.** Scoring rates are shrunk toward the league average
  in proportion to how many games they rest on, so two clean sheets cannot
  pass for an elite defence, and the match carries a note saying so. It is
  still two games of information — the shrinkage keeps the model honest about
  that rather than making it clairvoyant.
- **Player markets are team-level.** Lines come from the team's projected
  scoring and a positional share. Real names and positions, but there is no
  per-player feed, so no per-player edge is claimed.
- **No line-movement or multi-book comparison.** One price, one snapshot.
- **Corners** appear only when a feed supplies corner counts, which ESPN does
  not — so that market is normally absent by design.

## Future work

- [ ] Multi-bookmaker odds comparison and line-movement tracking
- [ ] Per-player statistics feed for genuine player props
- [ ] Weather for outdoor sports
- [ ] Backtesting against the logged prediction history
- [ ] Calibration report (predicted vs. realised, by probability bucket)
