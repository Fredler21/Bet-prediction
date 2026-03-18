# 🏆 Premium Bet Prediction AI Agent

AI-powered sports betting prediction system that pulls real-time data from SofaScore, runs multi-factor statistical analysis, and uses LLM reasoning to generate premium predictions, parlays, and value bets.

## Features

### 🔬 Data & Analysis
- **Live SofaScore Integration** — pulls schedules, team stats, H2H records, standings, lineups, injuries, and odds
- **Multi-Factor Statistical Model** — 8 weighted factors: form, home advantage, H2H, league position, scoring patterns, injuries, consistency, momentum
- **Sport-Specific Weights** — each sport has tuned factor weights (e.g., H2H matters more in tennis, home advantage matters more in soccer)
- **Poisson Scoring Model** — over/under predictions using Poisson distribution analysis
- **Value Bet Detection** — compares our estimated probability vs bookmaker implied probability to find +EV bets

### 🎯 Bet Types (like Hardwork Bet / DraftKings / FanDuel)
- **Moneyline / 1X2** — who wins
- **Spread / Handicap** — point spread betting
- **Over/Under (Totals)** — total goals/points
- **Both Teams to Score (BTTS)** — soccer specialty
- **Double Chance** — 1X, X2, 12
- **Parlay Builder** — multi-leg combination bets with optimizer

### 🏟️ Sports Covered
| Sport | Key Metrics |
|-------|-----------|
| ⚽ Soccer | Form, H2H, BTTS, O/U 2.5, possession, clean sheets |
| 🏀 Basketball | Scoring averages, pace, home court, point spread |
| 🎾 Tennis | H2H (heavily weighted), surface, recent form, injuries |
| ⚾ Baseball | Pitcher matchups, run lines, O/U 7.5 |
| 🏈 American Football | Spread, O/U 45.5, home field, injury report |
| 🏐 Volleyball | Set totals, form, home advantage |
| 🏒 Hockey | Puck line, O/U, goalie stats |
| 🥊 MMA | Fighter stats, reach, recent form |

### 💰 Bankroll Management
- **Kelly Criterion** — calculates mathematically optimal bet sizing
- **Fractional Kelly** — conservative version (25% Kelly by default)
- **Risk Assessment** — low/medium/high rating for every bet

### 🤖 AI Enhancement (Optional)
- **GPT-4o Analysis** — validates picks with LLM reasoning
- **Hidden Factor Detection** — identifies motivation, derbies, scheduling edges
- **Parlay Review** — AI reviews and critiques parlay selections

### 📊 Accuracy Tracking
- SQLite database logs every prediction
- Tracks win/loss record and ROI over time
- Per-sport accuracy breakdown

---

## Quick Start

### 1. Install Dependencies
```bash
cd /workspaces/Bet-prediction
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (optional — works without it)
```

### 3. Run

**Web Dashboard** (recommended):
```bash
python main.py
# Open http://localhost:8000 in your browser
```

**CLI Interface**:
```bash
python main.py cli
```

**Quick Parlay**:
```bash
python main.py parlay 6    # Build 6-leg parlay
```

**Daily Report**:
```bash
python main.py report
```

---

## Architecture

```
Bet-prediction/
├── main.py                  # Entry point (web/cli/report/parlay)
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py            # Settings & environment
│   ├── models.py            # Data models (Sport, Team, Event, Prediction, etc.)
│   ├── sofascore_client.py  # SofaScore API client (async, cached, retries)
│   ├── analyzer.py          # Statistical analysis engine (8-factor model)
│   ├── parlay_optimizer.py  # Parlay builder + Kelly Criterion bankroll
│   ├── agent.py             # AI Prediction Agent (orchestrator)
│   ├── database.py          # SQLite tracking for accuracy monitoring
│   ├── cli.py               # Rich terminal UI
│   └── web.py               # FastAPI REST API + web dashboard
└── data/                    # SQLite DB (auto-created)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/predictions` | Get predictions (filter by sport, confidence) |
| POST | `/api/parlay` | Build optimized parlay (legs, strategy) |
| GET | `/api/parlays` | Multiple parlay options |
| GET | `/api/value-bets` | Find +EV value bets |
| GET | `/api/report` | Full daily text report |
| GET | `/api/sports` | List supported sports |
| GET | `/` | Web dashboard |

### Example: Build a 6-Leg Parlay
```bash
curl -X POST http://localhost:8000/api/parlay \
  -H "Content-Type: application/json" \
  -d '{"num_legs": 6, "strategy": "balanced"}'
```

### Example: Get Soccer Predictions
```bash
curl "http://localhost:8000/api/predictions?sport=football&min_confidence=70"
```

---

## How the Prediction Model Works

### 8-Factor Weighted Analysis

Each prediction is scored across 8 factors, with sport-specific weights:

1. **Form (20-30%)** — Recent W/D/L with recency weighting (last game > 5 games ago)
2. **Home Advantage (3-12%)** — Sport-specific home boost + team's actual home record
3. **Head-to-Head (8-18%)** — Historical matchup record (weighted heavily for tennis)
4. **League Position (10-18%)** — Table position and points gap
5. **Scoring Patterns (10-20%)** — Attack strength vs defense weakness matchup
6. **Injuries (10-15%)** — Missing players penalty (key player identification)
7. **Consistency (6-10%)** — Win rate stability
8. **Momentum (5-7%)** — Last 3-5 game hot/cold streak

### Parlay Optimization Strategies

- **🛡️ Safe** — Picks highest confidence legs (most likely to hit)
- **⚖️ Balanced** — Balances confidence with value (default)
- **💎 Value** — Maximizes expected value (higher odds, slightly lower confidence)

Anti-correlation: limits 2 picks per league to reduce correlated losses.

---

## What This System Does That Others Don't

1. **Real SofaScore data** — not sample data; live schedules, lineups, injuries, odds
2. **Multi-sport** — not just soccer; covers 8+ sports with tuned models per sport
3. **Value detection** — finds where bookmakers undervalue outcomes
4. **Parlay optimization** — doesn't just pick random games; optimizes for max probability while minimizing correlation
5. **Bankroll math** — Kelly Criterion prevents overbetting
6. **Accuracy tracking** — logs results to measure and improve over time
7. **AI layer** — optional GPT-4o review catches angles pure stats miss

---

## Things to Know

- **SofaScore data** is fetched from their public API endpoints. Responses are cached (5 min TTL) to be respectful of rate limits.
- **OpenAI API key** is optional. Without it, predictions use pure statistical analysis. With it, you get enhanced reasoning.
- **Accuracy depends on data quality** — top leagues (Premier League, NBA, etc.) have richer stats than lower divisions.
- **This is for informational/entertainment purposes.** Bet responsibly.

---

## Future Enhancements

- [ ] Weather data integration (outdoor sports)
- [ ] Line movement tracking (sharp money detection)
- [ ] Player prop predictions
- [ ] Live/in-play prediction updates
- [ ] Telegram/Discord bot notifications
- [ ] Machine learning model training on historical accuracy data
- [ ] Multi-bookmaker odds comparison