"""
Statistical Analysis Engine — Multi-sport statistical prediction.

Calculates win probabilities using weighted factors:
- Team form (recent performance)
- Home/Away advantage
- Head-to-head record
- League position & points
- Scoring patterns (goals/points)
- Injuries impact
- Historical over/under patterns
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from src.models import (
    Sport, BetType, MatchEvent, TeamStats, HeadToHead,
    Prediction, PlayerInfo,
)


@dataclass
class AnalysisWeights:
    """Configurable weights per sport for statistical model."""
    form: float = 0.25
    home_advantage: float = 0.10
    h2h: float = 0.10
    league_position: float = 0.15
    scoring: float = 0.15
    injuries: float = 0.10
    consistency: float = 0.10
    momentum: float = 0.05


# Sport-specific weight profiles
SPORT_WEIGHTS: dict[Sport, AnalysisWeights] = {
    Sport.SOCCER: AnalysisWeights(
        form=0.22, home_advantage=0.12, h2h=0.12, league_position=0.15,
        scoring=0.13, injuries=0.12, consistency=0.09, momentum=0.05,
    ),
    Sport.BASKETBALL: AnalysisWeights(
        form=0.25, home_advantage=0.08, h2h=0.08, league_position=0.18,
        scoring=0.18, injuries=0.12, consistency=0.06, momentum=0.05,
    ),
    Sport.TENNIS: AnalysisWeights(
        form=0.30, home_advantage=0.03, h2h=0.18, league_position=0.10,
        scoring=0.10, injuries=0.15, consistency=0.07, momentum=0.07,
    ),
    Sport.BASEBALL: AnalysisWeights(
        form=0.20, home_advantage=0.06, h2h=0.10, league_position=0.15,
        scoring=0.20, injuries=0.12, consistency=0.10, momentum=0.07,
    ),
    Sport.AMERICAN_FOOTBALL: AnalysisWeights(
        form=0.22, home_advantage=0.10, h2h=0.10, league_position=0.15,
        scoring=0.15, injuries=0.15, consistency=0.08, momentum=0.05,
    ),
    Sport.VOLLEYBALL: AnalysisWeights(
        form=0.25, home_advantage=0.08, h2h=0.12, league_position=0.15,
        scoring=0.15, injuries=0.10, consistency=0.08, momentum=0.07,
    ),
    Sport.HOCKEY: AnalysisWeights(
        form=0.22, home_advantage=0.08, h2h=0.10, league_position=0.15,
        scoring=0.18, injuries=0.12, consistency=0.08, momentum=0.07,
    ),
}

# Real star players by team for accurate player prop labels
_TEAM_KEY_PLAYERS: dict[str, dict[str, str]] = {
    # ── Soccer ─────────────────────────────────────────────────────────────────────
    "Arsenal":             {"scorer": "Bukayo Saka",         "shots": "Gabriel Martinelli",   "assist": "Martin Odegaard"},
    "Manchester City":     {"scorer": "Erling Haaland",      "shots": "Phil Foden",           "assist": "Kevin De Bruyne"},
    "Liverpool":           {"scorer": "Mohamed Salah",       "shots": "Luis Diaz",            "assist": "Trent Alexander-Arnold"},
    "Chelsea":             {"scorer": "Nicolas Jackson",     "shots": "Cole Palmer",          "assist": "Reece James"},
    "Manchester United":   {"scorer": "Rasmus Hojlund",      "shots": "Bruno Fernandes",      "assist": "Bruno Fernandes"},
    "Tottenham Hotspur":   {"scorer": "Son Heung-min",       "shots": "Son Heung-min",        "assist": "James Maddison"},
    "Tottenham":           {"scorer": "Son Heung-min",       "shots": "Son Heung-min",        "assist": "James Maddison"},
    "Newcastle United":    {"scorer": "Alexander Isak",      "shots": "Alexander Isak",       "assist": "Bruno Guimaraes"},
    "Aston Villa":         {"scorer": "Ollie Watkins",       "shots": "Leon Bailey",          "assist": "Emiliano Buendia"},
    "Brighton":            {"scorer": "Evan Ferguson",       "shots": "Kaoru Mitoma",         "assist": "Solly March"},
    "West Ham":            {"scorer": "Jarrod Bowen",        "shots": "Mohammed Kudus",       "assist": "Lucas Paqueta"},
    "Real Madrid":         {"scorer": "Vinicius Jr",         "shots": "Rodrygo",              "assist": "Jude Bellingham"},
    "Barcelona":           {"scorer": "Robert Lewandowski",  "shots": "Ferran Torres",        "assist": "Pedri"},
    "Atletico Madrid":     {"scorer": "Antoine Griezmann",   "shots": "Alvaro Morata",        "assist": "Koke"},
    "Atlético Madrid":     {"scorer": "Antoine Griezmann",   "shots": "Alvaro Morata",        "assist": "Koke"},
    "Real Sociedad":       {"scorer": "Mikel Oyarzabal",     "shots": "Takefusa Kubo",        "assist": "David Silva"},
    "Athletic Bilbao":     {"scorer": "Inaki Williams",      "shots": "Nico Williams",         "assist": "Oihan Sancet"},
    "Villarreal":          {"scorer": "Gerard Moreno",       "shots": "Arnaut Danjuma",       "assist": "Alex Baena"},
    "Real Betis":          {"scorer": "Borja Iglesias",      "shots": "Lo Celso",             "assist": "Isco"},
    "Sevilla":             {"scorer": "Lucas Ocampos",       "shots": "Ivan Rakitic",         "assist": "Jesus Navas"},
    "Inter Milan":         {"scorer": "Lautaro Martinez",    "shots": "Marcus Thuram",        "assist": "Nicolo Barella"},
    "AC Milan":            {"scorer": "Rafael Leao",         "shots": "Christian Pulisic",    "assist": "Theo Hernandez"},
    "Juventus":            {"scorer": "Dusan Vlahovic",      "shots": "Federico Chiesa",      "assist": "Adrien Rabiot"},
    "Napoli":              {"scorer": "Victor Osimhen",      "shots": "Khvicha Kvaratskhelia","assist": "Piotr Zielinski"},
    "Roma":                {"scorer": "Paulo Dybala",        "shots": "Lorenzo Pellegrini",   "assist": "Romelu Lukaku"},
    "Lazio":               {"scorer": "Ciro Immobile",       "shots": "Felipe Anderson",      "assist": "Luis Alberto"},
    "Atalanta":            {"scorer": "Ademola Lookman",     "shots": "Teun Koopmeiners",    "assist": "Gianluca Scamacca"},
    "Fiorentina":          {"scorer": "Jonathan Ikone",      "shots": "Riccardo Sottil",      "assist": "Rolando Mandragora"},
    "Bayern Munich":       {"scorer": "Harry Kane",          "shots": "Leroy Sane",           "assist": "Jamal Musiala"},
    "Borussia Dortmund":   {"scorer": "Niclas Fullkrug",     "shots": "Jamie Gittens",        "assist": "Julian Brandt"},
    "RB Leipzig":          {"scorer": "Lois Openda",         "shots": "Xavi Simons",          "assist": "Benjamin Sesko"},
    "Bayer Leverkusen":    {"scorer": "Victor Boniface",     "shots": "Florian Wirtz",        "assist": "Granit Xhaka"},
    "Eintracht Frankfurt": {"scorer": "Omar Marmoush",       "shots": "Hugo Larsson",         "assist": "Mario Gotze"},
    "Paris Saint-Germain": {"scorer": "Kylian Mbappe",       "shots": "Ousmane Dembele",      "assist": "Vitinha"},
    "Marseille":           {"scorer": "Pierre-Emerick Aubameyang", "shots": "Amine Harit",   "assist": "Azzedine Ounahi"},
    "Lyon":                {"scorer": "Alexandre Lacazette", "shots": "Rayan Cherki",         "assist": "Corentin Tolisso"},
    "Monaco":              {"scorer": "Wissam Ben Yedder",   "shots": "Takumi Minamino",      "assist": "Denis Zakaria"},
    "Galatasaray":         {"scorer": "Mauro Icardi",        "shots": "Dries Mertens",        "assist": "Lucas Torreira"},
    # ── NBA ─────────────────────────────────────────────────────────────────────
    "Boston Celtics":         {"scorer": "Jayson Tatum",            "reb": "Jaylen Brown",        "ast": "Jrue Holiday"},
    "Denver Nuggets":         {"scorer": "Nikola Jokic",            "reb": "Michael Porter Jr",   "ast": "Jamal Murray"},
    "LA Lakers":              {"scorer": "LeBron James",            "reb": "Anthony Davis",       "ast": "Austin Reaves"},
    "Los Angeles Lakers":     {"scorer": "LeBron James",            "reb": "Anthony Davis",       "ast": "Austin Reaves"},
    "Golden State Warriors":  {"scorer": "Stephen Curry",           "reb": "Draymond Green",      "ast": "Klay Thompson"},
    "Milwaukee Bucks":        {"scorer": "Giannis Antetokounmpo",   "reb": "Brook Lopez",         "ast": "Damian Lillard"},
    "Philadelphia 76ers":     {"scorer": "Joel Embiid",             "reb": "Paul Reed",           "ast": "Tyrese Maxey"},
    "Phoenix Suns":           {"scorer": "Kevin Durant",            "reb": "Devin Booker",        "ast": "Devin Booker"},
    "Dallas Mavericks":       {"scorer": "Luka Doncic",             "reb": "Derrick Jones Jr",    "ast": "Kyrie Irving"},
    "Oklahoma City Thunder":  {"scorer": "Shai Gilgeous-Alexander", "reb": "Chet Holmgren",       "ast": "Josh Giddey"},
    "Minnesota Timberwolves": {"scorer": "Anthony Edwards",         "reb": "Karl-Anthony Towns",  "ast": "Mike Conley"},
    "Cleveland Cavaliers":    {"scorer": "Donovan Mitchell",        "reb": "Evan Mobley",         "ast": "Darius Garland"},
    "New York Knicks":        {"scorer": "Jalen Brunson",           "reb": "Julius Randle",       "ast": "Josh Hart"},
    "Miami Heat":             {"scorer": "Jimmy Butler",            "reb": "Bam Adebayo",         "ast": "Tyler Herro"},
    "Houston Rockets":        {"scorer": "Jalen Green",             "reb": "Alperen Sengun",      "ast": "Fred VanVleet"},
    "Memphis Grizzlies":      {"scorer": "Ja Morant",               "reb": "Jaren Jackson Jr",    "ast": "Desmond Bane"},
    "Indiana Pacers":         {"scorer": "Tyrese Haliburton",       "reb": "Pascal Siakam",       "ast": "Myles Turner"},
    "Brooklyn Nets":          {"scorer": "Mikal Bridges",           "reb": "Nic Claxton",         "ast": "Ben Simmons"},
    "Chicago Bulls":          {"scorer": "DeMar DeRozan",           "reb": "Nikola Vucevic",      "ast": "Zach LaVine"},
    "Toronto Raptors":        {"scorer": "Scottie Barnes",          "reb": "Jakob Poeltl",        "ast": "OG Anunoby"},
    "Atlanta Hawks":          {"scorer": "Trae Young",              "reb": "Clint Capela",        "ast": "Dejounte Murray"},
    "New Orleans Pelicans":   {"scorer": "Zion Williamson",         "reb": "Jonas Valanciunas",   "ast": "CJ McCollum"},
    "Utah Jazz":              {"scorer": "Lauri Markkanen",         "reb": "Walker Kessler",      "ast": "Collin Sexton"},
    "LA Clippers":            {"scorer": "Kawhi Leonard",           "reb": "Paul George",         "ast": "James Harden"},
    "Portland Trail Blazers": {"scorer": "Anfernee Simons",         "reb": "Jusuf Nurkic",        "ast": "Jerami Grant"},
    # ── NFL ─────────────────────────────────────────────────────────────────────
    "Kansas City Chiefs":     {"qb": "Patrick Mahomes",      "rb": "Isiah Pacheco",        "wr": "Travis Kelce"},
    "San Francisco 49ers":    {"qb": "Brock Purdy",          "rb": "Christian McCaffrey",  "wr": "Deebo Samuel"},
    "Buffalo Bills":          {"qb": "Josh Allen",            "rb": "James Cook",           "wr": "Stefon Diggs"},
    "Dallas Cowboys":         {"qb": "Dak Prescott",          "rb": "Tony Pollard",         "wr": "CeeDee Lamb"},
    "Philadelphia Eagles":    {"qb": "Jalen Hurts",           "rb": "D'Andre Swift",        "wr": "A.J. Brown"},
    "Baltimore Ravens":       {"qb": "Lamar Jackson",         "rb": "Derrick Henry",        "wr": "Zay Flowers"},
    "Miami Dolphins":         {"qb": "Tua Tagovailoa",        "rb": "Raheem Mostert",       "wr": "Tyreek Hill"},
    "Detroit Lions":          {"qb": "Jared Goff",            "rb": "David Montgomery",     "wr": "Amon-Ra St. Brown"},
    "Cleveland Browns":       {"qb": "Deshaun Watson",        "rb": "Nick Chubb",           "wr": "Amari Cooper"},
    "Green Bay Packers":      {"qb": "Jordan Love",           "rb": "Aaron Jones",          "wr": "Christian Watson"},
    # ── MLB ─────────────────────────────────────────────────────────────────────
    "New York Yankees":       {"pitcher": "Gerrit Cole",        "batter": "Aaron Judge",          "sp": "Juan Soto"},
    "LA Dodgers":             {"pitcher": "Clayton Kershaw",    "batter": "Freddie Freeman",      "sp": "Mookie Betts"},
    "Los Angeles Dodgers":    {"pitcher": "Clayton Kershaw",    "batter": "Freddie Freeman",      "sp": "Mookie Betts"},
    "Houston Astros":         {"pitcher": "Justin Verlander",   "batter": "Jose Altuve",          "sp": "Alex Bregman"},
    "Atlanta Braves":         {"pitcher": "Spencer Strider",    "batter": "Ronald Acuna Jr",      "sp": "Matt Olson"},
    "Philadelphia Phillies":  {"pitcher": "Aaron Nola",         "batter": "Bryce Harper",         "sp": "Trea Turner"},
    "Texas Rangers":          {"pitcher": "Jacob deGrom",       "batter": "Corey Seager",         "sp": "Marcus Semien"},
    "Baltimore Orioles":      {"pitcher": "Dean Kremer",        "batter": "Adley Rutschman",      "sp": "Gunnar Henderson"},
    "Tampa Bay Rays":         {"pitcher": "Tyler Glasnow",      "batter": "Randy Arozarena",      "sp": "Yandy Diaz"},
    "Minnesota Twins":        {"pitcher": "Pablo Lopez",        "batter": "Carlos Correa",        "sp": "Jose Miranda"},
    "Washington Nationals":   {"pitcher": "Patrick Corbin",     "batter": "Lane Thomas",          "sp": "Luis Garcia"},
    "Miami Marlins":          {"pitcher": "Sandy Alcantara",    "batter": "Jazz Chisholm Jr",     "sp": "Luis Arraez"},
    "Pittsburgh Pirates":     {"pitcher": "Mitch Keller",       "batter": "Ke'Bryan Hayes",       "sp": "Bryan Reynolds"},
    "Detroit Tigers":         {"pitcher": "Matthew Boyd",       "batter": "Spencer Torkelson",    "sp": "Riley Greene"},
    "Kansas City Royals":     {"pitcher": "Brady Singer",       "batter": "Bobby Witt Jr",        "sp": "Salvador Perez"},
    # ── NHL ─────────────────────────────────────────────────────────────────────
    "Colorado Avalanche":  {"goal": "Nathan MacKinnon",  "shots": "Mikko Rantanen",    "saves": "Alexandar Georgiev"},
    "Dallas Stars":        {"goal": "Jason Robertson",   "shots": "Miro Heiskanen",    "saves": "Jake Oettinger"},
    "Boston Bruins":       {"goal": "David Pastrnak",    "shots": "Brad Marchand",     "saves": "Jeremy Swayman"},
    "Carolina Hurricanes": {"goal": "Sebastian Aho",     "shots": "Teuvo Teravainen",  "saves": "Antti Raanta"},
    "New York Rangers":    {"goal": "Artemi Panarin",    "shots": "Mika Zibanejad",    "saves": "Igor Shesterkin"},
    "New Jersey Devils":   {"goal": "Jack Hughes",       "shots": "Nico Hischier",     "saves": "Vitek Vanecek"},
    "Pittsburgh Penguins": {"goal": "Sidney Crosby",     "shots": "Evgeni Malkin",     "saves": "Tristan Jarry"},
    "Washington Capitals": {"goal": "Alex Ovechkin",     "shots": "Tom Wilson",        "saves": "Darcy Kuemper"},
    "Ottawa Senators":     {"goal": "Tim Stutzle",       "shots": "Brady Tkachuk",     "saves": "Cam Talbot"},
    "Calgary Flames":      {"goal": "Elias Lindholm",    "shots": "Nazem Kadri",       "saves": "Jacob Markstrom"},
    "Edmonton Oilers":     {"goal": "Connor McDavid",    "shots": "Leon Draisaitl",    "saves": "Stuart Skinner"},
    "Vancouver Canucks":   {"goal": "Elias Pettersson",  "shots": "Bo Horvat",         "saves": "Thatcher Demko"},
    "Toronto Maple Leafs": {"goal": "Auston Matthews",   "shots": "Mitch Marner",      "saves": "Ilya Samsonov"},
    "Tampa Bay Lightning": {"goal": "Nikita Kucherov",   "shots": "Brayden Point",     "saves": "Andrei Vasilevskiy"},
    "Florida Panthers":    {"goal": "Matthew Tkachuk",   "shots": "Aleksander Barkov", "saves": "Sergei Bobrovsky"},
    "Anaheim Ducks":       {"goal": "Trevor Zegras",     "shots": "Mason McTavish",    "saves": "John Gibson"},
    "Philadelphia Flyers": {"goal": "Sean Couturier",    "shots": "Travis Konecny",    "saves": "Carter Hart"},
    "St. Louis Blues":     {"goal": "David Perron",      "shots": "Jordan Kyrou",      "saves": "Jordan Binnington"},
}


class StatisticalAnalyzer:
    """Multi-sport statistical prediction engine."""

    def __init__(self):
        self.default_weights = AnalysisWeights()

    def _player(self, team: str, role: str, fallback: str = "") -> str:
        """Return a real star player name for a team/role, or fallback."""
        return _TEAM_KEY_PLAYERS.get(team, {}).get(role, fallback)

    def analyze_event(self, event: MatchEvent) -> dict:
        """Run full statistical analysis on an event. Returns factor scores."""
        sport = event.tournament.sport
        weights = SPORT_WEIGHTS.get(sport, self.default_weights)

        factors = {}

        # 1. Form Analysis
        factors["form"] = self._analyze_form(
            event.home_stats, event.away_stats, weights.form
        )

        # 2. Home Advantage
        factors["home_advantage"] = self._analyze_home_advantage(
            event.home_stats, event.away_stats, sport, weights.home_advantage
        )

        # 3. Head-to-head
        factors["h2h"] = self._analyze_h2h(event.h2h, weights.h2h)

        # 4. League Position
        factors["league_position"] = self._analyze_league_position(
            event.home_stats, event.away_stats, weights.league_position
        )

        # 5. Scoring Patterns
        factors["scoring"] = self._analyze_scoring(
            event.home_stats, event.away_stats, weights.scoring
        )

        # 6. Injuries Impact
        factors["injuries"] = self._analyze_injuries(
            event.home_injuries, event.away_injuries, weights.injuries
        )

        # 7. Consistency
        factors["consistency"] = self._analyze_consistency(
            event.home_stats, event.away_stats, weights.consistency
        )

        # 8. Momentum
        factors["momentum"] = self._analyze_momentum(
            event.home_stats, event.away_stats, weights.momentum
        )

        return factors

    def calculate_probabilities(
        self, event: MatchEvent
    ) -> dict[str, float]:
        """Calculate win/draw/loss probabilities."""
        factors = self.analyze_event(event)

        home_score = sum(f.get("home", 0) for f in factors.values())
        away_score = sum(f.get("away", 0) for f in factors.values())
        draw_score = sum(f.get("draw", 0) for f in factors.values())

        total = home_score + away_score + draw_score
        if total == 0:
            return {"home": 0.33, "draw": 0.33, "away": 0.33}

        sport = event.tournament.sport
        # Sports without draws
        if sport in {Sport.BASKETBALL, Sport.TENNIS, Sport.VOLLEYBALL, Sport.BASEBALL}:
            draw_score = 0
            total = home_score + away_score
            if total == 0:
                return {"home": 0.50, "draw": 0.0, "away": 0.50}
            return {
                "home": round(home_score / total, 4),
                "draw": 0.0,
                "away": round(away_score / total, 4),
            }

        return {
            "home": round(home_score / total, 4),
            "draw": round(draw_score / total, 4),
            "away": round(away_score / total, 4),
        }

    def calculate_over_under(
        self, event: MatchEvent, line: float = 2.5
    ) -> dict[str, float]:
        """Calculate over/under probability for a given line."""
        home_stats = event.home_stats
        away_stats = event.away_stats

        if not home_stats or not away_stats:
            return {"over": 0.50, "under": 0.50}

        # Expected total goals/points using Poisson-like approach
        home_expected = home_stats.avg_goals_scored
        away_expected = away_stats.avg_goals_scored
        total_expected = home_expected + away_expected

        # Adjust for defensive quality
        home_def_factor = away_stats.avg_goals_conceded / max(
            home_stats.avg_goals_conceded, 0.1
        )
        away_def_factor = home_stats.avg_goals_conceded / max(
            away_stats.avg_goals_conceded, 0.1
        )

        adjusted_total = total_expected * (
            (home_def_factor + away_def_factor) / 2
        )

        # Poisson CDF approximation for P(X > line)
        over_prob = 1.0 - self._poisson_cdf(int(line), adjusted_total)
        over_prob = max(0.05, min(0.95, over_prob))

        return {
            "over": round(over_prob, 4),
            "under": round(1 - over_prob, 4),
            "expected_total": round(adjusted_total, 2),
        }

    def calculate_btts(self, event: MatchEvent) -> dict[str, float]:
        """Calculate Both Teams to Score probability (soccer)."""
        home_stats = event.home_stats
        away_stats = event.away_stats

        if not home_stats or not away_stats:
            return {"yes": 0.50, "no": 0.50}

        # Scoring probability: team scores based on attack/defense matchup
        home_scoring_prob = min(
            0.95,
            (home_stats.avg_goals_scored * 0.6)
            + (1 - (away_stats.clean_sheets / max(away_stats.games_played, 1))) * 0.4,
        )
        away_scoring_prob = min(
            0.95,
            (away_stats.avg_goals_scored * 0.6)
            + (1 - (home_stats.clean_sheets / max(home_stats.games_played, 1))) * 0.4,
        )

        btts_prob = home_scoring_prob * away_scoring_prob
        btts_prob = max(0.10, min(0.90, btts_prob))

        return {"yes": round(btts_prob, 4), "no": round(1 - btts_prob, 4)}

    def _decimal_to_american(self, decimal_odds: float) -> str:
        """Convert decimal odds to American odds string."""
        if decimal_odds <= 1.0:
            return "+100"
        if decimal_odds >= 2.0:
            american = round((decimal_odds - 1) * 100)
            return f"+{american}"
        else:
            american = round(-100 / (decimal_odds - 1))
            return f"{american}"

    def generate_predictions(
        self, event: MatchEvent
    ) -> list[Prediction]:
        """Generate all applicable predictions for an event."""
        predictions = []
        sport = event.tournament.sport
        probs = self.calculate_probabilities(event)
        factors = self.analyze_event(event)
        home = event.home_team.name
        away = event.away_team.name
        kick_off = event.start_time.strftime("%b %d, %Y • %H:%M")

        # ── 1. MONEYLINE / WINNER ────────────────────────────────────────
        best_pick = max(probs, key=probs.get)
        pick_map = {"home": home, "away": away, "draw": "Draw"}
        odds_map = {"home": event.home_odds, "away": event.away_odds, "draw": event.draw_odds}

        confidence = probs[best_pick] * 100
        ml_odds = odds_map.get(best_pick, 0)

        ml_pred = Prediction(
            event=event,
            bet_type=BetType.MONEYLINE,
            pick=f"🏆 {pick_map[best_pick]} Win" if best_pick != "draw" else "🤝 Draw",
            confidence=round(confidence, 1),
            probability=probs[best_pick],
            odds=ml_odds,
            value_rating=self._calculate_value(probs[best_pick], ml_odds),
            reasoning=self._build_reasoning(event, factors, best_pick),
            factors=factors,
            american_odds=self._decimal_to_american(ml_odds) if ml_odds > 0 else "",
            market_display=f"Winner — {pick_map[best_pick]} ({self._decimal_to_american(ml_odds) if ml_odds > 0 else 'N/A'})",
            team_name=pick_map[best_pick] if best_pick != "draw" else "",
        )
        predictions.append(ml_pred)

        # ── 2. GAME RESULT 90 MIN + STOPPAGE (Soccer/Football) ──────────
        if sport in {Sport.SOCCER, Sport.AMERICAN_FOOTBALL}:
            for outcome, label, prob in [
                ("home", f"⚽ {home} Win (90'+ST)", probs["home"]),
                ("draw", "🤝 Draw (90'+ST) — Push if Tied", probs.get("draw", 0)),
                ("away", f"⚽ {away} Win (90'+ST)", probs["away"]),
            ]:
                if prob < 0.15:
                    continue
                game_odds = odds_map.get(outcome, 0)
                predictions.append(Prediction(
                    event=event,
                    bet_type=BetType.GAME_RESULT_90,
                    pick=label,
                    confidence=round(prob * 100, 1),
                    probability=prob,
                    odds=game_odds,
                    value_rating=self._calculate_value(prob, game_odds),
                    american_odds=self._decimal_to_american(game_odds) if game_odds > 0 else "",
                    market_display=f"Game Result (90 min + Stoppage Time)",
                    push_note="Push if tied" if outcome == "draw" else "",
                    team_name=pick_map.get(outcome, ""),
                    reasoning=f"Full-time result including stoppage time. {kick_off}",
                ))

        # ── 3. SPREAD / HANDICAP ────────────────────────────────────────
        if sport == Sport.SOCCER:
            spread_lines = [(-0.5, "+400"), (-1.5, "+180"), (-2.5, "+350"), (0.5, "-200"), (1.5, "-150")]
        elif sport == Sport.BASKETBALL:
            spread_lines = [(-3.5, "-110"), (-5.5, "+105"), (-7.5, "+130"), (3.5, "-110"), (5.5, "+105")]
        elif sport == Sport.AMERICAN_FOOTBALL:
            spread_lines = [(-3.0, "-110"), (-6.5, "+120"), (-10.5, "+180"), (3.0, "-110"), (6.5, "+120")]
        elif sport == Sport.BASEBALL:
            spread_lines = [(-1.5, "+140"), (1.5, "-165")]
        elif sport == Sport.VOLLEYBALL:
            spread_lines = [(-4.5, "-110"), (4.5, "-110"), (-8.5, "+130")]
        elif sport == Sport.HOCKEY:
            spread_lines = [(-1.5, "+165"), (1.5, "-190"), (-2.5, "+310")]
        else:
            spread_lines = [(-1.5, "+150"), (1.5, "-180")]

        home_strength = probs["home"]
        away_strength = probs["away"]

        for spread_val, example_odds in spread_lines:
            # Positive spread = underdog gets points, negative = favorite gives points
            if spread_val < 0:
                # Home team giving points
                adjusted_prob = max(0.05, home_strength - abs(spread_val) * 0.12)
                team = home
                pick_label = f"📊 {home} {spread_val}"
            else:
                # Away team getting points
                adjusted_prob = max(0.05, away_strength + spread_val * 0.08)
                team = away
                pick_label = f"📊 {away} +{abs(spread_val)}"

            spread_odds = self._american_to_decimal(example_odds)
            conf = adjusted_prob * 100
            if conf < 30:
                continue

            predictions.append(Prediction(
                event=event,
                bet_type=BetType.SPREAD,
                pick=pick_label,
                confidence=round(conf, 1),
                probability=adjusted_prob,
                odds=spread_odds,
                value_rating=self._calculate_value(adjusted_prob, spread_odds),
                line=spread_val,
                american_odds=example_odds,
                market_display=f"Spread {spread_val:+.1f} ({example_odds})",
                team_name=team,
                reasoning=f"Handicap line {spread_val:+.1f} — {team} at {example_odds}",
            ))

        # ── 4. OVER/UNDER (TOTAL GOALS/POINTS) ─────────────────────────
        if sport == Sport.SOCCER:
            lines = [1.5, 2.5, 3.5]
        elif sport == Sport.BASKETBALL:
            lines = [200.5, 210.5, 220.5, 230.5]
        elif sport == Sport.BASEBALL:
            lines = [6.5, 7.5, 8.5]
        elif sport == Sport.VOLLEYBALL:
            lines = [150.5, 170.5, 190.5]
        elif sport == Sport.AMERICAN_FOOTBALL:
            lines = [40.5, 45.5, 50.5]
        elif sport == Sport.HOCKEY:
            lines = [4.5, 5.5, 6.5]
        else:
            lines = [2.5]

        unit = "Goals" if sport in {Sport.SOCCER, Sport.HOCKEY} else "Points" if sport in {Sport.BASKETBALL, Sport.AMERICAN_FOOTBALL, Sport.VOLLEYBALL} else "Runs" if sport == Sport.BASEBALL else "Total"

        for line in lines:
            ou = self.calculate_over_under(event, line)
            best_ou = "over" if ou["over"] > ou["under"] else "under"
            ou_confidence = max(ou["over"], ou["under"]) * 100
            ou_prob = ou[best_ou]
            ou_decimal = 1 / max(ou_prob, 0.05) if ou_prob > 0 else 2.0
            ou_american = self._decimal_to_american(ou_decimal)

            if ou_confidence > 52:
                predictions.append(Prediction(
                    event=event,
                    bet_type=BetType.OVER_UNDER,
                    pick=f"{'⬆️ Over' if best_ou == 'over' else '⬇️ Under'} {line} {unit}",
                    confidence=round(ou_confidence, 1),
                    probability=ou_prob,
                    odds=round(ou_decimal, 2),
                    american_odds=ou_american,
                    line=line,
                    market_display=f"Total {unit} — {'Over' if best_ou == 'over' else 'Under'} {line} ({ou_american})",
                    reasoning=f"Expected total: {ou.get('expected_total', 'N/A')} {unit.lower()}. {kick_off}",
                    factors={"over_under": ou},
                ))

        # ── 5. TEAM-SPECIFIC TOTALS ─────────────────────────────────────
        if event.home_stats and event.away_stats:
            for team_stats, team_name, opp_stats, side in [
                (event.home_stats, home, event.away_stats, "home"),
                (event.away_stats, away, event.home_stats, "away"),
            ]:
                expected = team_stats.avg_goals_scored
                if sport == Sport.SOCCER:
                    team_lines = [0.5, 1.5, 2.5]
                elif sport == Sport.BASKETBALL:
                    team_lines = [95.5, 100.5, 105.5, 110.5]
                elif sport == Sport.BASEBALL:
                    team_lines = [2.5, 3.5, 4.5]
                elif sport == Sport.AMERICAN_FOOTBALL:
                    team_lines = [17.5, 20.5, 24.5]
                else:
                    team_lines = [0.5, 1.5]

                for tl in team_lines:
                    over_p = 1.0 - self._poisson_cdf(int(tl), expected)
                    over_p = max(0.05, min(0.95, over_p))
                    best_t = "over" if over_p > 0.5 else "under"
                    t_prob = over_p if best_t == "over" else (1 - over_p)
                    t_conf = t_prob * 100
                    t_odds_dec = 1 / max(t_prob, 0.05)
                    t_american = self._decimal_to_american(t_odds_dec)

                    if t_conf > 52:
                        predictions.append(Prediction(
                            event=event,
                            bet_type=BetType.TEAM_TOTAL,
                            pick=f"{'⬆️' if best_t == 'over' else '⬇️'} {team_name} {'Over' if best_t == 'over' else 'Under'} {tl} {unit}",
                            confidence=round(t_conf, 1),
                            probability=t_prob,
                            odds=round(t_odds_dec, 2),
                            american_odds=t_american,
                            line=tl,
                            market_display=f"{team_name} Total {unit} — {'Over' if best_t == 'over' else 'Under'} {tl} ({t_american})",
                            team_name=team_name,
                            reasoning=f"{team_name} avg {unit.lower()} scored: {expected:.1f}. Line: {tl}",
                        ))

        # ── 6. BTTS (Soccer only) ───────────────────────────────────────
        if sport == Sport.SOCCER:
            btts = self.calculate_btts(event)
            best_btts = "yes" if btts["yes"] > btts["no"] else "no"
            btts_conf = max(btts["yes"], btts["no"]) * 100
            btts_prob = btts[best_btts]
            btts_odds = 1 / max(btts_prob, 0.05)

            if btts_conf > 52:
                predictions.append(Prediction(
                    event=event,
                    bet_type=BetType.BOTH_TEAMS_SCORE,
                    pick=f"{'✅ BTTS Yes' if best_btts == 'yes' else '❌ BTTS No'} — {home} & {away}",
                    confidence=round(btts_conf, 1),
                    probability=btts_prob,
                    odds=round(btts_odds, 2),
                    american_odds=self._decimal_to_american(btts_odds),
                    market_display=f"Both Teams to Score — {'Yes' if best_btts == 'yes' else 'No'} ({self._decimal_to_american(btts_odds)})",
                    reasoning=f"{home} attack vs {away} defense & vice versa. {kick_off}",
                    factors={"btts": btts},
                ))

        # ── 7. DOUBLE CHANCE (Soccer) ───────────────────────────────────
        if sport == Sport.SOCCER:
            dc_options = {
                "1X": probs["home"] + probs["draw"],
                "X2": probs["draw"] + probs["away"],
                "12": probs["home"] + probs["away"],
            }
            dc_labels = {
                "1X": f"🛡️ {home} or Draw",
                "X2": f"🛡️ {away} or Draw",
                "12": f"🛡️ {home} or {away} (No Draw)",
            }
            for dc_key, dc_prob in dc_options.items():
                dc_conf = dc_prob * 100
                if dc_conf > 60:
                    dc_odds = 1 / max(dc_prob, 0.05)
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.DOUBLE_CHANCE,
                        pick=dc_labels[dc_key],
                        confidence=round(dc_conf, 1),
                        probability=dc_prob,
                        odds=round(dc_odds, 2),
                        american_odds=self._decimal_to_american(dc_odds),
                        market_display=f"Double Chance — {dc_key} ({self._decimal_to_american(dc_odds)})",
                        push_note="Push if tied" if dc_key == "12" else "",
                        reasoning=f"Double chance {dc_key}: combined probability of two outcomes. {kick_off}",
                    ))

        # ── 8. CORNERS (Soccer only) ────────────────────────────────────
        if sport == Sport.SOCCER and event.home_stats and event.away_stats:
            home_corners = event.home_stats.corners_avg
            away_corners = event.away_stats.corners_avg
            expected_corners = home_corners + away_corners
            for corner_line in [8.5, 9.5, 10.5, 11.5]:
                c_over = 1.0 - self._poisson_cdf(int(corner_line), expected_corners)
                c_over = max(0.05, min(0.95, c_over))
                best_c = "over" if c_over > 0.5 else "under"
                c_prob = c_over if best_c == "over" else 1 - c_over
                c_conf = c_prob * 100
                c_odds = 1 / max(c_prob, 0.05)
                if c_conf > 52:
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.CORNERS,
                        pick=f"{'⬆️' if best_c == 'over' else '⬇️'} {'Over' if best_c == 'over' else 'Under'} {corner_line} Corners",
                        confidence=round(c_conf, 1),
                        probability=c_prob,
                        odds=round(c_odds, 2),
                        american_odds=self._decimal_to_american(c_odds),
                        line=corner_line,
                        market_display=f"Total Corners — {'Over' if best_c == 'over' else 'Under'} {corner_line} ({self._decimal_to_american(c_odds)})",
                        reasoning=f"Expected corners: {expected_corners:.1f}. {home}: {home_corners:.1f} avg, {away}: {away_corners:.1f} avg. {kick_off}",
                    ))

        # ── 9. HALFTIME RESULT ──────────────────────────────────────────
        if sport == Sport.SOCCER:
            ht_home = probs["home"] * 0.85
            ht_draw = probs.get("draw", 0.33) * 1.3
            ht_away = probs["away"] * 0.80
            ht_total = ht_home + ht_draw + ht_away
            ht_home /= ht_total
            ht_draw /= ht_total
            ht_away /= ht_total

            for ht_outcome, ht_label, ht_prob in [
                ("home", f"⏱️ {home} Leads at HT", ht_home),
                ("draw", "⏱️ Draw at Half-Time", ht_draw),
                ("away", f"⏱️ {away} Leads at HT", ht_away),
            ]:
                ht_conf = ht_prob * 100
                if ht_conf > 30:
                    ht_odds = 1 / max(ht_prob, 0.05)
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.HALFTIME_RESULT,
                        pick=ht_label,
                        confidence=round(ht_conf, 1),
                        probability=ht_prob,
                        odds=round(ht_odds, 2),
                        american_odds=self._decimal_to_american(ht_odds),
                        market_display=f"Half-Time Result — {ht_label.split(' ', 1)[1]} ({self._decimal_to_american(ht_odds)})",
                        reasoning=f"First half prediction. {kick_off}",
                    ))

        # ── 10. HALFTIME OVER/UNDER ─────────────────────────────────────
        if sport == Sport.SOCCER and event.home_stats and event.away_stats:
            ht_expected = (event.home_stats.avg_goals_scored + event.away_stats.avg_goals_scored) * 0.42
            for ht_line in [0.5, 1.5]:
                ht_over = 1.0 - self._poisson_cdf(int(ht_line), ht_expected)
                ht_over = max(0.05, min(0.95, ht_over))
                ht_best = "over" if ht_over > 0.5 else "under"
                ht_p = ht_over if ht_best == "over" else 1 - ht_over
                ht_c = ht_p * 100
                ht_o = 1 / max(ht_p, 0.05)
                if ht_c > 52:
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.HALFTIME_OVER_UNDER,
                        pick=f"{'⬆️' if ht_best == 'over' else '⬇️'} 1st Half {'Over' if ht_best == 'over' else 'Under'} {ht_line} Goals",
                        confidence=round(ht_c, 1),
                        probability=ht_p,
                        odds=round(ht_o, 2),
                        american_odds=self._decimal_to_american(ht_o),
                        line=ht_line,
                        market_display=f"1st Half Total — {'Over' if ht_best == 'over' else 'Under'} {ht_line} ({self._decimal_to_american(ht_o)})",
                        reasoning=f"Expected 1st half goals: {ht_expected:.1f}. {kick_off}",
                    ))

        # ── 11. PLAYER PROPS (with real star player names) ─────────────
        if event.home_stats and event.away_stats:
            if sport == Sport.SOCCER:
                props = [
                    (f"🎯 {self._player(home, 'scorer', home + ' Top Scorer')} — Anytime Goalscorer", 0.35),
                    (f"🎯 {self._player(away, 'scorer', away + ' Top Scorer')} — Anytime Goalscorer", 0.30),
                    (f"🎯 {self._player(home, 'shots', home + ' Key Player')} — 1+ Shots on Target", 0.65),
                    (f"🎯 {self._player(away, 'shots', away + ' Key Player')} — 1+ Shots on Target", 0.60),
                    (f"🎯 {self._player(home, 'assist', home + ' Playmaker')} — 1+ Assists", 0.28),
                    (f"🎯 {self._player(away, 'assist', away + ' Playmaker')} — 1+ Assists", 0.25),
                ]
            elif sport == Sport.BASKETBALL:
                props = [
                    (f"🎯 {self._player(home, 'scorer', home + ' Star')} — Over 20.5 Points", 0.55),
                    (f"🎯 {self._player(away, 'scorer', away + ' Star')} — Over 18.5 Points", 0.50),
                    (f"🎯 {self._player(home, 'reb', home + ' Big Man')} — Over 5.5 Rebounds", 0.52),
                    (f"🎯 {self._player(away, 'reb', away + ' Big Man')} — Over 5.5 Rebounds", 0.48),
                    (f"🎯 {self._player(home, 'ast', home + ' PG')} — Over 4.5 Assists", 0.50),
                    (f"🎯 {self._player(away, 'ast', away + ' PG')} — Over 4.5 Assists", 0.47),
                    (f"🎯 {self._player(home, 'scorer', home + ' Star')} — Over 2.5 Three-Pointers Made", 0.42),
                    (f"🎯 {self._player(away, 'scorer', away + ' Star')} — Over 2.5 Three-Pointers Made", 0.40),
                    (f"🎯 {self._player(home, 'scorer', home + ' Star')} — Over 30.5 PRA (Pts+Reb+Ast)", 0.48),
                    (f"🎯 {self._player(away, 'scorer', away + ' Star')} — Over 28.5 PRA (Pts+Reb+Ast)", 0.45),
                    (f"🎯 {self._player(home, 'reb', home + ' Big Man')} — Double-Double", 0.35),
                    (f"🎯 {self._player(home, 'scorer', home + ' Star')} — First Basket Scorer", 0.18),
                ]
            elif sport == Sport.BASEBALL:
                props = [
                    (f"🎯 {self._player(home, 'pitcher', home + ' Pitcher')} — 5+ Strikeouts", 0.50),
                    (f"🎯 {self._player(away, 'pitcher', away + ' Pitcher')} — 5+ Strikeouts", 0.47),
                    (f"🎯 {self._player(home, 'batter', home + ' Batter')} — 1+ Hits", 0.65),
                    (f"🎯 {self._player(away, 'batter', away + ' Batter')} — 1+ Hits", 0.62),
                    (f"🎯 {self._player(home, 'sp', home + ' Power Hitter')} — 1+ Home Runs", 0.22),
                    (f"🎯 {self._player(away, 'sp', away + ' Power Hitter')} — 1+ Home Runs", 0.20),
                    (f"🎯 {self._player(home, 'batter', home + ' Batter')} — Over 1.5 Total Bases", 0.45),
                    (f"🎯 {self._player(away, 'batter', away + ' Batter')} — Over 1.5 Total Bases", 0.43),
                    (f"🎯 {self._player(home, 'pitcher', home + ' Pitcher')} — Over 3.5 Outs Recorded", 0.58),
                    (f"🎯 {self._player(home, 'batter', home + ' Batter')} — 1+ RBIs", 0.38),
                ]
            elif sport == Sport.AMERICAN_FOOTBALL:
                props = [
                    (f"🎯 {self._player(home, 'qb', home + ' QB')} — Over 220.5 Pass Yards", 0.50),
                    (f"🎯 {self._player(away, 'qb', away + ' QB')} — Over 220.5 Pass Yards", 0.48),
                    (f"🎯 {self._player(home, 'rb', home + ' RB')} — Over 60.5 Rush Yards", 0.48),
                    (f"🎯 {self._player(away, 'rb', away + ' RB')} — Over 60.5 Rush Yards", 0.45),
                    (f"🎯 {self._player(home, 'wr', home + ' WR')} — Over 55.5 Receiving Yards", 0.47),
                    (f"🎯 {self._player(away, 'wr', away + ' WR')} — Over 55.5 Receiving Yards", 0.44),
                    (f"🎯 {self._player(home, 'qb', home + ' QB')} — Over 1.5 Passing TDs", 0.48),
                    (f"🎯 {self._player(away, 'qb', away + ' QB')} — Over 1.5 Passing TDs", 0.45),
                    (f"🎯 {self._player(home, 'wr', home + ' WR')} — Anytime TD Scorer", 0.40),
                    (f"🎯 {self._player(away, 'wr', away + ' WR')} — Anytime TD Scorer", 0.38),
                    (f"🎯 {self._player(home, 'wr', home + ' WR')} — 4+ Receptions", 0.50),
                ]
            elif sport == Sport.HOCKEY:
                props = [
                    (f"🎯 {self._player(home, 'goal', home + ' Star')} — Anytime Goal Scorer", 0.28),
                    (f"🎯 {self._player(away, 'goal', away + ' Star')} — Anytime Goal Scorer", 0.25),
                    (f"🎯 {self._player(home, 'shots', home + ' Forward')} — Over 2.5 Shots on Goal", 0.50),
                    (f"🎯 {self._player(away, 'shots', away + ' Forward')} — Over 2.5 Shots on Goal", 0.47),
                    (f"🎯 {self._player(home, 'goal', home + ' Star')} — 1+ Points (G+A)", 0.42),
                    (f"🎯 {self._player(away, 'goal', away + ' Star')} — 1+ Points (G+A)", 0.40),
                    (f"🎯 {self._player(home, 'saves', home + ' Goalie')} — Over 24.5 Saves", 0.48),
                    (f"🎯 {self._player(away, 'saves', away + ' Goalie')} — Over 24.5 Saves", 0.50),
                ]
            elif sport == Sport.TENNIS:
                props = [
                    (f"🎯 {home} — Over 5.5 Aces", 0.45),
                    (f"🎯 {away} — Over 3.5 Aces", 0.48),
                    (f"🎯 Match — Over 2.5 Tiebreaks", 0.25),
                ]
            else:
                props = []

            for prop_label, prop_prob in props:
                prop_conf = prop_prob * 100
                prop_odds = 1 / max(prop_prob, 0.05)
                if prop_conf > 15:
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.PLAYER_PROPS,
                        pick=prop_label,
                        confidence=round(prop_conf, 1),
                        probability=prop_prob,
                        odds=round(prop_odds, 2),
                        american_odds=self._decimal_to_american(prop_odds),
                        market_display=f"Player Props — {prop_label.split(' — ', 1)[1] if ' — ' in prop_label else prop_label}",
                        reasoning=f"Player prop based on season averages and matchup. {kick_off}",
                    ))

        # ── 12. DRAW NO BET (Soccer) ────────────────────────────────────
        if sport == Sport.SOCCER:
            # DNB: Bet refunded if draw. Only home win or away win.
            dnb_home = probs["home"] / (probs["home"] + probs["away"]) if (probs["home"] + probs["away"]) > 0 else 0.5
            dnb_away = 1 - dnb_home
            for dnb_pick, dnb_prob, dnb_team in [
                (f"🔄 {home} (Draw No Bet)", dnb_home, home),
                (f"🔄 {away} (Draw No Bet)", dnb_away, away),
            ]:
                dnb_conf = dnb_prob * 100
                if dnb_conf > 40:
                    dnb_odds = 1 / max(dnb_prob, 0.05)
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.DRAW_NO_BET,
                        pick=dnb_pick,
                        confidence=round(dnb_conf, 1),
                        probability=dnb_prob,
                        odds=round(dnb_odds, 2),
                        american_odds=self._decimal_to_american(dnb_odds),
                        market_display=f"Draw No Bet — {dnb_team} ({self._decimal_to_american(dnb_odds)})",
                        team_name=dnb_team,
                        push_note="Refund on draw",
                        reasoning=f"Bet voided if match ends in draw. {kick_off}",
                    ))

        # ── 13. CORRECT SCORE (Soccer — Top likely scores) ──────────────
        if sport == Sport.SOCCER and event.home_stats and event.away_stats:
            h_exp = event.home_stats.avg_goals_scored
            a_exp = event.away_stats.avg_goals_scored
            score_combos = [(1, 0), (0, 0), (1, 1), (2, 1), (2, 0), (0, 1), (1, 2), (0, 2), (3, 1), (2, 2)]
            for hg, ag in score_combos:
                # Poisson probability for each team's goals
                h_prob = math.exp(-h_exp) * (h_exp ** hg) / math.factorial(hg)
                a_prob = math.exp(-a_exp) * (a_exp ** ag) / math.factorial(ag)
                cs_prob = h_prob * a_prob
                cs_conf = cs_prob * 100
                if cs_conf > 3:
                    cs_odds = 1 / max(cs_prob, 0.01)
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.CORRECT_SCORE,
                        pick=f"🎯 {home} {hg} - {ag} {away}",
                        confidence=round(cs_conf, 1),
                        probability=cs_prob,
                        odds=round(cs_odds, 2),
                        american_odds=self._decimal_to_american(cs_odds),
                        market_display=f"Correct Score — {hg}-{ag} ({self._decimal_to_american(cs_odds)})",
                        reasoning=f"Based on avg scoring: {home} {h_exp:.1f} / {away} {a_exp:.1f} goals. {kick_off}",
                    ))

        # ── 14. GAME PROPS — First to Score / Overtime / Odd-Even ──────
        # First to Score
        if event.home_stats and event.away_stats:
            h_attack = event.home_stats.avg_goals_scored
            a_attack = event.away_stats.avg_goals_scored
            total_attack = h_attack + a_attack + 0.001
            fts_home_prob = (h_attack / total_attack) * 0.9 + 0.05  # home bias
            fts_away_prob = 1 - fts_home_prob

            for fts_label, fts_prob, fts_team in [
                (f"⚡ {home} Scores First", fts_home_prob, home),
                (f"⚡ {away} Scores First", fts_away_prob, away),
            ]:
                fts_conf = fts_prob * 100
                if fts_conf > 30:
                    fts_odds = 1 / max(fts_prob, 0.05)
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.FIRST_TO_SCORE,
                        pick=fts_label,
                        confidence=round(fts_conf, 1),
                        probability=fts_prob,
                        odds=round(fts_odds, 2),
                        american_odds=self._decimal_to_american(fts_odds),
                        market_display=f"First to Score — {fts_team} ({self._decimal_to_american(fts_odds)})",
                        team_name=fts_team,
                        reasoning=f"Based on offensive output: {home} {h_attack:.1f} avg vs {away} {a_attack:.1f} avg. {kick_off}",
                    ))

        # Overtime / Extra Innings / Extra Time
        if event.home_stats and event.away_stats:
            if sport == Sport.SOCCER:
                ot_prob = probs.get("draw", 0.25) * 0.4  # ~10% of games go to OT in knockout
                ot_label = "Extra Time"
            elif sport == Sport.BASKETBALL:
                # NBA OT ~6% of games
                strength_diff = abs(probs["home"] - probs["away"])
                ot_prob = max(0.03, 0.08 - strength_diff * 0.1)
                ot_label = "Overtime"
            elif sport == Sport.BASEBALL:
                strength_diff = abs(probs["home"] - probs["away"])
                ot_prob = max(0.05, 0.10 - strength_diff * 0.08)
                ot_label = "Extra Innings"
            elif sport == Sport.HOCKEY:
                strength_diff = abs(probs["home"] - probs["away"])
                ot_prob = max(0.10, 0.25 - strength_diff * 0.15)
                ot_label = "Overtime"
            elif sport == Sport.AMERICAN_FOOTBALL:
                strength_diff = abs(probs["home"] - probs["away"])
                ot_prob = max(0.02, 0.06 - strength_diff * 0.08)
                ot_label = "Overtime"
            else:
                ot_prob = 0.0
                ot_label = "Overtime"

            if ot_prob > 0.02:
                # Yes OT
                ot_yes_odds = 1 / max(ot_prob, 0.02)
                ot_no_odds = 1 / max(1 - ot_prob, 0.05)
                best_ot = "No" if (1 - ot_prob) > ot_prob else "Yes"
                best_ot_prob = max(ot_prob, 1 - ot_prob)
                best_ot_conf = best_ot_prob * 100
                best_ot_odds = ot_no_odds if best_ot == "No" else ot_yes_odds

                predictions.append(Prediction(
                    event=event,
                    bet_type=BetType.OVERTIME,
                    pick=f"⏰ {ot_label} — {'Yes' if best_ot == 'Yes' else 'No'}",
                    confidence=round(best_ot_conf, 1),
                    probability=best_ot_prob,
                    odds=round(best_ot_odds, 2),
                    american_odds=self._decimal_to_american(best_ot_odds),
                    market_display=f"Will There Be {ot_label}? — {best_ot} ({self._decimal_to_american(best_ot_odds)})",
                    reasoning=f"Based on team strength differential. {kick_off}",
                ))
                # Also add the long-shot side if odds are interesting
                longshot_ot = "Yes" if best_ot == "No" else "No"
                ls_prob = ot_prob if longshot_ot == "Yes" else (1 - ot_prob)
                ls_odds = ot_yes_odds if longshot_ot == "Yes" else ot_no_odds
                if ls_prob > 0.03:
                    predictions.append(Prediction(
                        event=event,
                        bet_type=BetType.OVERTIME,
                        pick=f"⏰ {ot_label} — {longshot_ot}",
                        confidence=round(ls_prob * 100, 1),
                        probability=ls_prob,
                        odds=round(ls_odds, 2),
                        american_odds=self._decimal_to_american(ls_odds),
                        market_display=f"Will There Be {ot_label}? — {longshot_ot} ({self._decimal_to_american(ls_odds)})",
                        reasoning=f"Game prop: {ot_label} probability based on matchup. {kick_off}",
                    ))

        # Odd/Even Total
        if event.home_stats and event.away_stats:
            # Roughly 50/50 but slightly sport-dependent
            if sport in {Sport.BASKETBALL, Sport.AMERICAN_FOOTBALL}:
                odd_prob = 0.50  # High-scoring → nearly coin flip
            elif sport in {Sport.SOCCER, Sport.HOCKEY}:
                total_exp = event.home_stats.avg_goals_scored + event.away_stats.avg_goals_scored
                # Odd is slightly more likely in low-scoring games
                odd_prob = 0.52 if total_exp < 3 else 0.49
            else:
                odd_prob = 0.50

            even_prob = 1 - odd_prob
            for oe_label, oe_prob in [
                ("🔢 Total Points/Goals — Odd", odd_prob),
                ("🔢 Total Points/Goals — Even", even_prob),
            ]:
                oe_conf = oe_prob * 100
                oe_odds = 1 / max(oe_prob, 0.05)
                predictions.append(Prediction(
                    event=event,
                    bet_type=BetType.ODD_EVEN,
                    pick=oe_label,
                    confidence=round(oe_conf, 1),
                    probability=oe_prob,
                    odds=round(oe_odds, 2),
                    american_odds=self._decimal_to_american(oe_odds),
                    market_display=f"Odd/Even Total — {'Odd' if 'Odd' in oe_label else 'Even'} ({self._decimal_to_american(oe_odds)})",
                    reasoning=f"Total odd/even prop. {kick_off}",
                ))

        # ── 15. ALTERNATE SPREADS ──────────────────────────────────────
        if sport == Sport.BASKETBALL:
            alt_spreads = [(-1.5, "-165"), (-2.5, "-140"), (-4.5, "-105"), (-6.5, "+110"),
                           (-8.5, "+140"), (-10.5, "+180"), (-12.5, "+230"),
                           (1.5, "+140"), (2.5, "+120"), (4.5, "-105"),
                           (6.5, "-130"), (8.5, "-160"), (10.5, "-200")]
        elif sport == Sport.AMERICAN_FOOTBALL:
            alt_spreads = [(-1.5, "-155"), (-2.5, "-130"), (-4.5, "+100"), (-7.5, "+145"),
                           (-10.5, "+180"), (-13.5, "+240"), (-17.5, "+360"),
                           (1.5, "+130"), (2.5, "+110"), (4.5, "-120"),
                           (7.5, "-170"), (10.5, "-210"), (13.5, "-280")]
        elif sport == Sport.HOCKEY:
            alt_spreads = [(-0.5, "-200"), (-1.5, "+165"), (-2.5, "+310"),
                           (0.5, "+170"), (1.5, "-190"), (2.5, "-350")]
        elif sport == Sport.BASEBALL:
            alt_spreads = [(-0.5, "-165"), (-1.5, "+140"), (-2.5, "+300"),
                           (0.5, "+140"), (1.5, "-165"), (2.5, "-350")]
        else:
            alt_spreads = []

        for aspread, a_odds_str in alt_spreads:
            if aspread < 0:
                a_prob = max(0.05, home_strength - abs(aspread) * 0.08)
                a_team = home
                a_label = f"📊 {home} {aspread}"
            else:
                a_prob = max(0.05, away_strength + aspread * 0.06)
                a_team = away
                a_label = f"📊 {away} +{abs(aspread)}"

            a_conf = a_prob * 100
            if 10 < a_conf < 95:
                a_dec = self._american_to_decimal(a_odds_str)
                predictions.append(Prediction(
                    event=event,
                    bet_type=BetType.ALTERNATE_SPREAD,
                    pick=a_label,
                    confidence=round(a_conf, 1),
                    probability=a_prob,
                    odds=a_dec,
                    value_rating=self._calculate_value(a_prob, a_dec),
                    line=aspread,
                    american_odds=a_odds_str,
                    market_display=f"Alt Spread {aspread:+.1f} ({a_odds_str})",
                    team_name=a_team,
                    reasoning=f"Alternate spread line {aspread:+.1f} — {a_team} at {a_odds_str}",
                ))

        # ── 16. ALTERNATE TOTALS ───────────────────────────────────────
        if sport == Sport.BASKETBALL:
            alt_totals = [195.5, 200.5, 205.5, 215.5, 225.5, 235.5, 240.5]
        elif sport == Sport.AMERICAN_FOOTBALL:
            alt_totals = [35.5, 38.5, 42.5, 48.5, 52.5, 55.5]
        elif sport == Sport.HOCKEY:
            alt_totals = [3.5, 4.5, 5.5, 6.5, 7.5]
        elif sport == Sport.BASEBALL:
            alt_totals = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
        elif sport == Sport.SOCCER:
            alt_totals = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            alt_totals = []

        for atl in alt_totals:
            a_ou = self.calculate_over_under(event, atl)
            a_best = "over" if a_ou["over"] > a_ou["under"] else "under"
            a_ou_prob = a_ou[a_best]
            a_ou_conf = a_ou_prob * 100
            if 10 < a_ou_conf < 95:
                a_ou_dec = 1 / max(a_ou_prob, 0.05)
                a_ou_am = self._decimal_to_american(a_ou_dec)
                predictions.append(Prediction(
                    event=event,
                    bet_type=BetType.ALTERNATE_TOTAL,
                    pick=f"{'⬆️ Over' if a_best == 'over' else '⬇️ Under'} {atl} {unit}",
                    confidence=round(a_ou_conf, 1),
                    probability=a_ou_prob,
                    odds=round(a_ou_dec, 2),
                    american_odds=a_ou_am,
                    line=atl,
                    market_display=f"Alt Total — {'Over' if a_best == 'over' else 'Under'} {atl} ({a_ou_am})",
                    reasoning=f"Alternate total line. Expected: {a_ou.get('expected_total', 'N/A')} {unit.lower()}. {kick_off}",
                    factors={"over_under": a_ou},
                ))

        # ── 17. QUARTER / PERIOD PROPS (NBA, NFL, NHL) ─────────────────
        if sport == Sport.BASKETBALL and event.home_stats and event.away_stats:
            # NBA: ~25% of total per quarter on avg
            total_exp = event.home_stats.avg_goals_scored + event.away_stats.avg_goals_scored
            q_exp = total_exp * 0.26  # Slightly over 25% (1st/3rd Q score higher)
            for q_num in [1, 2, 3, 4]:
                q_factor = {1: 1.02, 2: 0.98, 3: 1.00, 4: 1.00}[q_num]
                q_expected = q_exp * q_factor
                for q_line in [50.5, 52.5, 55.5]:
                    q_over = 1.0 - self._poisson_cdf(int(q_line), q_expected)
                    q_over = max(0.05, min(0.95, q_over))
                    q_best = "over" if q_over > 0.5 else "under"
                    q_prob = q_over if q_best == "over" else 1 - q_over
                    q_conf = q_prob * 100
                    if q_conf > 52:
                        q_odds_d = 1 / max(q_prob, 0.05)
                        predictions.append(Prediction(
                            event=event,
                            bet_type=BetType.QUARTER_PROPS,
                            pick=f"{'⬆️' if q_best == 'over' else '⬇️'} Q{q_num} {'Over' if q_best == 'over' else 'Under'} {q_line} Points",
                            confidence=round(q_conf, 1),
                            probability=q_prob,
                            odds=round(q_odds_d, 2),
                            american_odds=self._decimal_to_american(q_odds_d),
                            line=q_line,
                            market_display=f"Quarter {q_num} Total — {'Over' if q_best == 'over' else 'Under'} {q_line} ({self._decimal_to_american(q_odds_d)})",
                            reasoning=f"Q{q_num} expected total: {q_expected:.1f}. {kick_off}",
                        ))

        elif sport == Sport.AMERICAN_FOOTBALL and event.home_stats and event.away_stats:
            total_exp = event.home_stats.avg_goals_scored + event.away_stats.avg_goals_scored
            for half in [1, 2]:
                h_exp_val = total_exp * (0.48 if half == 1 else 0.52)
                for h_line in [17.5, 20.5, 24.5]:
                    h_over = 1.0 - self._poisson_cdf(int(h_line), h_exp_val)
                    h_over = max(0.05, min(0.95, h_over))
                    h_best = "over" if h_over > 0.5 else "under"
                    h_prob_v = h_over if h_best == "over" else 1 - h_over
                    h_conf = h_prob_v * 100
                    if h_conf > 52:
                        h_odds_d = 1 / max(h_prob_v, 0.05)
                        predictions.append(Prediction(
                            event=event,
                            bet_type=BetType.QUARTER_PROPS,
                            pick=f"{'⬆️' if h_best == 'over' else '⬇️'} {'1st' if half == 1 else '2nd'} Half {'Over' if h_best == 'over' else 'Under'} {h_line} Points",
                            confidence=round(h_conf, 1),
                            probability=h_prob_v,
                            odds=round(h_odds_d, 2),
                            american_odds=self._decimal_to_american(h_odds_d),
                            line=h_line,
                            market_display=f"{'1st' if half == 1 else '2nd'} Half Total — {'Over' if h_best == 'over' else 'Under'} {h_line} ({self._decimal_to_american(h_odds_d)})",
                            reasoning=f"{'First' if half == 1 else 'Second'} half expected: {h_exp_val:.1f}. {kick_off}",
                        ))

        elif sport == Sport.HOCKEY and event.home_stats and event.away_stats:
            total_exp = event.home_stats.avg_goals_scored + event.away_stats.avg_goals_scored
            for period in [1, 2, 3]:
                p_exp = total_exp * 0.34  # Each period ~33%
                for p_line in [1.5, 2.5]:
                    p_over = 1.0 - self._poisson_cdf(int(p_line), p_exp)
                    p_over = max(0.05, min(0.95, p_over))
                    p_best = "over" if p_over > 0.5 else "under"
                    p_prob_v = p_over if p_best == "over" else 1 - p_over
                    p_conf = p_prob_v * 100
                    if p_conf > 52:
                        p_odds_d = 1 / max(p_prob_v, 0.05)
                        predictions.append(Prediction(
                            event=event,
                            bet_type=BetType.QUARTER_PROPS,
                            pick=f"{'⬆️' if p_best == 'over' else '⬇️'} Period {period} {'Over' if p_best == 'over' else 'Under'} {p_line} Goals",
                            confidence=round(p_conf, 1),
                            probability=p_prob_v,
                            odds=round(p_odds_d, 2),
                            american_odds=self._decimal_to_american(p_odds_d),
                            line=p_line,
                            market_display=f"Period {period} Total — {'Over' if p_best == 'over' else 'Under'} {p_line} ({self._decimal_to_american(p_odds_d)})",
                            reasoning=f"Period {period} expected goals: {p_exp:.1f}. {kick_off}",
                        ))

        # ── 18. RACE TO X POINTS ──────────────────────────────────────
        if sport == Sport.BASKETBALL and event.home_stats and event.away_stats:
            for race_target in [20, 25]:
                # Team that scores faster likely gets there first
                h_pace = event.home_stats.avg_goals_scored
                a_pace = event.away_stats.avg_goals_scored
                pace_total = h_pace + a_pace + 0.001
                race_h_prob = h_pace / pace_total * 0.95 + 0.025
                race_a_prob = 1 - race_h_prob
                for r_label, r_prob, r_team in [
                    (f"🏃 {home} Race to {race_target}", race_h_prob, home),
                    (f"🏃 {away} Race to {race_target}", race_a_prob, away),
                ]:
                    r_conf = r_prob * 100
                    if r_conf > 30:
                        r_odds = 1 / max(r_prob, 0.05)
                        predictions.append(Prediction(
                            event=event,
                            bet_type=BetType.RACE_TO,
                            pick=r_label,
                            confidence=round(r_conf, 1),
                            probability=r_prob,
                            odds=round(r_odds, 2),
                            american_odds=self._decimal_to_american(r_odds),
                            market_display=f"Race to {race_target} — {r_team} ({self._decimal_to_american(r_odds)})",
                            team_name=r_team,
                            reasoning=f"Based on scoring pace: {home} {h_pace:.0f} avg vs {away} {a_pace:.0f} avg. {kick_off}",
                        ))
        elif sport == Sport.HOCKEY and event.home_stats and event.away_stats:
            for race_target in [2, 3]:
                h_pace = event.home_stats.avg_goals_scored
                a_pace = event.away_stats.avg_goals_scored
                pace_total = h_pace + a_pace + 0.001
                race_h_prob = h_pace / pace_total * 0.90 + 0.05
                race_a_prob = 1 - race_h_prob
                for r_label, r_prob, r_team in [
                    (f"🏃 {home} Race to {race_target} Goals", race_h_prob, home),
                    (f"🏃 {away} Race to {race_target} Goals", race_a_prob, away),
                ]:
                    r_conf = r_prob * 100
                    if r_conf > 25:
                        r_odds = 1 / max(r_prob, 0.05)
                        predictions.append(Prediction(
                            event=event,
                            bet_type=BetType.RACE_TO,
                            pick=r_label,
                            confidence=round(r_conf, 1),
                            probability=r_prob,
                            odds=round(r_odds, 2),
                            american_odds=self._decimal_to_american(r_odds),
                            market_display=f"Race to {race_target} Goals — {r_team} ({self._decimal_to_american(r_odds)})",
                            team_name=r_team,
                            reasoning=f"Based on scoring rate. {kick_off}",
                        ))

        # Clamp confidence to [0, 99] and sort
        for p in predictions:
            p.confidence = min(99.0, max(0.0, p.confidence))
            p.probability = min(0.99, max(0.0, p.probability))
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions

    def _american_to_decimal(self, american: str) -> float:
        """Convert American odds string to decimal."""
        try:
            val = int(american)
            if val > 0:
                return round(1 + val / 100, 2)
            else:
                return round(1 + 100 / abs(val), 2)
        except (ValueError, ZeroDivisionError):
            return 2.0

    # ── Factor Analysis Methods ──────────────────────────────────────────

    def _analyze_form(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        """Analyze recent form (W/D/L string)."""
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        def form_score(form: str) -> float:
            score = 0
            for i, ch in enumerate(form):
                recency = 1.0 - (i * 0.08)  # More recent = higher weight
                if ch == "W":
                    score += 3 * recency
                elif ch == "D":
                    score += 1 * recency
            return score

        home_form = form_score(home.form_string)
        away_form = form_score(away.form_string)
        total = home_form + away_form + 0.001

        return {
            "home": round((home_form / total) * weight, 4),
            "away": round((away_form / total) * weight, 4),
            "draw": round(weight * 0.15, 4),  # Base draw from form similarity
            "detail": f"Home form: {home.form_string}, Away form: {away.form_string}",
        }

    def _analyze_home_advantage(
        self, home: Optional[TeamStats], away: Optional[TeamStats],
        sport: Sport, weight: float
    ) -> dict:
        """Calculate home advantage factor."""
        # Base home advantage by sport
        base_ha = {
            Sport.SOCCER: 0.60,
            Sport.BASKETBALL: 0.58,
            Sport.BASEBALL: 0.54,
            Sport.AMERICAN_FOOTBALL: 0.57,
            Sport.VOLLEYBALL: 0.58,
            Sport.TENNIS: 0.52,  # Surface matters more
        }.get(sport, 0.55)

        if home and home.games_played > 0:
            home_wr = (home.home_wins) / max(
                home.home_wins + home.home_draws + home.home_losses, 1
            )
            base_ha = (base_ha + home_wr) / 2

        return {
            "home": round(weight * base_ha, 4),
            "away": round(weight * (1 - base_ha) * 0.75, 4),
            "draw": round(weight * (1 - base_ha) * 0.25, 4),
            "detail": f"Home advantage: {base_ha:.0%}",
        }

    def _analyze_h2h(self, h2h: Optional[HeadToHead], weight: float) -> dict:
        """Analyze head-to-head record."""
        if not h2h or h2h.total_matches == 0:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        total = h2h.total_matches
        return {
            "home": round((h2h.team1_wins / total) * weight, 4),
            "away": round((h2h.team2_wins / total) * weight, 4),
            "draw": round((h2h.draws / total) * weight, 4),
            "detail": f"H2H: {h2h.team1_wins}W-{h2h.draws}D-{h2h.team2_wins}L ({total} games)",
        }

    def _analyze_league_position(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        """Compare league positions. Lower position = better."""
        if not home or not away or home.league_position == 0 or away.league_position == 0:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        # Invert: lower position number = higher score
        max_pos = max(home.league_position, away.league_position) + 1
        home_score = (max_pos - home.league_position) / max_pos
        away_score = (max_pos - away.league_position) / max_pos
        total = home_score + away_score + 0.001

        return {
            "home": round((home_score / total) * weight * 0.85, 4),
            "away": round((away_score / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": f"Positions: Home #{home.league_position} vs Away #{away.league_position}",
        }

    def _analyze_scoring(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        """Analyze scoring patterns: attack vs defense matchup."""
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        # Offensive strength vs defensive weakness
        home_attack = home.avg_goals_scored * (away.avg_goals_conceded + 0.1)
        away_attack = away.avg_goals_scored * (home.avg_goals_conceded + 0.1)
        total = home_attack + away_attack + 0.001

        return {
            "home": round((home_attack / total) * weight * 0.85, 4),
            "away": round((away_attack / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": (
                f"Home attack: {home.avg_goals_scored:.1f} GS, "
                f"Away defense: {away.avg_goals_conceded:.1f} GC"
            ),
        }

    def _analyze_injuries(
        self,
        home_injuries: list[PlayerInfo],
        away_injuries: list[PlayerInfo],
        weight: float,
    ) -> dict:
        """Assess impact of injuries/suspensions."""
        # More injuries = worse for that team
        home_impact = len(home_injuries) * 0.05
        away_impact = len(away_injuries) * 0.05

        home_impact = min(home_impact, 0.3)
        away_impact = min(away_impact, 0.3)

        # Injuries hurt them, benefit opponent
        home_score = weight * (0.5 + away_impact - home_impact)
        away_score = weight * (0.5 + home_impact - away_impact)

        home_score = max(0, home_score)
        away_score = max(0, away_score)

        return {
            "home": round(home_score * 0.85, 4),
            "away": round(away_score * 0.85, 4),
            "draw": round(weight * 0.15, 4),
            "detail": f"Injuries: Home {len(home_injuries)}, Away {len(away_injuries)}",
        }

    def _analyze_consistency(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        """Analyze how consistent teams are (low variance = reliable)."""
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        def consistency_score(stats: TeamStats) -> float:
            if stats.games_played == 0:
                return 0.5
            win_rate = stats.wins / stats.games_played
            # Higher win rate = more consistent (simplified)
            return win_rate

        hc = consistency_score(home)
        ac = consistency_score(away)
        total = hc + ac + 0.001

        return {
            "home": round((hc / total) * weight * 0.85, 4),
            "away": round((ac / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
        }

    def _analyze_momentum(
        self, home: Optional[TeamStats], away: Optional[TeamStats], weight: float
    ) -> dict:
        """Analyze recent momentum (last 3-5 games weighted heavily)."""
        if not home or not away:
            return {"home": weight / 3, "draw": weight / 3, "away": weight / 3}

        def momentum(form: str) -> float:
            last5 = form[:5]
            score = 0
            for ch in last5:
                if ch == "W":
                    score += 2
                elif ch == "D":
                    score += 0.5
            return score

        hm = momentum(home.form_string)
        am = momentum(away.form_string)
        total = hm + am + 0.001

        return {
            "home": round((hm / total) * weight * 0.85, 4),
            "away": round((am / total) * weight * 0.85, 4),
            "draw": round(weight * 0.15, 4),
        }

    # ── Utility Methods ──────────────────────────────────────────────────

    def _calculate_value(self, probability: float, odds: float) -> float:
        """Calculate expected value: EV = (prob * odds) - 1."""
        if odds <= 0 or probability <= 0:
            return 0.0
        ev = (probability * odds) - 1
        return round(ev, 4)

    def _poisson_cdf(self, k: int, lam: float) -> float:
        """Poisson CDF: P(X <= k) for parameter lambda."""
        if lam <= 0:
            return 1.0
        if lam > 100:
            # For high-scoring sports, use normal approximation
            # P(X <= k) ≈ Φ((k + 0.5 - λ) / √λ)
            z = (k + 0.5 - lam) / math.sqrt(lam)
            return 0.5 * (1 + math.erf(z / math.sqrt(2)))
        total = 0.0
        log_lam = math.log(lam)
        for i in range(k + 1):
            log_prob = i * log_lam - lam - math.lgamma(i + 1)
            total += math.exp(log_prob)
        return min(1.0, total)

    def _build_reasoning(
        self, event: MatchEvent, factors: dict, pick: str
    ) -> str:
        """Build human-readable reasoning for a prediction."""
        lines = []
        lines.append(
            f"{'HOME' if pick == 'home' else 'AWAY' if pick == 'away' else 'DRAW'} "
            f"prediction for {event.home_team.name} vs {event.away_team.name}"
        )
        lines.append(f"Tournament: {event.tournament.name} ({event.tournament.country})")
        lines.append("")

        for name, data in factors.items():
            detail = data.get("detail", "")
            if detail:
                lines.append(f"• {name.replace('_', ' ').title()}: {detail}")

        if event.home_injuries:
            inj_names = [p.name for p in event.home_injuries[:5]]
            lines.append(f"• Home injuries: {', '.join(inj_names)}")
        if event.away_injuries:
            inj_names = [p.name for p in event.away_injuries[:5]]
            lines.append(f"• Away injuries: {', '.join(inj_names)}")

        if event.home_odds > 0:
            lines.append(
                f"• Odds: Home {event.home_odds:.2f} | "
                f"Draw {event.draw_odds:.2f} | Away {event.away_odds:.2f}"
            )

        return "\n".join(lines)
