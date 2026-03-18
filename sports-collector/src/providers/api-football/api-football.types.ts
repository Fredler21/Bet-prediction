// Raw response shapes from https://v3.football.api-sports.io

export interface ApiFootballResponse<T> {
  get: string;
  parameters: Record<string, string | number>;
  errors: string[] | Record<string, string>;
  results: number;
  paging: { current: number; total: number };
  response: T[];
}

// ── Leagues ─────────────────────────────────────────────────────────────

export interface ApiFootballLeague {
  league: {
    id: number;
    name: string;
    type: string;
    logo: string;
  };
  country: {
    name: string;
    code: string | null;
    flag: string | null;
  };
  seasons: Array<{
    year: number;
    start: string;
    end: string;
    current: boolean;
    coverage: Record<string, boolean | Record<string, boolean>>;
  }>;
}

// ── Teams ────────────────────────────────────────────────────────────────

export interface ApiFootballTeam {
  team: {
    id: number;
    name: string;
    code: string | null;
    country: string | null;
    founded: number | null;
    national: boolean;
    logo: string;
  };
  venue: {
    id: number | null;
    name: string | null;
    address: string | null;
    city: string | null;
    capacity: number | null;
    surface: string | null;
    image: string | null;
  };
}

// ── Fixtures ─────────────────────────────────────────────────────────────

export interface ApiFootballFixture {
  fixture: {
    id: number;
    referee: string | null;
    timezone: string;
    date: string;          // ISO 8601
    timestamp: number;
    periods: { first: number | null; second: number | null };
    venue: { id: number | null; name: string | null; city: string | null };
    status: {
      long: string;
      short: string;       // NS, 1H, HT, 2H, ET, FT, PST, CANC, …
      elapsed: number | null;
    };
  };
  league: {
    id: number;
    name: string;
    country: string;
    logo: string;
    flag: string | null;
    season: number;
    round: string;
  };
  teams: {
    home: { id: number; name: string; logo: string; winner: boolean | null };
    away: { id: number; name: string; logo: string; winner: boolean | null };
  };
  goals: {
    home: number | null;
    away: number | null;
  };
  score: {
    halftime: { home: number | null; away: number | null };
    fulltime: { home: number | null; away: number | null };
    extratime: { home: number | null; away: number | null };
    penalty: { home: number | null; away: number | null };
  };
  events?: ApiFootballEvent[];
}

export interface ApiFootballEvent {
  time: { elapsed: number; extra: number | null };
  team: { id: number; name: string; logo: string };
  player: { id: number; name: string };
  assist: { id: number | null; name: string | null };
  type: string;    // "Goal" | "Card" | "subst" | "Var"
  detail: string;  // "Normal Goal" | "Yellow Card" | "Red Card" | …
  comments: string | null;
}

// ── Standings ─────────────────────────────────────────────────────────────

export interface ApiFootballStandingWrapper {
  league: {
    id: number;
    name: string;
    country: string;
    logo: string;
    flag: string | null;
    season: number;
    standings: ApiFootballStanding[][];
  };
}

export interface ApiFootballStanding {
  rank: number;
  team: { id: number; name: string; logo: string };
  points: number;
  goalsDiff: number;
  group: string;
  form: string;
  status: string;
  description: string | null;
  all: ApiFootballStandingRecord;
  home: ApiFootballStandingRecord;
  away: ApiFootballStandingRecord;
  update: string;
}

export interface ApiFootballStandingRecord {
  played: number;
  win: number;
  draw: number;
  lose: number;
  goals: { for: number; against: number };
}

// ── Players ───────────────────────────────────────────────────────────────

export interface ApiFootballPlayerWrapper {
  player: {
    id: number;
    name: string;
    firstname: string;
    lastname: string;
    age: number;
    birth: { date: string | null; place: string | null; country: string | null };
    nationality: string | null;
    height: string | null;
    weight: string | null;
    injured: boolean;
    photo: string;
  };
  statistics: Array<{
    team: { id: number; name: string; logo: string };
    league: { id: number; name: string; country: string; logo: string; season: number };
    games: {
      appearences: number | null;
      lineups: number | null;
      minutes: number | null;
      number: number | null;
      position: string | null;
      rating: string | null;
      captain: boolean;
    };
  }>;
}
