"""
Database layer — Track predictions and results for accuracy monitoring.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Text, Boolean,
    create_engine,
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker

from src.config import settings

Base = declarative_base()


class PredictionRecord(Base):
    """Stores every prediction made for accuracy tracking."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    event_id = Column(Integer, index=True)
    sport = Column(String(50))
    tournament = Column(String(200))
    home_team = Column(String(200))
    away_team = Column(String(200))
    bet_type = Column(String(50))
    pick = Column(String(200))
    confidence = Column(Float)
    probability = Column(Float)
    odds = Column(Float)
    value_rating = Column(Float)
    reasoning = Column(Text)
    factors_json = Column(Text)

    # Result tracking
    result = Column(String(20), nullable=True)  # "win", "loss", "push", "void"
    actual_score_home = Column(Integer, nullable=True)
    actual_score_away = Column(Integer, nullable=True)
    settled_at = Column(DateTime, nullable=True)
    profit_loss = Column(Float, nullable=True)


class ParlayRecord(Base):
    """Stores parlay recommendations."""
    __tablename__ = "parlays"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_legs = Column(Integer)
    strategy = Column(String(50))
    combined_confidence = Column(Float)
    combined_odds = Column(Float)
    expected_value = Column(Float)
    recommended_stake = Column(Float)
    risk_level = Column(String(20))
    legs_json = Column(Text)  # JSON of leg details
    result = Column(String(20), nullable=True)
    profit_loss = Column(Float, nullable=True)


class AccuracyStats(Base):
    """Tracks running accuracy statistics."""
    __tablename__ = "accuracy_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, default=datetime.utcnow)
    sport = Column(String(50))
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy_pct = Column(Float, default=0.0)
    total_profit_loss = Column(Float, default=0.0)
    roi_pct = Column(Float, default=0.0)


def get_engine():
    """Create database engine (synchronous SQLite)."""
    import os
    from src.config import BASE_DIR
    # On Vercel (read-only filesystem) use /tmp; otherwise use data/ dir
    if os.getenv("VERCEL") or not os.access(str(BASE_DIR), os.W_OK):
        db_path = "/tmp/predictions.db"
    else:
        db_path = str(BASE_DIR / "data" / "predictions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session() -> Session:
    """Get a database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def save_prediction(prediction, session: Optional[Session] = None):
    """Save a prediction to the database."""
    own_session = session is None
    if own_session:
        session = get_session()

    record = PredictionRecord(
        event_id=prediction.event.id,
        sport=prediction.event.tournament.sport.value,
        tournament=prediction.event.tournament.name,
        home_team=prediction.event.home_team.name,
        away_team=prediction.event.away_team.name,
        bet_type=prediction.bet_type.value,
        pick=prediction.pick,
        confidence=prediction.confidence,
        probability=prediction.probability,
        odds=prediction.odds,
        value_rating=prediction.value_rating,
        reasoning=prediction.reasoning[:2000],
        factors_json=json.dumps(
            {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
             for k, v in prediction.factors.items()}
        ),
    )
    session.add(record)
    session.commit()

    if own_session:
        session.close()


def get_accuracy_report(sport: Optional[str] = None) -> dict:
    """Get accuracy statistics."""
    session = get_session()
    try:
        query = session.query(PredictionRecord).filter(
            PredictionRecord.result.isnot(None)
        )
        if sport:
            query = query.filter(PredictionRecord.sport == sport)

        records = query.all()
        if not records:
            return {"total": 0, "wins": 0, "accuracy": 0, "roi": 0}

        total = len(records)
        wins = sum(1 for r in records if r.result == "win")
        total_pl = sum(r.profit_loss or 0 for r in records)
        total_staked = total  # Simplified: $1 per bet

        return {
            "total": total,
            "wins": wins,
            "losses": total - wins,
            "accuracy": round(wins / total * 100, 1) if total > 0 else 0,
            "profit_loss": round(total_pl, 2),
            "roi": round(total_pl / total_staked * 100, 1) if total_staked > 0 else 0,
        }
    finally:
        session.close()
