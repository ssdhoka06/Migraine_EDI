"""
SQLAlchemy Database Models
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, date
import uuid

from app.core.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    """User profile model"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Demographics
    gender = Column(String(1), nullable=False)  # M or F
    age = Column(Integer, nullable=False)
    height = Column(Float, nullable=False)  # cm
    weight = Column(Float, nullable=False)  # kg
    bmi = Column(Float, nullable=False)
    
    # Location
    location_city = Column(String(100))
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Migraine baseline
    attacks_per_month = Column(Integer, default=4)
    
    # Menstrual cycle tracking
    has_menstrual_cycle = Column(Boolean, default=False)
    cycle_start_day = Column(Date, nullable=True)
    cycle_length = Column(Integer, default=28)
    
    # User state
    current_phase = Column(String(20), default="foundation")  # foundation, temporal, personalized
    days_logged = Column(Integer, default=0)
    last_log_date = Column(Date, nullable=True)
    
    # Model state
    model_version = Column(String(50), default="foundation_v1")
    last_retrain_date = Column(DateTime, nullable=True)
    
    # Relationships
    daily_logs = relationship("DailyLog", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class DailyLog(Base):
    """Daily migraine log entry"""
    __tablename__ = "daily_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Sleep factors
    sleep_hours = Column(Float, nullable=False)
    sleep_quality_good = Column(Boolean, default=True)
    
    # Stress
    stress_level = Column(Integer, nullable=False)  # 1-10
    
    # Diet
    skipped_meals = Column(JSON, default=list)  # ["breakfast", "lunch", "dinner"]
    had_snack = Column(Boolean, default=False)
    alcohol_drinks = Column(Integer, default=0)
    caffeine_drinks = Column(Integer, default=0)
    water_glasses = Column(Integer, default=0)
    
    # Environment
    bright_light_exposure = Column(Boolean, default=False)
    screen_time_hours = Column(Float, default=0)
    
    # Weather (auto-fetched)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    pressure_change = Column(Float, nullable=True)  # Change from previous day
    
    # Menstrual cycle phase (auto-calculated)
    cycle_day = Column(Integer, nullable=True)
    cycle_phase = Column(String(20), nullable=True)  # menstrual, follicular, ovulation, luteal
    
    # Prodromal symptoms
    symptoms = Column(JSON, default=dict)
    
    # Outcome (filled next day)
    migraine_occurred = Column(Boolean, nullable=True)
    migraine_severity = Column(Integer, nullable=True)  # 1-10
    migraine_duration_hours = Column(Float, nullable=True)
    migraine_location = Column(String(20), nullable=True)  # left, right, both
    migraine_with_aura = Column(Boolean, nullable=True)
    medications_taken = Column(JSON, default=list)
    
    # Prediction tracking
    predicted_probability = Column(Float, nullable=True)
    predicted_risk_level = Column(String(20), nullable=True)
    prediction_was_correct = Column(Boolean, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="daily_logs")


class Prediction(Base):
    """Prediction history"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    prediction_date = Column(Date, nullable=False)  # Date being predicted
    
    # Prediction results
    attack_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    severity_prediction = Column(Float, nullable=True)
    
    # Model info
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(20), nullable=False)  # foundation, temporal, personalized
    
    # Trigger contributions
    top_triggers = Column(JSON, default=list)
    contributing_factors = Column(JSON, default=list)
    protective_factors = Column(JSON, default=list)
    
    # Recommendations
    recommendations = Column(JSON, default=list)
    
    # Outcome tracking
    actual_outcome = Column(Boolean, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="predictions")


class TriggerAnalysis(Base):
    """User's trigger analysis (updated weekly)"""
    __tablename__ = "trigger_analyses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Analysis results
    triggers = Column(JSON, default=list)  # List of trigger objects with OR, contribution, etc.
    patterns = Column(JSON, default=list)  # Discovered patterns
    
    # Stats
    total_logs = Column(Integer, default=0)
    total_migraines = Column(Integer, default=0)
    prediction_accuracy = Column(Float, default=0.0)


class WeeklyStats(Base):
    """Weekly statistics snapshot"""
    __tablename__ = "weekly_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    week_start = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Stats
    logs_count = Column(Integer, default=0)
    migraines_count = Column(Integer, default=0)
    predictions_made = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    
    # Averages
    avg_sleep = Column(Float, nullable=True)
    avg_stress = Column(Float, nullable=True)
    avg_risk = Column(Float, nullable=True)