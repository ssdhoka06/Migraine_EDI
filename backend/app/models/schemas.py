"""
Pydantic Schemas for API Request/Response
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum


# Enums
class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class Phase(str, Enum):
    FOUNDATION = "foundation"
    TEMPORAL = "temporal"
    PERSONALIZED = "personalized"


class CyclePhase(str, Enum):
    MENSTRUAL = "menstrual"
    FOLLICULAR = "follicular"
    OVULATION = "ovulation"
    LUTEAL = "luteal"


# User Schemas
class OnboardingRequest(BaseModel):
    """Onboarding request schema"""
    gender: Gender
    age: int = Field(..., ge=10, le=100)
    height: float = Field(..., ge=100, le=250)  # cm
    weight: float = Field(..., ge=20, le=300)  # kg
    bmi: Optional[float] = None
    attacks_per_month: int = Field(default=4, ge=0, le=31)
    location_city: str = Field(..., min_length=1, max_length=100)
    has_menstrual_cycle: bool = False
    cycle_start_day: Optional[str] = None  # ISO date string


class OnboardingResponse(BaseModel):
    """Onboarding response schema"""
    success: bool
    user_id: str
    message: str
    current_phase: Phase


class UserProfile(BaseModel):
    """User profile schema"""
    id: str
    gender: Gender
    age: int
    bmi: float
    location_city: Optional[str]
    attacks_per_month: int
    has_menstrual_cycle: bool
    current_phase: Phase
    days_logged: int
    model_version: str
    created_at: datetime

    class Config:
        from_attributes = True


# Daily Log Schemas
class SymptomsInput(BaseModel):
    """Prodromal symptoms input"""
    fatigue: bool = False
    stiff_neck: bool = False
    yawning: bool = False
    food_cravings: bool = False
    mood_change: bool = False
    concentration: bool = False
    light_sensitivity: bool = False
    sound_sensitivity: bool = False
    nausea: bool = False
    visual_disturbance: bool = False


class MigraineDetails(BaseModel):
    """Migraine occurrence details"""
    severity: int = Field(..., ge=1, le=10)
    duration_hours: float = Field(..., ge=0, le=72)
    location: str = "both"  # left, right, both
    with_aura: bool = False
    medications_taken: List[str] = []


class DailyLogRequest(BaseModel):
    """Daily log submission request"""
    user_id: str
    date: str  # ISO date string
    
    # Sleep
    sleep_hours: float = Field(..., ge=0, le=24)
    sleep_quality_good: bool = True
    
    # Stress
    stress_level: int = Field(..., ge=1, le=10)
    
    # Diet
    skipped_meals: List[str] = []
    had_snack: bool = False
    alcohol_drinks: int = Field(default=0, ge=0)
    caffeine_drinks: int = Field(default=0, ge=0)
    water_glasses: int = Field(default=0, ge=0)
    
    # Environment
    bright_light_exposure: bool = False
    screen_time_hours: float = Field(default=0, ge=0, le=24)
    
    # Symptoms
    symptoms: Dict[str, bool] = {}
    
    # Outcome (for previous day)
    migraine_occurred: bool = False
    migraine_details: Optional[MigraineDetails] = None


class DailyLogResponse(BaseModel):
    """Daily log response"""
    success: bool
    message: str
    log_id: int
    days_logged: int
    phase_update: Optional[str] = None


class DailyLogEntry(BaseModel):
    """Daily log entry for history"""
    date: str
    sleep_hours: float
    sleep_quality_good: bool
    stress_level: int
    migraine_occurred: Optional[bool]
    migraine_details: Optional[MigraineDetails]
    predicted_risk_level: Optional[str]
    prediction_was_correct: Optional[bool]

    class Config:
        from_attributes = True


# Prediction Schemas
class TriggerContribution(BaseModel):
    """Individual trigger contribution"""
    trigger: str
    contribution: float
    icon: str
    color: str
    description: str


class ContributingFactor(BaseModel):
    """Contributing factor detail"""
    factor: str
    value: str
    threshold: str
    status: str  # normal, warning, critical


class Recommendation(BaseModel):
    """Action recommendation"""
    action: str
    reason: str
    priority: str  # low, medium, high
    icon: str


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    user_id: str
    prediction_date: str
    attack_probability: float
    risk_level: RiskLevel
    confidence: float
    severity_prediction: Optional[float]
    model_version: str
    model_type: str
    top_triggers: List[TriggerContribution]
    contributing_factors: List[ContributingFactor]
    protective_factors: List[str]
    recommendations: List[Recommendation]


# Insights Schemas
class TriggerInfo(BaseModel):
    """Trigger analysis info"""
    name: str
    odds_ratio: float
    contribution: float
    occurrences: int
    icon: str
    description: str


class PatternInfo(BaseModel):
    """Discovered pattern info"""
    title: str
    description: str
    icon: str
    confidence: float


class TriggerAnalysisResponse(BaseModel):
    """Trigger analysis response"""
    user_id: str
    total_logs: int
    total_migraines: int
    triggers: List[TriggerInfo]
    patterns: List[PatternInfo]


class WeeklyAccuracy(BaseModel):
    """Weekly accuracy data point"""
    week: str
    accuracy: float


class WeeklyStatsResponse(BaseModel):
    """Weekly stats response"""
    prediction_accuracy: float
    total_migraines: int
    streak_days: int
    weekly_accuracy: List[WeeklyAccuracy]


# Generic API Response
class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None