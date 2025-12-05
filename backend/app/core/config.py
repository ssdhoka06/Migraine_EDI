"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # App
    APP_NAME: str = "MigraineMamba"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./migrainemamba.db"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ]
    
    # JWT
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Weather API (OpenWeatherMap)
    WEATHER_API_KEY: str = ""
    WEATHER_API_URL: str = "https://api.openweathermap.org/data/2.5"
    
    # ML Model Settings
    MODEL_PATH: str = "./models"
    FOUNDATION_MODEL_PATH: str = "./models/foundation_lightgbm.joblib"
    TEMPORAL_MODEL_PATH: str = "./models/temporal_mamba.pt"
    
    # Prediction Thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    MODERATE_RISK_THRESHOLD: float = 0.5
    HIGH_RISK_THRESHOLD: float = 0.7
    
    # Phase Thresholds (days)
    FOUNDATION_PHASE_DAYS: int = 14
    TEMPORAL_PHASE_DAYS: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()