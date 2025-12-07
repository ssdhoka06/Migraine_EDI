"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get the project root directory (Migraine_EDI/).
    
    Works by going up from this file's location:
    config.py -> core/ -> app/ -> backend/ -> Migraine_EDI/
    """
    return Path(__file__).resolve().parent.parent.parent.parent


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
    
    # ==========================================================================
    # ML Model Settings
    # ==========================================================================
    # Project structure:
    # Migraine_EDI/           <- PROJECT_ROOT
    # ├── backend/
    # │   └── app/
    # │       └── core/
    # │           └── config.py  <- THIS FILE
    # └── migraine_mamba/
    #     ├── models/
    #     │   ├── mamba_finetuned.pth
    #     │   └── mamba_ssl.pth
    #     └── src/
    #         ├── model.py
    #         └── personalization.py
    
    @property
    def PROJECT_ROOT(self) -> Path:
        return get_project_root()
    
    @property  
    def MIGRAINE_MAMBA_DIR(self) -> str:
        env_val = os.getenv("MIGRAINE_MAMBA_DIR")
        if env_val:
            return env_val
        return str(get_project_root() / "migraine_mamba")
    
    @property
    def MODELS_DIR(self) -> str:
        env_val = os.getenv("MODELS_DIR")
        if env_val:
            return env_val
        return str(get_project_root() / "migraine_mamba" / "models")
    
    @property
    def USER_DATA_DIR(self) -> str:
        env_val = os.getenv("USER_DATA_DIR")
        if env_val:
            return env_val
        return str(get_project_root() / "backend" / "user_data")
    
    # Device for PyTorch inference ("cpu", "cuda", "mps", or "auto")
    ML_DEVICE: str = os.getenv("ML_DEVICE", "auto")
    
    # Legacy paths (kept for backward compatibility)
    MODEL_PATH: str = "./models"
    FOUNDATION_MODEL_PATH: str = "./models/foundation_lightgbm.joblib"
    TEMPORAL_MODEL_PATH: str = "./models/temporal_mamba.pt"
    
    # Prediction Thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    MODERATE_RISK_THRESHOLD: float = 0.5
    HIGH_RISK_THRESHOLD: float = 0.7
    
    # Phase Thresholds (days) - aligned with PersonalizationSystem
    FOUNDATION_PHASE_DAYS: int = 14
    TEMPORAL_PHASE_DAYS: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# Debug: Print paths on import (remove in production)
if os.getenv("DEBUG_PATHS", "").lower() == "true":
    print(f"[Config] PROJECT_ROOT: {settings.PROJECT_ROOT}")
    print(f"[Config] MIGRAINE_MAMBA_DIR: {settings.MIGRAINE_MAMBA_DIR}")
    print(f"[Config] MODELS_DIR: {settings.MODELS_DIR}")
    print(f"[Config] USER_DATA_DIR: {settings.USER_DATA_DIR}")
