"""
Predictions API Router
Handles migraine risk predictions
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, date, timedelta
import logging

from app.core.database import get_db
from app.models.database import User, DailyLog, Prediction
from app.models.schemas import (
    PredictionResponse,
    APIResponse,
)
from app.services.prediction_service import prediction_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{user_id}", response_model=APIResponse)
async def get_prediction(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get today's migraine prediction for a user
    """
    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get recent logs for temporal context
    result = await db.execute(
        select(DailyLog)
        .where(DailyLog.user_id == user_id)
        .order_by(desc(DailyLog.date))
        .limit(14)
    )
    recent_logs = result.scalars().all()
    
    # Prepare user data
    user_data = {
        "id": user.id,
        "gender": user.gender,
        "age": user.age,
        "bmi": user.bmi,
        "has_menstrual_cycle": user.has_menstrual_cycle,
        "cycle_start_day": user.cycle_start_day,
        "cycle_length": user.cycle_length,
        "days_logged": user.days_logged,
        "attacks_per_month": user.attacks_per_month,
    }
    
    # Get most recent log for current features
    today = date.today()
    
    if recent_logs and recent_logs[0].date == today:
        current_log = recent_logs[0]
        log_data = {
            "sleep_hours": current_log.sleep_hours,
            "sleep_quality_good": current_log.sleep_quality_good,
            "stress_level": current_log.stress_level,
            "skipped_meals": current_log.skipped_meals or [],
            "had_snack": current_log.had_snack,
            "alcohol_drinks": current_log.alcohol_drinks,
            "caffeine_drinks": current_log.caffeine_drinks,
            "water_glasses": current_log.water_glasses,
            "bright_light_exposure": current_log.bright_light_exposure,
            "screen_time_hours": current_log.screen_time_hours,
            "symptoms": current_log.symptoms or {},
            "pressure_change": current_log.pressure_change or 0,
        }
    elif recent_logs:
        recent = recent_logs[0]
        log_data = {
            "sleep_hours": recent.sleep_hours,
            "sleep_quality_good": recent.sleep_quality_good,
            "stress_level": recent.stress_level,
            "skipped_meals": recent.skipped_meals or [],
            "had_snack": recent.had_snack,
            "alcohol_drinks": recent.alcohol_drinks,
            "caffeine_drinks": recent.caffeine_drinks,
            "water_glasses": recent.water_glasses,
            "bright_light_exposure": recent.bright_light_exposure,
            "screen_time_hours": recent.screen_time_hours,
            "symptoms": recent.symptoms or {},
            "pressure_change": 0,
        }
    else:
        log_data = {
            "sleep_hours": 7,
            "sleep_quality_good": True,
            "stress_level": 5,
            "skipped_meals": [],
            "had_snack": False,
            "alcohol_drinks": 0,
            "caffeine_drinks": 2,
            "water_glasses": 6,
            "bright_light_exposure": False,
            "screen_time_hours": 4,
            "symptoms": {},
            "pressure_change": 0,
        }
    
    # Convert historical logs
    historical_logs = []
    for log in recent_logs[1:]:
        historical_logs.append({
            "date": log.date.isoformat(),
            "sleep_hours": log.sleep_hours,
            "sleep_quality_good": log.sleep_quality_good,
            "stress_level": log.stress_level,
            "skipped_meals": log.skipped_meals or [],
            "had_snack": log.had_snack,
            "alcohol_drinks": log.alcohol_drinks,
            "caffeine_drinks": log.caffeine_drinks,
            "water_glasses": log.water_glasses,
            "bright_light_exposure": log.bright_light_exposure,
            "screen_time_hours": log.screen_time_hours,
            "symptoms": log.symptoms or {},
            "migraine_occurred": log.migraine_occurred,
        })
    
    # Generate prediction
    prediction_result = prediction_service.predict(
        user_data=user_data,
        log_data=log_data,
        historical_logs=historical_logs
    )
    
    # Store prediction
    prediction = Prediction(
        user_id=user_id,
        prediction_date=today,
        attack_probability=prediction_result["attack_probability"],
        risk_level=prediction_result["risk_level"].value,
        confidence=prediction_result["confidence"],
        severity_prediction=prediction_result["severity_prediction"],
        model_version=prediction_result["model_version"],
        model_type=prediction_result["model_type"],
        top_triggers=[t.model_dump() for t in prediction_result["top_triggers"]],
        contributing_factors=[f.model_dump() for f in prediction_result["contributing_factors"]],
        protective_factors=prediction_result["protective_factors"],
        recommendations=[r.model_dump() for r in prediction_result["recommendations"]],
    )
    
    db.add(prediction)
    
    if recent_logs and recent_logs[0].date == today:
        recent_logs[0].predicted_probability = prediction_result["attack_probability"]
        recent_logs[0].predicted_risk_level = prediction_result["risk_level"].value
    
    await db.commit()
    
    response = PredictionResponse(
        user_id=user_id,
        prediction_date=today.isoformat(),
        attack_probability=prediction_result["attack_probability"],
        risk_level=prediction_result["risk_level"],
        confidence=prediction_result["confidence"],
        severity_prediction=prediction_result["severity_prediction"],
        model_version=prediction_result["model_version"],
        model_type=prediction_result["model_type"],
        top_triggers=prediction_result["top_triggers"],
        contributing_factors=prediction_result["contributing_factors"],
        protective_factors=prediction_result["protective_factors"],
        recommendations=prediction_result["recommendations"],
    )
    
    return APIResponse(
        success=True,
        data=response
    )


@router.get("/history/{user_id}", response_model=APIResponse)
async def get_prediction_history(
    user_id: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's prediction history
    """
    cutoff_date = date.today() - timedelta(days=days)
    
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == user_id)
        .where(Prediction.prediction_date >= cutoff_date)
        .order_by(desc(Prediction.prediction_date))
    )
    predictions = result.scalars().all()
    
    history = []
    for pred in predictions:
        history.append({
            "date": pred.prediction_date.isoformat(),
            "probability": pred.attack_probability,
            "risk_level": pred.risk_level,
            "confidence": pred.confidence,
            "actual_outcome": pred.actual_outcome,
            "was_correct": pred.was_correct,
            "model_type": pred.model_type,
        })
    
    return APIResponse(
        success=True,
        data=history
    )


@router.get("/accuracy/{user_id}", response_model=APIResponse)
async def get_prediction_accuracy(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's overall prediction accuracy statistics
    """
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == user_id)
        .where(Prediction.was_correct.isnot(None))
    )
    predictions = result.scalars().all()
    
    if not predictions:
        return APIResponse(
            success=True,
            data={
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "by_risk_level": {}
            }
        )
    
    total = len(predictions)
    correct = sum(1 for p in predictions if p.was_correct)
    
    # Accuracy by risk level
    by_risk_level = {}
    for pred in predictions:
        level = pred.risk_level
        if level not in by_risk_level:
            by_risk_level[level] = {"total": 0, "correct": 0}
        by_risk_level[level]["total"] += 1
        if pred.was_correct:
            by_risk_level[level]["correct"] += 1
    
    for level in by_risk_level:
        t = by_risk_level[level]["total"]
        c = by_risk_level[level]["correct"]
        by_risk_level[level]["accuracy"] = c / t if t > 0 else 0
    
    return APIResponse(
        success=True,
        data={
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": correct / total if total > 0 else 0,
            "by_risk_level": by_risk_level
        }
    )
