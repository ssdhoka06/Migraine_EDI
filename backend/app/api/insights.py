"""
Insights API Router
Handles trigger analysis and user insights
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, date, timedelta
import logging

from app.core.database import get_db
from app.models.database import User, DailyLog
from app.models.schemas import (
    TriggerAnalysisResponse,
    WeeklyStatsResponse,
    WeeklyAccuracy,
    APIResponse,
)
from app.services.trigger_analysis_service import trigger_analysis_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/triggers/{user_id}", response_model=APIResponse)
async def get_trigger_analysis(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive trigger analysis for a user
    """
    # Verify user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get all logs with outcomes
    result = await db.execute(
        select(DailyLog)
        .where(DailyLog.user_id == user_id)
        .where(DailyLog.migraine_occurred.isnot(None))
        .order_by(desc(DailyLog.date))
    )
    logs = result.scalars().all()
    
    # Convert to dict format
    log_dicts = []
    for log in logs:
        log_dicts.append({
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
            "predicted_probability": log.predicted_probability,
            "prediction_was_correct": log.prediction_was_correct,
        })
    
    # Analyze triggers
    triggers = trigger_analysis_service.analyze_all_triggers(log_dicts)
    
    # Discover patterns
    patterns = trigger_analysis_service.discover_patterns(log_dicts)
    
    # Calculate totals
    total_logs = len(log_dicts)
    total_migraines = sum(1 for l in log_dicts if l.get("migraine_occurred"))
    
    response = TriggerAnalysisResponse(
        user_id=user_id,
        total_logs=total_logs,
        total_migraines=total_migraines,
        triggers=triggers,
        patterns=patterns
    )
    
    return APIResponse(
        success=True,
        data=response
    )


@router.get("/weekly-stats/{user_id}", response_model=APIResponse)
async def get_weekly_stats(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get weekly statistics and accuracy trends
    """
    # Verify user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get logs from last 90 days
    cutoff_date = date.today() - timedelta(days=90)
    result = await db.execute(
        select(DailyLog)
        .where(DailyLog.user_id == user_id)
        .where(DailyLog.date >= cutoff_date)
        .order_by(desc(DailyLog.date))
    )
    logs = result.scalars().all()
    
    # Convert to dict format
    log_dicts = []
    for log in logs:
        log_dicts.append({
            "date": log.date.isoformat(),
            "migraine_occurred": log.migraine_occurred,
            "predicted_probability": log.predicted_probability,
            "prediction_was_correct": log.prediction_was_correct,
        })
    
    # Calculate stats
    stats = trigger_analysis_service.calculate_stats(log_dicts)
    
    # Get weekly accuracy trend
    weekly_accuracy_data = trigger_analysis_service.get_weekly_accuracy(log_dicts, weeks=8)
    weekly_accuracy = [
        WeeklyAccuracy(week=w["week"], accuracy=w["accuracy"])
        for w in weekly_accuracy_data
    ]
    
    response = WeeklyStatsResponse(
        prediction_accuracy=stats["prediction_accuracy"],
        total_migraines=stats["total_migraines"],
        streak_days=stats["streak_days"],
        weekly_accuracy=weekly_accuracy
    )
    
    return APIResponse(
        success=True,
        data=response
    )


@router.get("/summary/{user_id}", response_model=APIResponse)
async def get_insights_summary(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a summary of user insights for dashboard
    """
    # Verify user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get recent logs
    cutoff_date = date.today() - timedelta(days=30)
    result = await db.execute(
        select(DailyLog)
        .where(DailyLog.user_id == user_id)
        .where(DailyLog.date >= cutoff_date)
        .order_by(desc(DailyLog.date))
    )
    logs = result.scalars().all()
    
    # Calculate summary stats
    total_logs = len(logs)
    migraines = sum(1 for l in logs if l.migraine_occurred)
    migraine_free_days = total_logs - migraines
    
    # Average stats
    avg_sleep = sum(l.sleep_hours for l in logs) / total_logs if total_logs > 0 else 0
    avg_stress = sum(l.stress_level for l in logs) / total_logs if total_logs > 0 else 0
    
    # Prediction accuracy
    logs_with_predictions = [l for l in logs if l.prediction_was_correct is not None]
    correct = sum(1 for l in logs_with_predictions if l.prediction_was_correct)
    accuracy = correct / len(logs_with_predictions) if logs_with_predictions else 0
    
    # Top trigger (simplified)
    poor_sleep_migraines = sum(1 for l in logs if l.migraine_occurred and l.sleep_hours < 6)
    high_stress_migraines = sum(1 for l in logs if l.migraine_occurred and l.stress_level >= 7)
    
    top_trigger = "Sleep" if poor_sleep_migraines >= high_stress_migraines else "Stress"
    
    return APIResponse(
        success=True,
        data={
            "period_days": 30,
            "total_logs": total_logs,
            "migraines": migraines,
            "migraine_free_days": migraine_free_days,
            "avg_sleep_hours": round(avg_sleep, 1),
            "avg_stress_level": round(avg_stress, 1),
            "prediction_accuracy": round(accuracy * 100, 1),
            "top_trigger": top_trigger,
            "days_logged": user.days_logged,
            "current_phase": user.current_phase,
        }
    )


@router.get("/recommendations/{user_id}", response_model=APIResponse)
async def get_personalized_recommendations(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized recommendations based on trigger analysis
    """
    # Verify user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get trigger analysis
    result = await db.execute(
        select(DailyLog)
        .where(DailyLog.user_id == user_id)
        .where(DailyLog.migraine_occurred.isnot(None))
    )
    logs = result.scalars().all()
    
    log_dicts = [{
        "sleep_hours": log.sleep_hours,
        "sleep_quality_good": log.sleep_quality_good,
        "stress_level": log.stress_level,
        "skipped_meals": log.skipped_meals or [],
        "had_snack": log.had_snack,
        "alcohol_drinks": log.alcohol_drinks,
        "water_glasses": log.water_glasses,
        "migraine_occurred": log.migraine_occurred,
    } for log in logs]
    
    triggers = trigger_analysis_service.analyze_all_triggers(log_dicts)
    
    # Generate recommendations based on top triggers
    recommendations = []
    
    for trigger in triggers[:3]:  # Top 3 triggers
        if "Sleep" in trigger.name:
            recommendations.append({
                "category": "Sleep",
                "title": "Prioritize Sleep Hygiene",
                "description": f"Sleep issues are your #{triggers.index(trigger)+1} trigger (OR: {trigger.odds_ratio}). Aim for 7-8 hours nightly.",
                "actions": [
                    "Set a consistent bedtime",
                    "Avoid screens 1 hour before bed",
                    "Keep bedroom cool and dark"
                ],
                "priority": "high" if trigger.odds_ratio > 2 else "medium"
            })
        elif "Stress" in trigger.name:
            recommendations.append({
                "category": "Stress",
                "title": "Manage Stress Levels",
                "description": f"High stress is a significant trigger for you (OR: {trigger.odds_ratio}).",
                "actions": [
                    "Practice 10-minute daily meditation",
                    "Take regular breaks during work",
                    "Try progressive muscle relaxation"
                ],
                "priority": "high" if trigger.odds_ratio > 2 else "medium"
            })
        elif "Meal" in trigger.name or "Dehydration" in trigger.name:
            recommendations.append({
                "category": "Diet",
                "title": "Maintain Regular Eating Habits",
                "description": f"Dietary factors affect your migraines (OR: {trigger.odds_ratio}).",
                "actions": [
                    "Don't skip meals",
                    "Drink at least 8 glasses of water daily",
                    "Have healthy snacks available"
                ],
                "priority": "medium"
            })
    
    # Add general recommendations
    recommendations.append({
        "category": "Tracking",
        "title": "Keep Logging Daily",
        "description": f"You've logged {user.days_logged} days. More data improves predictions!",
        "actions": [
            "Log every morning for best results",
            "Record migraine outcomes honestly",
            "Note any unusual symptoms"
        ],
        "priority": "low"
    })
    
    return APIResponse(
        success=True,
        data=recommendations
    )
