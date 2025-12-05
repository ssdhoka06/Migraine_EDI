"""
Daily Logs API Router
Handles daily migraine log submissions and history
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, date, timedelta
from typing import List
import logging

from app.core.database import get_db
from app.core.config import settings
from app.models.database import User, DailyLog
from app.models.schemas import (
    DailyLogRequest,
    DailyLogResponse,
    DailyLogEntry,
    APIResponse,
)
from app.services.prediction_service import prediction_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/submit", response_model=APIResponse)
async def submit_daily_log(
    request: DailyLogRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a daily migraine log entry
    """
    try:
        # Get user
        result = await db.execute(select(User).where(User.id == request.user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Parse date
        log_date = datetime.strptime(request.date, "%Y-%m-%d").date()
        
        # Check if log already exists for this date
        existing = await db.execute(
            select(DailyLog).where(
                DailyLog.user_id == request.user_id,
                DailyLog.date == log_date
            )
        )
        existing_log = existing.scalar_one_or_none()
        
        if existing_log:
            # Update existing log
            log = existing_log
        else:
            # Create new log
            log = DailyLog(
                user_id=request.user_id,
                date=log_date,
            )
        
        # Update log fields
        log.sleep_hours = request.sleep_hours
        log.sleep_quality_good = request.sleep_quality_good
        log.stress_level = request.stress_level
        log.skipped_meals = request.skipped_meals
        log.had_snack = request.had_snack
        log.alcohol_drinks = request.alcohol_drinks
        log.caffeine_drinks = request.caffeine_drinks
        log.water_glasses = request.water_glasses
        log.bright_light_exposure = request.bright_light_exposure
        log.screen_time_hours = request.screen_time_hours
        log.symptoms = request.symptoms
        
        # Handle migraine outcome (from previous day check-in)
        if request.migraine_occurred is not None:
            log.migraine_occurred = request.migraine_occurred
            
            if request.migraine_occurred and request.migraine_details:
                log.migraine_severity = request.migraine_details.severity
                log.migraine_duration_hours = request.migraine_details.duration_hours
                log.migraine_location = request.migraine_details.location
                log.migraine_with_aura = request.migraine_details.with_aura
                log.medications_taken = request.migraine_details.medications_taken
        
        # Calculate menstrual cycle phase if applicable
        if user.has_menstrual_cycle and user.cycle_start_day:
            cycle_day, cycle_phase = prediction_service.calculate_cycle_phase(
                user.cycle_start_day,
                log_date,
                user.cycle_length or 28
            )
            log.cycle_day = cycle_day
            log.cycle_phase = cycle_phase
        
        # Save log
        if not existing_log:
            db.add(log)
        
        await db.commit()
        await db.refresh(log)
        
        # Update user stats
        if not existing_log:
            user.days_logged += 1
            user.last_log_date = log_date
            
            # Check for phase upgrade
            phase_update = None
            old_phase = user.current_phase
            new_phase = prediction_service.calculate_phase(user.days_logged)
            
            if new_phase != old_phase:
                user.current_phase = new_phase
                user.model_version = f"{new_phase}_v1"
                phase_update = f"Upgraded to {new_phase} model!"
                logger.info(f"User {user.id} upgraded to {new_phase} phase")
            
            await db.commit()
        else:
            phase_update = None
        
        logger.info(f"Daily log submitted for user {request.user_id} on {log_date}")
        
        return APIResponse(
            success=True,
            data=DailyLogResponse(
                success=True,
                message="Log submitted successfully",
                log_id=log.id,
                days_logged=user.days_logged,
                phase_update=phase_update
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Log submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/history/{user_id}", response_model=APIResponse)
async def get_log_history(
    user_id: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's daily log history
    """
    # Verify user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get logs
    cutoff_date = date.today() - timedelta(days=days)
    result = await db.execute(
        select(DailyLog)
        .where(DailyLog.user_id == user_id)
        .where(DailyLog.date >= cutoff_date)
        .order_by(desc(DailyLog.date))
    )
    logs = result.scalars().all()
    
    # Convert to response format
    history = []
    for log in logs:
        entry = DailyLogEntry(
            date=log.date.isoformat(),
            sleep_hours=log.sleep_hours,
            sleep_quality_good=log.sleep_quality_good,
            stress_level=log.stress_level,
            migraine_occurred=log.migraine_occurred,
            migraine_details=None,
            predicted_risk_level=log.predicted_risk_level,
            prediction_was_correct=log.prediction_was_correct,
        )
        
        if log.migraine_occurred and log.migraine_severity:
            entry.migraine_details = {
                "severity": log.migraine_severity,
                "duration_hours": log.migraine_duration_hours,
                "location": log.migraine_location,
                "with_aura": log.migraine_with_aura,
            }
        
        history.append(entry)
    
    return APIResponse(
        success=True,
        data=history
    )


@router.put("/outcome/{user_id}/{date_str}", response_model=APIResponse)
async def update_migraine_outcome(
    user_id: str,
    date_str: str,
    had_migraine: bool,
    severity: int = None,
    duration: float = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Update the migraine outcome for a specific date (next-day verification)
    """
    try:
        log_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    
    # Get log
    result = await db.execute(
        select(DailyLog).where(
            DailyLog.user_id == user_id,
            DailyLog.date == log_date
        )
    )
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Log not found for this date"
        )
    
    # Update outcome
    log.migraine_occurred = had_migraine
    
    if had_migraine:
        log.migraine_severity = severity or 5
        log.migraine_duration_hours = duration or 4
    
    # Check if prediction was correct
    if log.predicted_probability is not None:
        predicted_migraine = log.predicted_probability >= 0.5
        log.prediction_was_correct = predicted_migraine == had_migraine
    
    await db.commit()
    
    return APIResponse(
        success=True,
        message="Outcome updated successfully"
    )


@router.get("/recent/{user_id}", response_model=APIResponse)
async def get_recent_logs(
    user_id: str,
    count: int = 14,
    db: AsyncSession = Depends(get_db)
):
    """
    Get most recent logs for prediction context
    """
    result = await db.execute(
        select(DailyLog)
        .where(DailyLog.user_id == user_id)
        .order_by(desc(DailyLog.date))
        .limit(count)
    )
    logs = result.scalars().all()
    
    # Convert to dict format for prediction service
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
    
    return APIResponse(
        success=True,
        data=log_dicts
    )