"""
Users API Router
Handles user registration, profile management, and onboarding
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, date
import logging

from app.core.database import get_db
from app.models.database import User
from app.models.schemas import (
    OnboardingRequest,
    OnboardingResponse,
    UserProfile,
    APIResponse,
    Phase,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/onboarding", response_model=APIResponse)
async def onboard_user(
    request: OnboardingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user profile during onboarding
    """
    try:
        # Calculate BMI if not provided
        bmi = request.bmi
        if not bmi:
            height_m = request.height / 100
            bmi = round(request.weight / (height_m ** 2), 1)
        
        # Parse cycle start day if provided
        cycle_start = None
        if request.cycle_start_day:
            try:
                cycle_start = datetime.strptime(request.cycle_start_day, "%Y-%m-%d").date()
            except ValueError:
                pass
        
        # Create user
        user = User(
            gender=request.gender.value,
            age=request.age,
            height=request.height,
            weight=request.weight,
            bmi=bmi,
            attacks_per_month=request.attacks_per_month,
            location_city=request.location_city,
            has_menstrual_cycle=request.has_menstrual_cycle,
            cycle_start_day=cycle_start,
            current_phase="foundation",
            days_logged=0,
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"New user created: {user.id}")
        
        return APIResponse(
            success=True,
            data=OnboardingResponse(
                success=True,
                user_id=user.id,
                message="Profile created successfully! Start logging to get predictions.",
                current_phase=Phase.FOUNDATION
            ),
            message="User created successfully"
        )
        
    except Exception as e:
        logger.error(f"Onboarding error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/profile/{user_id}", response_model=APIResponse)
async def get_user_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user profile by ID
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return APIResponse(
        success=True,
        data=UserProfile(
            id=user.id,
            gender=user.gender,
            age=user.age,
            bmi=user.bmi,
            location_city=user.location_city,
            attacks_per_month=user.attacks_per_month,
            has_menstrual_cycle=user.has_menstrual_cycle,
            current_phase=Phase(user.current_phase),
            days_logged=user.days_logged,
            model_version=user.model_version,
            created_at=user.created_at
        )
    )


@router.put("/profile/{user_id}", response_model=APIResponse)
async def update_user_profile(
    user_id: str,
    updates: dict,
    db: AsyncSession = Depends(get_db)
):
    """
    Update user profile
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update allowed fields
    allowed_fields = [
        "age", "height", "weight", "location_city",
        "attacks_per_month", "has_menstrual_cycle", "cycle_start_day"
    ]
    
    for field, value in updates.items():
        if field in allowed_fields and hasattr(user, field):
            setattr(user, field, value)
    
    # Recalculate BMI if height or weight changed
    if "height" in updates or "weight" in updates:
        height_m = user.height / 100
        user.bmi = round(user.weight / (height_m ** 2), 1)
    
    user.updated_at = datetime.utcnow()
    await db.commit()
    
    return APIResponse(
        success=True,
        message="Profile updated successfully"
    )


@router.delete("/profile/{user_id}", response_model=APIResponse)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete user and all associated data
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    await db.delete(user)
    await db.commit()
    
    logger.info(f"User deleted: {user_id}")
    
    return APIResponse(
        success=True,
        message="User deleted successfully"
    )
