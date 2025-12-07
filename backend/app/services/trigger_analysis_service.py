"""
Trigger Analysis Service - FIXED
Analyzes user's migraine triggers and patterns
Shows risk factors based on logged data, even without migraine occurrences
"""
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter
import logging

from app.models.schemas import TriggerInfo, PatternInfo

logger = logging.getLogger(__name__)


class TriggerAnalysisService:
    """Service for analyzing migraine triggers and patterns"""
    
    def __init__(self):
        self.min_logs_for_analysis = 1  # Changed from 7 - show data even with 1 log
    
    def calculate_odds_ratio(
        self, 
        trigger_present_migraine: int,
        trigger_present_no_migraine: int,
        trigger_absent_migraine: int,
        trigger_absent_no_migraine: int
    ) -> float:
        """Calculate odds ratio for a trigger"""
        # Add small constant to avoid division by zero
        eps = 0.5
        
        a = trigger_present_migraine + eps
        b = trigger_present_no_migraine + eps
        c = trigger_absent_migraine + eps
        d = trigger_absent_no_migraine + eps
        
        odds_ratio = (a * d) / (b * c)
        return round(odds_ratio, 2)
    
    def analyze_trigger(
        self,
        logs: List[Dict[str, Any]],
        trigger_key: str,
        trigger_check: callable,
        base_risk: float = 1.5  # Default risk multiplier from literature
    ) -> Dict[str, Any]:
        """Analyze a single trigger across all logs"""
        trigger_present_migraine = 0
        trigger_present_no_migraine = 0
        trigger_absent_migraine = 0
        trigger_absent_no_migraine = 0
        trigger_occurrences = 0
        
        for log in logs:
            trigger_active = trigger_check(log)
            if trigger_active:
                trigger_occurrences += 1
            
            # If migraine_occurred is None, we still count trigger occurrences
            # but can't calculate correlation
            if log.get("migraine_occurred") is None:
                continue
                
            migraine = log.get("migraine_occurred", False)
            
            if trigger_active and migraine:
                trigger_present_migraine += 1
            elif trigger_active and not migraine:
                trigger_present_no_migraine += 1
            elif not trigger_active and migraine:
                trigger_absent_migraine += 1
            else:
                trigger_absent_no_migraine += 1
        
        total_migraines = trigger_present_migraine + trigger_absent_migraine
        total_logs = len(logs)
        
        # Calculate odds ratio if we have migraine data
        if total_migraines > 0:
            odds_ratio = self.calculate_odds_ratio(
                trigger_present_migraine,
                trigger_present_no_migraine,
                trigger_absent_migraine,
                trigger_absent_no_migraine
            )
            contribution = trigger_present_migraine / total_migraines if total_migraines > 0 else 0
        else:
            # No migraines yet - use literature-based risk and occurrence frequency
            # Show how often this trigger is present (potential risk factor)
            odds_ratio = base_risk  # Use literature-based default
            # Contribution = how often this trigger occurs relative to other triggers
            contribution = trigger_occurrences / total_logs if total_logs > 0 else 0
        
        return {
            "odds_ratio": odds_ratio,
            "occurrences": trigger_occurrences,
            "contribution": contribution,
            "migraine_with_trigger": trigger_present_migraine,
            "total_with_trigger": trigger_occurrences,
        }
    
    def analyze_all_triggers(self, logs: List[Dict[str, Any]]) -> List[TriggerInfo]:
        """Analyze all triggers for a user"""
        triggers = []
        
        if not logs:
            return triggers
        
        # Define trigger checks with literature-based risk multipliers
        trigger_definitions = [
            {
                "key": "sleep_deficit",
                "name": "Sleep Deficit",
                "check": lambda l: l.get("sleep_hours", 7) < 6,
                "icon": "🌙",
                "description": "Less than 6 hours of sleep",
                "base_risk": 3.98  # OR from migraine literature
            },
            {
                "key": "high_stress",
                "name": "High Stress",
                "check": lambda l: l.get("stress_level", 5) >= 7,
                "icon": "😰",
                "description": "Stress level 7 or higher",
                "base_risk": 2.7
            },
            {
                "key": "poor_sleep",
                "name": "Poor Sleep Quality",
                "check": lambda l: not l.get("sleep_quality_good", True),
                "icon": "😫",
                "description": "Restless or interrupted sleep",
                "base_risk": 2.5
            },
            {
                "key": "skipped_meals",
                "name": "Skipped Meals",
                "check": lambda l: len(l.get("skipped_meals", [])) > 0,
                "icon": "🍽️",
                "description": "One or more meals skipped",
                "base_risk": 2.1
            },
            {
                "key": "alcohol",
                "name": "Alcohol",
                "check": lambda l: l.get("alcohol_drinks", 0) >= 2,  # Lowered threshold
                "icon": "🍷",
                "description": "2 or more alcoholic drinks",
                "base_risk": 3.0
            },
            {
                "key": "dehydration",
                "name": "Dehydration",
                "check": lambda l: l.get("water_glasses", 8) < 5,  # Adjusted threshold
                "icon": "💧",
                "description": "Less than 5 glasses of water",
                "base_risk": 1.8
            },
            {
                "key": "bright_light",
                "name": "Bright Light",
                "check": lambda l: l.get("bright_light_exposure", False),
                "icon": "☀️",
                "description": "Extended bright light exposure",
                "base_risk": 2.0
            },
            {
                "key": "high_screen_time",
                "name": "High Screen Time",
                "check": lambda l: l.get("screen_time_hours", 4) > 6,  # Lowered threshold
                "icon": "📱",
                "description": "More than 6 hours of screen time",
                "base_risk": 1.6
            },
            {
                "key": "caffeine_high",
                "name": "High Caffeine",
                "check": lambda l: l.get("caffeine_drinks", 2) >= 4,  # Lowered threshold
                "icon": "☕",
                "description": "4 or more caffeinated drinks",
                "base_risk": 1.5
            },
            {
                "key": "multiple_symptoms",
                "name": "Multiple Symptoms",
                "check": lambda l: sum(1 for v in l.get("symptoms", {}).values() if v) >= 2,
                "icon": "⚠️",
                "description": "2 or more prodromal symptoms",
                "base_risk": 4.0
            },
        ]
        
        for trigger_def in trigger_definitions:
            result = self.analyze_trigger(
                logs,
                trigger_def["key"],
                trigger_def["check"],
                trigger_def.get("base_risk", 1.5)
            )
            
            # Include triggers that have occurred at least once
            if result["occurrences"] > 0:
                triggers.append(TriggerInfo(
                    name=trigger_def["name"],
                    odds_ratio=result["odds_ratio"],
                    contribution=result["contribution"],
                    occurrences=result["occurrences"],
                    icon=trigger_def["icon"],
                    description=trigger_def["description"]
                ))
        
        # Sort by odds ratio (risk level)
        triggers.sort(key=lambda x: x.odds_ratio, reverse=True)
        
        # Calculate contribution based on odds ratio weight (not just occurrences)
        # Higher OR = higher contribution to overall risk
        total_or = sum(t.odds_ratio for t in triggers)
        if total_or > 0:
            for trigger in triggers:
                # Weight contribution by odds ratio
                trigger.contribution = trigger.odds_ratio / total_or
        
        return triggers
    
    def discover_patterns(self, logs: List[Dict[str, Any]]) -> List[PatternInfo]:
        """Discover patterns in user's migraine data"""
        patterns = []
        
        if not logs:
            return patterns
        
        total_logs = len(logs)
        migraine_logs = [l for l in logs if l.get("migraine_occurred")]
        
        # Always analyze logged risk factors (regardless of migraines)
        poor_sleep_days = sum(1 for l in logs if l.get("sleep_hours", 7) < 6)
        high_stress_days = sum(1 for l in logs if l.get("stress_level", 5) >= 7)
        poor_quality_sleep = sum(1 for l in logs if not l.get("sleep_quality_good", True))
        dehydration_days = sum(1 for l in logs if l.get("water_glasses", 8) < 5)
        skipped_meal_days = sum(1 for l in logs if len(l.get("skipped_meals", [])) > 0)
        alcohol_days = sum(1 for l in logs if l.get("alcohol_drinks", 0) >= 2)
        
        # Combined risk factors
        sleep_stress_combo = sum(1 for l in logs if l.get("sleep_hours", 7) < 6 and l.get("stress_level", 5) >= 7)
        morning_cascade = sum(1 for l in logs if (l.get("sleep_hours", 7) < 6 or not l.get("sleep_quality_good", True)) and "breakfast" in l.get("skipped_meals", []))
        
        # Pattern 1: Sleep issues
        if poor_sleep_days > 0 or poor_quality_sleep > 0:
            sleep_issue_days = max(poor_sleep_days, poor_quality_sleep)
            pct = round(sleep_issue_days / total_logs * 100)
            patterns.append(PatternInfo(
                title="Sleep Issues Detected",
                description=f"Poor sleep on {sleep_issue_days} of {total_logs} days ({pct}%). Sleep deficit is the #1 migraine trigger with 3.98x risk increase.",
                icon="🌙",
                confidence=min(0.9, 0.5 + (sleep_issue_days / total_logs) * 0.4)
            ))
        
        # Pattern 2: High stress
        if high_stress_days > 0:
            pct = round(high_stress_days / total_logs * 100)
            patterns.append(PatternInfo(
                title="Elevated Stress Levels",
                description=f"High stress (7+/10) on {high_stress_days} of {total_logs} days ({pct}%). Stress is a major trigger with 2.7x risk increase.",
                icon="😰",
                confidence=min(0.85, 0.5 + (high_stress_days / total_logs) * 0.35)
            ))
        
        # Pattern 3: Sleep + Stress combo (highest risk)
        if sleep_stress_combo > 0:
            patterns.append(PatternInfo(
                title="⚠️ High Risk Combination",
                description=f"Poor sleep AND high stress occurred together on {sleep_stress_combo} days. This combination dramatically increases migraine risk.",
                icon="🔥",
                confidence=0.9
            ))
        
        # Pattern 4: Dehydration
        if dehydration_days > 0:
            pct = round(dehydration_days / total_logs * 100)
            patterns.append(PatternInfo(
                title="Hydration Warning",
                description=f"Low water intake (<5 glasses) on {dehydration_days} of {total_logs} days ({pct}%). Aim for 8+ glasses daily.",
                icon="💧",
                confidence=min(0.75, 0.5 + (dehydration_days / total_logs) * 0.25)
            ))
        
        # Pattern 5: Skipped meals
        if skipped_meal_days > 0:
            pct = round(skipped_meal_days / total_logs * 100)
            patterns.append(PatternInfo(
                title="Meal Skipping Pattern",
                description=f"Meals skipped on {skipped_meal_days} of {total_logs} days ({pct}%). Regular meals help prevent migraines.",
                icon="🍽️",
                confidence=min(0.7, 0.5 + (skipped_meal_days / total_logs) * 0.2)
            ))
        
        # Pattern 6: Morning cascade
        if morning_cascade > 0:
            patterns.append(PatternInfo(
                title="Morning Risk Pattern",
                description=f"Poor sleep + skipped breakfast on {morning_cascade} days. This morning combination often triggers attacks.",
                icon="🌅",
                confidence=0.8
            ))
        
        # Pattern 7: Alcohol
        if alcohol_days > 0:
            pct = round(alcohol_days / total_logs * 100)
            patterns.append(PatternInfo(
                title="Alcohol Trigger",
                description=f"Alcohol consumption (2+ drinks) on {alcohol_days} of {total_logs} days ({pct}%). Alcohol is a common trigger.",
                icon="🍷",
                confidence=min(0.75, 0.5 + (alcohol_days / total_logs) * 0.25)
            ))
        
        # If we have migraine data, add migraine-specific patterns
        if migraine_logs:
            # Day of week pattern
            weekday_counts = Counter()
            for log in migraine_logs:
                log_date = log.get("date")
                if isinstance(log_date, str):
                    log_date = datetime.strptime(log_date, "%Y-%m-%d").date()
                if log_date:
                    weekday_counts[log_date.weekday()] += 1
            
            if weekday_counts:
                most_common_day = weekday_counts.most_common(1)[0]
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                patterns.append(PatternInfo(
                    title=f"{day_names[most_common_day[0]]} Migraines",
                    description=f"{most_common_day[1]} of your migraines occurred on {day_names[most_common_day[0]]}s",
                    icon="📅",
                    confidence=min(0.9, 0.5 + (most_common_day[1] / max(len(migraine_logs), 1)) * 0.4)
                ))
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns[:5]  # Return top 5 patterns
    
    def get_weekly_accuracy(self, logs: List[Dict[str, Any]], weeks: int = 8) -> List[Dict[str, Any]]:
        """Calculate weekly prediction accuracy"""
        weekly_data = []
        
        # Group logs by week
        logs_with_predictions = [
            l for l in logs 
            if l.get("predicted_probability") is not None and l.get("migraine_occurred") is not None
        ]
        
        if not logs_with_predictions:
            # No validated predictions yet - show 0% (not enough data)
            current_week = date.today().isocalendar()[1]
            return [{
                "week": f"W{current_week}",
                "accuracy": 0  # 0 = no data to calculate, NOT 100%
            }]
        
        # Sort by date
        logs_with_predictions.sort(key=lambda x: x.get("date", ""))
        
        # Calculate weekly accuracy
        week_logs = {}
        for log in logs_with_predictions:
            log_date = log.get("date")
            if isinstance(log_date, str):
                log_date = datetime.strptime(log_date, "%Y-%m-%d").date()
            
            week_num = log_date.isocalendar()[1]
            week_key = f"W{week_num}"
            
            if week_key not in week_logs:
                week_logs[week_key] = {"correct": 0, "total": 0}
            
            # Check if prediction was correct
            prob = log.get("predicted_probability", 0.5)
            predicted_migraine = prob >= 0.5
            actual_migraine = log.get("migraine_occurred", False)
            
            week_logs[week_key]["total"] += 1
            if predicted_migraine == actual_migraine:
                week_logs[week_key]["correct"] += 1
        
        # Convert to list
        for week, data in week_logs.items():
            if data["total"] > 0:
                accuracy = (data["correct"] / data["total"]) * 100
                weekly_data.append({
                    "week": week,
                    "accuracy": round(accuracy, 1)
                })
        
        if not weekly_data:
            current_week = date.today().isocalendar()[1]
            return [{"week": f"W{current_week}", "accuracy": 0}]
        
        return weekly_data[-weeks:]
    
    def calculate_stats(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics"""
        total_logs = len(logs)
        total_migraines = sum(1 for l in logs if l.get("migraine_occurred"))
        
        # Prediction accuracy - only calculate if we have validated predictions
        logs_with_predictions = [
            l for l in logs 
            if l.get("prediction_was_correct") is not None
        ]
        
        if logs_with_predictions:
            correct_predictions = sum(1 for l in logs_with_predictions if l.get("prediction_was_correct"))
            accuracy = correct_predictions / len(logs_with_predictions)
        else:
            accuracy = 0  # No validated predictions yet
        
        # Streak (consecutive days logged)
        streak = 0
        if logs:
            sorted_logs = sorted(logs, key=lambda x: x.get("date", ""), reverse=True)
            today = date.today()
            for i, log in enumerate(sorted_logs):
                log_date = log.get("date")
                if isinstance(log_date, str):
                    log_date = datetime.strptime(log_date, "%Y-%m-%d").date()
                
                expected_date = today - timedelta(days=i)
                if log_date == expected_date:
                    streak += 1
                else:
                    break
        
        return {
            "total_logs": total_logs,
            "total_migraines": total_migraines,
            "prediction_accuracy": accuracy,
            "streak_days": streak,
        }


# Singleton instance
trigger_analysis_service = TriggerAnalysisService()
