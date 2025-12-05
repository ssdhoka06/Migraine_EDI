"""
Trigger Analysis Service
Analyzes user's migraine triggers and patterns
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
        self.min_logs_for_analysis = 7
    
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
        trigger_check: callable
    ) -> Dict[str, Any]:
        """Analyze a single trigger across all logs"""
        trigger_present_migraine = 0
        trigger_present_no_migraine = 0
        trigger_absent_migraine = 0
        trigger_absent_no_migraine = 0
        
        for log in logs:
            if log.get("migraine_occurred") is None:
                continue
                
            trigger_active = trigger_check(log)
            migraine = log.get("migraine_occurred", False)
            
            if trigger_active and migraine:
                trigger_present_migraine += 1
            elif trigger_active and not migraine:
                trigger_present_no_migraine += 1
            elif not trigger_active and migraine:
                trigger_absent_migraine += 1
            else:
                trigger_absent_no_migraine += 1
        
        odds_ratio = self.calculate_odds_ratio(
            trigger_present_migraine,
            trigger_present_no_migraine,
            trigger_absent_migraine,
            trigger_absent_no_migraine
        )
        
        occurrences = trigger_present_migraine + trigger_present_no_migraine
        total_migraines = trigger_present_migraine + trigger_absent_migraine
        
        # Calculate contribution (relative importance)
        contribution = 0
        if total_migraines > 0:
            contribution = trigger_present_migraine / total_migraines
        
        return {
            "odds_ratio": odds_ratio,
            "occurrences": occurrences,
            "contribution": contribution,
            "migraine_with_trigger": trigger_present_migraine,
            "total_with_trigger": occurrences,
        }
    
    def analyze_all_triggers(self, logs: List[Dict[str, Any]]) -> List[TriggerInfo]:
        """Analyze all triggers for a user"""
        triggers = []
        
        # Define trigger checks
        trigger_definitions = [
            {
                "key": "sleep_deficit",
                "name": "Sleep Deficit",
                "check": lambda l: l.get("sleep_hours", 7) < 6,
                "icon": "🌙",
                "description": "Less than 6 hours of sleep"
            },
            {
                "key": "high_stress",
                "name": "High Stress",
                "check": lambda l: l.get("stress_level", 5) >= 7,
                "icon": "😰",
                "description": "Stress level 7 or higher"
            },
            {
                "key": "poor_sleep",
                "name": "Poor Sleep Quality",
                "check": lambda l: not l.get("sleep_quality_good", True),
                "icon": "😫",
                "description": "Restless or interrupted sleep"
            },
            {
                "key": "skipped_meals",
                "name": "Skipped Meals",
                "check": lambda l: len(l.get("skipped_meals", [])) > 0,
                "icon": "🍽️",
                "description": "One or more meals skipped"
            },
            {
                "key": "alcohol",
                "name": "Alcohol",
                "check": lambda l: l.get("alcohol_drinks", 0) >= 3,
                "icon": "🍷",
                "description": "3 or more alcoholic drinks"
            },
            {
                "key": "dehydration",
                "name": "Dehydration",
                "check": lambda l: l.get("water_glasses", 8) < 4,
                "icon": "💧",
                "description": "Less than 4 glasses of water"
            },
            {
                "key": "bright_light",
                "name": "Bright Light",
                "check": lambda l: l.get("bright_light_exposure", False),
                "icon": "☀️",
                "description": "Extended bright light exposure"
            },
            {
                "key": "high_screen_time",
                "name": "High Screen Time",
                "check": lambda l: l.get("screen_time_hours", 4) > 8,
                "icon": "📱",
                "description": "More than 8 hours of screen time"
            },
            {
                "key": "caffeine_high",
                "name": "High Caffeine",
                "check": lambda l: l.get("caffeine_drinks", 2) >= 5,
                "icon": "☕",
                "description": "5 or more caffeinated drinks"
            },
            {
                "key": "multiple_symptoms",
                "name": "Multiple Symptoms",
                "check": lambda l: sum(1 for v in l.get("symptoms", {}).values() if v) >= 3,
                "icon": "⚠️",
                "description": "3 or more prodromal symptoms"
            },
        ]
        
        for trigger_def in trigger_definitions:
            result = self.analyze_trigger(
                logs,
                trigger_def["key"],
                trigger_def["check"]
            )
            
            # Only include triggers with meaningful data
            if result["occurrences"] > 0:
                triggers.append(TriggerInfo(
                    name=trigger_def["name"],
                    odds_ratio=result["odds_ratio"],
                    contribution=result["contribution"],
                    occurrences=result["occurrences"],
                    icon=trigger_def["icon"],
                    description=trigger_def["description"]
                ))
        
        # Sort by odds ratio
        triggers.sort(key=lambda x: x.odds_ratio, reverse=True)
        
        return triggers
    
    def discover_patterns(self, logs: List[Dict[str, Any]]) -> List[PatternInfo]:
        """Discover patterns in user's migraine data"""
        patterns = []
        
        if len(logs) < self.min_logs_for_analysis:
            return patterns
        
        migraine_logs = [l for l in logs if l.get("migraine_occurred")]
        
        if not migraine_logs:
            return patterns
        
        # Pattern 1: Day of week analysis
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
            if most_common_day[1] >= 2:
                patterns.append(PatternInfo(
                    title=f"{day_names[most_common_day[0]]} Pattern",
                    description=f"You tend to get migraines on {day_names[most_common_day[0]]}s ({most_common_day[1]} occurrences)",
                    icon="📅",
                    confidence=min(0.9, 0.5 + (most_common_day[1] / len(migraine_logs)) * 0.4)
                ))
        
        # Pattern 2: Sleep-stress combination
        sleep_stress_count = sum(
            1 for l in migraine_logs 
            if l.get("sleep_hours", 7) < 6 and l.get("stress_level", 5) >= 7
        )
        if sleep_stress_count >= 2:
            patterns.append(PatternInfo(
                title="Sleep + Stress Combo",
                description=f"Poor sleep combined with high stress preceded {sleep_stress_count} migraines",
                icon="🔥",
                confidence=min(0.85, 0.5 + (sleep_stress_count / len(migraine_logs)) * 0.35)
            ))
        
        # Pattern 3: Weekend pattern
        weekend_migraines = sum(
            1 for l in migraine_logs
            if isinstance(l.get("date"), (str, date)) and (
                (datetime.strptime(l["date"], "%Y-%m-%d").weekday() if isinstance(l["date"], str) else l["date"].weekday()) >= 5
            )
        )
        if weekend_migraines >= 2 and weekend_migraines / len(migraine_logs) > 0.3:
            patterns.append(PatternInfo(
                title="Weekend Warrior",
                description=f"{weekend_migraines} of your migraines occurred on weekends - watch caffeine/sleep changes",
                icon="🏖️",
                confidence=0.7
            ))
        
        # Pattern 4: Morning trigger accumulation
        morning_triggers = sum(
            1 for l in migraine_logs
            if (l.get("sleep_hours", 7) < 6 or not l.get("sleep_quality_good", True))
               and "breakfast" in l.get("skipped_meals", [])
        )
        if morning_triggers >= 2:
            patterns.append(PatternInfo(
                title="Morning Cascade",
                description=f"Poor sleep + skipped breakfast combination found in {morning_triggers} migraine days",
                icon="🌅",
                confidence=min(0.8, 0.5 + (morning_triggers / len(migraine_logs)) * 0.3)
            ))
        
        # Pattern 5: Dehydration pattern
        dehydration_count = sum(
            1 for l in migraine_logs
            if l.get("water_glasses", 8) < 4
        )
        if dehydration_count >= 2:
            patterns.append(PatternInfo(
                title="Dehydration Link",
                description=f"Low water intake preceded {dehydration_count} migraines - aim for 8+ glasses",
                icon="💧",
                confidence=min(0.75, 0.5 + (dehydration_count / len(migraine_logs)) * 0.25)
            ))
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns[:5]
    
    def get_weekly_accuracy(self, logs: List[Dict[str, Any]], weeks: int = 8) -> List[Dict[str, Any]]:
        """Calculate weekly prediction accuracy"""
        weekly_data = []
        
        # Group logs by week
        logs_with_predictions = [
            l for l in logs 
            if l.get("predicted_probability") is not None and l.get("migraine_occurred") is not None
        ]
        
        if not logs_with_predictions:
            # Return simulated data for demo
            for i in range(weeks):
                weekly_data.append({
                    "week": f"W{weeks - i}",
                    "accuracy": 60 + (i * 3) + np.random.randint(-5, 5)
                })
            return weekly_data
        
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
        
        return weekly_data[-weeks:]
    
    def calculate_stats(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics"""
        total_logs = len(logs)
        total_migraines = sum(1 for l in logs if l.get("migraine_occurred"))
        
        # Prediction accuracy
        logs_with_predictions = [
            l for l in logs 
            if l.get("prediction_was_correct") is not None
        ]
        correct_predictions = sum(1 for l in logs_with_predictions if l.get("prediction_was_correct"))
        accuracy = correct_predictions / len(logs_with_predictions) if logs_with_predictions else 0
        
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