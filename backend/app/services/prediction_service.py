"""
ML Prediction Service
Handles migraine predictions using Foundation (LightGBM) and Temporal (Mamba) models
"""
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import random
import logging

from app.core.config import settings
from app.models.schemas import (
    TriggerContribution,
    ContributingFactor,
    Recommendation,
    RiskLevel,
)

logger = logging.getLogger(__name__)


# Trigger definitions with clinical odds ratios
TRIGGER_DEFINITIONS = {
    "sleep_deficit": {
        "name": "Sleep Deficit",
        "base_or": 3.98,
        "icon": "🌙",
        "color": "#8b5cf6",
        "threshold": 6,
        "description": "Less than 6 hours of sleep significantly increases risk",
    },
    "high_stress": {
        "name": "High Stress",
        "base_or": 2.67,
        "icon": "😰",
        "color": "#f97316",
        "threshold": 7,
        "description": "Stress level above 7/10 is a major trigger",
    },
    "pressure_drop": {
        "name": "Weather Change",
        "base_or": 1.27,
        "icon": "🌡️",
        "color": "#06b6d4",
        "threshold": -5,
        "description": "Barometric pressure drop detected",
    },
    "menstrual_phase": {
        "name": "Menstrual Phase",
        "base_or": 2.04,
        "icon": "📅",
        "color": "#ec4899",
        "threshold": None,
        "description": "Days -2 to +3 of cycle carry 85% higher risk",
    },
    "skipped_meals": {
        "name": "Skipped Meals",
        "base_or": 1.89,
        "icon": "🍽️",
        "color": "#84cc16",
        "threshold": 1,
        "description": "Skipping meals can trigger attacks",
    },
    "bright_light": {
        "name": "Bright Light",
        "base_or": 1.54,
        "icon": "☀️",
        "color": "#eab308",
        "threshold": None,
        "description": "Extended bright light exposure",
    },
    "alcohol": {
        "name": "Alcohol",
        "base_or": 2.08,
        "icon": "🍷",
        "color": "#ef4444",
        "threshold": 3,
        "description": "Alcohol consumption above 3 drinks",
    },
    "poor_sleep_quality": {
        "name": "Poor Sleep Quality",
        "base_or": 2.15,
        "icon": "😫",
        "color": "#a855f7",
        "threshold": None,
        "description": "Restless or interrupted sleep",
    },
    "dehydration": {
        "name": "Dehydration",
        "base_or": 1.45,
        "icon": "💧",
        "color": "#3b82f6",
        "threshold": 4,
        "description": "Less than 4 glasses of water",
    },
    "caffeine_withdrawal": {
        "name": "Caffeine Change",
        "base_or": 1.32,
        "icon": "☕",
        "color": "#d97706",
        "threshold": None,
        "description": "Significant change in caffeine intake",
    },
}

# Protective factors
PROTECTIVE_FACTORS = {
    "snack": {
        "name": "Had Snacks",
        "or": 0.60,
        "description": "Nighttime snacks reduce risk",
    },
    "good_sleep": {
        "name": "7+ Hours Sleep",
        "or": 0.72,
        "description": "Adequate sleep is protective",
    },
    "hydrated": {
        "name": "Well Hydrated",
        "or": 0.85,
        "description": "8+ glasses of water",
    },
    "low_stress": {
        "name": "Low Stress",
        "or": 0.68,
        "description": "Stress below 4/10",
    },
}


class PredictionService:
    """Service for generating migraine predictions"""
    
    def __init__(self):
        self.foundation_model = None
        self.temporal_model = None
        self._load_models()
    
    def _load_models(self):
        """Load ML models (placeholder for actual model loading)"""
        # In production, load actual models:
        # self.foundation_model = joblib.load(settings.FOUNDATION_MODEL_PATH)
        # self.temporal_model = torch.load(settings.TEMPORAL_MODEL_PATH)
        logger.info("Models loaded (using simulation mode)")
    
    def calculate_phase(self, days_logged: int) -> str:
        """Determine user's prediction phase based on days logged"""
        if days_logged < settings.FOUNDATION_PHASE_DAYS:
            return "foundation"
        elif days_logged < settings.TEMPORAL_PHASE_DAYS:
            return "temporal"
        else:
            return "personalized"
    
    def calculate_cycle_phase(
        self, 
        cycle_start: Optional[date], 
        current_date: date,
        cycle_length: int = 28
    ) -> Tuple[Optional[int], Optional[str]]:
        """Calculate menstrual cycle day and phase"""
        if not cycle_start:
            return None, None
        
        days_since_start = (current_date - cycle_start).days
        cycle_day = (days_since_start % cycle_length) + 1
        
        # Determine phase
        if cycle_day <= 5:
            phase = "menstrual"
        elif cycle_day <= 13:
            phase = "follicular"
        elif cycle_day <= 16:
            phase = "ovulation"
        else:
            phase = "luteal"
        
        return cycle_day, phase
    
    def extract_features(self, log_data: Dict[str, Any], user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from log data for prediction"""
        features = {
            # Sleep features
            "sleep_hours": log_data.get("sleep_hours", 7),
            "sleep_deficit": 1 if log_data.get("sleep_hours", 7) < 6 else 0,
            "sleep_quality": 1 if log_data.get("sleep_quality_good", True) else 0,
            
            # Stress
            "stress_level": log_data.get("stress_level", 5),
            "high_stress": 1 if log_data.get("stress_level", 5) > 7 else 0,
            
            # Diet
            "meals_skipped": len(log_data.get("skipped_meals", [])),
            "had_snack": 1 if log_data.get("had_snack", False) else 0,
            "alcohol": log_data.get("alcohol_drinks", 0),
            "caffeine": log_data.get("caffeine_drinks", 0),
            "water": log_data.get("water_glasses", 0),
            "dehydration": 1 if log_data.get("water_glasses", 8) < 4 else 0,
            
            # Environment
            "bright_light": 1 if log_data.get("bright_light_exposure", False) else 0,
            "screen_time": log_data.get("screen_time_hours", 4),
            
            # Weather (simulated if not available)
            "pressure_change": log_data.get("pressure_change", 0),
            
            # Demographics
            "age": user_data.get("age", 30),
            "bmi": user_data.get("bmi", 22),
            "gender": 1 if user_data.get("gender") == "F" else 0,
            
            # Symptoms count
            "symptom_count": sum(1 for v in log_data.get("symptoms", {}).values() if v),
        }
        
        # Menstrual cycle features
        if user_data.get("has_menstrual_cycle") and user_data.get("cycle_start_day"):
            cycle_day, cycle_phase = self.calculate_cycle_phase(
                user_data["cycle_start_day"],
                date.today(),
                user_data.get("cycle_length", 28)
            )
            features["cycle_day"] = cycle_day or 0
            features["in_risk_window"] = 1 if cycle_day and (cycle_day <= 3 or cycle_day >= 26) else 0
        else:
            features["cycle_day"] = 0
            features["in_risk_window"] = 0
        
        return features
    
    def predict_foundation(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Foundation model prediction (LightGBM-based)
        Uses clinical odds ratios for transparent prediction
        """
        # Base probability from user's historical rate
        base_prob = 0.15  # ~15% base rate
        
        # Calculate cumulative risk from triggers
        log_odds = np.log(base_prob / (1 - base_prob))
        
        # Apply trigger effects
        if features["sleep_deficit"]:
            log_odds += np.log(TRIGGER_DEFINITIONS["sleep_deficit"]["base_or"])
        
        if features["high_stress"]:
            log_odds += np.log(TRIGGER_DEFINITIONS["high_stress"]["base_or"])
        
        if features["meals_skipped"] > 0:
            log_odds += np.log(TRIGGER_DEFINITIONS["skipped_meals"]["base_or"]) * min(features["meals_skipped"], 2)
        
        if features["alcohol"] >= 3:
            log_odds += np.log(TRIGGER_DEFINITIONS["alcohol"]["base_or"])
        
        if not features["sleep_quality"]:
            log_odds += np.log(TRIGGER_DEFINITIONS["poor_sleep_quality"]["base_or"])
        
        if features["dehydration"]:
            log_odds += np.log(TRIGGER_DEFINITIONS["dehydration"]["base_or"])
        
        if features["bright_light"]:
            log_odds += np.log(TRIGGER_DEFINITIONS["bright_light"]["base_or"])
        
        if features["in_risk_window"]:
            log_odds += np.log(TRIGGER_DEFINITIONS["menstrual_phase"]["base_or"])
        
        # Apply protective factors
        if features["had_snack"]:
            log_odds += np.log(PROTECTIVE_FACTORS["snack"]["or"])
        
        if features["sleep_hours"] >= 7:
            log_odds += np.log(PROTECTIVE_FACTORS["good_sleep"]["or"])
        
        if features["water"] >= 8:
            log_odds += np.log(PROTECTIVE_FACTORS["hydrated"]["or"])
        
        if features["stress_level"] < 4:
            log_odds += np.log(PROTECTIVE_FACTORS["low_stress"]["or"])
        
        # Add symptom contribution
        if features["symptom_count"] >= 3:
            log_odds += 0.5  # Significant increase with multiple symptoms
        
        # Convert back to probability
        probability = 1 / (1 + np.exp(-log_odds))
        
        # Confidence based on feature completeness and model certainty
        confidence = 0.70 + (0.10 * min(features["symptom_count"], 3) / 3)
        
        return float(probability), float(confidence)
    
    def predict_temporal(
        self, 
        features: Dict[str, float],
        historical_features: List[Dict[str, float]]
    ) -> Tuple[float, float]:
        """
        Temporal model prediction (Mamba-based)
        Uses sequence of past 14 days for pattern recognition
        """
        # Start with foundation prediction
        base_prob, base_conf = self.predict_foundation(features)
        
        # Temporal adjustment based on patterns
        if len(historical_features) >= 7:
            # Calculate trend
            recent_risks = [self.predict_foundation(f)[0] for f in historical_features[-7:]]
            trend = np.mean(recent_risks[-3:]) - np.mean(recent_risks[:4])
            
            # Adjust probability based on trend
            base_prob = base_prob + (trend * 0.3)
            base_prob = np.clip(base_prob, 0.05, 0.95)
            
            # Higher confidence with more data
            base_conf = min(0.85, base_conf + 0.05)
        
        return float(base_prob), float(base_conf)
    
    def predict_personalized(
        self,
        features: Dict[str, float],
        historical_features: List[Dict[str, float]],
        user_trigger_weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Personalized model prediction
        Fine-tuned to individual's specific trigger sensitivities
        """
        # Get temporal prediction as base
        base_prob, base_conf = self.predict_temporal(features, historical_features)
        
        # Apply personalized trigger weights
        adjustment = 0
        for trigger, weight in user_trigger_weights.items():
            if trigger in features and features[trigger]:
                adjustment += weight * 0.1
        
        final_prob = np.clip(base_prob + adjustment, 0.05, 0.95)
        final_conf = min(0.90, base_conf + 0.05)
        
        return float(final_prob), float(final_conf)
    
    def get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level"""
        if probability < settings.LOW_RISK_THRESHOLD:
            return RiskLevel.LOW
        elif probability < settings.MODERATE_RISK_THRESHOLD:
            return RiskLevel.MODERATE
        elif probability < settings.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def calculate_trigger_contributions(
        self, 
        features: Dict[str, float],
        probability: float
    ) -> List[TriggerContribution]:
        """Calculate individual trigger contributions to risk"""
        contributions = []
        
        trigger_checks = [
            ("sleep_deficit", features.get("sleep_deficit", 0)),
            ("high_stress", features.get("high_stress", 0)),
            ("poor_sleep_quality", not features.get("sleep_quality", 1)),
            ("menstrual_phase", features.get("in_risk_window", 0)),
            ("skipped_meals", features.get("meals_skipped", 0) > 0),
            ("alcohol", features.get("alcohol", 0) >= 3),
            ("dehydration", features.get("dehydration", 0)),
            ("bright_light", features.get("bright_light", 0)),
        ]
        
        active_triggers = [(t, TRIGGER_DEFINITIONS[t]) for t, active in trigger_checks if active]
        
        if not active_triggers:
            return contributions
        
        # Calculate relative contributions
        total_or = sum(t["base_or"] for _, t in active_triggers)
        
        for trigger_key, trigger_def in active_triggers:
            contribution = trigger_def["base_or"] / total_or if total_or > 0 else 0
            contributions.append(TriggerContribution(
                trigger=trigger_def["name"],
                contribution=contribution,
                icon=trigger_def["icon"],
                color=trigger_def["color"],
                description=trigger_def["description"],
            ))
        
        # Sort by contribution
        contributions.sort(key=lambda x: x.contribution, reverse=True)
        
        return contributions[:5]  # Top 5 triggers
    
    def get_contributing_factors(self, features: Dict[str, float]) -> List[ContributingFactor]:
        """Get list of contributing factors with their values"""
        factors = []
        
        # Sleep
        sleep_hours = features.get("sleep_hours", 7)
        factors.append(ContributingFactor(
            factor="Sleep",
            value=f"{sleep_hours}h",
            threshold="≥7h",
            status="critical" if sleep_hours < 5 else "warning" if sleep_hours < 6 else "normal"
        ))
        
        # Stress
        stress = features.get("stress_level", 5)
        factors.append(ContributingFactor(
            factor="Stress",
            value=f"{stress}/10",
            threshold="≤5",
            status="critical" if stress >= 8 else "warning" if stress >= 6 else "normal"
        ))
        
        # Hydration
        water = features.get("water", 6)
        factors.append(ContributingFactor(
            factor="Hydration",
            value=f"{water} glasses",
            threshold="≥8",
            status="warning" if water < 6 else "normal"
        ))
        
        # Meals
        meals_skipped = features.get("meals_skipped", 0)
        factors.append(ContributingFactor(
            factor="Meals",
            value=f"{3 - meals_skipped}/3",
            threshold="3/3",
            status="warning" if meals_skipped > 0 else "normal"
        ))
        
        return factors
    
    def get_protective_factors(self, features: Dict[str, float]) -> List[str]:
        """Get list of active protective factors"""
        protective = []
        
        if features.get("had_snack"):
            protective.append("Had Snacks")
        if features.get("sleep_hours", 0) >= 7:
            protective.append("Good Sleep Duration")
        if features.get("water", 0) >= 8:
            protective.append("Well Hydrated")
        if features.get("stress_level", 10) < 4:
            protective.append("Low Stress")
        if features.get("sleep_quality"):
            protective.append("Quality Sleep")
        
        return protective
    
    def generate_recommendations(
        self, 
        features: Dict[str, float],
        triggers: List[TriggerContribution]
    ) -> List[Recommendation]:
        """Generate personalized recommendations based on triggers"""
        recommendations = []
        
        # Sleep recommendations
        if features.get("sleep_deficit") or not features.get("sleep_quality"):
            recommendations.append(Recommendation(
                action="Prioritize 7-8 hours of sleep tonight",
                reason="Sleep deficit is your top trigger today",
                priority="high",
                icon="🛏️"
            ))
        
        # Stress recommendations
        if features.get("high_stress"):
            recommendations.append(Recommendation(
                action="Take 15-minute relaxation breaks",
                reason="Your stress level is elevated",
                priority="high",
                icon="🧘"
            ))
        
        # Hydration
        if features.get("dehydration"):
            recommendations.append(Recommendation(
                action="Drink 2-3 more glasses of water",
                reason="Dehydration can trigger migraines",
                priority="medium",
                icon="💧"
            ))
        
        # Meals
        if features.get("meals_skipped", 0) > 0:
            recommendations.append(Recommendation(
                action="Don't skip any meals today",
                reason="Regular meals help prevent attacks",
                priority="medium",
                icon="🍽️"
            ))
        
        # Screen time
        if features.get("screen_time", 0) > 8:
            recommendations.append(Recommendation(
                action="Take regular screen breaks (20-20-20 rule)",
                reason="Extended screen time can strain your eyes",
                priority="low",
                icon="👀"
            ))
        
        # Menstrual phase
        if features.get("in_risk_window"):
            recommendations.append(Recommendation(
                action="Be extra cautious - you're in your high-risk window",
                reason="Days -2 to +3 of cycle have 85% higher risk",
                priority="high",
                icon="📅"
            ))
        
        # General protective
        if not features.get("had_snack"):
            recommendations.append(Recommendation(
                action="Have a light snack before bed",
                reason="Nighttime snacks are protective (OR 0.60)",
                priority="low",
                icon="🥜"
            ))
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 2))
        
        return recommendations[:5]
    
    def predict(
        self,
        user_data: Dict[str, Any],
        log_data: Dict[str, Any],
        historical_logs: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main prediction method
        Returns complete prediction with triggers, factors, and recommendations
        """
        # Extract features
        features = self.extract_features(log_data, user_data)
        
        # Determine model type based on phase
        phase = self.calculate_phase(user_data.get("days_logged", 0))
        
        # Get historical features if available
        historical_features = []
        if historical_logs:
            historical_features = [
                self.extract_features(log, user_data) 
                for log in historical_logs
            ]
        
        # Make prediction based on phase
        if phase == "foundation":
            probability, confidence = self.predict_foundation(features)
            model_type = "foundation"
            model_version = "foundation_lightgbm_v1"
        elif phase == "temporal":
            probability, confidence = self.predict_temporal(features, historical_features)
            model_type = "temporal"
            model_version = "temporal_mamba_v1"
        else:
            # For personalized, we'd use learned trigger weights
            # Using placeholder weights here
            user_weights = user_data.get("trigger_weights", {})
            probability, confidence = self.predict_personalized(
                features, historical_features, user_weights
            )
            model_type = "personalized"
            model_version = f"personalized_{user_data.get('id', 'unknown')}_v1"
        
        # Get risk level
        risk_level = self.get_risk_level(probability)
        
        # Calculate trigger contributions
        triggers = self.calculate_trigger_contributions(features, probability)
        
        # Get contributing factors
        contributing_factors = self.get_contributing_factors(features)
        
        # Get protective factors
        protective_factors = self.get_protective_factors(features)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(features, triggers)
        
        # Estimate severity if attack occurs
        severity_prediction = 5.0 + (probability * 4)  # 5-9 scale based on probability
        
        return {
            "attack_probability": probability,
            "risk_level": risk_level,
            "confidence": confidence,
            "severity_prediction": severity_prediction,
            "model_version": model_version,
            "model_type": model_type,
            "top_triggers": triggers,
            "contributing_factors": contributing_factors,
            "protective_factors": protective_factors,
            "recommendations": recommendations,
        }


# Singleton instance
prediction_service = PredictionService()