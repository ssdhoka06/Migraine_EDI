"""
Phase 3: Continual Learning & Personalization System
=====================================================
Implements the complete user lifecycle:
- Days 1-14: Foundation model (static)
- Days 15-30: Generic temporal model (Mamba)
- Days 31+: Personalized model (fine-tuned per user)

This module provides:
1. User state management
2. Model selection based on user history
3. Weekly personalization pipeline
4. Prediction generation with explanations

Author: Dhoka
Date: December 2025
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from model import MigraineMamba, MigraineModelConfig


class UserPhase(Enum):
    """User lifecycle phases."""
    FOUNDATION = "foundation"      # Days 1-14
    GENERIC_TEMPORAL = "generic"   # Days 15-30
    PERSONALIZED = "personalized"  # Days 31+


@dataclass
class UserState:
    """Tracks user's current state in the system."""
    user_id: str
    days_logged: int
    total_attacks: int
    last_log_date: str
    current_phase: str
    model_version: str
    personalization_week: int
    created_at: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass  
class DailyPrediction:
    """Structure for daily prediction output."""
    user_id: str
    date: str
    risk_probability: float
    risk_level: str
    severity_prediction: float
    trigger_importance: Dict[str, float]
    recommendations: List[str]
    model_version: str
    confidence: float


class PersonalizationSystem:
    """
    Main system for personalized migraine prediction.
    
    Manages the complete user lifecycle from initial signup
    through personalized predictions.
    """
    
    def __init__(
        self,
        models_dir: str = "../models",
        user_data_dir: str = "../user_data",
        device: str = None,
    ):
        self.models_dir = Path(models_dir)
        self.user_data_dir = Path(user_data_dir)
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Device selection
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Model config (must match training)
        self.model_config = MigraineModelConfig(
            n_continuous_features=8,
            n_binary_features=6,
            seq_len=14,
            d_model=64,
            n_mamba_layers=2,
            dropout=0.3,
        )
        
        # Load generic model
        self.generic_model = self._load_generic_model()
        
        # Cache for user models
        self._user_model_cache = {}
        
        # Trigger names for explanations
        self.trigger_names = [
            'Sleep', 'Stress', 'Weather', 'Fasting',
            'Alcohol', 'Menstrual', 'Light'
        ]
        
        # Feature names
        self.continuous_features = [
            'sleep_hours', 'stress_level', 'barometric_pressure', 'pressure_change',
            'temperature', 'humidity', 'hours_fasting', 'alcohol_drinks'
        ]
        self.binary_features = [
            'had_breakfast', 'had_lunch', 'had_dinner', 'had_snack',
            'bright_light_exposure', 'sleep_quality'
        ]
        
        print(f"✓ PersonalizationSystem initialized on {self.device}")
    
    def _load_generic_model(self) -> MigraineMamba:
        """Load the generic (pre-trained) Mamba model."""
        model = MigraineMamba(self.model_config)
        
        # Try v2 first, then v1
        model_paths = [
            self.models_dir / "models_v2" / "mamba_finetuned_v2.pth",
            self.models_dir / "mamba_finetuned.pth",
            self.models_dir.parent / "models_v2" / "mamba_finetuned_v2.pth",
            self.models_dir.parent / "models" / "mamba_finetuned.pth",
        ]
        
        for path in model_paths:
            if path.exists():
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                print(f"  ✓ Loaded generic model from {path}")
                return model
        
        print("  ⚠ No pre-trained model found, using random weights")
        model.to(self.device)
        model.eval()
        return model
    
    # =========================================================================
    # USER MANAGEMENT
    # =========================================================================
    
    def get_or_create_user(self, user_id: str) -> UserState:
        """Get existing user state or create new user."""
        state_path = self.user_data_dir / f"{user_id}" / "state.json"
        
        if state_path.exists():
            with open(state_path) as f:
                return UserState.from_dict(json.load(f))
        
        # Create new user
        user_dir = self.user_data_dir / f"{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        state = UserState(
            user_id=user_id,
            days_logged=0,
            total_attacks=0,
            last_log_date="",
            current_phase=UserPhase.FOUNDATION.value,
            model_version="foundation_v1",
            personalization_week=0,
            created_at=datetime.now().isoformat(),
        )
        
        self._save_user_state(state)
        print(f"  ✓ Created new user: {user_id}")
        return state
    
    def _save_user_state(self, state: UserState):
        """Save user state to disk."""
        state_path = self.user_data_dir / f"{state.user_id}" / "state.json"
        with open(state_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
    
    def get_user_phase(self, user_id: str) -> UserPhase:
        """Determine which phase a user is in."""
        state = self.get_or_create_user(user_id)
        
        if state.days_logged < 14:
            return UserPhase.FOUNDATION
        elif state.days_logged < 30:
            return UserPhase.GENERIC_TEMPORAL
        else:
            return UserPhase.PERSONALIZED
    
    def update_user_log(
        self,
        user_id: str,
        date: str,
        features: Dict,
        attack_occurred: Optional[bool] = None,
    ):
        """
        Record a daily log entry for user.
        
        Args:
            user_id: User identifier
            date: Date string (YYYY-MM-DD)
            features: Dict of daily features
            attack_occurred: Whether migraine occurred (can be None if not yet known)
        """
        state = self.get_or_create_user(user_id)
        
        # Save log entry
        logs_dir = self.user_data_dir / f"{user_id}" / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        log_entry = {
            'date': date,
            'features': features,
            'attack_occurred': attack_occurred,
            'logged_at': datetime.now().isoformat(),
        }
        
        with open(logs_dir / f"{date}.json", 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        # Update state
        state.days_logged += 1
        state.last_log_date = date
        if attack_occurred:
            state.total_attacks += 1
        
        # Update phase
        state.current_phase = self.get_user_phase(user_id).value
        
        self._save_user_state(state)
    
    def get_user_history(
        self,
        user_id: str,
        n_days: int = 14,
    ) -> List[Dict]:
        """Get last N days of user logs."""
        logs_dir = self.user_data_dir / f"{user_id}" / "logs"
        
        if not logs_dir.exists():
            return []
        
        # Get all log files
        log_files = sorted(logs_dir.glob("*.json"), reverse=True)
        
        history = []
        for log_file in log_files[:n_days]:
            with open(log_file) as f:
                history.append(json.load(f))
        
        return list(reversed(history))  # Chronological order
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict(self, user_id: str, current_features: Dict) -> DailyPrediction:
        """
        Generate prediction for user based on their phase.
        
        Args:
            user_id: User identifier
            current_features: Today's feature values
            
        Returns:
            DailyPrediction with risk, triggers, and recommendations
        """
        state = self.get_or_create_user(user_id)
        phase = UserPhase(state.current_phase)
        
        if phase == UserPhase.FOUNDATION:
            return self._predict_foundation(user_id, current_features, state)
        elif phase == UserPhase.GENERIC_TEMPORAL:
            return self._predict_generic(user_id, current_features, state)
        else:
            return self._predict_personalized(user_id, current_features, state)
    
    def _predict_foundation(
        self,
        user_id: str,
        features: Dict,
        state: UserState,
    ) -> DailyPrediction:
        """
        Days 1-14: Simple rule-based prediction.
        Uses clinical odds ratios directly.
        """
        # Base rate
        risk = 0.17
        
        # Apply odds ratios
        if features.get('sleep_hours', 7) < 6:
            risk *= 3.98
        if features.get('alcohol_drinks', 0) >= 5:
            risk *= 2.08
        if features.get('menstrual_cycle_day', -1) in [0, 1]:
            risk *= 2.04
        if features.get('pressure_change', 0) < -10:
            risk *= 1.27
        if features.get('stress_level', 5) > 7:
            risk *= 1.5
        if features.get('had_snack', 0) == 1:
            risk *= 0.60
        
        risk = min(risk, 0.95)  # Cap at 95%
        
        # Identify triggers
        triggers = self._identify_triggers(features)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk, triggers, features)
        
        return DailyPrediction(
            user_id=user_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            risk_probability=risk,
            risk_level=self._risk_level(risk),
            severity_prediction=5.0,  # Default
            trigger_importance=triggers,
            recommendations=recommendations,
            model_version="foundation_v1",
            confidence=0.6,  # Lower confidence for rule-based
        )
    
    def _predict_generic(
        self,
        user_id: str,
        current_features: Dict,
        state: UserState,
    ) -> DailyPrediction:
        """
        Days 15-30: Use generic Mamba model.
        """
        # Get history and create sequence
        history = self.get_user_history(user_id, n_days=13)
        sequence = self._create_sequence(history, current_features)
        
        # Run model
        prob, severity, trigger_scores = self._run_model(
            self.generic_model, sequence
        )
        
        triggers = {
            name: float(score) 
            for name, score in zip(self.trigger_names, trigger_scores)
        }
        
        recommendations = self._generate_recommendations(
            prob, triggers, current_features
        )
        
        return DailyPrediction(
            user_id=user_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            risk_probability=prob,
            risk_level=self._risk_level(prob),
            severity_prediction=severity,
            trigger_importance=triggers,
            recommendations=recommendations,
            model_version="generic_mamba_v1",
            confidence=0.75,
        )
    
    def _predict_personalized(
        self,
        user_id: str,
        current_features: Dict,
        state: UserState,
    ) -> DailyPrediction:
        """
        Days 31+: Use personalized model if available.
        """
        # Try to load personalized model
        model = self._get_user_model(user_id)
        
        if model is None:
            # Fall back to generic
            model = self.generic_model
            model_version = "generic_mamba_v1"
            confidence = 0.75
        else:
            model_version = f"personalized_week_{state.personalization_week}"
            confidence = 0.85
        
        # Get history and create sequence
        history = self.get_user_history(user_id, n_days=13)
        sequence = self._create_sequence(history, current_features)
        
        # Run model
        prob, severity, trigger_scores = self._run_model(model, sequence)
        
        triggers = {
            name: float(score)
            for name, score in zip(self.trigger_names, trigger_scores)
        }
        
        recommendations = self._generate_recommendations(
            prob, triggers, current_features
        )
        
        return DailyPrediction(
            user_id=user_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            risk_probability=prob,
            risk_level=self._risk_level(prob),
            severity_prediction=severity,
            trigger_importance=triggers,
            recommendations=recommendations,
            model_version=model_version,
            confidence=confidence,
        )
    
    def _get_user_model(self, user_id: str) -> Optional[MigraineMamba]:
        """Load user's personalized model if it exists."""
        if user_id in self._user_model_cache:
            return self._user_model_cache[user_id]
        
        model_dir = self.user_data_dir / f"{user_id}" / "models"
        if not model_dir.exists():
            return None
        
        # Find latest model
        model_files = sorted(model_dir.glob("*.pth"), reverse=True)
        if not model_files:
            return None
        
        # Load model
        model = MigraineMamba(self.model_config)
        checkpoint = torch.load(model_files[0], map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Cache it
        self._user_model_cache[user_id] = model
        
        return model
    
    def _create_sequence(
        self,
        history: List[Dict],
        current_features: Dict,
    ) -> Dict[str, np.ndarray]:
        """Create 14-day input sequence from history + current day."""
        
        # Default values for missing days
        default_continuous = [7.0, 5.0, 1013.0, 0.0, 22.0, 50.0, 4.0, 0.0]
        default_binary = [1, 1, 1, 0, 0, 1]
        
        # Normalization (should match training)
        means = np.array([7.0, 5.0, 1013.0, 0.0, 20.0, 50.0, 6.0, 0.5])
        stds = np.array([1.5, 2.5, 10.0, 5.0, 10.0, 20.0, 4.0, 1.0])
        stds = np.where(stds == 0, 1.0, stds)
        
        continuous_seq = np.zeros((14, 8))
        binary_seq = np.zeros((14, 6))
        menstrual_seq = np.full(14, -1)
        
        # Fill from history
        for i, entry in enumerate(history[-13:]):  # Last 13 days
            idx = i
            features = entry.get('features', {})
            
            continuous_seq[idx] = [
                features.get(f, default_continuous[j])
                for j, f in enumerate(self.continuous_features)
            ]
            binary_seq[idx] = [
                features.get(f, default_binary[j])
                for j, f in enumerate(self.binary_features)
            ]
            menstrual_seq[idx] = features.get('menstrual_cycle_day', -1)
        
        # Add current day as day 14
        continuous_seq[13] = [
            current_features.get(f, default_continuous[j])
            for j, f in enumerate(self.continuous_features)
        ]
        binary_seq[13] = [
            current_features.get(f, default_binary[j])
            for j, f in enumerate(self.binary_features)
        ]
        menstrual_seq[13] = current_features.get('menstrual_cycle_day', -1)
        
        # Normalize continuous
        continuous_seq = (continuous_seq - means) / stds
        
        # Clinical features
        clinical = np.zeros(8)
        if current_features.get('sleep_hours', 7) < 6:
            clinical[6] = (7 - current_features.get('sleep_hours', 7)) / 3
        clinical[7] = current_features.get('stress_level', 5) / 10
        
        return {
            'continuous': continuous_seq.astype(np.float32),
            'binary': binary_seq.astype(np.float32),
            'menstrual': menstrual_seq.astype(np.int64),
            'day_of_week': (np.arange(14) % 7).astype(np.int64),
            'clinical': clinical.astype(np.float32),
        }
    
    @torch.no_grad()
    def _run_model(
        self,
        model: MigraineMamba,
        sequence: Dict,
    ) -> Tuple[float, float, np.ndarray]:
        """Run model and extract outputs."""
        continuous = torch.tensor(sequence['continuous']).unsqueeze(0).to(self.device)
        binary = torch.tensor(sequence['binary']).unsqueeze(0).to(self.device)
        menstrual = torch.tensor(sequence['menstrual']).unsqueeze(0).to(self.device)
        day_of_week = torch.tensor(sequence['day_of_week']).unsqueeze(0).to(self.device)
        clinical = torch.tensor(sequence['clinical']).unsqueeze(0).to(self.device)
        
        outputs = model(continuous, binary, menstrual, day_of_week, clinical)
        
        prob = torch.sigmoid(outputs['attack_logits']).item()
        severity = outputs.get('severity', torch.tensor([5.0])).item()
        
        # Get trigger importance if available
        if 'trigger_importance' in outputs:
            trigger_scores = outputs['trigger_importance'].cpu().numpy().flatten()
        else:
            # Estimate from feature importance
            trigger_scores = self._estimate_trigger_importance(sequence)
        
        return prob, severity, trigger_scores
    
    def _estimate_trigger_importance(self, sequence: Dict) -> np.ndarray:
        """Estimate trigger importance from feature values."""
        last_day = sequence['continuous'][-1]  # Normalized values
        
        scores = np.zeros(7)
        
        # Sleep (index 0) - negative values mean less sleep
        scores[0] = max(0, -last_day[0] * 0.3)
        
        # Stress (index 1) - positive values mean more stress
        scores[1] = max(0, last_day[1] * 0.2)
        
        # Weather/pressure change (index 3)
        scores[2] = max(0, -last_day[3] * 0.15)
        
        # Fasting (index 6)
        scores[3] = max(0, last_day[6] * 0.2)
        
        # Alcohol (index 7)
        scores[4] = max(0, last_day[7] * 0.25)
        
        # Menstrual
        menstrual_day = sequence['menstrual'][-1]
        if menstrual_day in [0, 1, 26, 27]:
            scores[5] = 0.3
        
        # Light (binary index 4)
        scores[6] = sequence['binary'][-1, 4] * 0.15
        
        # Normalize
        total = scores.sum()
        if total > 0:
            scores = scores / total
        
        return scores
    
    def _identify_triggers(self, features: Dict) -> Dict[str, float]:
        """Identify active triggers from feature values."""
        triggers = {}
        
        if features.get('sleep_hours', 7) < 6:
            triggers['Sleep'] = 0.8
        if features.get('stress_level', 5) > 7:
            triggers['Stress'] = 0.6
        if features.get('pressure_change', 0) < -10:
            triggers['Weather'] = 0.5
        if features.get('hours_fasting', 4) > 10:
            triggers['Fasting'] = 0.4
        if features.get('alcohol_drinks', 0) >= 3:
            triggers['Alcohol'] = 0.7
        if features.get('menstrual_cycle_day', -1) in [0, 1]:
            triggers['Menstrual'] = 0.65
        if features.get('bright_light_exposure', 0) == 1:
            triggers['Light'] = 0.3
        
        return triggers
    
    def _risk_level(self, prob: float) -> str:
        """Convert probability to risk level."""
        if prob < 0.3:
            return "LOW"
        elif prob < 0.5:
            return "MODERATE"
        elif prob < 0.7:
            return "HIGH"
        else:
            return "VERY HIGH"
    
    def _generate_recommendations(
        self,
        risk: float,
        triggers: Dict[str, float],
        features: Dict,
    ) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        # Sort triggers by importance
        sorted_triggers = sorted(triggers.items(), key=lambda x: x[1], reverse=True)
        
        if risk >= 0.7:
            recommendations.append("⚠️ Consider taking preventive medication")
            recommendations.append("📱 Keep rescue medication nearby")
        
        for trigger, score in sorted_triggers[:3]:
            if trigger == 'Sleep' and score > 0.3:
                recommendations.append("🛏️ Prioritize 8+ hours of sleep tonight")
            elif trigger == 'Stress' and score > 0.3:
                recommendations.append("🧘 Try relaxation techniques or meditation")
            elif trigger == 'Weather' and score > 0.3:
                recommendations.append("🌡️ Barometric pressure dropping - stay hydrated")
            elif trigger == 'Fasting' and score > 0.3:
                recommendations.append("🍽️ Don't skip meals today")
            elif trigger == 'Alcohol' and score > 0.3:
                recommendations.append("🍷 Avoid alcohol today")
            elif trigger == 'Menstrual' and score > 0.3:
                recommendations.append("📅 Menstrual phase - extra caution advised")
            elif trigger == 'Light' and score > 0.3:
                recommendations.append("😎 Avoid bright lights and screens")
        
        if risk >= 0.5:
            recommendations.append("💧 Drink at least 8 glasses of water")
        
        if not recommendations:
            recommendations.append("✅ Continue your healthy routine")
        
        return recommendations[:5]  # Max 5 recommendations


# =========================================================================
# CLI for testing
# =========================================================================

def demo():
    """Demo the personalization system."""
    print("\n" + "=" * 60)
    print("🧠 MIGRAINE PERSONALIZATION SYSTEM - DEMO")
    print("=" * 60)
    
    # Initialize system
    system = PersonalizationSystem(
        models_dir="../models_v2",
        user_data_dir="../demo_users",
    )
    
    # Create demo user
    user_id = "demo_user_001"
    
    print(f"\n📱 Creating user: {user_id}")
    state = system.get_or_create_user(user_id)
    print(f"   Phase: {state.current_phase}")
    print(f"   Days logged: {state.days_logged}")
    
    # Simulate some daily logs
    print("\n📝 Simulating 20 days of logs...")
    
    base_date = datetime.now() - timedelta(days=20)
    
    for i in range(20):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        
        # Simulate varying conditions
        features = {
            'sleep_hours': 6.5 + np.random.normal(0, 1),
            'stress_level': 5 + np.random.normal(0, 2),
            'barometric_pressure': 1013 + np.random.normal(0, 5),
            'pressure_change': np.random.normal(0, 3),
            'temperature': 22,
            'humidity': 50,
            'hours_fasting': 4 + np.random.exponential(2),
            'alcohol_drinks': max(0, np.random.poisson(0.5)),
            'had_breakfast': 1,
            'had_lunch': 1,
            'had_dinner': 1,
            'had_snack': np.random.choice([0, 1]),
            'bright_light_exposure': np.random.choice([0, 1], p=[0.7, 0.3]),
            'sleep_quality': np.random.choice([0, 1], p=[0.3, 0.7]),
            'menstrual_cycle_day': i % 28 if np.random.random() > 0.5 else -1,
        }
        
        # Simulate some attacks
        attack = np.random.random() < 0.15
        
        system.update_user_log(user_id, date, features, attack)
    
    # Check state
    state = system.get_or_create_user(user_id)
    print(f"\n✓ After 20 days:")
    print(f"   Phase: {state.current_phase}")
    print(f"   Days logged: {state.days_logged}")
    print(f"   Total attacks: {state.total_attacks}")
    
    # Get prediction
    print("\n🔮 Getting today's prediction...")
    
    today_features = {
        'sleep_hours': 5.0,  # Poor sleep
        'stress_level': 7,   # High stress
        'barometric_pressure': 1005,
        'pressure_change': -8,
        'temperature': 22,
        'humidity': 60,
        'hours_fasting': 6,
        'alcohol_drinks': 2,
        'had_breakfast': 1,
        'had_lunch': 1,
        'had_dinner': 1,
        'had_snack': 0,
        'bright_light_exposure': 1,
        'sleep_quality': 0,
        'menstrual_cycle_day': 1,
    }
    
    prediction = system.predict(user_id, today_features)
    
    print(f"\n" + "=" * 60)
    print(f"📊 PREDICTION FOR {prediction.date}")
    print("=" * 60)
    print(f"\n   Risk: {prediction.risk_probability*100:.1f}% ({prediction.risk_level})")
    print(f"   Model: {prediction.model_version}")
    print(f"   Confidence: {prediction.confidence*100:.0f}%")
    
    print(f"\n   🎯 Top Triggers:")
    sorted_triggers = sorted(
        prediction.trigger_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for trigger, score in sorted_triggers[:3]:
        bar = "█" * int(score * 20)
        print(f"      {trigger}: {score:.2f} {bar}")
    
    print(f"\n   💡 Recommendations:")
    for rec in prediction.recommendations:
        print(f"      {rec}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()