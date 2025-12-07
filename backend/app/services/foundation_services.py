"""
Foundation Model Service
========================
Loads the trained LightGBM model for baseline risk calculation.
Used on Day 0 (onboarding) before user starts logging data.

From Day 1+, the Mamba model takes over for daily predictions.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import joblib

logger = logging.getLogger(__name__)

# Path to foundation_ready folder
FOUNDATION_DIR = Path(__file__).resolve().parent.parent.parent.parent / "foundation_ready"


class FoundationModelService:
    """
    Service for LightGBM foundation model predictions.
    
    Used for:
    - Initial baseline risk calculation at onboarding
    - Static risk assessment before user has logged data
    """
    
    def __init__(self, foundation_dir: str = None):
        self.foundation_dir = Path(foundation_dir) if foundation_dir else FOUNDATION_DIR
        self.model = None
        self.preprocessor = None
        self.feature_columns = []
        self._load_model()
    
    def _load_model(self):
        """Load LightGBM model and preprocessor."""
        try:
            # Load model
            model_path = self.foundation_dir / "best_model_LightGBM.joblib"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"✓ Loaded LightGBM model from {model_path}")
            else:
                logger.warning(f"LightGBM model not found at {model_path}")
                return
            
            # Load preprocessor
            preprocessor_path = self.foundation_dir / "preprocessor_CLEAN.joblib"
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info(f"✓ Loaded preprocessor from {preprocessor_path}")
            
            # Load feature columns
            features_path = self.foundation_dir / "features_CLEAN.csv"
            if features_path.exists():
                import pandas as pd
                features_df = pd.read_csv(features_path)
                self.feature_columns = features_df['Column'].tolist()
                logger.info(f"✓ Loaded {len(self.feature_columns)} feature columns")
            
        except Exception as e:
            logger.error(f"Error loading foundation model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def _map_onboarding_to_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Map onboarding data to LightGBM feature format.
        
        Onboarding collects:
        - gender (M/F)
        - age
        - height (cm)
        - weight (kg)
        - bmi
        - attacks_per_month
        - location_city
        - has_menstrual_cycle
        - cycle_start_day
        
        LightGBM expects 140+ features, so we map what we have and default the rest.
        """
        features = {}
        
        # Direct mappings
        features['sex'] = 1 if user_data.get('gender') == 'F' else 0
        features['Age at first visit'] = user_data.get('age', 30)
        features['height'] = user_data.get('height', 165)
        features['body weight'] = user_data.get('weight', 60)
        features['BMI'] = user_data.get('bmi', 22.0)
        
        # Estimate sleep time from typical values (default 7 hours)
        features['sleep time'] = 7.0
        
        # Age at onset - estimate from current age (assume started 5-10 years ago for adults)
        age = user_data.get('age', 30)
        features['Age at onset'] = max(10, age - 5)
        
        # Frequency - convert attacks_per_month to "Frequency: Months" 
        # (this seems to be days between attacks)
        attacks = user_data.get('attacks_per_month', 4)
        if attacks > 0:
            features['Frequency: Months'] = 30 / attacks  # days between attacks
        else:
            features['Frequency: Months'] = 30  # 1 attack per month
        
        # Binary symptom features - default to 0 (unknown)
        binary_features = [
            'Properties Tightening', 'Characteristics: Throat', 'Both pain',
            'temporal pain', 'Occipital pain', 'Varies depending on the change date',
            'Need to tolerate obstacles', 'Aggravating factor: Shaking your head',
            'Aggravating factors: Worsening weather', 'Accompanying symptoms: nausea',
            'Accompanying symptoms: stiff shoulders and neck', 'No prodromal symptoms',
            'Solution: Lie down', 'Applicable Child motion sickness',
            'Applicable: Lack of exercise', 'MOH Headache upon waking up',
            'History of headaches_can be', 'Usual parts_both sides', 'Usual parts_one side',
            'Taking headache medicine_y', 'Medication Over-the-counter medicines_1',
            'Family history_can be'
        ]
        for feat in binary_features:
            features[feat] = 0
        
        # One-hot encoded features - all 0 except one
        # Dominant hand - assume right
        features['Dominant hand_0'] = 0
        features['Dominant hand_left'] = 0
        features['Dominant hand_right'] = 1
        
        # Drinking - assume occasional
        features['drinking_0'] = 0
        features['drinking_Occasional drinking'] = 1
        features['drinking_habitual'] = 0
        
        # Smoking - assume non-smoker
        features['smoking_0'] = 1
        features['smoking_In the past'] = 0
        features['smoking_Passive smoking available'] = 0
        features['smoking_Unknown'] = 0
        
        # Bedtime - assume 23:00 (most common)
        bedtime_cols = [c for c in self.feature_columns if c.startswith('Bedtime_')]
        for col in bedtime_cols:
            features[col] = 1 if col == 'Bedtime_23:00' else 0
        
        # Wake up time - assume 07:00 (most common)
        wakeup_cols = [c for c in self.feature_columns if c.startswith('Wake up time_')]
        for col in wakeup_cols:
            features[col] = 1 if col == 'Wake up time_07:00' else 0
        
        # Employment - assume full-time
        features['Schooling/employment_Unknown'] = 0
        features['Schooling/employment details_Full-time'] = 1
        features['Schooling/employment details_Part-time'] = 0
        features['Schooling/employment details_Unknown'] = 0
        features['Schooling/employment details_student'] = 0
        
        # Onset time features
        features['Onset time: 1 week_1'] = 0
        features['Onset time: 1 week_Nothing\nYes'] = 0
        
        # Medication features - assume yes (taking OTC meds)
        features['Medication Over-the-counter medicines_Yes\nNothing'] = 0
        
        return features
    
    def _create_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Create feature vector in the correct order for the model."""
        if not self.feature_columns:
            raise ValueError("Feature columns not loaded")
        
        vector = []
        for col in self.feature_columns:
            # Handle special column names with newlines
            clean_col = col.replace('\n', '\\n')
            value = features.get(col, features.get(clean_col, 0))
            vector.append(value)
        
        return np.array(vector).reshape(1, -1)
    
    def predict_baseline_risk(self, user_data: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """
        Predict baseline migraine risk for a new user.
        
        Args:
            user_data: Onboarding data (gender, age, height, weight, bmi, attacks_per_month, etc.)
            
        Returns:
            Tuple of (probability, confidence, metadata)
        """
        if not self.is_available():
            # Fallback to simple estimation
            return self._fallback_prediction(user_data)
        
        try:
            # Map onboarding data to features
            features = self._map_onboarding_to_features(user_data)
            
            # Create feature vector
            X = self._create_feature_vector(features)
            
            # Apply preprocessor if available
            if self.preprocessor is not None:
                try:
                    X = self.preprocessor.transform(X)
                except Exception as e:
                    logger.warning(f"Preprocessor transform failed: {e}, using raw features")
            
            # Predict probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                prediction = self.model.predict(X)[0]
                probability = float(prediction)
            
            # Confidence based on how much data we have
            confidence = 0.76  # Model's validated AUC
            
            metadata = {
                'model_type': 'foundation',
                'model_version': 'lightgbm_v1',
                'features_used': len(self.feature_columns),
                'features_from_user': 6,  # gender, age, height, weight, bmi, attacks
            }
            
            return probability, confidence, metadata
            
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            return self._fallback_prediction(user_data)
    
    def _fallback_prediction(self, user_data: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Simple fallback when model isn't available."""
        # Base rate from epidemiology
        base_rate = 0.17
        
        # Adjust for attacks per month
        attacks = user_data.get('attacks_per_month', 4)
        if attacks >= 15:
            probability = 0.50  # Chronic migraine
        elif attacks >= 8:
            probability = 0.35  # High-frequency episodic
        elif attacks >= 4:
            probability = 0.25  # Moderate episodic
        else:
            probability = 0.15  # Low episodic
        
        metadata = {
            'model_type': 'foundation',
            'model_version': 'fallback_v1',
            'features_used': 1,
            'features_from_user': 1,
        }
        
        return probability, 0.60, metadata
    
    def get_risk_factors(self, user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key risk factors from user profile."""
        risk_factors = []
        
        # BMI risk
        bmi = user_data.get('bmi', 22)
        if bmi >= 30:
            risk_factors.append({
                'factor': 'BMI',
                'value': f'{bmi:.1f}',
                'risk_level': 'high',
                'description': 'Obesity is associated with increased migraine frequency'
            })
        elif bmi >= 25:
            risk_factors.append({
                'factor': 'BMI',
                'value': f'{bmi:.1f}',
                'risk_level': 'moderate',
                'description': 'Overweight may increase migraine risk'
            })
        
        # Attack frequency
        attacks = user_data.get('attacks_per_month', 4)
        if attacks >= 15:
            risk_factors.append({
                'factor': 'Frequency',
                'value': f'{attacks}/month',
                'risk_level': 'high',
                'description': 'Chronic migraine (≥15 days/month)'
            })
        elif attacks >= 8:
            risk_factors.append({
                'factor': 'Frequency',
                'value': f'{attacks}/month',
                'risk_level': 'moderate',
                'description': 'High-frequency episodic migraine'
            })
        
        # Gender
        if user_data.get('gender') == 'F':
            risk_factors.append({
                'factor': 'Gender',
                'value': 'Female',
                'risk_level': 'moderate',
                'description': 'Women are 3x more likely to have migraines'
            })
        
        # Menstrual cycle
        if user_data.get('has_menstrual_cycle'):
            risk_factors.append({
                'factor': 'Hormonal',
                'value': 'Menstrual cycle tracked',
                'risk_level': 'moderate',
                'description': 'Hormonal fluctuations can trigger migraines (OR 2.04)'
            })
        
        return risk_factors


# Singleton instance
_foundation_service = None

def get_foundation_service() -> FoundationModelService:
    """Get or create foundation service singleton."""
    global _foundation_service
    if _foundation_service is None:
        _foundation_service = FoundationModelService()
    return _foundation_service
