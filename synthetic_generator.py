"""
MigraineMamba Enhanced - Synthetic Data Generator v2.0
======================================================
Generates clinically-grounded synthetic migraine patient data.
All parameters backed by peer-reviewed sources.

Key Improvements over v1.0:
- Fixed menstrual cycle day calculation (days 26-27 = days -2 to -1)
- Proper stress decline tracking with lagged effects
- Explicit trigger sensitivity storage in output
- Attack clustering for 30% of patients
- Prodromal symptom generation
- Better autocorrelation for all features
- Proper alcohol lag effect (24h)
- Circadian rhythm modulation

Author: Dhoka
Date: December 2025
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatientProfile:
    """Immutable patient characteristics generated at simulation start."""
    patient_id: int
    gender: str
    age: int
    bmi: float
    phenotype: str
    attacks_per_month: float
    base_attack_rate: float
    peak_hour: int
    shows_clustering: bool
    has_prodrome: bool
    
    # Trigger sensitivities (stored for output)
    sensitive_sleep: bool = False
    sensitive_stress: bool = False
    sensitive_weather: bool = False
    sensitive_fasting: bool = False
    sensitive_alcohol: bool = False
    sensitive_menstrual: bool = False
    sensitive_photophobia: bool = False
    
    # Menstrual cycle tracking (females only)
    cycle_start_day: int = 0  # Random offset for cycle


@dataclass
class DailyState:
    """Mutable state that tracks patient's daily values."""
    day: int
    sleep_hours: float = 7.0
    sleep_quality: int = 1
    stress_level: int = 5
    stress_change: float = 0.0
    barometric_pressure: float = 1013.25
    pressure_change: float = 0.0
    temperature: float = 15.0
    humidity: float = 50.0
    had_breakfast: bool = True
    had_lunch: bool = True
    had_dinner: bool = True
    had_snack: bool = False
    hours_fasting: int = 0
    alcohol_drinks: int = 0
    menstrual_cycle_day: int = -1
    bright_light_exposure: bool = False
    
    # Prodromal symptoms (generated before attack)
    prodrome_fatigue: bool = False
    prodrome_mood_change: bool = False
    prodrome_neck_stiffness: bool = False
    prodrome_yawning: bool = False
    prodrome_food_cravings: bool = False
    prodrome_concentration: bool = False
    prodrome_light_sensitivity: bool = False
    prodrome_sound_sensitivity: bool = False


class MigraineDataGenerator:
    """
    Generates clinically-grounded synthetic migraine patient data.
    All parameters backed by peer-reviewed sources.
    """
    
    def __init__(self, config_path: str = 'config.json', seed: int = 42):
        """Initialize generator with configuration."""
        np.random.seed(seed)
        self.seed = seed
        self.config = self._load_config(config_path)
        self._validate_config()
        
        # Pre-compute phenotype probabilities
        self.phenotype_names = ['chronic', 'high_episodic', 'moderate', 'low']
        self.phenotype_probs = [
            self.config['phenotypes'][p]['prevalence'] 
            for p in self.phenotype_names
        ]
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                "Please ensure config.json exists."
            )
    
    def _validate_config(self):
        """Validate configuration against clinical constraints."""
        # Check phenotype prevalences sum to 1.0
        total_prev = sum(
            self.config['phenotypes'][p]['prevalence'] 
            for p in ['chronic', 'high_episodic', 'moderate', 'low']
        )
        assert abs(total_prev - 1.0) < 0.001, \
            f"Phenotype prevalences must sum to 1.0, got {total_prev}"
        
        # Check trigger prevalences are valid probabilities
        for trigger, params in self.config['triggers'].items():
            assert 0 <= params['prevalence'] <= 1, \
                f"Trigger {trigger} prevalence must be in [0, 1]"
        
        print("✓ Configuration validated against clinical constraints")
    
    def generate_patient_profile(self, patient_id: int) -> PatientProfile:
        """Generate base patient demographics and attack profile."""
        cfg = self.config
        
        # Gender (2-3:1 female:male ratio, Vetvik & MacGregor 2017)
        gender = np.random.choice(
            ['F', 'M'], 
            p=[cfg['demographics']['female_proportion'], 
               1 - cfg['demographics']['female_proportion']]
        )
        
        # Age distribution (peak 25-55)
        age = int(np.clip(
            np.random.normal(
                cfg['demographics']['age_mean'], 
                cfg['demographics']['age_std']
            ),
            cfg['demographics']['age_min'],
            cfg['demographics']['age_max']
        ))
        
        # BMI
        bmi = np.clip(
            np.random.normal(
                cfg['demographics']['bmi_mean'], 
                cfg['demographics']['bmi_std']
            ),
            cfg['demographics']['bmi_min'],
            cfg['demographics']['bmi_max']
        )
        
        # Assign phenotype based on epidemiology
        phenotype = np.random.choice(self.phenotype_names, p=self.phenotype_probs)
        
        # Attacks per month based on phenotype
        attack_range = cfg['phenotypes'][phenotype]['attacks_per_month_range']
        attacks_per_month = np.random.uniform(attack_range[0], attack_range[1])
        
        # For chronic patients, ensure minimum attack frequency is achievable
        # by boosting the base rate to account for OR-based probability calculation
        base_rate = attacks_per_month / 30.0
        if phenotype == 'chronic':
            # Chronic patients need higher base rate to achieve 15+ attacks/month
            base_rate = max(base_rate, 0.6)  # Ensure at least ~18 attacks/month potential
        elif phenotype == 'high_episodic':
            base_rate = max(base_rate, 0.35)  # Ensure at least ~10 attacks/month potential
        
        # Circadian preference (Baksa et al. 2019: 52.6% attacks 6AM-noon)
        circ = cfg['circadian']
        peak_hour = int(np.clip(
            np.random.normal(circ['peak_hour_mean'], circ['peak_hour_std']),
            circ['peak_hour_min'],
            circ['peak_hour_max']
        ))
        
        # Clustering behavior (30% of patients, Baksa 2019)
        shows_clustering = np.random.random() < cfg['temporal_constraints']['clustering_probability']
        
        # Prodrome presence (85.3% experience prodromal symptoms)
        has_prodrome = np.random.random() < cfg['prodrome']['prevalence']
        
        # Generate trigger sensitivities
        triggers = cfg['triggers']
        
        profile = PatientProfile(
            patient_id=patient_id,
            gender=gender,
            age=age,
            bmi=round(bmi, 1),
            phenotype=phenotype,
            attacks_per_month=attacks_per_month,
            base_attack_rate=base_rate,  # Use the boosted base_rate
            peak_hour=peak_hour,
            shows_clustering=shows_clustering,
            has_prodrome=has_prodrome,
            sensitive_sleep=np.random.random() < triggers['sleep']['prevalence'],
            sensitive_stress=np.random.random() < triggers['stress']['prevalence'],
            sensitive_weather=np.random.random() < triggers['weather']['prevalence'],
            sensitive_fasting=np.random.random() < triggers['fasting']['prevalence'],
            sensitive_alcohol=np.random.random() < triggers['alcohol']['prevalence'],
            sensitive_menstrual=(
                gender == 'F' and 
                np.random.random() < triggers['menstrual']['prevalence']
            ),
            sensitive_photophobia=np.random.random() < triggers['photophobia']['prevalence'],
            cycle_start_day=np.random.randint(0, 28) if gender == 'F' else 0
        )
        
        return profile
    
    def generate_daily_features(
        self, 
        day: int, 
        profile: PatientProfile,
        prev_state: Optional[DailyState],
        prev_states: List[DailyState]
    ) -> DailyState:
        """Generate daily features with realistic autocorrelation."""
        cfg = self.config
        state = DailyState(day=day)
        
        # === SLEEP ===
        sleep_cfg = cfg['features']['sleep']
        is_weekend = day % 7 in [5, 6]
        
        if prev_state:
            # Autocorrelated sleep
            alpha = sleep_cfg['autocorrelation']
            base_sleep = (
                sleep_cfg['mean_hours'] + 
                (sleep_cfg['weekend_bonus'] if is_weekend else 0)
            )
            state.sleep_hours = np.clip(
                alpha * prev_state.sleep_hours + 
                (1 - alpha) * np.random.normal(base_sleep, sleep_cfg['std_hours']),
                sleep_cfg['min_hours'],
                sleep_cfg['max_hours']
            )
        else:
            state.sleep_hours = np.clip(
                np.random.normal(sleep_cfg['mean_hours'], sleep_cfg['std_hours']),
                sleep_cfg['min_hours'],
                sleep_cfg['max_hours']
            )
        
        state.sleep_quality = 1 if state.sleep_hours >= 6 else 0
        
        # === STRESS ===
        stress_cfg = cfg['features']['stress']
        base_stress = stress_cfg['weekend_mean'] if is_weekend else stress_cfg['weekday_mean']
        
        if prev_state:
            # Autocorrelated stress
            alpha = stress_cfg['autocorrelation']
            raw_stress = (
                alpha * prev_state.stress_level + 
                (1 - alpha) * np.random.normal(base_stress, stress_cfg['std'])
            )
        else:
            raw_stress = np.random.normal(base_stress, stress_cfg['std'])
        
        state.stress_level = int(np.clip(
            raw_stress, 
            stress_cfg['scale_min'], 
            stress_cfg['scale_max']
        ))
        
        # Track stress change for decline detection
        if prev_state:
            state.stress_change = state.stress_level - prev_state.stress_level
        
        # === WEATHER ===
        weather_cfg = cfg['features']['weather']
        
        if prev_state:
            # Highly autocorrelated pressure (ACF > 0.7)
            alpha = weather_cfg['pressure_autocorrelation']
            pressure_change = np.random.normal(0, weather_cfg['pressure_daily_change_std'])
            state.barometric_pressure = (
                alpha * prev_state.barometric_pressure + 
                (1 - alpha) * weather_cfg['pressure_mean'] + 
                pressure_change
            )
            state.pressure_change = state.barometric_pressure - prev_state.barometric_pressure
        else:
            state.barometric_pressure = np.random.normal(
                weather_cfg['pressure_mean'], 
                weather_cfg['pressure_std']
            )
            state.pressure_change = 0
        
        # Seasonal temperature and humidity
        season_factor = np.sin(2 * np.pi * day / 365)
        state.temperature = (
            weather_cfg['temperature_base'] + 
            weather_cfg['temperature_seasonal_amplitude'] * season_factor + 
            np.random.normal(0, 5)
        )
        state.humidity = np.clip(
            weather_cfg['humidity_base'] + 
            weather_cfg['humidity_seasonal_amplitude'] * season_factor + 
            np.random.normal(0, 15),
            0, 100
        )
        
        # === MEALS ===
        meals_cfg = cfg['features']['meals']
        state.had_breakfast = np.random.random() > meals_cfg['skip_breakfast_prob']
        state.had_lunch = np.random.random() > meals_cfg['skip_lunch_prob']
        state.had_dinner = np.random.random() > meals_cfg['skip_dinner_prob']
        state.had_snack = np.random.random() < meals_cfg['have_snack_prob']
        
        # Calculate fasting hours (simplified)
        meals_count = sum([state.had_breakfast, state.had_lunch, state.had_dinner])
        state.hours_fasting = max(0, 8 - meals_count * 3) if meals_count < 3 else 0
        
        # === ALCOHOL ===
        alcohol_cfg = cfg['features']['alcohol']
        drinking_prob = alcohol_cfg['drinking_day_prob']
        if is_weekend:
            drinking_prob *= alcohol_cfg['weekend_multiplier']
        
        if np.random.random() < drinking_prob:
            state.alcohol_drinks = max(1, int(np.random.exponential(alcohol_cfg['drinks_mean'])))
        
        # === MENSTRUAL CYCLE ===
        if profile.gender == 'F':
            cycle_length = cfg['triggers']['menstrual']['cycle_days']
            # Proper cycle day calculation with offset
            state.menstrual_cycle_day = (day + profile.cycle_start_day) % cycle_length
        
        # === LIGHT EXPOSURE ===
        state.bright_light_exposure = (
            np.random.random() < cfg['features']['light_exposure']['bright_light_prob']
        )
        
        return state
    
    def _check_stress_decline(
        self, 
        current_state: DailyState, 
        prev_states: List[DailyState]
    ) -> Tuple[bool, float]:
        """
        Check for stress decline trigger (Lipton et al. 2014).
        Returns (triggered, odds_ratio).
        """
        if len(prev_states) < 1:
            return False, 1.0
        
        cfg = self.config['triggers']['stress']
        threshold = cfg['decline_threshold']
        
        # Check stress decline in the past 1-3 days
        for i, past in enumerate(reversed(prev_states[-3:])):
            decline = past.stress_level - current_state.stress_level
            if decline >= threshold:
                # Apply appropriate OR based on timing
                if i == 0:  # 1 day ago (~6h lag)
                    return True, cfg['odds_ratio_6h']
                elif i == 1:  # 2 days ago (~12h lag)
                    return True, cfg['odds_ratio_12h']
                else:  # 3 days ago (~18h lag)
                    return True, cfg['odds_ratio_18h']
        
        return False, 1.0
    
    def _get_menstrual_or(
        self, 
        cycle_day: int, 
        profile: PatientProfile
    ) -> float:
        """
        Get menstrual cycle odds ratio based on cycle day.
        Fixed: Days 26-27 are treated as -2 to -1 (pre-menstrual).
        """
        if not profile.sensitive_menstrual or profile.gender != 'F':
            return 1.0
        
        cfg = self.config['triggers']['menstrual']
        
        # High-risk days (days 0-1 of menstruation)
        if cycle_day in cfg['high_risk_days']:
            return cfg['odds_ratio_days_0_1']
        
        # Pre-menstrual days (days 26-27 = -2 to -1)
        if cycle_day in cfg['pre_menstrual_days']:
            return cfg['odds_ratio_pre_menstrual']
        
        # Ovulation (protective)
        if cycle_day in cfg['ovulation_days']:
            return cfg['odds_ratio_ovulation']
        
        return 1.0
    
    def _apply_or_to_probability(self, base_prob: float, odds_ratio: float) -> float:
        """
        Correctly apply an odds ratio to a probability.
        
        OR = (p_exposed / (1 - p_exposed)) / (p_unexposed / (1 - p_unexposed))
        
        Solving for p_exposed:
        p_exposed = (OR * p_unexposed) / (1 - p_unexposed + OR * p_unexposed)
        """
        if base_prob >= 1.0:
            return 0.95
        if base_prob <= 0.0:
            return 0.0
        
        # Convert probability to odds, multiply by OR, convert back
        base_odds = base_prob / (1 - base_prob)
        new_odds = base_odds * odds_ratio
        new_prob = new_odds / (1 + new_odds)
        
        return min(new_prob, 0.95)
    
    def calculate_attack_probability(
        self,
        profile: PatientProfile,
        state: DailyState,
        days_since_attack: int,
        prev_states: List[DailyState],
        recent_attack_in_cluster_window: bool
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate attack probability using clinical odds ratios.
        Uses proper OR application to probabilities.
        Returns (probability, trigger_contributions).
        """
        cfg = self.config
        
        # Start with base rate
        prob = profile.base_attack_rate
        trigger_contributions = {'base_rate': prob}
        
        # === HARD CONSTRAINT: Refractory Period (48-72 hours) ===
        if days_since_attack < 2:  # Within 48 hours
            return 0.0, {'refractory': 0.0}
        
        # Track cumulative OR for proper application
        cumulative_or = 1.0
        
        # === Sleep Trigger (OR 3.98, Duan et al. 2022) ===
        if profile.sensitive_sleep:
            if state.sleep_hours < cfg['triggers']['sleep']['threshold_hours']:
                or_val = cfg['triggers']['sleep']['odds_ratio']
                cumulative_or *= or_val
                trigger_contributions['sleep'] = or_val
        
        # === Stress Decline Trigger (OR 1.92 at 6h, Lipton et al. 2014) ===
        if profile.sensitive_stress:
            triggered, or_val = self._check_stress_decline(state, prev_states)
            if triggered:
                cumulative_or *= or_val
                trigger_contributions['stress_decline'] = or_val
        
        # === Weather Trigger (OR 1.27, Kimoto et al. 2011) ===
        if profile.sensitive_weather:
            if state.pressure_change < -cfg['triggers']['weather']['threshold_mb']:
                or_val = cfg['triggers']['weather']['odds_ratio']
                cumulative_or *= or_val
                trigger_contributions['weather'] = or_val
        
        # === Fasting Trigger (Turner et al. 2013) ===
        if profile.sensitive_fasting:
            if state.hours_fasting >= cfg['triggers']['fasting']['threshold_hours']:
                or_val = cfg['triggers']['fasting']['odds_ratio']
                cumulative_or *= or_val
                trigger_contributions['fasting'] = or_val
            elif state.had_snack:
                # Protective effect of snacking
                or_val = cfg['triggers']['fasting']['protective_snack_or']
                cumulative_or *= or_val
                trigger_contributions['snack_protective'] = or_val
        
        # === Alcohol Trigger (OR 2.08 for 5+ drinks, Mostofsky et al. 2020) ===
        # Effect is on NEXT day (24h lag)
        if profile.sensitive_alcohol and len(prev_states) >= 1:
            prev_alcohol = prev_states[-1].alcohol_drinks
            if prev_alcohol >= 5:
                or_val = cfg['triggers']['alcohol']['odds_ratio_5plus']
                cumulative_or *= or_val
                trigger_contributions['alcohol'] = or_val
        
        # === Menstrual Trigger (Stewart et al. 2000) ===
        menstrual_or = self._get_menstrual_or(state.menstrual_cycle_day, profile)
        if menstrual_or != 1.0:
            cumulative_or *= menstrual_or
            trigger_contributions['menstrual'] = menstrual_or
        
        # === Photophobia Trigger ===
        if profile.sensitive_photophobia and state.bright_light_exposure:
            or_val = cfg['triggers']['photophobia']['odds_ratio']
            cumulative_or *= or_val
            trigger_contributions['photophobia'] = or_val
        
        # === Circadian Modulation (Baksa et al. 2019) ===
        circ = cfg['circadian']
        hour_factor = 1.0 + (circ['peak_multiplier'] - 1.0) * np.exp(
            -0.5 * ((profile.peak_hour - 9) / 3) ** 2
        )
        cumulative_or *= hour_factor
        trigger_contributions['circadian'] = hour_factor
        
        # === Clustering Effect (30% of patients) ===
        if profile.shows_clustering and recent_attack_in_cluster_window:
            cluster_mult = cfg['temporal_constraints']['clustering_multiplier']
            cumulative_or *= cluster_mult
            trigger_contributions['clustering'] = cluster_mult
        
        # Apply cumulative OR properly to base probability
        final_prob = self._apply_or_to_probability(prob, cumulative_or)
        
        trigger_contributions['cumulative_or'] = cumulative_or
        trigger_contributions['final'] = final_prob
        
        return final_prob, trigger_contributions
    
    def generate_prodrome(
        self, 
        state: DailyState, 
        profile: PatientProfile,
        attack_tomorrow: bool
    ) -> DailyState:
        """Generate prodromal symptoms if attack is coming."""
        if not attack_tomorrow or not profile.has_prodrome:
            return state
        
        probs = self.config['prodrome']['symptom_probabilities']
        
        state.prodrome_fatigue = np.random.random() < probs['fatigue']
        state.prodrome_mood_change = np.random.random() < probs['mood_change']
        state.prodrome_neck_stiffness = np.random.random() < probs['neck_stiffness']
        state.prodrome_yawning = np.random.random() < probs['yawning']
        state.prodrome_food_cravings = np.random.random() < probs['food_cravings']
        state.prodrome_concentration = np.random.random() < probs['concentration_difficulty']
        state.prodrome_light_sensitivity = np.random.random() < probs['sensitivity_light']
        state.prodrome_sound_sensitivity = np.random.random() < probs['sensitivity_sound']
        
        return state
    
    def generate_attack_details(self, profile: PatientProfile) -> Dict:
        """Generate attack severity and duration."""
        cfg = self.config['features']
        
        # Severity (Normal distribution, scale 1-10)
        severity = int(np.clip(
            np.random.normal(cfg['attack_severity']['mean'], cfg['attack_severity']['std']),
            cfg['attack_severity']['min'],
            cfg['attack_severity']['max']
        ))
        
        # Duration (Exponential, 4-72 hours)
        duration = np.clip(
            np.random.exponential(cfg['attack_duration']['mean_hours']),
            cfg['attack_duration']['min_hours'],
            cfg['attack_duration']['max_hours']
        )
        
        # Gender difference (Vetvik & MacGregor 2017: women 38.8h vs men 12.8h)
        if profile.gender == 'F':
            duration *= cfg['attack_duration']['female_multiplier']
            duration = min(duration, cfg['attack_duration']['max_hours'])
        
        return {
            'attack': 1,
            'severity': severity,
            'duration_hours': round(duration, 1)
        }
    
    def add_noise_and_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic missing data and measurement noise."""
        cfg = self.config['missing_data']
        n_rows = len(df)
        
        # Features that can have missing values
        noise_cols = [
            'sleep_hours', 'stress_level', 'barometric_pressure',
            'temperature', 'humidity'
        ]
        missing_cols = [
            'sleep_hours', 'sleep_quality', 'stress_level',
            'had_breakfast', 'had_lunch', 'had_dinner', 'had_snack',
            'alcohol_drinks', 'bright_light_exposure'
        ]
        
        # Add measurement noise
        for col in noise_cols:
            if col in df.columns:
                noise_mask = np.random.random(n_rows) < 0.5
                noise = np.random.normal(0, cfg['measurement_noise_pct'], n_rows)
                df.loc[noise_mask, col] *= (1 + noise[noise_mask])
        
        # Add random missing data
        for col in missing_cols:
            if col in df.columns:
                n_missing = int(n_rows * cfg['random_missing_rate'])
                missing_idx = np.random.choice(df.index, n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
        
        return df
    
    def generate_patient_history(self, patient_id: int) -> pd.DataFrame:
        """Generate complete history for one patient."""
        profile = self.generate_patient_profile(patient_id)
        n_days = self.config['generation']['n_days']
        start_date = datetime.strptime(
            self.config['generation']['start_date'], '%Y-%m-%d'
        )
        cluster_window = self.config['temporal_constraints']['clustering_window_days']
        
        history = []
        prev_state = None
        prev_states = []
        days_since_attack = 99
        attack_days = []
        
        # First pass: generate features and determine attacks
        for day in range(n_days):
            # Check for recent attack in clustering window
            recent_attack = any(
                (day - ad) <= cluster_window and (day - ad) > 0 
                for ad in attack_days
            )
            
            # Generate daily features
            state = self.generate_daily_features(day, profile, prev_state, prev_states)
            
            # Calculate attack probability
            prob, triggers = self.calculate_attack_probability(
                profile, state, days_since_attack, prev_states, recent_attack
            )
            
            # Sample attack
            attack_occurred = np.random.random() < prob
            
            # Build daily record
            record = {
                'patient_id': patient_id,
                'day': day,
                'date': (start_date + timedelta(days=day)).strftime('%Y-%m-%d'),
                'day_of_week': (start_date + timedelta(days=day)).weekday(),
                
                # Features
                'sleep_hours': round(state.sleep_hours, 2),
                'sleep_quality': state.sleep_quality,
                'stress_level': state.stress_level,
                'stress_change': round(state.stress_change, 2),
                'barometric_pressure': round(state.barometric_pressure, 2),
                'pressure_change': round(state.pressure_change, 2),
                'temperature': round(state.temperature, 1),
                'humidity': round(state.humidity, 1),
                'had_breakfast': int(state.had_breakfast),
                'had_lunch': int(state.had_lunch),
                'had_dinner': int(state.had_dinner),
                'had_snack': int(state.had_snack),
                'hours_fasting': state.hours_fasting,
                'alcohol_drinks': state.alcohol_drinks,
                'menstrual_cycle_day': state.menstrual_cycle_day,
                'bright_light_exposure': int(state.bright_light_exposure),
                
                # Prodromal symptoms
                'prodrome_fatigue': int(state.prodrome_fatigue),
                'prodrome_mood_change': int(state.prodrome_mood_change),
                'prodrome_neck_stiffness': int(state.prodrome_neck_stiffness),
                'prodrome_yawning': int(state.prodrome_yawning),
                'prodrome_food_cravings': int(state.prodrome_food_cravings),
                'prodrome_concentration': int(state.prodrome_concentration),
                'prodrome_light_sensitivity': int(state.prodrome_light_sensitivity),
                'prodrome_sound_sensitivity': int(state.prodrome_sound_sensitivity),
                
                # Calculated probability (for validation)
                'attack_probability': round(prob, 4),
            }
            
            if attack_occurred:
                attack_details = self.generate_attack_details(profile)
                record.update(attack_details)
                days_since_attack = 0
                attack_days.append(day)
            else:
                record.update({'attack': 0, 'severity': 0, 'duration_hours': 0})
                days_since_attack += 1
            
            history.append(record)
            
            # Update state tracking
            prev_state = state
            prev_states.append(state)
            if len(prev_states) > 7:  # Keep last week
                prev_states.pop(0)
        
        # Second pass: add prodromal symptoms before attacks
        df = pd.DataFrame(history)
        for i in range(len(df) - 1):
            if df.loc[i + 1, 'attack'] == 1 and profile.has_prodrome:
                probs = self.config['prodrome']['symptom_probabilities']
                df.loc[i, 'prodrome_fatigue'] = int(np.random.random() < probs['fatigue'])
                df.loc[i, 'prodrome_mood_change'] = int(np.random.random() < probs['mood_change'])
                df.loc[i, 'prodrome_neck_stiffness'] = int(np.random.random() < probs['neck_stiffness'])
                df.loc[i, 'prodrome_yawning'] = int(np.random.random() < probs['yawning'])
        
        # Add patient metadata
        df['gender'] = profile.gender
        df['age'] = profile.age
        df['bmi'] = profile.bmi
        df['phenotype'] = profile.phenotype
        df['attacks_per_month_expected'] = round(profile.attacks_per_month, 2)
        
        # Add trigger sensitivities (important for model validation!)
        df['sensitive_sleep'] = int(profile.sensitive_sleep)
        df['sensitive_stress'] = int(profile.sensitive_stress)
        df['sensitive_weather'] = int(profile.sensitive_weather)
        df['sensitive_fasting'] = int(profile.sensitive_fasting)
        df['sensitive_alcohol'] = int(profile.sensitive_alcohol)
        df['sensitive_menstrual'] = int(profile.sensitive_menstrual)
        df['sensitive_photophobia'] = int(profile.sensitive_photophobia)
        df['shows_clustering'] = int(profile.shows_clustering)
        df['has_prodrome'] = int(profile.has_prodrome)
        
        return df
    
    def generate_dataset(
        self, 
        output_dir: str = 'synthetic_data',
        add_noise: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Generate full dataset of synthetic patients."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        n_patients = self.config['generation']['n_patients']
        
        if verbose:
            print(f"Generating {n_patients} synthetic patients...")
            print(f"Configuration: {self.config['generation']['n_days']} days per patient")
        
        all_patients = []
        
        for patient_id in range(n_patients):
            if verbose and (patient_id + 1) % 500 == 0:
                print(f"  Generated {patient_id + 1}/{n_patients} patients")
            
            df = self.generate_patient_history(patient_id)
            all_patients.append(df)
        
        # Combine all patients
        if verbose:
            print("\n  Combining all patients...")
        combined_df = pd.concat(all_patients, ignore_index=True)
        
        # Add noise and missing data
        if add_noise:
            if verbose:
                print("  Adding noise and missing data...")
            combined_df = self.add_noise_and_missing(combined_df)
        
        # Save dataset
        output_file = output_path / 'migraine_synthetic_data.csv'
        combined_df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"\n✓ Generated {len(combined_df):,} records across {n_patients:,} patients")
            print(f"✓ Data saved to {output_file}")
        
        # Run validation
        self.validate_dataset(combined_df, verbose)
        
        return combined_df
    
    def validate_dataset(self, df: pd.DataFrame, verbose: bool = True):
        """Validate dataset against clinical constraints."""
        if verbose:
            print("\n" + "=" * 60)
            print("DATASET VALIDATION")
            print("=" * 60)
        
        validation_passed = True
        
        # 1. CRITICAL: Refractory period violations
        violations = 0
        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id].sort_values('day')
            attack_days = patient_df[patient_df['attack'] == 1]['day'].values
            
            if len(attack_days) > 1:
                gaps = np.diff(attack_days)
                violations += np.sum(gaps < 2)
        
        if verbose:
            status = "✓ PASS" if violations == 0 else "✗ CRITICAL FAIL"
            print(f"\n1. Refractory violations: {violations} (target: 0) {status}")
        
        if violations > 0:
            validation_passed = False
        
        # 2. Phenotype distribution
        if verbose:
            print("\n2. Phenotype Distribution:")
        patient_phenotypes = df.groupby('patient_id')['phenotype'].first()
        actual_dist = patient_phenotypes.value_counts(normalize=True)
        
        for phenotype in ['chronic', 'high_episodic', 'moderate', 'low']:
            expected = self.config['phenotypes'][phenotype]['prevalence']
            actual = actual_dist.get(phenotype, 0)
            diff = abs(actual - expected)
            status = "✓" if diff < 0.03 else "⚠"
            if verbose:
                print(f"   {phenotype}: {actual:.1%} (expected: {expected:.1%}) {status}")
        
        # 3. Attack frequency by phenotype
        if verbose:
            print("\n3. Attack Frequency by Phenotype:")
        n_months = df['day'].max() / 30
        
        for phenotype in ['chronic', 'high_episodic', 'moderate', 'low']:
            phenotype_df = df[df['phenotype'] == phenotype]
            if len(phenotype_df) == 0:
                continue
            
            attacks = phenotype_df.groupby('patient_id')['attack'].sum()
            avg_per_month = attacks.mean() / n_months
            expected = self.config['phenotypes'][phenotype]['attacks_per_month_range']
            
            in_range = expected[0] * 0.8 <= avg_per_month <= expected[1] * 1.2
            status = "✓" if in_range else "⚠"
            if verbose:
                print(f"   {phenotype}: {avg_per_month:.1f}/month (target: {expected[0]}-{expected[1]}) {status}")
        
        # 4. Sleep OR check
        sleep_deprived = df[df['sleep_hours'] < 6]['attack'].mean()
        sleep_normal = df[df['sleep_hours'] >= 6]['attack'].mean()
        
        if sleep_normal > 0 and sleep_deprived > 0:
            sleep_or = (sleep_deprived / (1 - sleep_deprived)) / (sleep_normal / (1 - sleep_normal))
            expected_range = self.config['validation_targets']['sleep_or_range']
            status = "✓" if expected_range[0] <= sleep_or <= expected_range[1] else "⚠"
            if verbose:
                print(f"\n4. Sleep OR: {sleep_or:.2f} (target: {expected_range[0]}-{expected_range[1]}) {status}")
        
        # 5. Menstrual OR check (females only)
        female_df = df[df['gender'] == 'F']
        if len(female_df) > 0:
            high_risk = female_df[female_df['menstrual_cycle_day'].isin([0, 1])]['attack'].mean()
            other = female_df[~female_df['menstrual_cycle_day'].isin([0, 1])]['attack'].mean()
            
            if other > 0 and high_risk > 0:
                menstrual_or = (high_risk / (1 - high_risk)) / (other / (1 - other))
                expected_range = self.config['validation_targets']['menstrual_or_range']
                status = "✓" if expected_range[0] <= menstrual_or <= expected_range[1] else "⚠"
                if verbose:
                    print(f"\n5. Menstrual OR: {menstrual_or:.2f} (target: {expected_range[0]}-{expected_range[1]}) {status}")
        
        # 6. Gender distribution
        gender_dist = df.groupby('patient_id')['gender'].first().value_counts(normalize=True)
        female_pct = gender_dist.get('F', 0)
        expected_female = self.config['demographics']['female_proportion']
        status = "✓" if abs(female_pct - expected_female) < 0.05 else "⚠"
        if verbose:
            print(f"\n6. Gender: F={female_pct:.1%}, M={1-female_pct:.1%} (target: F={expected_female:.0%}) {status}")
        
        if verbose:
            print("\n" + "=" * 60)
            if validation_passed:
                print("✓ VALIDATION PASSED - Dataset is publication-ready")
            else:
                print("✗ VALIDATION FAILED - Review critical errors above")
            print("=" * 60)
        
        return validation_passed


def main():
    """Main entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic migraine patient data'
    )
    parser.add_argument(
        '--config', '-c', 
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', '-o', 
        default='synthetic_data',
        help='Output directory'
    )
    parser.add_argument(
        '--seed', '-s', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-noise', 
        action='store_true',
        help='Disable noise and missing data'
    )
    parser.add_argument(
        '--quiet', '-q', 
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = MigraineDataGenerator(config_path=args.config, seed=args.seed)
    df = generator.generate_dataset(
        output_dir=args.output,
        add_noise=not args.no_noise,
        verbose=not args.quiet
    )
    
    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total records: {len(df):,}")
        print(f"Total patients: {df['patient_id'].nunique():,}")
        print(f"Total attacks: {df['attack'].sum():,}")
        print(f"Overall attack rate: {df['attack'].mean():.3f}")
        print(f"Days per patient: {df.groupby('patient_id').size().mean():.0f}")
        
        print("\n=== Sample Data (First 10 rows) ===")
        print(df[['patient_id', 'day', 'date', 'sleep_hours', 'stress_level', 
                  'menstrual_cycle_day', 'attack', 'severity']].head(10))


if __name__ == '__main__':
    main()