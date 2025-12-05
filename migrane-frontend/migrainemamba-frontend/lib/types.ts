// lib/types.ts

export interface OnboardingData {
  gender: 'M' | 'F';
  age: number;
  height: number; // cm
  weight: number; // kg
  bmi?: number; // calculated
  attacks_per_month: number;
  location_city: string;
  location_lat?: number;
  location_lon?: number;
  has_menstrual_cycle: boolean;
  cycle_start_day?: string; // ISO date format
}

export interface DailyLogData {
  user_id: string;
  date: string; // ISO date YYYY-MM-DD
  
  // Sleep & Stress
  sleep_hours: number;
  sleep_quality_good: boolean;
  stress_level: number; // 1-10
  
  // Diet & Lifestyle
  skipped_meals: string[]; // ['breakfast', 'lunch', 'dinner']
  had_snack: boolean;
  alcohol_drinks: number;
  bright_light_exposure: boolean;
  
  // Prodromal Symptoms
  symptoms: {
    fatigue: boolean;
    stiff_neck: boolean;
    yawning: boolean;
    food_cravings: boolean;
    mood_change: boolean;
    concentration: boolean;
    light_sensitivity: boolean;
    sound_sensitivity: boolean;
  };
  
  // Outcome (Ground Truth)
  migraine_occurred: boolean;
  migraine_details?: {
    severity: number; // 1-10
    duration_hours: number;
  };
}

export interface PredictionResponse {
  user_id: string;
  date: string;
  attack_probability: number; // 0-1
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  top_triggers: Array<{
    trigger: string;
    contribution: number;
  }>;
  recommendations: string[];
}