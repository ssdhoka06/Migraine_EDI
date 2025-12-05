// User and Profile Types
export interface UserProfile {
  id: string;
  gender: 'M' | 'F';
  age: number;
  height: number;
  weight: number;
  bmi: number;
  attacks_per_month: number;
  location_city: string;
  has_menstrual_cycle: boolean;
  cycle_start_day?: string;
  created_at: string;
  days_logged: number;
  current_phase: UserPhase;
  model_version: string;
  personalization_week: number;
}

export type UserPhase = 'foundation' | 'generic' | 'personalized';

// Onboarding Types
export interface OnboardingData {
  gender: 'M' | 'F';
  age: number;
  height: number;
  weight: number;
  bmi?: number;
  attacks_per_month: number;
  location_city: string;
  has_menstrual_cycle: boolean;
  cycle_start_day?: string;
}

// Daily Log Types
export interface DailyLogData {
  user_id: string;
  date: string;
  
  // Sleep & Stress
  sleep_hours: number;
  sleep_quality_good: boolean;
  stress_level: number;
  
  // Diet & Lifestyle
  skipped_meals: string[];
  had_snack: boolean;
  alcohol_drinks: number;
  caffeine_drinks: number;
  water_glasses: number;
  bright_light_exposure: boolean;
  screen_time_hours: number;
  
  // Weather (auto-fetched or manual)
  barometric_pressure?: number;
  pressure_change?: number;
  temperature?: number;
  humidity?: number;
  
  // Menstrual (if applicable)
  menstrual_cycle_day?: number;
  
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
    nausea: boolean;
    visual_disturbance: boolean;
  };
  
  // Outcome (Ground Truth)
  migraine_occurred: boolean;
  migraine_details?: {
    severity: number;
    duration_hours: number;
    location: 'left' | 'right' | 'both' | 'frontal' | 'back';
    with_aura: boolean;
    medications_taken: string[];
    relief_time_hours?: number;
  };
}

// Prediction Types
export interface PredictionResponse {
  user_id: string;
  date: string;
  prediction_time: string;
  
  // Main prediction
  attack_probability: number;
  risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'VERY_HIGH';
  severity_prediction: number;
  
  // Model info
  model_version: string;
  confidence: number;
  phase: UserPhase;
  
  // Trigger analysis
  top_triggers: TriggerContribution[];
  
  // Recommendations
  recommendations: Recommendation[];
  
  // Additional context
  contributing_factors: ContributingFactor[];
  protective_factors: string[];
}

export interface TriggerContribution {
  trigger: string;
  contribution: number;
  icon: string;
  color: string;
  description: string;
}

export interface Recommendation {
  priority: 'high' | 'medium' | 'low';
  action: string;
  reason: string;
  icon: string;
}

export interface ContributingFactor {
  factor: string;
  value: string | number;
  threshold: string | number;
  status: 'warning' | 'critical' | 'normal';
}

// History & Analytics Types
export interface DailyLogEntry {
  date: string;
  migraine_occurred: boolean;
  severity?: number;
  prediction_accuracy?: 'correct' | 'false_positive' | 'false_negative' | 'true_negative';
  risk_level: string;
  predicted_probability: number;
  actual_probability?: number;
  // Additional fields for history page
  prediction_was_correct?: boolean;
  predicted_risk_level?: string;
  sleep_hours: number;
  stress_level: number;
  sleep_quality_good: boolean;
  top_triggers?: string[];
  migraine_details?: {
    severity: number;
    duration_hours: number;
    location: 'left' | 'right' | 'both' | 'frontal' | 'back';
    with_aura: boolean;
    medications_taken?: string[];
  };
}

export interface WeeklyStats {
  week_start: string;
  total_attacks: number;
  total_migraines: number;
  avg_severity: number;
  prediction_accuracy: number;
  most_common_triggers: string[];
  improvement_from_last_week: number;
  streak_days: number;
  weekly_accuracy: { week: string; accuracy: number }[];
}

export interface TriggerAnalysis {
  total_logs: number;
  triggers: {
    name: string;
    occurrence_rate: number;
    occurrences: number;
    attack_correlation: number;
    odds_ratio: number;
    contribution: number;
    confidence_interval: [number, number];
    trend: 'increasing' | 'decreasing' | 'stable';
    personalized_threshold?: number;
    icon: string;
    description: string;
  }[];
  patterns?: {
    title: string;
    description: string;
    icon: string;
    confidence: number;
  }[];
}

export interface InsightData {
  title: string;
  description: string;
  type: 'pattern' | 'trigger' | 'improvement' | 'warning';
  data?: any;
  action_items?: string[];
}

// Chart Data Types
export interface ChartDataPoint {
  date: string;
  value: number;
  label?: string;
}

export interface AttackHistoryData {
  date: string;
  attack: boolean;
  severity?: number;
  predicted_risk: number;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Form Types
export interface FormStep {
  id: string;
  title: string;
  description: string;
  icon: string;
}

// Notification Types
export interface Notification {
  id: string;
  type: 'prediction' | 'reminder' | 'insight' | 'achievement';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action_url?: string;
}

// Settings Types
export interface UserSettings {
  notifications_enabled: boolean;
  morning_reminder_time: string;
  evening_reminder_time: string;
  weather_alerts: boolean;
  weekly_report: boolean;
  share_anonymous_data: boolean;
  dark_mode: boolean;
}