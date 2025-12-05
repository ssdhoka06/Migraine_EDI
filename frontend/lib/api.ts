import axios from 'axios';
import { 
  OnboardingData, 
  DailyLogData, 
  PredictionResponse, 
  UserProfile,
  DailyLogEntry,
  TriggerAnalysis,
  WeeklyStats,
  ApiResponse 
} from './types';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

// Request interceptor for adding auth token
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Mock Data Generator (for demo purposes when backend is not available)
const generateMockPrediction = (userId: string): PredictionResponse => {
  const probability = Math.random() * 0.6 + 0.2; // 20-80%
  const riskLevel = probability < 0.3 ? 'LOW' : probability < 0.5 ? 'MODERATE' : probability < 0.7 ? 'HIGH' : 'VERY_HIGH';
  
  return {
    user_id: userId,
    date: new Date().toISOString().split('T')[0],
    prediction_time: new Date().toISOString(),
    attack_probability: probability,
    risk_level: riskLevel,
    severity_prediction: Math.random() * 4 + 5, // 5-9
    model_version: 'mamba_personalized_v2',
    confidence: 0.78 + Math.random() * 0.15,
    phase: 'personalized',
    top_triggers: [
      {
        trigger: 'Sleep Deficit',
        contribution: 0.35,
        icon: '😴',
        color: '#8b5cf6',
        description: 'Only 5 hours of sleep last night'
      },
      {
        trigger: 'Weather Change',
        contribution: 0.28,
        icon: '🌡️',
        color: '#06b6d4',
        description: 'Barometric pressure dropping 8mb'
      },
      {
        trigger: 'Stress Level',
        contribution: 0.22,
        icon: '😰',
        color: '#f97316',
        description: 'Elevated stress for 3 consecutive days'
      },
      {
        trigger: 'Menstrual Phase',
        contribution: 0.15,
        icon: '📅',
        color: '#ec4899',
        description: 'Day 2 of menstrual cycle'
      }
    ],
    recommendations: [
      {
        priority: 'high',
        action: 'Take preventive medication',
        reason: 'High risk detected - consider preventive measures',
        icon: '💊'
      },
      {
        priority: 'high',
        action: 'Ensure 8+ hours of sleep tonight',
        reason: 'Sleep deficit is your #1 trigger',
        icon: '🛏️'
      },
      {
        priority: 'medium',
        action: 'Stay well hydrated',
        reason: 'Drink at least 8 glasses of water',
        icon: '💧'
      },
      {
        priority: 'medium',
        action: 'Avoid screens 1 hour before bed',
        reason: 'Reduce visual strain and improve sleep quality',
        icon: '📱'
      },
      {
        priority: 'low',
        action: 'Practice relaxation techniques',
        reason: 'Help manage elevated stress levels',
        icon: '🧘'
      }
    ],
    contributing_factors: [
      { factor: 'Sleep Hours', value: 5, threshold: 6, status: 'critical' },
      { factor: 'Pressure Change', value: -8, threshold: -5, status: 'warning' },
      { factor: 'Stress Level', value: 7, threshold: 6, status: 'warning' },
      { factor: 'Caffeine', value: 2, threshold: 3, status: 'normal' }
    ],
    protective_factors: [
      'Regular meal timing maintained',
      'No alcohol consumption',
      'Adequate hydration'
    ]
  };
};

const generateMockProfile = (userId: string): UserProfile => ({
  id: userId,
  gender: 'F',
  age: 32,
  height: 165,
  weight: 60,
  bmi: 22.0,
  attacks_per_month: 4,
  location_city: 'San Francisco',
  has_menstrual_cycle: true,
  cycle_start_day: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
  created_at: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000).toISOString(),
  days_logged: 45,
  current_phase: 'personalized',
  model_version: 'mamba_personalized_week_2',
  personalization_week: 2
});

const generateMockHistory = (days: number): DailyLogEntry[] => {
  const history: DailyLogEntry[] = [];
  const today = new Date();
  
  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    const migraine = Math.random() < 0.15; // 15% base rate
    const predicted = Math.random() * 0.6 + 0.2;
    
    history.push({
      date: date.toISOString().split('T')[0],
      migraine_occurred: migraine,
      severity: migraine ? Math.floor(Math.random() * 4) + 6 : undefined,
      risk_level: predicted < 0.3 ? 'LOW' : predicted < 0.5 ? 'MODERATE' : 'HIGH',
      predicted_probability: predicted,
      prediction_accuracy: migraine 
        ? (predicted > 0.5 ? 'correct' : 'false_negative')
        : (predicted < 0.5 ? 'true_negative' : 'false_positive')
    });
  }
  
  return history;
};

// API Client
export const apiClient = {
  // Onboarding
  async submitOnboarding(data: OnboardingData): Promise<ApiResponse<{ user_id: string }>> {
    try {
      const response = await api.post('/onboarding', data);
      return { success: true, data: response.data };
    } catch (error) {
      // Mock response for demo
      const userId = `user_${Date.now()}`;
      if (typeof window !== 'undefined') {
        localStorage.setItem('user_id', userId);
        localStorage.setItem('user_profile', JSON.stringify({
          ...data,
          id: userId,
          created_at: new Date().toISOString(),
          days_logged: 0,
          current_phase: 'foundation',
          model_version: 'foundation_v1',
          personalization_week: 0
        }));
      }
      return { success: true, data: { user_id: userId } };
    }
  },

  // Daily Log
  async submitDailyLog(data: DailyLogData): Promise<ApiResponse<{ success: boolean }>> {
    try {
      const response = await api.post('/log/daily', data);
      return { success: true, data: response.data };
    } catch (error) {
      // Mock: save to localStorage
      if (typeof window !== 'undefined') {
        const logs = JSON.parse(localStorage.getItem('daily_logs') || '[]');
        logs.push({ ...data, submitted_at: new Date().toISOString() });
        localStorage.setItem('daily_logs', JSON.stringify(logs));
        
        // Update days logged in profile
        const profile = JSON.parse(localStorage.getItem('user_profile') || '{}');
        profile.days_logged = (profile.days_logged || 0) + 1;
        
        // Update phase based on days logged
        if (profile.days_logged >= 31) {
          profile.current_phase = 'personalized';
        } else if (profile.days_logged >= 15) {
          profile.current_phase = 'generic';
        }
        
        localStorage.setItem('user_profile', JSON.stringify(profile));
      }
      return { success: true, data: { success: true } };
    }
  },

  // Get Prediction
  async getPrediction(userId: string, date?: string): Promise<ApiResponse<PredictionResponse>> {
    try {
      const params = date ? { date } : {};
      const response = await api.get<PredictionResponse>(`/predict/${userId}`, { params });
      return { success: true, data: response.data };
    } catch (error) {
      // Return mock prediction
      return { success: true, data: generateMockPrediction(userId) };
    }
  },

  // Get User Profile
  async getProfile(userId: string): Promise<ApiResponse<UserProfile>> {
    try {
      const response = await api.get<UserProfile>(`/users/${userId}`);
      return { success: true, data: response.data };
    } catch (error) {
      // Return from localStorage or generate mock
      if (typeof window !== 'undefined') {
        const stored = localStorage.getItem('user_profile');
        if (stored) {
          return { success: true, data: JSON.parse(stored) };
        }
      }
      return { success: true, data: generateMockProfile(userId) };
    }
  },

  // Get Log History
  async getLogHistory(userId: string, limit: number = 30): Promise<ApiResponse<DailyLogEntry[]>> {
    try {
      const response = await api.get(`/logs/${userId}?limit=${limit}`);
      return { success: true, data: response.data };
    } catch (error) {
      return { success: true, data: generateMockHistory(limit) };
    }
  },

  // Get Trigger Analysis
  async getTriggerAnalysis(userId: string): Promise<ApiResponse<TriggerAnalysis[]>> {
    try {
      const response = await api.get(`/analysis/triggers/${userId}`);
      return { success: true, data: response.data };
    } catch (error) {
      // Mock trigger analysis
      return {
        success: true,
        data: [
          {
            trigger_name: 'Sleep Deprivation',
            occurrence_rate: 0.42,
            attack_correlation: 0.68,
            odds_ratio: 3.98,
            confidence_interval: [2.1, 5.8],
            trend: 'stable',
            personalized_threshold: 5.5
          },
          {
            trigger_name: 'Barometric Pressure Drop',
            occurrence_rate: 0.31,
            attack_correlation: 0.52,
            odds_ratio: 1.27,
            confidence_interval: [0.9, 1.7],
            trend: 'stable'
          },
          {
            trigger_name: 'High Stress',
            occurrence_rate: 0.55,
            attack_correlation: 0.45,
            odds_ratio: 1.92,
            confidence_interval: [1.3, 2.8],
            trend: 'decreasing'
          },
          {
            trigger_name: 'Menstrual Phase',
            occurrence_rate: 0.07,
            attack_correlation: 0.78,
            odds_ratio: 2.04,
            confidence_interval: [1.5, 2.8],
            trend: 'stable'
          },
          {
            trigger_name: 'Alcohol (5+ drinks)',
            occurrence_rate: 0.05,
            attack_correlation: 0.62,
            odds_ratio: 2.08,
            confidence_interval: [1.4, 3.1],
            trend: 'decreasing'
          }
        ]
      };
    }
  },

  // Get Weekly Stats
  async getWeeklyStats(userId: string): Promise<ApiResponse<WeeklyStats>> {
    try {
      const response = await api.get(`/stats/weekly/${userId}`);
      return { success: true, data: response.data };
    } catch (error) {
      return {
        success: true,
        data: {
          week_start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          total_attacks: Math.floor(Math.random() * 3),
          avg_severity: 6.5 + Math.random() * 2,
          prediction_accuracy: 0.75 + Math.random() * 0.15,
          most_common_triggers: ['Sleep Deficit', 'Weather Change', 'Stress'],
          improvement_from_last_week: Math.random() * 20 - 10
        }
      };
    }
  },

  // Update yesterday's outcome
  async updateOutcome(userId: string, date: string, migraineOccurred: boolean, details?: any): Promise<ApiResponse<{ success: boolean }>> {
    try {
      const response = await api.put(`/logs/${userId}/${date}/outcome`, {
        migraine_occurred: migraineOccurred,
        details
      });
      return { success: true, data: response.data };
    } catch (error) {
      return { success: true, data: { success: true } };
    }
  }
};

export default api;