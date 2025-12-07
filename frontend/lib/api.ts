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

// Mock Data Generators (fallback when backend unavailable)
const generateMockPrediction = (userId: string): PredictionResponse => {
  const probability = Math.random() * 0.4 + 0.2; // 20-60%
  const riskLevel = probability < 0.3 ? 'LOW' : probability < 0.5 ? 'MODERATE' : probability < 0.7 ? 'HIGH' : 'VERY_HIGH';
  
  return {
    user_id: userId,
    date: new Date().toISOString().split('T')[0],
    prediction_time: new Date().toISOString(),
    attack_probability: probability,
    risk_level: riskLevel,
    severity_prediction: Math.random() * 4 + 5,
    model_version: 'lightgbm_foundation_v1',
    confidence: 0.76,
    phase: 'foundation',
    top_triggers: [
      { trigger: 'Sleep Deficit', contribution: 0.35, icon: '😴', color: '#8b5cf6', description: 'Sleep patterns affect risk' },
      { trigger: 'Stress Level', contribution: 0.25, icon: '😰', color: '#f97316', description: 'Elevated stress detected' },
    ],
    recommendations: [
      { priority: 'high', action: 'Ensure 7-8 hours of sleep', reason: 'Sleep is key for prevention', icon: '🛏️' },
      { priority: 'medium', action: 'Stay hydrated', reason: 'Drink 8 glasses of water', icon: '💧' },
    ],
    contributing_factors: [],
    protective_factors: []
  };
};

const generateMockProfile = (userId: string): UserProfile => {
  const stored = typeof window !== 'undefined' ? localStorage.getItem('user_profile') : null;
  if (stored) {
    return JSON.parse(stored);
  }
  return {
    id: userId,
    gender: 'F',
    age: 30,
    height: 165,
    weight: 60,
    bmi: 22.0,
    attacks_per_month: 4,
    location_city: 'Unknown',
    has_menstrual_cycle: false,
    created_at: new Date().toISOString(),
    days_logged: 0,
    current_phase: 'foundation',
    model_version: 'foundation_v1',
    personalization_week: 0
  };
};

const generateMockHistory = (days: number): DailyLogEntry[] => {
  const history: DailyLogEntry[] = [];
  const today = new Date();
  
  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const predicted = Math.random() * 0.5 + 0.2;
    const migraine = Math.random() < 0.15;
    
    history.push({
      date: date.toISOString().split('T')[0],
      migraine_occurred: migraine,
      severity: migraine ? Math.floor(Math.random() * 4) + 5 : undefined,
      risk_level: predicted < 0.3 ? 'LOW' : predicted < 0.5 ? 'MODERATE' : 'HIGH',
      predicted_probability: predicted,
      predicted_risk_level: predicted < 0.3 ? 'LOW' : predicted < 0.5 ? 'MODERATE' : 'HIGH',
      prediction_was_correct: true,
      sleep_hours: Math.random() * 3 + 5,
      stress_level: Math.floor(Math.random() * 5) + 3,
      sleep_quality_good: Math.random() > 0.3,
      top_triggers: [],
      migraine_details: undefined,
      prediction_accuracy: 'true_negative' as const,
    });
  }
  return history;
};

// API Client - Updated to match backend routes
export const apiClient = {
  // Onboarding - POST /api/v1/users/onboarding
  async submitOnboarding(data: OnboardingData): Promise<ApiResponse<{ user_id: string }>> {
    try {
      const response = await api.post('/users/onboarding', data);
      // Backend returns { success, data: { user_id, ... }, message }
      const userId = response.data?.data?.user_id || response.data?.user_id;
      if (userId) {
        localStorage.setItem('user_id', userId);
        localStorage.setItem('user_profile', JSON.stringify({
          ...data,
          id: userId,
          created_at: new Date().toISOString(),
          days_logged: 0,
          current_phase: 'foundation',
          model_version: 'lightgbm_foundation_v1',
          personalization_week: 0
        }));
      }
      return { success: true, data: { user_id: userId } };
    } catch (error: any) {
      console.error('Onboarding failed, using local fallback:', error.response?.data || error.message);
      // Fallback: create local user ID
      const userId = `user_${Date.now()}`;
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
      return { success: true, data: { user_id: userId } };
    }
  },

  // Daily Log - POST /api/v1/logs/submit
  async submitDailyLog(data: DailyLogData): Promise<ApiResponse<{ success: boolean }>> {
    try {
      const response = await api.post('/logs/submit', data);
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error('Daily log submission failed:', error.response?.data || error.message);
      // Save locally as fallback
      if (typeof window !== 'undefined') {
        const logs = JSON.parse(localStorage.getItem('daily_logs') || '[]');
        logs.push({ ...data, submitted_at: new Date().toISOString() });
        localStorage.setItem('daily_logs', JSON.stringify(logs));
      }
      return { success: true, data: { success: true } };
    }
  },

  // Get Prediction - GET /api/v1/predictions/{user_id}
  async getPrediction(userId: string, date?: string): Promise<ApiResponse<PredictionResponse>> {
    try {
      const params = date ? { date } : {};
      const response = await api.get(`/predictions/${userId}`, { params });
      // Backend returns { success, data: { ... }, message, error }
      const predictionData = response.data?.data || response.data;
      
      // Map backend field names to frontend expected names
      const mappedData: PredictionResponse = {
        user_id: predictionData.user_id,
        date: predictionData.prediction_date || predictionData.date || new Date().toISOString().split('T')[0],
        prediction_time: predictionData.prediction_time || new Date().toISOString(),
        attack_probability: predictionData.attack_probability,
        risk_level: predictionData.risk_level,
        severity_prediction: predictionData.severity_prediction,
        model_version: predictionData.model_version,
        confidence: predictionData.confidence,
        phase: predictionData.model_type === 'foundation' ? 'foundation' : 
               predictionData.model_type === 'personalized' ? 'personalized' : 'generic',
        top_triggers: (predictionData.top_triggers || []).map((t: any) => ({
          trigger: t.trigger || t.name,
          contribution: t.contribution,
          icon: t.icon || '⚡',
          color: t.color || '#8b5cf6',
          description: t.description || ''
        })),
        recommendations: (predictionData.recommendations || []).map((r: any) => ({
          priority: r.priority || 'medium',
          action: r.action,
          reason: r.reason,
          icon: r.icon || '💡'
        })),
        contributing_factors: (predictionData.contributing_factors || []).map((f: any) => ({
          factor: f.factor,
          value: f.value,
          threshold: f.threshold,
          status: f.status || 'normal'
        })),
        protective_factors: predictionData.protective_factors || []
      };
      
      return { success: true, data: mappedData };
    } catch (error: any) {
      console.error('Get prediction failed, using mock:', error.response?.data || error.message);
      return { success: true, data: generateMockPrediction(userId) };
    }
  },

  // Get User Profile - GET /api/v1/users/profile/{user_id}
  async getProfile(userId: string): Promise<ApiResponse<UserProfile>> {
    try {
      const response = await api.get(`/users/profile/${userId}`);
      // Backend returns { success, data: { ... }, message, error }
      const profileData = response.data?.data || response.data;
      return { success: true, data: profileData };
    } catch (error: any) {
      console.error('Get profile failed, using mock:', error.response?.data || error.message);
      return { success: true, data: generateMockProfile(userId) };
    }
  },

  // Get Log History - GET /api/v1/logs/history/{user_id}
  async getLogHistory(userId: string, limit: number = 30): Promise<ApiResponse<DailyLogEntry[]>> {
    try {
      const response = await api.get(`/logs/history/${userId}`, { params: { limit } });
      // Backend returns { success, data: [...], message, error }
      const historyData = response.data?.data || response.data || [];
      return { success: true, data: Array.isArray(historyData) ? historyData : [] };
    } catch (error: any) {
      console.error('Get log history failed, using mock:', error.response?.data || error.message);
      return { success: true, data: generateMockHistory(limit) };
    }
  },

  // Get Recent Logs - GET /api/v1/logs/recent/{user_id}
  async getRecentLogs(userId: string, limit: number = 7): Promise<ApiResponse<DailyLogEntry[]>> {
    try {
      const response = await api.get(`/logs/recent/${userId}`, { params: { limit } });
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error('Get recent logs failed:', error.response?.data || error.message);
      return { success: true, data: generateMockHistory(limit) };
    }
  },

  // Get Trigger Analysis - GET /api/v1/insights/triggers/{user_id}
  async getTriggerAnalysis(userId: string): Promise<ApiResponse<TriggerAnalysis>> {
    try {
      const response = await api.get(`/insights/triggers/${userId}`);
      const triggerData = response.data?.data || response.data;
      return { success: true, data: triggerData };
    } catch (error: any) {
      console.error('Get trigger analysis failed:', error.response?.data || error.message);
      return {
        success: true,
        data: {
          total_logs: 0,
          triggers: [],
          patterns: []
        }
      };
    }
  },

  // Get Weekly Stats - GET /api/v1/insights/weekly-stats/{user_id}
  async getWeeklyStats(userId: string): Promise<ApiResponse<WeeklyStats>> {
    try {
      const response = await api.get(`/insights/weekly-stats/${userId}`);
      const statsData = response.data?.data || response.data;
      return { success: true, data: statsData };
    } catch (error: any) {
      console.error('Get weekly stats failed:', error.response?.data || error.message);
      return {
        success: true,
        data: {
          week_start: new Date().toISOString().split('T')[0],
          total_attacks: 0,
          total_migraines: 0,
          avg_severity: 0,
          prediction_accuracy: 0,
          most_common_triggers: [],
          improvement_from_last_week: 0,
          streak_days: 0,
          weekly_accuracy: []
        }
      };
    }
  },

  // Get Insights Summary - GET /api/v1/insights/summary/{user_id}
  async getInsightsSummary(userId: string): Promise<ApiResponse<any>> {
    try {
      const response = await api.get(`/insights/summary/${userId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error('Get insights summary failed:', error.response?.data || error.message);
      return { success: true, data: {} };
    }
  },

  // Get Recommendations - GET /api/v1/insights/recommendations/{user_id}
  async getRecommendations(userId: string): Promise<ApiResponse<any>> {
    try {
      const response = await api.get(`/insights/recommendations/${userId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error('Get recommendations failed:', error.response?.data || error.message);
      return { success: true, data: [] };
    }
  },

  // Update Migraine Outcome - PUT /api/v1/logs/outcome/{user_id}/{date_str}
  async updateOutcome(
    userId: string, 
    date: string, 
    migraineOccurred: boolean, 
    details?: any
  ): Promise<ApiResponse<{ success: boolean }>> {
    try {
      const response = await api.put(`/logs/outcome/${userId}/${date}`, {
        migraine_occurred: migraineOccurred,
        ...details
      });
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error('Update outcome failed:', error.response?.data || error.message);
      return { success: true, data: { success: true } };
    }
  },

  // Get Prediction History - GET /api/v1/predictions/history/{user_id}
  async getPredictionHistory(userId: string, limit: number = 30): Promise<ApiResponse<any[]>> {
    try {
      const response = await api.get(`/predictions/history/${userId}`, { params: { limit } });
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error('Get prediction history failed:', error.response?.data || error.message);
      return { success: true, data: [] };
    }
  },

  // Get Prediction Accuracy - GET /api/v1/predictions/accuracy/{user_id}
  async getPredictionAccuracy(userId: string): Promise<ApiResponse<any>> {
    try {
      const response = await api.get(`/predictions/accuracy/${userId}`);
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error('Get prediction accuracy failed:', error.response?.data || error.message);
      return { success: true, data: { accuracy: 0 } };
    }
  },
};

export default api;
