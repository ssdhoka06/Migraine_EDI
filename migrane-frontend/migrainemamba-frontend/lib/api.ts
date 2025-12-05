// lib/api.ts

import axios from 'axios';
import { OnboardingData, DailyLogData, PredictionResponse } from './types';

// Change this to your backend URL when ready
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API Functions
export const apiClient = {
  // Onboarding
  async submitOnboarding(data: OnboardingData) {
    const response = await api.post('/onboarding', data);
    return response.data;
  },

  // Daily Log
  async submitDailyLog(data: DailyLogData) {
    const response = await api.post('/log/daily', data);
    return response.data;
  },

  // Get Prediction
  async getPrediction(userId: string, date?: string) {
    const params = date ? { date } : {};
    const response = await api.get<PredictionResponse>(`/predict/${userId}`, { params });
    return response.data;
  },

  // Get user's log history
  async getLogHistory(userId: string, limit: number = 30) {
    const response = await api.get(`/logs/${userId}?limit=${limit}`);
    return response.data;
  },
};

export default api;