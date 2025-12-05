import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Tailwind class merge utility
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Format date for display
export function formatDate(date: string | Date, format: 'short' | 'long' | 'time' = 'short'): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  
  if (format === 'short') {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } else if (format === 'long') {
    return d.toLocaleDateString('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  } else {
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  }
}

// Get relative time string
export function getRelativeTime(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return formatDate(d, 'short');
}

// Calculate BMI
export function calculateBMI(height: number, weight: number): number {
  const heightInMeters = height / 100;
  return Number((weight / (heightInMeters * heightInMeters)).toFixed(1));
}

// Get risk level color
export function getRiskColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'LOW':
      return 'text-emerald-400';
    case 'MODERATE':
      return 'text-amber-400';
    case 'HIGH':
      return 'text-orange-500';
    case 'VERY_HIGH':
    case 'CRITICAL':
      return 'text-red-500';
    default:
      return 'text-slate-400';
  }
}

// Get risk background color
export function getRiskBgColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'LOW':
      return 'bg-emerald-500/20 border-emerald-500/30';
    case 'MODERATE':
      return 'bg-amber-500/20 border-amber-500/30';
    case 'HIGH':
      return 'bg-orange-500/20 border-orange-500/30';
    case 'VERY_HIGH':
    case 'CRITICAL':
      return 'bg-red-500/20 border-red-500/30';
    default:
      return 'bg-slate-500/20 border-slate-500/30';
  }
}

// Get phase display info
export function getPhaseInfo(phase: string): { label: string; description: string; color: string } {
  switch (phase) {
    case 'foundation':
      return {
        label: 'Foundation',
        description: 'Building your baseline with general predictions',
        color: 'text-blue-400'
      };
    case 'generic':
      return {
        label: 'Temporal',
        description: 'Using 14-day patterns for predictions',
        color: 'text-purple-400'
      };
    case 'personalized':
      return {
        label: 'Personalized',
        description: 'Custom model trained on your unique patterns',
        color: 'text-emerald-400'
      };
    default:
      return {
        label: 'Unknown',
        description: '',
        color: 'text-slate-400'
      };
  }
}

// Get days until next menstrual phase
export function getMenstrualInfo(cycleStartDay: string | undefined): { day: number; phase: string; isHighRisk: boolean } | null {
  if (!cycleStartDay) return null;
  
  const start = new Date(cycleStartDay);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - start.getTime()) / 86400000);
  const day = ((diffDays % 28) + 28) % 28; // Handle negative modulo
  
  let phase: string;
  let isHighRisk = false;
  
  if (day <= 1 || day >= 26) {
    phase = 'Menstrual';
    isHighRisk = true;
  } else if (day <= 5) {
    phase = 'Follicular';
  } else if (day <= 14) {
    phase = 'Ovulation';
  } else {
    phase = 'Luteal';
  }
  
  return { day, phase, isHighRisk };
}

// Format percentage
export function formatPercentage(value: number, decimals: number = 0): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

// Format number with suffix
export function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num.toString();
}

// Get greeting based on time of day
export function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 17) return 'Good afternoon';
  return 'Good evening';
}

// Validate email
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Debounce function
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

// Generate unique ID
export function generateId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

// Sleep function for animations
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Interpolate color between two hex values
export function interpolateColor(color1: string, color2: string, factor: number): string {
  const hex = (x: number) => {
    const h = Math.round(x).toString(16);
    return h.length === 1 ? '0' + h : h;
  };

  const r1 = parseInt(color1.slice(1, 3), 16);
  const g1 = parseInt(color1.slice(3, 5), 16);
  const b1 = parseInt(color1.slice(5, 7), 16);

  const r2 = parseInt(color2.slice(1, 3), 16);
  const g2 = parseInt(color2.slice(3, 5), 16);
  const b2 = parseInt(color2.slice(5, 7), 16);

  const r = Math.round(r1 + factor * (r2 - r1));
  const g = Math.round(g1 + factor * (g2 - g1));
  const b = Math.round(b1 + factor * (b2 - b1));

  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

// Get color for risk probability
export function getRiskGradient(probability: number): string {
  if (probability < 0.3) {
    return interpolateColor('#10b981', '#f59e0b', probability / 0.3);
  } else if (probability < 0.6) {
    return interpolateColor('#f59e0b', '#f97316', (probability - 0.3) / 0.3);
  } else {
    return interpolateColor('#f97316', '#ef4444', (probability - 0.6) / 0.4);
  }
}