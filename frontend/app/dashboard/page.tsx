'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Sun,
  Moon,
  AlertTriangle,
  TrendingUp,
  Calendar,
  Zap,
  ChevronRight,
  RefreshCw,
  Info,
  CheckCircle2,
  XCircle,
  Clock,
  Sparkles,
} from 'lucide-react';
import { Button, Card, Badge, ProgressRing, Skeleton } from '@/components/ui';
import { apiClient } from '@/lib/api';
import { PredictionResponse, UserProfile, DailyLogEntry } from '@/lib/types';
import { 
  cn, 
  formatDate, 
  getGreeting, 
  getRiskColor, 
  getRiskBgColor,
  getPhaseInfo,
  formatPercentage,
  getRiskGradient,
} from '@/lib/utils';

export default function DashboardPage() {
  const router = useRouter();
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [history, setHistory] = useState<DailyLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showOutcomeModal, setShowOutcomeModal] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    const userId = localStorage.getItem('user_id');
    if (!userId) {
      router.push('/onboarding');
      return;
    }

    try {
      const [predictionRes, profileRes, historyRes] = await Promise.all([
        apiClient.getPrediction(userId),
        apiClient.getProfile(userId),
        apiClient.getLogHistory(userId, 7),
      ]);

      if (predictionRes.data) setPrediction(predictionRes.data);
      if (profileRes.data) setProfile(profileRes.data);
      if (historyRes.data) setHistory(historyRes.data);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshPrediction = async () => {
    setIsRefreshing(true);
    const userId = localStorage.getItem('user_id');
    if (userId) {
      const res = await apiClient.getPrediction(userId);
      if (res.data) setPrediction(res.data);
    }
    setIsRefreshing(false);
  };

  const handleOutcomeSubmit = async (hadMigraine: boolean) => {
    const userId = localStorage.getItem('user_id');
    if (!userId) return;

    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const dateStr = yesterday.toISOString().split('T')[0];

    await apiClient.updateOutcome(userId, dateStr, hadMigraine);
    setShowOutcomeModal(false);
    loadData();
  };

  if (isLoading) {
    return <DashboardSkeleton />;
  }

  const riskPercentage = prediction ? Math.round(prediction.attack_probability * 100) : 0;
  const phaseInfo = profile ? getPhaseInfo(profile.current_phase) : null;

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="flex flex-col md:flex-row md:items-center justify-between gap-4"
        >
          <div>
            <h1 className="text-2xl md:text-3xl font-display font-bold">
              {getGreeting()}, {profile?.gender === 'F' ? '👩' : '👨'}
            </h1>
            <p className="text-slate-400 mt-1">
              {formatDate(new Date(), 'long')}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {phaseInfo && (
              <Badge variant="info" className="flex items-center gap-1">
                <Sparkles className="w-3 h-3" />
                {phaseInfo.label} Model
              </Badge>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={refreshPrediction}
              className={cn(isRefreshing && 'animate-spin')}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </motion.div>

        {/* Yesterday's Outcome Prompt */}
        {showOutcomeModal && (
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
          >
            <Card className="p-6 border-amber-500/50 bg-amber-500/10">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="font-semibold text-lg">Yesterday&apos;s Check-in</h3>
                  <p className="text-slate-400 text-sm">Did you have a migraine yesterday?</p>
                </div>
                <div className="flex gap-3">
                  <Button
                    variant="danger"
                    onClick={() => handleOutcomeSubmit(true)}
                    leftIcon={<XCircle className="w-4 h-4" />}
                  >
                    Yes, I did
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => handleOutcomeSubmit(false)}
                    leftIcon={<CheckCircle2 className="w-4 h-4" />}
                  >
                    No migraine
                  </Button>
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        {/* Main Prediction Card */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="p-6 md:p-8 neural-glow">
            <div className="flex flex-col lg:flex-row items-center gap-8">
              {/* Risk Circle */}
              <div className="flex-shrink-0">
                <ProgressRing
                  progress={riskPercentage}
                  size={180}
                  strokeWidth={12}
                  color={getRiskGradient(prediction?.attack_probability || 0)}
                >
                  <div className="text-center">
                    <div className="text-4xl font-bold" style={{ color: getRiskGradient(prediction?.attack_probability || 0) }}>
                      {riskPercentage}%
                    </div>
                    <div className="text-sm text-slate-400">Risk</div>
                  </div>
                </ProgressRing>
              </div>

              {/* Prediction Details */}
              <div className="flex-1 text-center lg:text-left">
                <div className="flex items-center justify-center lg:justify-start gap-2 mb-2">
                  {prediction?.risk_level === 'VERY_HIGH' || prediction?.risk_level === 'HIGH' ? (
                    <AlertTriangle className="w-6 h-6 text-red-400" />
                  ) : (
                    <Sun className="w-6 h-6 text-amber-400" />
                  )}
                  <h2 className="text-2xl font-bold">
                    <span className={getRiskColor(prediction?.risk_level || 'LOW')}>
                      {prediction?.risk_level?.replace('_', ' ')}
                    </span>{' '}
                    Risk Today
                  </h2>
                </div>
                <p className="text-slate-400 mb-4">
                  Based on your 14-day pattern analysis
                </p>

                <div className="flex flex-wrap justify-center lg:justify-start gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-sm text-slate-500">Confidence</div>
                    <div className="font-semibold text-neural-400">
                      {formatPercentage(prediction?.confidence || 0)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-slate-500">Expected Severity</div>
                    <div className="font-semibold text-amber-400">
                      {prediction?.severity_prediction?.toFixed(1)}/10
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-slate-500">Model Version</div>
                    <div className="font-semibold text-slate-300 text-xs font-mono">
                      {prediction?.model_version?.split('_').slice(-2).join('_')}
                    </div>
                  </div>
                </div>

                <Link href="/daily-log">
                  <Button>
                    Log Today&apos;s Data
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                </Link>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Triggers and Recommendations */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Top Triggers */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="p-6 h-full">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Zap className="w-5 h-5 text-amber-400" />
                  Top Triggers Today
                </h3>
                <Link href="/insights" className="text-sm text-neural-400 hover:text-neural-300">
                  View all
                </Link>
              </div>

              <div className="space-y-4">
                {prediction?.top_triggers?.slice(0, 4).map((trigger, index) => (
                  <motion.div
                    key={trigger.trigger}
                    initial={{ x: -20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    className="flex items-center gap-4"
                  >
                    <div 
                      className="w-10 h-10 rounded-lg flex items-center justify-center text-xl"
                      style={{ backgroundColor: `${trigger.color}20` }}
                    >
                      {trigger.icon}
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-medium text-sm">{trigger.trigger}</span>
                        <span className="text-sm text-slate-400">
                          {formatPercentage(trigger.contribution)}
                        </span>
                      </div>
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${trigger.contribution * 100}%` }}
                          transition={{ delay: 0.5 + index * 0.1, duration: 0.5 }}
                          className="h-full rounded-full"
                          style={{ backgroundColor: trigger.color }}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">{trigger.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>

          {/* Recommendations */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="p-6 h-full">
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-6">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
                Recommended Actions
              </h3>

              <div className="space-y-3">
                {prediction?.recommendations?.map((rec, index) => (
                  <motion.div
                    key={rec.action}
                    initial={{ x: 20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                    className={cn(
                      'p-4 rounded-xl border transition-all duration-300 hover:border-opacity-100',
                      rec.priority === 'high' 
                        ? 'bg-red-500/10 border-red-500/30' 
                        : rec.priority === 'medium'
                        ? 'bg-amber-500/10 border-amber-500/30'
                        : 'bg-slate-700/30 border-slate-600/30'
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-xl">{rec.icon}</span>
                      <div>
                        <p className="font-medium text-sm">{rec.action}</p>
                        <p className="text-xs text-slate-500 mt-1">{rec.reason}</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        </div>

        {/* Contributing Factors */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="p-6">
            <h3 className="text-lg font-semibold flex items-center gap-2 mb-6">
              <Info className="w-5 h-5 text-neural-400" />
              Contributing Factors
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {prediction?.contributing_factors?.map((factor, index) => (
                <div
                  key={factor.factor}
                  className={cn(
                    'p-4 rounded-xl border',
                    factor.status === 'critical' 
                      ? 'bg-red-500/10 border-red-500/30' 
                      : factor.status === 'warning'
                      ? 'bg-amber-500/10 border-amber-500/30'
                      : 'bg-slate-700/30 border-slate-600/30'
                  )}
                >
                  <div className="text-sm text-slate-400">{factor.factor}</div>
                  <div className={cn(
                    'text-2xl font-bold mt-1',
                    factor.status === 'critical' ? 'text-red-400' :
                    factor.status === 'warning' ? 'text-amber-400' : 'text-emerald-400'
                  )}>
                    {factor.value}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Threshold: {factor.threshold}
                  </div>
                </div>
              ))}
            </div>

            {/* Protective Factors */}
            {prediction?.protective_factors && prediction.protective_factors.length > 0 && (
              <div className="mt-6 pt-6 border-t border-slate-700">
                <h4 className="text-sm font-medium text-emerald-400 mb-3 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                  Protective Factors Active
                </h4>
                <div className="flex flex-wrap gap-2">
                  {prediction.protective_factors.map((factor) => (
                    <Badge key={factor} variant="success">
                      {factor}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </motion.div>

        {/* Recent History Mini */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Calendar className="w-5 h-5 text-purple-400" />
                Last 7 Days
              </h3>
              <Link href="/history" className="text-sm text-neural-400 hover:text-neural-300">
                View full history
              </Link>
            </div>

            <div className="flex items-end justify-between gap-2">
              {history.slice(0, 7).map((day, index) => (
                <div key={day.date} className="flex-1 text-center">
                  <div 
                    className={cn(
                      'mx-auto w-8 h-8 rounded-lg flex items-center justify-center mb-2',
                      day.migraine_occurred 
                        ? 'bg-red-500/20 text-red-400' 
                        : 'bg-emerald-500/20 text-emerald-400'
                    )}
                  >
                    {day.migraine_occurred ? '😣' : '✓'}
                  </div>
                  <div className="text-xs text-slate-500">
                    {formatDate(day.date, 'short')}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* User Phase Progress */}
        {profile && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Your Personalization Journey</h3>
              
              <div className="relative">
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-slate-400">Days Logged</span>
                  <span className="text-sm font-medium">{profile.days_logged} / 31</span>
                </div>
                <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-neural-500 via-purple-500 to-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min((profile.days_logged / 31) * 100, 100)}%` }}
                  />
                </div>
                
                {/* Milestones */}
                <div className="flex justify-between mt-4 text-xs">
                  <div className={cn(
                    'flex flex-col items-center',
                    profile.days_logged >= 1 ? 'text-emerald-400' : 'text-slate-500'
                  )}>
                    <div className={cn(
                      'w-6 h-6 rounded-full flex items-center justify-center mb-1',
                      profile.days_logged >= 1 ? 'bg-emerald-500' : 'bg-slate-700'
                    )}>
                      {profile.days_logged >= 1 ? '✓' : '1'}
                    </div>
                    <span>Foundation</span>
                  </div>
                  <div className={cn(
                    'flex flex-col items-center',
                    profile.days_logged >= 15 ? 'text-emerald-400' : 'text-slate-500'
                  )}>
                    <div className={cn(
                      'w-6 h-6 rounded-full flex items-center justify-center mb-1',
                      profile.days_logged >= 15 ? 'bg-emerald-500' : 'bg-slate-700'
                    )}>
                      {profile.days_logged >= 15 ? '✓' : '15'}
                    </div>
                    <span>Temporal</span>
                  </div>
                  <div className={cn(
                    'flex flex-col items-center',
                    profile.days_logged >= 31 ? 'text-emerald-400' : 'text-slate-500'
                  )}>
                    <div className={cn(
                      'w-6 h-6 rounded-full flex items-center justify-center mb-1',
                      profile.days_logged >= 31 ? 'bg-emerald-500' : 'bg-slate-700'
                    )}>
                      {profile.days_logged >= 31 ? '✓' : '31'}
                    </div>
                    <span>Personalized</span>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <Skeleton className="h-8 w-48 mb-2" />
            <Skeleton className="h-4 w-32" />
          </div>
        </div>

        <Card className="p-8">
          <div className="flex flex-col lg:flex-row items-center gap-8">
            <Skeleton className="w-44 h-44 rounded-full" variant="circular" />
            <div className="flex-1 space-y-4">
              <Skeleton className="h-8 w-64" />
              <Skeleton className="h-4 w-48" />
              <div className="flex gap-4">
                <Skeleton className="h-16 w-24" />
                <Skeleton className="h-16 w-24" />
                <Skeleton className="h-16 w-24" />
              </div>
            </div>
          </div>
        </Card>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="p-6">
            <Skeleton className="h-6 w-40 mb-6" />
            <div className="space-y-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="flex items-center gap-4">
                  <Skeleton className="w-10 h-10 rounded-lg" />
                  <div className="flex-1">
                    <Skeleton className="h-4 w-full mb-2" />
                    <Skeleton className="h-2 w-full" />
                  </div>
                </div>
              ))}
            </div>
          </Card>
          <Card className="p-6">
            <Skeleton className="h-6 w-40 mb-6" />
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-20 w-full rounded-xl" />
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}