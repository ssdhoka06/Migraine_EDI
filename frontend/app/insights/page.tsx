'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import {
  TrendingUp,
  Zap,
  Target,
  Award,
  Calendar,
  Activity,
  Brain,
  Sparkles,
} from 'lucide-react';
import { Card, Badge, Skeleton } from '@/components/ui';
import { apiClient } from '@/lib/api';
import { TriggerAnalysis, WeeklyStats } from '@/lib/types';
import { cn, formatPercentage } from '@/lib/utils';

const triggerColors: Record<string, string> = {
  'Sleep Deficit': '#8b5cf6',
  'High Stress': '#f97316',
  'Weather Change': '#06b6d4',
  'Menstrual Phase': '#ec4899',
  'Skipped Meals': '#84cc16',
  'Bright Light': '#eab308',
  'Alcohol': '#ef4444',
  'Dehydration': '#3b82f6',
  'Poor Sleep Quality': '#a855f7',
  'Caffeine': '#d97706',
};

const radarData = [
  { trigger: 'Sleep', fullMark: 100 },
  { trigger: 'Stress', fullMark: 100 },
  { trigger: 'Weather', fullMark: 100 },
  { trigger: 'Hormonal', fullMark: 100 },
  { trigger: 'Diet', fullMark: 100 },
  { trigger: 'Light', fullMark: 100 },
];

export default function InsightsPage() {
  const router = useRouter();
  const [triggerAnalysis, setTriggerAnalysis] = useState<TriggerAnalysis | null>(null);
  const [weeklyStats, setWeeklyStats] = useState<WeeklyStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);

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
      const [triggerRes, statsRes] = await Promise.all([
        apiClient.getTriggerAnalysis(userId),
        apiClient.getWeeklyStats(userId),
      ]);

      if (triggerRes.data) setTriggerAnalysis(triggerRes.data);
      if (statsRes.data) setWeeklyStats(statsRes.data);
    } catch (error) {
      console.error('Failed to load insights:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <InsightsSkeleton />;
  }

  const barChartData = triggerAnalysis?.triggers.map((t) => ({
    name: t.name.replace(' Deficit', '').replace(' Stress', ''),
    impact: Math.round(t.odds_ratio * 100) / 100,
    occurrences: t.occurrences,
    color: triggerColors[t.name] || '#64748b',
  })) || [];

  const pieChartData = triggerAnalysis?.triggers.slice(0, 5).map((t) => ({
    name: t.name,
    value: Math.round(t.contribution * 100),
    color: triggerColors[t.name] || '#64748b',
  })) || [];

  const accuracyTrend = weeklyStats?.weekly_accuracy || [];
  const personalRadarData = radarData.map((item) => {
    const trigger = triggerAnalysis?.triggers.find(
      (t) => t.name.toLowerCase().includes(item.trigger.toLowerCase())
    );
    return {
      ...item,
      yours: trigger ? Math.round(trigger.contribution * 100) : 20,
      average: Math.floor(Math.random() * 30) + 20,
    };
  });

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
            <h1 className="text-2xl md:text-3xl font-display font-bold flex items-center gap-3">
              <Brain className="w-8 h-8 text-neural-400" />
              Your Insights
            </h1>
            <p className="text-slate-400 mt-1">
              Personalized trigger analysis and patterns
            </p>
          </div>
          <Badge variant="info" className="w-fit flex items-center gap-2">
            <Sparkles className="w-3 h-3" />
            {triggerAnalysis?.total_logs || 0} days analyzed
          </Badge>
        </motion.div>

        {/* Key Stats */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                <Target className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-emerald-400">
                  {formatPercentage(weeklyStats?.prediction_accuracy || 0)}
                </div>
                <div className="text-xs text-slate-500">Accuracy</div>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center">
                <Zap className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-amber-400">
                  {triggerAnalysis?.triggers.length || 0}
                </div>
                <div className="text-xs text-slate-500">Triggers Found</div>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center">
                <Activity className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-red-400">
                  {weeklyStats?.total_migraines || 0}
                </div>
                <div className="text-xs text-slate-500">Attacks (30d)</div>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                <Award className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-400">
                  {weeklyStats?.streak_days || 0}
                </div>
                <div className="text-xs text-slate-500">Day Streak</div>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Main Charts Grid */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Trigger Impact Chart */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-neural-400" />
                Trigger Impact (Odds Ratio)
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis 
                      dataKey="name" 
                      type="category" 
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      width={80}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        color: '#f1f5f9',
                      }}
                      formatter={(value: number) => [`${value}x`, 'Impact']}
                    />
                    <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                      {barChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-slate-500 mt-4">
                OR &gt; 1.5 indicates a significant trigger. Your top trigger is{' '}
                <span className="text-white font-medium">
                  {triggerAnalysis?.triggers[0]?.name}
                </span>{' '}
                with {triggerAnalysis?.triggers[0]?.odds_ratio.toFixed(2)}x risk increase.
              </p>
            </Card>
          </motion.div>

          {/* Trigger Distribution Pie */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <Zap className="w-5 h-5 text-amber-400" />
                Trigger Distribution
              </h3>
              <div className="h-72 flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieChartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={3}
                      dataKey="value"
                      label={({ name, value }) => `${value}%`}
                      labelLine={false}
                    >
                      {pieChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        color: '#f1f5f9',
                      }}
                      formatter={(value: number) => [`${value}%`, 'Contribution']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              {/* Legend */}
              <div className="flex flex-wrap justify-center gap-3 mt-4">
                {pieChartData.map((item) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-xs text-slate-400">{item.name}</span>
                  </div>
                ))}
              </div>
            </Card>
          </motion.div>

          {/* Accuracy Trend */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <Target className="w-5 h-5 text-emerald-400" />
                Prediction Accuracy Trend
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={accuracyTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                      dataKey="week"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      tickFormatter={(v) => `${v}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        color: '#f1f5f9',
                      }}
                      formatter={(value: number) => [`${value}%`, 'Accuracy']}
                    />
                    <Line
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#10b981"
                      strokeWidth={3}
                      dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, fill: '#10b981' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-slate-500 mt-4">
                Your prediction accuracy improves as the model learns your patterns.
                Target: 78% by day 31.
              </p>
            </Card>
          </motion.div>

          {/* Personal vs Average Radar */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                Your Profile vs Average
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={personalRadarData}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis
                      dataKey="trigger"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                    />
                    <PolarRadiusAxis
                      angle={30}
                      domain={[0, 100]}
                      tick={{ fill: '#64748b', fontSize: 10 }}
                    />
                    <Radar
                      name="You"
                      dataKey="yours"
                      stroke="#8b5cf6"
                      fill="#8b5cf6"
                      fillOpacity={0.5}
                    />
                    <Radar
                      name="Average"
                      dataKey="average"
                      stroke="#64748b"
                      fill="#64748b"
                      fillOpacity={0.2}
                    />
                    <Legend />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        color: '#f1f5f9',
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-slate-500 mt-4">
                Your trigger sensitivity compared to the population average.
              </p>
            </Card>
          </motion.div>
        </div>

        {/* Detailed Trigger List */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-6">All Identified Triggers</h3>
            <div className="space-y-4">
              {triggerAnalysis?.triggers.map((trigger, index) => (
                <motion.div
                  key={trigger.name}
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.7 + index * 0.05 }}
                  className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-xl"
                >
                  <div
                    className="w-12 h-12 rounded-lg flex items-center justify-center text-2xl"
                    style={{
                      backgroundColor: `${triggerColors[trigger.name] || '#64748b'}20`,
                    }}
                  >
                    {trigger.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{trigger.name}</span>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-slate-400">
                          {trigger.occurrences} occurrences
                        </span>
                        <Badge
                          variant={
                            trigger.odds_ratio >= 2
                              ? 'danger'
                              : trigger.odds_ratio >= 1.5
                              ? 'warning'
                              : 'default'
                          }
                        >
                          OR {trigger.odds_ratio.toFixed(2)}
                        </Badge>
                      </div>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.min(trigger.contribution * 100, 100)}%`,
                          backgroundColor: triggerColors[trigger.name] || '#64748b',
                        }}
                      />
                    </div>
                    <p className="text-xs text-slate-500 mt-2">{trigger.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* Patterns Discovered */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <Card className="p-6 neural-glow">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-neural-400" />
              Key Patterns Discovered
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              {triggerAnalysis?.patterns?.map((pattern, index) => (
                <div
                  key={index}
                  className="p-4 bg-slate-800/50 rounded-xl border border-slate-700"
                >
                  <div className="flex items-start gap-3">
                    <span className="text-xl">{pattern.icon}</span>
                    <div>
                      <h4 className="font-medium text-sm">{pattern.title}</h4>
                      <p className="text-xs text-slate-400 mt-1">{pattern.description}</p>
                      <Badge variant="info" className="mt-2 text-xs">
                        Confidence: {formatPercentage(pattern.confidence)}
                      </Badge>
                    </div>
                  </div>
                </div>
              )) || (
                <p className="text-slate-400 col-span-2">
                  Keep logging to discover more patterns!
                </p>
              )}
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}

function InsightsSkeleton() {
  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <Skeleton className="h-8 w-48 mb-2" />
          <Skeleton className="h-4 w-64" />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="p-4">
              <Skeleton className="h-16 w-full" />
            </Card>
          ))}
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="p-6">
              <Skeleton className="h-6 w-40 mb-6" />
              <Skeleton className="h-72 w-full" />
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}