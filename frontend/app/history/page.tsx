'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import {
  ChevronLeft,
  ChevronRight,
  Calendar as CalendarIcon,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Info,
} from 'lucide-react';
import { Button, Card, Badge, Skeleton } from '@/components/ui';
import { apiClient } from '@/lib/api';
import { DailyLogEntry } from '@/lib/types';
import { cn, formatDate, getRiskColor } from '@/lib/utils';

const DAYS_OF_WEEK = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];

export default function HistoryPage() {
  const router = useRouter();
  const [currentDate, setCurrentDate] = useState(new Date());
  const [history, setHistory] = useState<DailyLogEntry[]>([]);
  const [selectedDay, setSelectedDay] = useState<DailyLogEntry | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [currentDate]);

  const loadData = async () => {
    const userId = localStorage.getItem('user_id');
    if (!userId) {
      router.push('/onboarding');
      return;
    }

    try {
      const res = await apiClient.getLogHistory(userId, 90);
      if (res.data) setHistory(res.data);
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getDaysInMonth = (date: Date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDay = firstDay.getDay();

    return { daysInMonth, startingDay };
  };

  const getLogForDate = (day: number) => {
    const dateStr = `${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    return history.find((h) => h.date === dateStr);
  };

  const { daysInMonth, startingDay } = getDaysInMonth(currentDate);

  const prevMonth = () => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
    setSelectedDay(null);
  };

  const nextMonth = () => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
    setSelectedDay(null);
  };

  const goToToday = () => {
    setCurrentDate(new Date());
    setSelectedDay(null);
  };

  // Calculate stats for current month
  const monthLogs = history.filter((h) => {
    const logDate = new Date(h.date);
    return (
      logDate.getMonth() === currentDate.getMonth() &&
      logDate.getFullYear() === currentDate.getFullYear()
    );
  });

  const migrainesDays = monthLogs.filter((h) => h.migraine_occurred).length;
  const totalLogs = monthLogs.length;
  const correctPredictions = monthLogs.filter(
    (h) => h.prediction_was_correct
  ).length;
  const accuracy = totalLogs > 0 ? (correctPredictions / totalLogs) * 100 : 0;

  if (isLoading) {
    return <HistorySkeleton />;
  }

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
              <CalendarIcon className="w-8 h-8 text-purple-400" />
              History
            </h1>
            <p className="text-slate-400 mt-1">Track your migraine patterns over time</p>
          </div>
          <Button variant="secondary" onClick={goToToday}>
            Go to Today
          </Button>
        </motion.div>

        {/* Month Stats */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          <Card className="p-4 text-center">
            <div className="text-3xl font-bold text-red-400">{migrainesDays}</div>
            <div className="text-sm text-slate-500">Migraine Days</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-3xl font-bold text-emerald-400">{totalLogs - migrainesDays}</div>
            <div className="text-sm text-slate-500">Clear Days</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-3xl font-bold text-neural-400">{totalLogs}</div>
            <div className="text-sm text-slate-500">Days Logged</div>
          </Card>
          <Card className="p-4 text-center">
            <div className="text-3xl font-bold text-purple-400">{accuracy.toFixed(0)}%</div>
            <div className="text-sm text-slate-500">Prediction Accuracy</div>
          </Card>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Calendar */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <Card className="p-6">
              {/* Month Navigation */}
              <div className="flex items-center justify-between mb-6">
                <button
                  onClick={prevMonth}
                  className="p-2 rounded-lg hover:bg-slate-700 transition-colors"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <h2 className="text-xl font-semibold">
                  {MONTHS[currentDate.getMonth()]} {currentDate.getFullYear()}
                </h2>
                <button
                  onClick={nextMonth}
                  className="p-2 rounded-lg hover:bg-slate-700 transition-colors"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>

              {/* Day Headers */}
              <div className="grid grid-cols-7 gap-1 mb-2">
                {DAYS_OF_WEEK.map((day) => (
                  <div
                    key={day}
                    className="text-center text-xs font-medium text-slate-500 py-2"
                  >
                    {day}
                  </div>
                ))}
              </div>

              {/* Calendar Grid */}
              <div className="grid grid-cols-7 gap-1">
                {/* Empty cells for days before start */}
                {Array.from({ length: startingDay }).map((_, i) => (
                  <div key={`empty-${i}`} className="aspect-square" />
                ))}

                {/* Day cells */}
                {Array.from({ length: daysInMonth }).map((_, i) => {
                  const day = i + 1;
                  const log = getLogForDate(day);
                  const isToday =
                    new Date().toDateString() ===
                    new Date(
                      currentDate.getFullYear(),
                      currentDate.getMonth(),
                      day
                    ).toDateString();
                  const isSelected = selectedDay?.date === log?.date;
                  const isFuture = new Date(currentDate.getFullYear(), currentDate.getMonth(), day) > new Date();

                  return (
                    <button
                      key={day}
                      onClick={() => log && setSelectedDay(log)}
                      disabled={!log || isFuture}
                      className={cn(
                        'aspect-square rounded-lg flex flex-col items-center justify-center relative transition-all duration-200',
                        isToday && 'ring-2 ring-neural-500',
                        isSelected && 'ring-2 ring-white',
                        log && !isFuture
                          ? 'hover:bg-slate-700 cursor-pointer'
                          : 'cursor-default opacity-50',
                        log?.migraine_occurred
                          ? 'bg-red-500/20'
                          : log
                          ? 'bg-emerald-500/20'
                          : 'bg-slate-800/30'
                      )}
                    >
                      <span
                        className={cn(
                          'text-sm font-medium',
                          isToday && 'text-neural-400',
                          log?.migraine_occurred && 'text-red-400',
                          log && !log.migraine_occurred && 'text-emerald-400'
                        )}
                      >
                        {day}
                      </span>
                      {log && (
                        <span className="text-xs mt-0.5">
                          {log.migraine_occurred ? '😣' : '✓'}
                        </span>
                      )}
                      {log?.prediction_was_correct === false && (
                        <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-amber-500" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Legend */}
              <div className="flex items-center justify-center gap-6 mt-6 pt-4 border-t border-slate-700">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-emerald-500/20" />
                  <span className="text-xs text-slate-400">Clear</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-red-500/20" />
                  <span className="text-xs text-slate-400">Migraine</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-amber-500" />
                  <span className="text-xs text-slate-400">Wrong Prediction</span>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Day Detail Panel */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="p-6 h-full">
              {selectedDay ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold">
                      {formatDate(selectedDay.date, 'long')}
                    </h3>
                    <Badge
                      variant={selectedDay.migraine_occurred ? 'danger' : 'success'}
                    >
                      {selectedDay.migraine_occurred ? 'Migraine' : 'Clear'}
                    </Badge>
                  </div>

                  {/* Prediction vs Outcome */}
                  <div className="p-4 bg-slate-800/50 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-400">Predicted Risk</span>
                      <span
                        className={cn(
                          'font-bold',
                          getRiskColor(selectedDay.predicted_risk_level || 'LOW')
                        )}
                      >
                        {selectedDay.predicted_risk_level?.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {selectedDay.prediction_was_correct ? (
                        <>
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          <span className="text-sm text-emerald-400">
                            Correct prediction
                          </span>
                        </>
                      ) : (
                        <>
                          <XCircle className="w-4 h-4 text-amber-400" />
                          <span className="text-sm text-amber-400">
                            Prediction was off
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Key Factors */}
                  <div>
                    <h4 className="text-sm font-medium text-slate-400 mb-3">
                      Key Factors That Day
                    </h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-500">Sleep</span>
                        <span
                          className={cn(
                            selectedDay.sleep_hours < 6
                              ? 'text-red-400'
                              : 'text-emerald-400'
                          )}
                        >
                          {selectedDay.sleep_hours}h
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-500">Stress</span>
                        <span
                          className={cn(
                            selectedDay.stress_level > 7
                              ? 'text-red-400'
                              : selectedDay.stress_level > 5
                              ? 'text-amber-400'
                              : 'text-emerald-400'
                          )}
                        >
                          {selectedDay.stress_level}/10
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-500">Sleep Quality</span>
                        <span
                          className={
                            selectedDay.sleep_quality_good
                              ? 'text-emerald-400'
                              : 'text-amber-400'
                          }
                        >
                          {selectedDay.sleep_quality_good ? 'Good' : 'Poor'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Migraine Details */}
                  {selectedDay.migraine_occurred && selectedDay.migraine_details && (
                    <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                      <h4 className="text-sm font-medium text-red-400 mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        Migraine Details
                      </h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Severity</span>
                          <span className="text-white">
                            {selectedDay.migraine_details.severity}/10
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Duration</span>
                          <span className="text-white">
                            {selectedDay.migraine_details.duration_hours}h
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Location</span>
                          <span className="text-white capitalize">
                            {selectedDay.migraine_details.location}
                          </span>
                        </div>
                        {selectedDay.migraine_details.with_aura && (
                          <Badge variant="info" className="text-xs">
                            With Aura
                          </Badge>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Top Triggers */}
                  {selectedDay.top_triggers && selectedDay.top_triggers.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-3">
                        Top Triggers
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedDay.top_triggers.map((trigger) => (
                          <Badge key={trigger} variant="default">
                            {trigger}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center">
                  <Info className="w-12 h-12 text-slate-600 mb-4" />
                  <h3 className="font-medium text-slate-400 mb-2">Select a Day</h3>
                  <p className="text-sm text-slate-500">
                    Click on a logged day in the calendar to view details
                  </p>
                </div>
              )}
            </Card>
          </motion.div>
        </div>

        {/* Recent Log List */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-neural-400" />
              Recent Activity
            </h3>
            <div className="space-y-3">
              {history.slice(0, 10).map((log, index) => (
                <motion.div
                  key={log.date}
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.5 + index * 0.05 }}
                  onClick={() => setSelectedDay(log)}
                  className={cn(
                    'flex items-center justify-between p-4 rounded-xl cursor-pointer transition-all duration-200',
                    log.migraine_occurred
                      ? 'bg-red-500/10 hover:bg-red-500/20'
                      : 'bg-slate-800/50 hover:bg-slate-700/50'
                  )}
                >
                  <div className="flex items-center gap-4">
                    <div
                      className={cn(
                        'w-10 h-10 rounded-lg flex items-center justify-center text-xl',
                        log.migraine_occurred ? 'bg-red-500/20' : 'bg-emerald-500/20'
                      )}
                    >
                      {log.migraine_occurred ? '😣' : '😊'}
                    </div>
                    <div>
                      <div className="font-medium">{formatDate(log.date, 'long')}</div>
                      <div className="text-sm text-slate-500">
                        Sleep: {log.sleep_hours}h • Stress: {log.stress_level}/10
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {log.prediction_was_correct ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <XCircle className="w-5 h-5 text-amber-400" />
                    )}
                    <Badge
                      variant={log.migraine_occurred ? 'danger' : 'success'}
                    >
                      {log.migraine_occurred ? 'Migraine' : 'Clear'}
                    </Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}

function HistorySkeleton() {
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

        <div className="grid lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 p-6">
            <Skeleton className="h-8 w-48 mb-6 mx-auto" />
            <div className="grid grid-cols-7 gap-1">
              {Array.from({ length: 35 }).map((_, i) => (
                <Skeleton key={i} className="aspect-square rounded-lg" />
              ))}
            </div>
          </Card>
          <Card className="p-6">
            <Skeleton className="h-full min-h-[300px] w-full" />
          </Card>
        </div>
      </div>
    </div>
  );
}