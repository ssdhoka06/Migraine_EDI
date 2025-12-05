'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm, Controller } from 'react-hook-form';
import {
  Moon,
  Coffee,
  Utensils,
  Wine,
  Sun,
  Brain,
  ChevronRight,
  ChevronLeft,
  Check,
  AlertTriangle,
  Droplets,
  Calendar,
} from 'lucide-react';
import { Button, Card, Slider, Toggle } from '@/components/ui';
import { apiClient } from '@/lib/api';
import { DailyLogData } from '@/lib/types';
import { cn, formatDate } from '@/lib/utils';

const steps = [
  { id: 'sleep', title: 'Sleep & Rest', icon: Moon, color: 'from-indigo-500 to-purple-500' },
  { id: 'diet', title: 'Diet & Hydration', icon: Utensils, color: 'from-green-500 to-emerald-500' },
  { id: 'lifestyle', title: 'Lifestyle', icon: Sun, color: 'from-amber-500 to-orange-500' },
  { id: 'symptoms', title: 'Warning Signs', icon: Brain, color: 'from-purple-500 to-pink-500' },
  { id: 'outcome', title: 'Yesterday', icon: Calendar, color: 'from-red-500 to-rose-500' },
];

const prodromalSymptoms = [
  { key: 'fatigue', label: 'Extreme Fatigue', emoji: '😴', description: 'Unusually tired' },
  { key: 'stiff_neck', label: 'Stiff Neck', emoji: '💢', description: 'Neck tension or pain' },
  { key: 'yawning', label: 'Excessive Yawning', emoji: '🥱', description: 'Frequent yawning' },
  { key: 'food_cravings', label: 'Food Cravings', emoji: '🍫', description: 'Unusual cravings' },
  { key: 'mood_change', label: 'Mood Changes', emoji: '😠', description: 'Irritability or depression' },
  { key: 'concentration', label: 'Poor Focus', emoji: '🤔', description: 'Difficulty concentrating' },
  { key: 'light_sensitivity', label: 'Light Sensitive', emoji: '💡', description: 'Bothered by lights' },
  { key: 'sound_sensitivity', label: 'Sound Sensitive', emoji: '🔊', description: 'Bothered by sounds' },
  { key: 'nausea', label: 'Nausea', emoji: '🤢', description: 'Feeling sick' },
  { key: 'visual_disturbance', label: 'Visual Issues', emoji: '👁️', description: 'Aura or spots' },
];

const mealOptions = [
  { value: 'breakfast', label: 'Breakfast', time: 'Morning' },
  { value: 'lunch', label: 'Lunch', time: 'Midday' },
  { value: 'dinner', label: 'Dinner', time: 'Evening' },
];

type FormData = {
  sleep_hours: number;
  sleep_quality_good: boolean;
  stress_level: number;
  skipped_meals: string[];
  had_snack: boolean;
  alcohol_drinks: number;
  caffeine_drinks: number;
  water_glasses: number;
  bright_light_exposure: boolean;
  screen_time_hours: number;
  symptoms: Record<string, boolean>;
  migraine_occurred: boolean;
  migraine_severity?: number;
  migraine_duration?: number;
  migraine_location?: string;
  migraine_with_aura?: boolean;
};

export default function DailyLogPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  const { control, watch, setValue, handleSubmit } = useForm<FormData>({
    defaultValues: {
      sleep_hours: 7,
      sleep_quality_good: true,
      stress_level: 5,
      skipped_meals: [],
      had_snack: false,
      alcohol_drinks: 0,
      caffeine_drinks: 2,
      water_glasses: 6,
      bright_light_exposure: false,
      screen_time_hours: 4,
      symptoms: {},
      migraine_occurred: false,
      migraine_severity: 5,
      migraine_duration: 4,
      migraine_location: 'both',
      migraine_with_aura: false,
    },
  });

  const watchedValues = watch();

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    
    const userId = localStorage.getItem('user_id');
    if (!userId) {
      router.push('/onboarding');
      return;
    }

    const payload: DailyLogData = {
      user_id: userId,
      date: new Date().toISOString().split('T')[0],
      sleep_hours: data.sleep_hours,
      sleep_quality_good: data.sleep_quality_good,
      stress_level: data.stress_level,
      skipped_meals: data.skipped_meals,
      had_snack: data.had_snack,
      alcohol_drinks: data.alcohol_drinks,
      caffeine_drinks: data.caffeine_drinks,
      water_glasses: data.water_glasses,
      bright_light_exposure: data.bright_light_exposure,
      screen_time_hours: data.screen_time_hours,
      symptoms: data.symptoms as any,
      migraine_occurred: data.migraine_occurred,
      migraine_details: data.migraine_occurred ? {
        severity: data.migraine_severity || 5,
        duration_hours: data.migraine_duration || 4,
        location: data.migraine_location as any || 'both',
        with_aura: data.migraine_with_aura || false,
        medications_taken: [],
      } : undefined,
    };

    try {
      await apiClient.submitDailyLog(payload);
      setSubmitSuccess(true);
      setTimeout(() => {
        router.push('/dashboard');
      }, 2000);
    } catch (error) {
      console.error('Failed to submit log:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleSubmit(onSubmit)();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  if (submitSuccess) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="text-center"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring' }}
            className="w-24 h-24 rounded-full bg-emerald-500 flex items-center justify-center mx-auto mb-6"
          >
            <Check className="w-12 h-12 text-white" />
          </motion.div>
          <h2 className="text-2xl font-bold mb-2">Log Submitted!</h2>
          <p className="text-slate-400">Redirecting to your dashboard...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl font-display font-bold mb-2">
            Good Morning! ☀️
          </h1>
          <p className="text-slate-400">
            Tell us about the last 24 hours • {formatDate(new Date(), 'long')}
          </p>
        </motion.div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const isActive = index === currentStep;
              const isCompleted = index < currentStep;

              return (
                <div key={step.id} className="flex items-center">
                  <button
                    onClick={() => setCurrentStep(index)}
                    className={cn(
                      'flex flex-col items-center transition-all duration-300',
                      isActive || isCompleted ? 'opacity-100' : 'opacity-50'
                    )}
                  >
                    <div
                      className={cn(
                        'w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300',
                        isCompleted
                          ? 'bg-emerald-500'
                          : isActive
                          ? `bg-gradient-to-br ${step.color}`
                          : 'bg-slate-700'
                      )}
                    >
                      {isCompleted ? (
                        <Check className="w-6 h-6 text-white" />
                      ) : (
                        <Icon className="w-6 h-6 text-white" />
                      )}
                    </div>
                    <span
                      className={cn(
                        'text-xs mt-2 hidden sm:block font-medium',
                        isActive ? 'text-white' : 'text-slate-500'
                      )}
                    >
                      {step.title}
                    </span>
                  </button>
                  {index < steps.length - 1 && (
                    <div
                      className={cn(
                        'w-8 sm:w-16 h-0.5 mx-1 sm:mx-2',
                        isCompleted ? 'bg-emerald-500' : 'bg-slate-700'
                      )}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Form Content */}
        <Card className="p-6 md:p-8">
          <AnimatePresence mode="wait">
            {/* Step 1: Sleep */}
            {currentStep === 0 && (
              <motion.div
                key="sleep"
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: -20, opacity: 0 }}
                className="space-y-8"
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                    <Moon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">Sleep & Rest</h2>
                    <p className="text-sm text-slate-400">Sleep deprivation is a major trigger (OR 3.98)</p>
                  </div>
                </div>

                <Controller
                  name="sleep_hours"
                  control={control}
                  render={({ field }) => (
                    <Slider
                      value={field.value}
                      onChange={field.onChange}
                      min={0}
                      max={12}
                      step={0.5}
                      label="Hours of sleep"
                      valueSuffix="h"
                    />
                  )}
                />

                {watchedValues.sleep_hours < 6 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-start gap-3"
                  >
                    <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-amber-400 font-medium">Sleep deficit detected</p>
                      <p className="text-xs text-slate-400 mt-1">
                        Less than 6 hours of sleep increases migraine risk by 3.98x
                      </p>
                    </div>
                  </motion.div>
                )}

                <Controller
                  name="sleep_quality_good"
                  control={control}
                  render={({ field }) => (
                    <div className="space-y-3">
                      <label className="text-sm font-medium text-slate-300">Was your sleep restful?</label>
                      <div className="grid grid-cols-2 gap-4">
                        <button
                          type="button"
                          onClick={() => field.onChange(true)}
                          className={cn(
                            'p-4 rounded-xl border-2 transition-all duration-300 flex items-center gap-3',
                            field.value
                              ? 'border-emerald-500 bg-emerald-500/10'
                              : 'border-slate-700 hover:border-slate-600'
                          )}
                        >
                          <span className="text-2xl">😴</span>
                          <span>Yes, restful</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => field.onChange(false)}
                          className={cn(
                            'p-4 rounded-xl border-2 transition-all duration-300 flex items-center gap-3',
                            !field.value
                              ? 'border-amber-500 bg-amber-500/10'
                              : 'border-slate-700 hover:border-slate-600'
                          )}
                        >
                          <span className="text-2xl">😫</span>
                          <span>Poor quality</span>
                        </button>
                      </div>
                    </div>
                  )}
                />

                <Controller
                  name="stress_level"
                  control={control}
                  render={({ field }) => (
                    <div className="space-y-4">
                      <Slider
                        value={field.value}
                        onChange={field.onChange}
                        min={1}
                        max={10}
                        label="Stress level yesterday"
                        valueSuffix="/10"
                      />
                      <div className="flex justify-between text-xs text-slate-500">
                        <span>😌 Zen</span>
                        <span>😰 Extremely Stressed</span>
                      </div>
                    </div>
                  )}
                />
              </motion.div>
            )}

            {/* Step 2: Diet */}
            {currentStep === 1 && (
              <motion.div
                key="diet"
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: -20, opacity: 0 }}
                className="space-y-8"
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
                    <Utensils className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">Diet & Hydration</h2>
                    <p className="text-sm text-slate-400">Snacks are protective (OR 0.60)</p>
                  </div>
                </div>

                <Controller
                  name="skipped_meals"
                  control={control}
                  render={({ field }) => (
                    <div className="space-y-3">
                      <label className="text-sm font-medium text-slate-300">Did you skip any meals?</label>
                      <div className="grid grid-cols-3 gap-3">
                        {mealOptions.map((meal) => (
                          <button
                            key={meal.value}
                            type="button"
                            onClick={() => {
                              const current = field.value || [];
                              if (current.includes(meal.value)) {
                                field.onChange(current.filter(m => m !== meal.value));
                              } else {
                                field.onChange([...current, meal.value]);
                              }
                            }}
                            className={cn(
                              'p-4 rounded-xl border-2 transition-all duration-300 flex flex-col items-center gap-2',
                              field.value?.includes(meal.value)
                                ? 'border-red-500 bg-red-500/10'
                                : 'border-slate-700 hover:border-slate-600'
                            )}
                          >
                            <span className="text-2xl">
                              {meal.value === 'breakfast' ? '🍳' : meal.value === 'lunch' ? '🥗' : '🍽️'}
                            </span>
                            <span className="text-sm font-medium">{meal.label}</span>
                            <span className="text-xs text-slate-500">{field.value?.includes(meal.value) ? 'Skipped' : 'Had it'}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                />

                <Controller
                  name="had_snack"
                  control={control}
                  render={({ field }) => (
                    <Toggle
                      checked={field.value}
                      onChange={field.onChange}
                      label="Had snacks yesterday"
                      description="Nighttime snacks are protective (OR 0.60)"
                    />
                  )}
                />

                <Controller
                  name="water_glasses"
                  control={control}
                  render={({ field }) => (
                    <div className="space-y-3">
                      <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
                        <Droplets className="w-4 h-4 text-blue-400" />
                        Water intake (glasses)
                      </label>
                      <div className="flex items-center gap-4">
                        <button
                          type="button"
                          onClick={() => field.onChange(Math.max(0, field.value - 1))}
                          className="w-12 h-12 rounded-xl bg-slate-700 hover:bg-slate-600 text-xl font-bold transition-colors"
                        >
                          −
                        </button>
                        <div className="flex-1 text-center">
                          <span className="text-4xl font-bold text-blue-400">{field.value}</span>
                          <span className="text-slate-400 ml-2">glasses</span>
                        </div>
                        <button
                          type="button"
                          onClick={() => field.onChange(field.value + 1)}
                          className="w-12 h-12 rounded-xl bg-slate-700 hover:bg-slate-600 text-xl font-bold transition-colors"
                        >
                          +
                        </button>
                      </div>
                      {field.value < 6 && (
                        <p className="text-xs text-amber-400">💧 Try to drink at least 8 glasses</p>
                      )}
                    </div>
                  )}
                />

                <div className="grid grid-cols-2 gap-4">
                  <Controller
                    name="caffeine_drinks"
                    control={control}
                    render={({ field }) => (
                      <div className="p-4 bg-slate-800/50 rounded-xl">
                        <label className="text-sm font-medium text-slate-300 flex items-center gap-2 mb-3">
                          <Coffee className="w-4 h-4 text-amber-600" />
                          Caffeine
                        </label>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => field.onChange(Math.max(0, field.value - 1))}
                            className="w-8 h-8 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
                          >
                            −
                          </button>
                          <span className="text-2xl font-bold flex-1 text-center">{field.value}</span>
                          <button
                            type="button"
                            onClick={() => field.onChange(field.value + 1)}
                            className="w-8 h-8 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
                          >
                            +
                          </button>
                        </div>
                        <p className="text-xs text-slate-500 text-center mt-2">cups</p>
                      </div>
                    )}
                  />

                  <Controller
                    name="alcohol_drinks"
                    control={control}
                    render={({ field }) => (
                      <div className="p-4 bg-slate-800/50 rounded-xl">
                        <label className="text-sm font-medium text-slate-300 flex items-center gap-2 mb-3">
                          <Wine className="w-4 h-4 text-red-400" />
                          Alcohol
                        </label>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => field.onChange(Math.max(0, field.value - 1))}
                            className="w-8 h-8 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
                          >
                            −
                          </button>
                          <span className="text-2xl font-bold flex-1 text-center">{field.value}</span>
                          <button
                            type="button"
                            onClick={() => field.onChange(field.value + 1)}
                            className="w-8 h-8 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
                          >
                            +
                          </button>
                        </div>
                        <p className="text-xs text-slate-500 text-center mt-2">drinks</p>
                        {field.value >= 5 && (
                          <p className="text-xs text-red-400 text-center mt-1">⚠️ High risk (OR 2.08)</p>
                        )}
                      </div>
                    )}
                  />
                </div>
              </motion.div>
            )}

            {/* Step 3: Lifestyle */}
            {currentStep === 2 && (
              <motion.div
                key="lifestyle"
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: -20, opacity: 0 }}
                className="space-y-8"
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                    <Sun className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">Lifestyle Factors</h2>
                    <p className="text-sm text-slate-400">Environmental triggers matter</p>
                  </div>
                </div>

                <Controller
                  name="bright_light_exposure"
                  control={control}
                  render={({ field }) => (
                    <Toggle
                      checked={field.value}
                      onChange={field.onChange}
                      label="Bright light or sun exposure"
                      description="Extended time in bright environments"
                    />
                  )}
                />

                <Controller
                  name="screen_time_hours"
                  control={control}
                  render={({ field }) => (
                    <Slider
                      value={field.value}
                      onChange={field.onChange}
                      min={0}
                      max={16}
                      step={0.5}
                      label="Screen time"
                      valueSuffix=" hours"
                    />
                  )}
                />

                <div className="p-4 bg-neural-500/10 border border-neural-500/30 rounded-xl">
                  <h4 className="font-medium text-neural-400 mb-2">💡 Did you know?</h4>
                  <p className="text-sm text-slate-400">
                    Blue light from screens can affect sleep quality and contribute to eye strain,
                    both of which are associated with migraines.
                  </p>
                </div>
              </motion.div>
            )}

            {/* Step 4: Symptoms */}
            {currentStep === 3 && (
              <motion.div
                key="symptoms"
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: -20, opacity: 0 }}
                className="space-y-6"
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">Warning Signs</h2>
                    <p className="text-sm text-slate-400">Prodromal symptoms can predict attacks</p>
                  </div>
                </div>

                <p className="text-slate-400 text-sm">
                  Did you notice any of these symptoms yesterday? (Select all that apply)
                </p>

                <Controller
                  name="symptoms"
                  control={control}
                  render={({ field }) => (
                    <div className="grid grid-cols-2 gap-3">
                      {prodromalSymptoms.map((symptom) => (
                        <button
                          key={symptom.key}
                          type="button"
                          onClick={() => {
                            const current = field.value || {};
                            field.onChange({
                              ...current,
                              [symptom.key]: !current[symptom.key],
                            });
                          }}
                          className={cn(
                            'p-4 rounded-xl border-2 transition-all duration-300 text-left',
                            field.value?.[symptom.key]
                              ? 'border-purple-500 bg-purple-500/10'
                              : 'border-slate-700 hover:border-slate-600'
                          )}
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">{symptom.emoji}</span>
                            <div>
                              <div className="font-medium text-sm">{symptom.label}</div>
                              <div className="text-xs text-slate-500">{symptom.description}</div>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                />

                {Object.values(watchedValues.symptoms || {}).filter(Boolean).length >= 3 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-start gap-3"
                  >
                    <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-amber-400 font-medium">Multiple warning signs detected</p>
                      <p className="text-xs text-slate-400 mt-1">
                        Having 3+ prodromal symptoms significantly increases tomorrow&apos;s migraine risk
                      </p>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* Step 5: Yesterday's Outcome */}
            {currentStep === 4 && (
              <motion.div
                key="outcome"
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: -20, opacity: 0 }}
                className="space-y-6"
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-red-500 to-rose-500 flex items-center justify-center">
                    <Calendar className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">Yesterday&apos;s Outcome</h2>
                    <p className="text-sm text-slate-400">This helps us improve your predictions</p>
                  </div>
                </div>

                <Controller
                  name="migraine_occurred"
                  control={control}
                  render={({ field }) => (
                    <div className="space-y-4">
                      <label className="text-sm font-medium text-slate-300">Did you have a migraine yesterday?</label>
                      <div className="grid grid-cols-2 gap-4">
                        <button
                          type="button"
                          onClick={() => field.onChange(false)}
                          className={cn(
                            'p-6 rounded-xl border-2 transition-all duration-300 flex flex-col items-center gap-3',
                            !field.value
                              ? 'border-emerald-500 bg-emerald-500/10'
                              : 'border-slate-700 hover:border-slate-600'
                          )}
                        >
                          <span className="text-4xl">😊</span>
                          <span className="font-medium">No migraine</span>
                          <span className="text-xs text-slate-500">Migraine-free day</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => field.onChange(true)}
                          className={cn(
                            'p-6 rounded-xl border-2 transition-all duration-300 flex flex-col items-center gap-3',
                            field.value
                              ? 'border-red-500 bg-red-500/10'
                              : 'border-slate-700 hover:border-slate-600'
                          )}
                        >
                          <span className="text-4xl">😣</span>
                          <span className="font-medium">Had a migraine</span>
                          <span className="text-xs text-slate-500">Tell us about it</span>
                        </button>
                      </div>
                    </div>
                  )}
                />

                {watchedValues.migraine_occurred && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="space-y-6 pt-4 border-t border-slate-700"
                  >
                    <Controller
                      name="migraine_severity"
                      control={control}
                      render={({ field }) => (
                        <div className="space-y-4">
                          <Slider
                            value={field.value || 5}
                            onChange={field.onChange}
                            min={1}
                            max={10}
                            label="Pain severity"
                            valueSuffix="/10"
                          />
                          <div className="flex justify-between text-xs text-slate-500">
                            <span>😐 Mild</span>
                            <span>😖 Severe</span>
                          </div>
                        </div>
                      )}
                    />

                    <Controller
                      name="migraine_duration"
                      control={control}
                      render={({ field }) => (
                        <Slider
                          value={field.value || 4}
                          onChange={field.onChange}
                          min={1}
                          max={72}
                          label="Duration"
                          valueSuffix=" hours"
                        />
                      )}
                    />

                    <Controller
                      name="migraine_location"
                      control={control}
                      render={({ field }) => (
                        <div className="space-y-3">
                          <label className="text-sm font-medium text-slate-300">Pain location</label>
                          <div className="grid grid-cols-3 gap-3">
                            {[
                              { value: 'left', label: 'Left side', emoji: '⬅️' },
                              { value: 'right', label: 'Right side', emoji: '➡️' },
                              { value: 'both', label: 'Both sides', emoji: '↔️' },
                            ].map((option) => (
                              <button
                                key={option.value}
                                type="button"
                                onClick={() => field.onChange(option.value)}
                                className={cn(
                                  'p-3 rounded-xl border-2 transition-all duration-300 flex flex-col items-center gap-2',
                                  field.value === option.value
                                    ? 'border-neural-500 bg-neural-500/10'
                                    : 'border-slate-700 hover:border-slate-600'
                                )}
                              >
                                <span className="text-xl">{option.emoji}</span>
                                <span className="text-xs">{option.label}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    />

                    <Controller
                      name="migraine_with_aura"
                      control={control}
                      render={({ field }) => (
                        <Toggle
                          checked={field.value || false}
                          onChange={field.onChange}
                          label="Had visual aura"
                          description="Flashing lights, blind spots, or other visual disturbances"
                        />
                      )}
                    />
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8 pt-6 border-t border-slate-700">
            <Button
              type="button"
              variant="ghost"
              onClick={prevStep}
              disabled={currentStep === 0}
              className={currentStep === 0 ? 'invisible' : ''}
            >
              <ChevronLeft className="w-5 h-5" />
              Back
            </Button>

            <Button
              type="button"
              onClick={nextStep}
              isLoading={isSubmitting}
            >
              {currentStep === steps.length - 1 ? 'Submit Log' : 'Continue'}
              {currentStep < steps.length - 1 && <ChevronRight className="w-5 h-5" />}
            </Button>
          </div>
        </Card>

        {/* Quick Stats */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mt-6 grid grid-cols-4 gap-4"
        >
          <div className="text-center p-3 bg-slate-800/50 rounded-xl">
            <div className="text-lg font-bold text-indigo-400">{watchedValues.sleep_hours}h</div>
            <div className="text-xs text-slate-500">Sleep</div>
          </div>
          <div className="text-center p-3 bg-slate-800/50 rounded-xl">
            <div className="text-lg font-bold text-amber-400">{watchedValues.stress_level}/10</div>
            <div className="text-xs text-slate-500">Stress</div>
          </div>
          <div className="text-center p-3 bg-slate-800/50 rounded-xl">
            <div className="text-lg font-bold text-blue-400">{watchedValues.water_glasses}</div>
            <div className="text-xs text-slate-500">Water</div>
          </div>
          <div className="text-center p-3 bg-slate-800/50 rounded-xl">
            <div className="text-lg font-bold text-purple-400">
              {Object.values(watchedValues.symptoms || {}).filter(Boolean).length}
            </div>
            <div className="text-xs text-slate-500">Symptoms</div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}