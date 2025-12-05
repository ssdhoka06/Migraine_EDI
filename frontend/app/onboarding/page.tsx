'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
import {
  User,
  Ruler,
  Scale,
  MapPin,
  Calendar,
  Activity,
  ChevronRight,
  ChevronLeft,
  Check,
  Brain,
} from 'lucide-react';
import { Button, Card, Input, Slider, Toggle } from '@/components/ui';
import { apiClient } from '@/lib/api';
import { OnboardingData } from '@/lib/types';
import { calculateBMI, cn } from '@/lib/utils';

const steps = [
  { id: 'personal', title: 'About You', icon: User },
  { id: 'body', title: 'Body Metrics', icon: Ruler },
  { id: 'location', title: 'Location', icon: MapPin },
  { id: 'history', title: 'Migraine History', icon: Activity },
  { id: 'cycle', title: 'Cycle Tracking', icon: Calendar },
];

export default function OnboardingPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm<OnboardingData>({
    defaultValues: {
      gender: 'F',
      age: 30,
      height: 165,
      weight: 60,
      attacks_per_month: 4,
      location_city: '',
      has_menstrual_cycle: false,
      cycle_start_day: '',
    },
  });

  const gender = watch('gender');
  const height = watch('height');
  const weight = watch('weight');
  const age = watch('age');
  const attacksPerMonth = watch('attacks_per_month');
  const hasMenstrualCycle = watch('has_menstrual_cycle');

  const bmi = calculateBMI(height, weight);

  const onSubmit = async (data: OnboardingData) => {
    setIsSubmitting(true);
    setError(null);

    try {
      const payload: OnboardingData = {
        ...data,
        bmi,
      };

      const response = await apiClient.submitOnboarding(payload);

      if (response.success && response.data?.user_id) {
        localStorage.setItem('user_id', response.data.user_id);
        router.push('/dashboard');
      }
    } catch (err: any) {
      console.error('Onboarding error:', err);
      setError(err.message || 'Failed to create profile. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      // Skip cycle step for males
      if (currentStep === 3 && gender === 'M') {
        handleSubmit(onSubmit)();
      } else {
        setCurrentStep(currentStep + 1);
      }
    } else {
      handleSubmit(onSubmit)();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const isLastStep = gender === 'M' ? currentStep === 3 : currentStep === steps.length - 1;

  return (
    <div className="min-h-screen flex items-center justify-center p-4 py-12">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="text-center mb-8"
        >
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-neural-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-3xl font-display font-bold gradient-text">
              Welcome to MigraineMamba
            </h1>
          </div>
          <p className="text-slate-400">
            Let&apos;s set up your profile to start predicting migraine attacks
          </p>
        </motion.div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.slice(0, gender === 'M' ? 4 : 5).map((step, index) => {
              const Icon = step.icon;
              const isActive = index === currentStep;
              const isCompleted = index < currentStep;

              return (
                <div key={step.id} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <div
                      className={cn(
                        'w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300',
                        isCompleted
                          ? 'bg-emerald-500'
                          : isActive
                          ? 'bg-neural-500'
                          : 'bg-slate-700'
                      )}
                    >
                      {isCompleted ? (
                        <Check className="w-5 h-5 text-white" />
                      ) : (
                        <Icon className="w-5 h-5 text-white" />
                      )}
                    </div>
                    <span
                      className={cn(
                        'text-xs mt-2 hidden sm:block',
                        isActive ? 'text-neural-400' : 'text-slate-500'
                      )}
                    >
                      {step.title}
                    </span>
                  </div>
                  {index < (gender === 'M' ? 3 : 4) && (
                    <div
                      className={cn(
                        'w-12 sm:w-20 h-0.5 mx-2',
                        isCompleted ? 'bg-emerald-500' : 'bg-slate-700'
                      )}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-400"
          >
            {error}
          </motion.div>
        )}

        {/* Form Steps */}
        <Card className="p-8">
          <form onSubmit={handleSubmit(onSubmit)}>
            <AnimatePresence mode="wait">
              {/* Step 1: Personal Info */}
              {currentStep === 0 && (
                <motion.div
                  key="personal"
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -20, opacity: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <h2 className="text-xl font-semibold mb-2">About You</h2>
                    <p className="text-slate-400 text-sm">Basic information helps us personalize predictions</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-3">Gender</label>
                    <div className="grid grid-cols-2 gap-4">
                      {[
                        { value: 'F', label: 'Female', emoji: '👩' },
                        { value: 'M', label: 'Male', emoji: '👨' },
                      ].map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          onClick={() => setValue('gender', option.value as 'M' | 'F')}
                          className={cn(
                            'p-4 rounded-xl border-2 transition-all duration-300 flex items-center gap-3',
                            gender === option.value
                              ? 'border-neural-500 bg-neural-500/10'
                              : 'border-slate-700 hover:border-slate-600'
                          )}
                        >
                          <span className="text-2xl">{option.emoji}</span>
                          <span className="font-medium">{option.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <Slider
                    value={age}
                    onChange={(v) => setValue('age', v)}
                    min={10}
                    max={90}
                    label="Age"
                    valueSuffix=" years"
                  />
                </motion.div>
              )}

              {/* Step 2: Body Metrics */}
              {currentStep === 1 && (
                <motion.div
                  key="body"
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -20, opacity: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <h2 className="text-xl font-semibold mb-2">Body Metrics</h2>
                    <p className="text-slate-400 text-sm">BMI can influence migraine patterns</p>
                  </div>

                  <Slider
                    value={height}
                    onChange={(v) => setValue('height', v)}
                    min={120}
                    max={220}
                    label="Height"
                    valueSuffix=" cm"
                  />

                  <Slider
                    value={weight}
                    onChange={(v) => setValue('weight', v)}
                    min={30}
                    max={150}
                    label="Weight"
                    valueSuffix=" kg"
                  />

                  <div className="p-4 bg-slate-800/50 rounded-xl">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">Calculated BMI</span>
                      <span className={cn(
                        'text-2xl font-bold',
                        bmi < 18.5 || bmi > 30 ? 'text-amber-400' : 'text-emerald-400'
                      )}>
                        {bmi}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">
                      {bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese'}
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Step 3: Location */}
              {currentStep === 2 && (
                <motion.div
                  key="location"
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -20, opacity: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <h2 className="text-xl font-semibold mb-2">Your Location</h2>
                    <p className="text-slate-400 text-sm">Weather patterns can trigger migraines</p>
                  </div>

                  <Input
                    label="City"
                    placeholder="e.g., San Francisco, London, Tokyo"
                    leftIcon={<MapPin className="w-5 h-5" />}
                    {...register('location_city', { required: 'City is required' })}
                    error={errors.location_city?.message}
                    hint="We use this to track barometric pressure changes"
                  />

                  <div className="p-4 bg-neural-500/10 border border-neural-500/30 rounded-xl">
                    <p className="text-sm text-neural-400">
                      🌡️ Weather data helps us identify atmospheric triggers like pressure drops, 
                      which are correlated with migraines (OR 1.27).
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Step 4: Migraine History */}
              {currentStep === 3 && (
                <motion.div
                  key="history"
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -20, opacity: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <h2 className="text-xl font-semibold mb-2">Migraine History</h2>
                    <p className="text-slate-400 text-sm">Help us understand your baseline</p>
                  </div>

                  <Slider
                    value={attacksPerMonth}
                    onChange={(v) => setValue('attacks_per_month', v)}
                    min={0}
                    max={30}
                    label="Attacks per month"
                    valueSuffix=" attacks"
                  />

                  <div className="p-4 bg-slate-800/50 rounded-xl space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Classification</span>
                      <span className={cn(
                        'font-medium',
                        attacksPerMonth < 4 ? 'text-emerald-400' : 
                        attacksPerMonth < 8 ? 'text-amber-400' : 
                        attacksPerMonth < 15 ? 'text-orange-400' : 'text-red-400'
                      )}>
                        {attacksPerMonth < 4 ? 'Episodic (Low)' : 
                         attacksPerMonth < 8 ? 'Episodic (Moderate)' : 
                         attacksPerMonth < 15 ? 'High-Frequency Episodic' : 'Chronic'}
                      </span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className={cn(
                          'h-full rounded-full transition-all',
                          attacksPerMonth < 4 ? 'bg-emerald-500' : 
                          attacksPerMonth < 8 ? 'bg-amber-500' : 
                          attacksPerMonth < 15 ? 'bg-orange-500' : 'bg-red-500'
                        )}
                        style={{ width: `${Math.min(attacksPerMonth / 30 * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Step 5: Menstrual Cycle (Female only) */}
              {currentStep === 4 && gender === 'F' && (
                <motion.div
                  key="cycle"
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -20, opacity: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <h2 className="text-xl font-semibold mb-2">Menstrual Cycle Tracking</h2>
                    <p className="text-slate-400 text-sm">Hormonal changes are a major trigger (OR 2.04)</p>
                  </div>

                  <Toggle
                    checked={hasMenstrualCycle}
                    onChange={(checked) => setValue('has_menstrual_cycle', checked)}
                    label="Track menstrual cycle"
                    description="Significantly improves prediction accuracy"
                  />

                  {hasMenstrualCycle && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="space-y-4"
                    >
                      <Input
                        type="date"
                        label="First day of your last period"
                        {...register('cycle_start_day')}
                        hint="This helps us track your cycle phases"
                      />

                      <div className="p-4 bg-pink-500/10 border border-pink-500/30 rounded-xl">
                        <p className="text-sm text-pink-400">
                          📅 Days -2 to +3 of your cycle carry an 85% higher migraine probability. 
                          We&apos;ll factor this into your daily predictions.
                        </p>
                      </div>
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
                {isLastStep ? 'Complete Setup' : 'Continue'}
                {!isLastStep && <ChevronRight className="w-5 h-5" />}
              </Button>
            </div>
          </form>
        </Card>

        {/* Back to Home */}
        <div className="text-center mt-6">
          <button
            onClick={() => router.push('/')}
            className="text-slate-500 hover:text-slate-300 text-sm transition-colors"
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}