// app/onboarding/page.tsx

'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { apiClient } from '@/lib/api';
import { useRouter } from 'next/navigation';

export default function OnboardingPage() {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm({
    defaultValues: {
      gender: 'F' as 'M' | 'F',
      age: 30,
      height: 165,
      weight: 60,
      attacks_per_month: 2,
      location_city: '',
      has_menstrual_cycle: false,
      cycle_start_day: '',
    },
  });

  const gender = watch('gender');
  const height = watch('height');
  const weight = watch('weight');
  const hasMenstrualCycle = watch('has_menstrual_cycle');

  // Calculate BMI
  const bmi = height && weight ? (weight / ((height / 100) ** 2)).toFixed(1) : '0.0';

  const onSubmit = async (data: any) => {
    setIsSubmitting(true);
    setError(null);

    try {
      const payload = {
        gender: data.gender,
        age: parseInt(data.age),
        height: parseFloat(data.height),
        weight: parseFloat(data.weight),
        bmi: parseFloat(bmi),
        attacks_per_month: parseInt(data.attacks_per_month),
        location_city: data.location_city,
        has_menstrual_cycle: data.has_menstrual_cycle,
        cycle_start_day: data.has_menstrual_cycle && data.cycle_start_day
          ? data.cycle_start_day
          : undefined,
      };

      console.log('Submitting onboarding data:', payload);

      const response = await apiClient.submitOnboarding(payload);

      // Store user_id in localStorage
      if (response.user_id) {
        localStorage.setItem('user_id', response.user_id);
        alert('✅ Profile created! Let\'s start tracking your data.');
        router.push('/dashboard');
      }
    } catch (err: any) {
      console.error('Failed to submit onboarding:', err);
      setError(
        err.response?.data?.message ||
          'Failed to create profile. Please try again.'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            Welcome to MigraineMamba
          </h1>
          <p className="text-gray-600 text-lg">
            Let's set up your profile to start predicting migraine attacks
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Basic Info */}
          <section className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-6 text-gray-800">
              👤 Basic Information
            </h2>

            <div className="space-y-4">
              {/* Gender */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Gender
                </label>
                <div className="flex gap-4">
                  <label className="flex items-center cursor-pointer bg-gray-50 px-6 py-3 rounded-lg hover:bg-gray-100 transition">
                    <input
                      type="radio"
                      value="F"
                      {...register('gender', { required: true })}
                      className="mr-3 w-4 h-4"
                    />
                    <span>Female</span>
                  </label>
                  <label className="flex items-center cursor-pointer bg-gray-50 px-6 py-3 rounded-lg hover:bg-gray-100 transition">
                    <input
                      type="radio"
                      value="M"
                      {...register('gender', { required: true })}
                      className="mr-3 w-4 h-4"
                    />
                    <span>Male</span>
                  </label>
                </div>
              </div>

              {/* Age */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Age
                </label>
                <input
                  type="number"
                  {...register('age', { required: true, min: 1, max: 120 })}
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="30"
                />
              </div>

              {/* Height */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Height (cm)
                </label>
                <input
                  type="number"
                  {...register('height', { required: true, min: 50, max: 250 })}
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="165"
                />
              </div>

              {/* Weight */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Weight (kg)
                </label>
                <input
                  type="number"
                  {...register('weight', { required: true, min: 20, max: 300 })}
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="60"
                />
              </div>

              {/* BMI Display */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">
                  Calculated BMI: <span className="font-semibold text-blue-600 text-lg">{bmi}</span>
                </p>
              </div>
            </div>
          </section>

          {/* Migraine History */}
          <section className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-6 text-gray-800">
              🤕 Migraine History
            </h2>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                How many migraine attacks do you typically have per month?
              </label>
              <input
                type="number"
                {...register('attacks_per_month', { required: true, min: 0 })}
                className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="2"
              />
            </div>
          </section>

          {/* Location */}
          <section className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-6 text-gray-800">
              📍 Location
            </h2>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                City (for weather data)
              </label>
              <input
                type="text"
                {...register('location_city', { required: true })}
                className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="e.g., New York, London, Tokyo"
              />
              <p className="text-xs text-gray-500 mt-2">
                We use your location to track weather patterns that may trigger migraines
              </p>
            </div>
          </section>

          {/* Menstrual Cycle */}
          {gender === 'F' && (
            <section className="bg-white p-6 rounded-lg shadow-lg">
              <h2 className="text-xl font-semibold mb-6 text-gray-800">
                📅 Menstrual Cycle (Optional)
              </h2>

              <div className="space-y-4">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    {...register('has_menstrual_cycle')}
                    className="mr-3 w-5 h-5"
                  />
                  <span className="text-gray-700">
                    Track menstrual cycle (helps improve predictions)
                  </span>
                </label>

                {hasMenstrualCycle && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Start date of last period
                    </label>
                    <input
                      type="date"
                      {...register('cycle_start_day')}
                      className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-5 rounded-lg font-bold text-xl hover:from-blue-700 hover:to-purple-700 transition shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? 'Creating Profile...' : 'Create Profile →'}
          </button>
        </form>

        <div className="text-center mt-6">
          <button
            onClick={() => router.push('/')}
            className="text-blue-600 hover:text-blue-700 underline"
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
