// app/daily-log/page.tsx

'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { apiClient } from '@/lib/api';
import { useRouter } from 'next/navigation';

export default function DailyLogPage() {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { register, handleSubmit, watch, setValue } = useForm({
    defaultValues: {
      sleep_hours: 7,
      sleep_quality: 'yes',
      stress_level: 5,
      skipped_meals: [] as string[],
      had_snack: false,
      alcohol_drinks: 0,
      bright_light: false,
      migraine_occurred: 'false',
      severity: 5,
      duration: 4,
      symptom_fatigue: false,
      symptom_stiff_neck: false,
      symptom_yawning: false,
      symptom_food_cravings: false,
      symptom_mood_change: false,
      symptom_concentration: false,
      symptom_light_sensitivity: false,
      symptom_sound_sensitivity: false,
    },
  });

  const migraineOccurred = watch('migraine_occurred');
  const sleepHours = watch('sleep_hours');
  const stressLevel = watch('stress_level');
  const alcoholDrinks = watch('alcohol_drinks');
  const severity = watch('severity');
  const duration = watch('duration');

  const onSubmit = async (data: any) => {
    setIsSubmitting(true);
    setError(null);

    try {
      const userId = localStorage.getItem('user_id');

      if (!userId) {
        setError('Please complete onboarding first');
        router.push('/onboarding');
        return;
      }

      const payload = {
        user_id: userId,
        date: new Date().toISOString().split('T')[0], // YYYY-MM-DD

        // Sleep & Stress
        sleep_hours: parseFloat(data.sleep_hours),
        sleep_quality_good: data.sleep_quality === 'yes',
        stress_level: parseInt(data.stress_level),

        // Diet & Lifestyle
        skipped_meals: data.skipped_meals || [],
        had_snack: data.had_snack || false,
        alcohol_drinks: parseInt(data.alcohol_drinks) || 0,
        bright_light_exposure: data.bright_light || false,

        // Symptoms
        symptoms: {
          fatigue: data.symptom_fatigue || false,
          stiff_neck: data.symptom_stiff_neck || false,
          yawning: data.symptom_yawning || false,
          food_cravings: data.symptom_food_cravings || false,
          mood_change: data.symptom_mood_change || false,
          concentration: data.symptom_concentration || false,
          light_sensitivity: data.symptom_light_sensitivity || false,
          sound_sensitivity: data.symptom_sound_sensitivity || false,
        },

        // Outcome (Ground Truth)
        migraine_occurred: data.migraine_occurred === 'true',
        migraine_details:
          data.migraine_occurred === 'true'
            ? {
                severity: parseInt(data.severity),
                duration_hours: parseFloat(data.duration),
              }
            : undefined,
      };

      console.log('Submitting payload:', payload);

      await apiClient.submitDailyLog(payload);

      alert('✅ Daily log saved! See you tomorrow morning.');
      router.push('/dashboard');
    } catch (err: any) {
      console.error('Failed to save log:', err);
      setError(
        err.response?.data?.message ||
          'Failed to save daily log. Please try again.'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Good Morning!
          </h1>
          <p className="text-gray-600">
            Please tell us about yesterday (the last 24 hours)
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
          {/* SECTION A: Sleep & Stress */}
          <section className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
            <h2 className="text-xl font-semibold mb-6 text-gray-800">
              Sleep & Stress
            </h2>

            <div className="space-y-6">
              {/* Sleep Hours */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  How many hours did you sleep?
                </label>
                <input
                  type="range"
                  {...register('sleep_hours')}
                  min="0"
                  max="12"
                  step="0.5"
                  className="w-full h-3 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-sm text-gray-600 mt-2">
                  <span>0h</span>
                  <span className="font-semibold text-blue-600 text-xl">
                    {sleepHours} hours
                  </span>
                  <span>12h</span>
                </div>
              </div>

              {/* Sleep Quality */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Was your sleep restful?
                </label>
                <div className="flex gap-4">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      value="yes"
                      {...register('sleep_quality')}
                      className="mr-2 w-4 h-4"
                    />
                    <span className="text-gray-700">Yes, restful</span>
                  </label>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      value="no"
                      {...register('sleep_quality')}
                      className="mr-2 w-4 h-4"
                    />
                    <span className="text-gray-700">No, poor quality</span>
                  </label>
                </div>
              </div>

              {/* Stress Level */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Stress level yesterday? (1 = Zen, 10 = Panic)
                </label>
                <input
                  type="range"
                  {...register('stress_level')}
                  min="1"
                  max="10"
                  className="w-full h-3 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-sm text-gray-600 mt-2">
                  <span>Zen</span>
                  <span className="font-semibold text-blue-600 text-xl">
                    {stressLevel}/10
                  </span>
                  <span>Panic</span>
                </div>
              </div>
            </div>
          </section>

          {/* SECTION B: Diet & Lifestyle */}
          <section className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-green-500">
            <h2 className="text-xl font-semibold mb-6 text-gray-800">
              Diet & Lifestyle
            </h2>

            <div className="space-y-6">
              {/* Skipped Meals */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Did you skip any meals?
                </label>
                <div className="flex flex-wrap gap-3">
                  {['breakfast', 'lunch', 'dinner'].map((meal) => (
                    <label
                      key={meal}
                      className="flex items-center cursor-pointer bg-gray-50 px-4 py-2 rounded-lg hover:bg-gray-100 transition"
                    >
                      <input
                        type="checkbox"
                        value={meal}
                        {...register('skipped_meals')}
                        className="mr-2 w-4 h-4"
                      />
                      <span className="capitalize text-gray-700">{meal}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Snacks */}
              <div>
                <label className="flex items-center cursor-pointer bg-gray-50 px-4 py-3 rounded-lg hover:bg-gray-100 transition">
                  <input
                    type="checkbox"
                    {...register('had_snack')}
                    className="mr-3 w-4 h-4"
                  />
                  <span className="text-gray-700">Did you have any snacks?</span>
                </label>
              </div>

              {/* Alcohol */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Alcohol consumed?
                </label>
                <div className="flex items-center gap-4">
                  <button
                    type="button"
                    onClick={() =>
                      setValue(
                        'alcohol_drinks',
                        Math.max(0, (alcoholDrinks || 0) - 1)
                      )
                    }
                    className="w-10 h-10 bg-red-500 text-white rounded-lg hover:bg-red-600 transition font-bold"
                  >
                    −
                  </button>
                  <span className="text-2xl font-semibold text-gray-700 min-w-[60px] text-center">
                    {alcoholDrinks || 0}
                  </span>
                  <button
                    type="button"
                    onClick={() => setValue('alcohol_drinks', (alcoholDrinks || 0) + 1)}
                    className="w-10 h-10 bg-green-500 text-white rounded-lg hover:bg-green-600 transition font-bold"
                  >
                    +
                  </button>
                  <span className="text-sm text-gray-600 ml-2">drinks</span>
                </div>
              </div>

              {/* Bright Light */}
              <div>
                <label className="flex items-center cursor-pointer bg-gray-50 px-4 py-3 rounded-lg hover:bg-gray-100 transition">
                  <input
                    type="checkbox"
                    {...register('bright_light')}
                    className="mr-3 w-4 h-4"
                  />
                  <span className="text-gray-700">
                    Were you in bright light/sun?
                  </span>
                </label>
              </div>
            </div>
          </section>

          {/* SECTION C: Prodromal Symptoms */}
          <section className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-purple-500">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Warning Signs
            </h2>
            <p className="text-sm text-gray-600 mb-6">
              Did you notice any of these symptoms yesterday?
            </p>

            <div className="grid md:grid-cols-2 gap-3">
              {[
                { key: 'fatigue', label: 'Extreme Fatigue', icon: '😴' },
                { key: 'mood_change', label: 'Mood Changes', icon: '😠' },
                { key: 'stiff_neck', label: 'Stiff Neck', icon: '💢' },
                { key: 'yawning', label: 'Excessive Yawning', icon: '🥱' },
                { key: 'food_cravings', label: 'Food Cravings', icon: '🍫' },
                {
                  key: 'concentration',
                  label: 'Difficulty Concentrating',
                  icon: '🤔',
                },
                {
                  key: 'light_sensitivity',
                  label: 'Sensitive to Light',
                  icon: '💡',
                },
                {
                  key: 'sound_sensitivity',
                  label: 'Sensitive to Sound',
                  icon: '🔊',
                },
              ].map((symptom) => (
                <label
                  key={symptom.key}
                  className="flex items-center cursor-pointer bg-gray-50 px-4 py-3 rounded-lg hover:bg-purple-50 transition"
                >
                  <input
                    type="checkbox"
                    {...register(`symptom_${symptom.key}` as any)}
                    className="mr-3 w-4 h-4"
                  />
                  <span className="mr-2">{symptom.icon}</span>
                  <span className="text-gray-700">{symptom.label}</span>
                </label>
              ))}
            </div>
          </section>

          {/* SECTION D: Outcome (Ground Truth) */}
          <section className="bg-yellow-50 p-6 rounded-lg border-2 border-yellow-400 shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">
              Did you have a migraine yesterday?
            </h2>

            <div className="flex gap-4 mb-6">
              <label className="flex items-center cursor-pointer bg-white px-6 py-3 rounded-lg hover:bg-gray-50 transition border-2 border-transparent hover:border-red-300">
                <input
                  type="radio"
                  value="true"
                  {...register('migraine_occurred')}
                  className="mr-3 w-5 h-5"
                />
                <span className="font-medium text-gray-700">
                  Yes, I had a migraine
                </span>
              </label>
              <label className="flex items-center cursor-pointer bg-white px-6 py-3 rounded-lg hover:bg-gray-50 transition border-2 border-transparent hover:border-green-300">
                <input
                  type="radio"
                  value="false"
                  {...register('migraine_occurred')}
                  className="mr-3 w-5 h-5"
                />
                <span className="font-medium text-gray-700">No migraine</span>
              </label>
            </div>

            {/* Migraine Details */}
            {migraineOccurred === 'true' && (
              <div className="space-y-6 pt-6 border-t-2 border-yellow-200">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Severity? (1-10)
                  </label>
                  <input
                    type="range"
                    {...register('severity')}
                    min="1"
                    max="10"
                    className="w-full h-3 bg-red-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-sm text-gray-600 mt-2">
                    <span>Mild</span>
                    <span className="font-semibold text-red-600 text-xl">
                      {severity}/10
                    </span>
                    <span>Severe</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    How long did it last? (hours)
                  </label>
                  <input
                    type="number"
                    {...register('duration')}
                    min="0.5"
                    step="0.5"
                    className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-yellow-400 focus:border-transparent text-lg"
                    placeholder="e.g., 4.5"
                  />
                </div>
              </div>
            )}
          </section>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-5 rounded-lg font-bold text-xl hover:from-blue-700 hover:to-purple-700 transition shadow-lg disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isSubmitting ? 'Submitting...' : 'Submit Daily Log →'}
          </button>
        </form>

        <div className="text-center mt-6">
          <button
            onClick={() => router.push('/dashboard')}
            className="text-blue-600 hover:text-blue-700 underline"
          >
            ← Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
