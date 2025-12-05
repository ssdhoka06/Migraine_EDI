// app/page.tsx

'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

export default function HomePage() {
  const [hasUserId, setHasUserId] = useState(false);

  useEffect(() => {
    // Check if user has already completed onboarding
    const userId = localStorage.getItem('user_id');
    setHasUserId(!!userId);
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full text-center space-y-8">
        {/* Logo/Header */}
        <div className="space-y-4">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            MigraineMamba
          </h1>
          <p className="text-xl text-gray-600">
            Predict migraine attacks 24 hours in advance using AI
          </p>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 my-12">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <div className="text-4xl mb-3">🧠</div>
            <h3 className="font-semibold mb-2">AI-Powered</h3>
            <p className="text-sm text-gray-600">
              Advanced Mamba architecture learns your unique patterns
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <div className="text-4xl mb-3">⏰</div>
            <h3 className="font-semibold mb-2">24hr Prediction</h3>
            <p className="text-sm text-gray-600">
              Know your risk a day in advance
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <div className="text-4xl mb-3">📊</div>
            <h3 className="font-semibold mb-2">Personal Insights</h3>
            <p className="text-sm text-gray-600">
              Discover your unique triggers
            </p>
          </div>
        </div>

        {/* CTA Buttons */}
        <div className="space-y-4">
          {hasUserId ? (
            <>
              <Link
                href="/dashboard"
                className="block w-full max-w-md mx-auto bg-blue-600 text-white py-4 px-8 rounded-lg font-semibold text-lg hover:bg-blue-700 transition"
              >
                Go to Dashboard
              </Link>
              <Link
                href="/daily-log"
                className="block w-full max-w-md mx-auto bg-purple-600 text-white py-4 px-8 rounded-lg font-semibold text-lg hover:bg-purple-700 transition"
              >
                Submit Today's Log
              </Link>
            </>
          ) : (
            <Link
              href="/onboarding"
              className="block w-full max-w-md mx-auto bg-blue-600 text-white py-4 px-8 rounded-lg font-semibold text-lg hover:bg-blue-700 transition"
            >
              Get Started
            </Link>
          )}
        </div>

        {/* Info */}
        <p className="text-sm text-gray-500 mt-8">
          Built with Next.js, FastAPI, and PyTorch Mamba
        </p>
      </div>
    </div>
  );
}