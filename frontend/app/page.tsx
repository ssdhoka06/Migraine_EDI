'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import {
  Brain,
  Clock,
  TrendingUp,
  Shield,
  Zap,
  ChevronRight,
  Activity,
  Target,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui';

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Predictions',
    description: 'Advanced Mamba architecture learns your unique patterns with self-supervised learning.',
    gradient: 'from-purple-500 to-pink-500',
  },
  {
    icon: Clock,
    title: '24-Hour Forecast',
    description: 'Know your migraine risk a full day in advance, giving you time to prepare.',
    gradient: 'from-blue-500 to-cyan-500',
  },
  {
    icon: TrendingUp,
    title: 'Personalized Insights',
    description: 'Discover your unique triggers and track improvement over time.',
    gradient: 'from-emerald-500 to-teal-500',
  },
  {
    icon: Shield,
    title: 'Clinically Grounded',
    description: 'Built on peer-reviewed research with validated trigger correlations.',
    gradient: 'from-orange-500 to-amber-500',
  },
];

const phases = [
  {
    day: 'Day 1',
    title: 'Instant Diagnosis',
    description: 'Foundation model provides immediate risk assessment',
    icon: Zap,
  },
  {
    day: 'Days 2-14',
    title: 'Build Your Baseline',
    description: 'Daily logging creates your personal pattern database',
    icon: Activity,
  },
  {
    day: 'Day 15+',
    title: 'Temporal Predictions',
    description: 'AI analyzes 14-day patterns for daily forecasts',
    icon: Target,
  },
  {
    day: 'Day 31+',
    title: 'Fully Personalized',
    description: 'Custom model fine-tuned to your unique triggers',
    icon: Sparkles,
  },
];

const stats = [
  { value: '78%', label: 'Prediction Accuracy' },
  { value: '24h', label: 'Advance Warning' },
  { value: '7+', label: 'Trigger Categories' },
  { value: '3.98x', label: 'Sleep Trigger OR' },
];

export default function HomePage() {
  const [hasUserId, setHasUserId] = useState(false);

  useEffect(() => {
    const userId = localStorage.getItem('user_id');
    setHasUserId(!!userId);
  }, []);

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden px-4">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-neural-500/20 rounded-full blur-[100px]"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.5, 0.3],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
          <motion.div
            className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/20 rounded-full blur-[100px]"
            animate={{
              scale: [1.2, 1, 1.2],
              opacity: [0.4, 0.2, 0.4],
            }}
            transition={{
              duration: 10,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        </div>

        <div className="relative z-10 max-w-5xl mx-auto text-center">
          {/* Logo */}
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-3 mb-8"
          >
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-neural-500 to-purple-600 flex items-center justify-center shadow-lg shadow-neural-500/50">
              <Brain className="w-9 h-9 text-white" />
            </div>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-5xl md:text-7xl font-display font-bold mb-6"
          >
            <span className="gradient-text">MigraineMamba</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-xl md:text-2xl text-slate-400 mb-8 max-w-2xl mx-auto"
          >
            Predict migraine attacks{' '}
            <span className="text-neural-400 font-semibold">24 hours in advance</span>{' '}
            using advanced AI and personalized trigger analysis
          </motion.p>

          {/* Stats */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12 max-w-3xl mx-auto"
          >
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-white mb-1">
                  {stat.value}
                </div>
                <div className="text-sm text-slate-500">{stat.label}</div>
              </div>
            ))}
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            {hasUserId ? (
              <>
                <Link href="/dashboard">
                  <Button size="lg" className="min-w-[200px]">
                    Go to Dashboard
                    <ChevronRight className="w-5 h-5" />
                  </Button>
                </Link>
                <Link href="/daily-log">
                  <Button variant="secondary" size="lg" className="min-w-[200px]">
                    Submit Today&apos;s Log
                  </Button>
                </Link>
              </>
            ) : (
              <>
                <Link href="/onboarding">
                  <Button size="lg" className="min-w-[200px]">
                    Get Started Free
                    <ChevronRight className="w-5 h-5" />
                  </Button>
                </Link>
                <Button variant="secondary" size="lg" className="min-w-[200px]">
                  Watch Demo
                </Button>
              </>
            )}
          </motion.div>

          {/* Scroll indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="absolute bottom-8 left-1/2 -translate-x-1/2"
          >
            <motion.div
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-6 h-10 rounded-full border-2 border-slate-600 flex items-start justify-center p-2"
            >
              <motion.div className="w-1.5 h-2.5 bg-slate-500 rounded-full" />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-display font-bold mb-4">
              Why <span className="gradient-text">MigraineMamba</span>?
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Our AI system combines clinical research with state-of-the-art machine learning
              to give you accurate, actionable predictions.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ y: 20, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="glass-card p-6 card-hover"
                >
                  <div
                    className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4`}
                  >
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                  <p className="text-slate-400 text-sm">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 px-4 bg-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-display font-bold mb-4">
              Your Personalization Journey
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              The system learns and adapts to your unique patterns over time
            </p>
          </motion.div>

          <div className="relative">
            {/* Timeline line */}
            <div className="hidden md:block absolute left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-neural-500 via-purple-500 to-emerald-500" />

            <div className="space-y-12">
              {phases.map((phase, index) => {
                const Icon = phase.icon;
                const isLeft = index % 2 === 0;

                return (
                  <motion.div
                    key={phase.title}
                    initial={{ x: isLeft ? -50 : 50, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className={`flex items-center gap-8 ${
                      isLeft ? 'md:flex-row' : 'md:flex-row-reverse'
                    }`}
                  >
                    <div className={`flex-1 ${isLeft ? 'md:text-right' : 'md:text-left'}`}>
                      <div className="glass-card p-6 inline-block">
                        <span className="text-neural-400 font-mono text-sm">{phase.day}</span>
                        <h3 className="text-xl font-semibold text-white mt-1 mb-2">
                          {phase.title}
                        </h3>
                        <p className="text-slate-400 text-sm">{phase.description}</p>
                      </div>
                    </div>

                    <div className="hidden md:flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 border-4 border-neural-500 z-10">
                      <Icon className="w-7 h-7 text-neural-400" />
                    </div>

                    <div className="flex-1 hidden md:block" />
                  </motion.div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            viewport={{ once: true }}
            className="glass-card p-12 text-center neural-glow"
          >
            <h2 className="text-3xl md:text-4xl font-display font-bold mb-4">
              Ready to Take Control?
            </h2>
            <p className="text-slate-400 mb-8 max-w-xl mx-auto">
              Start predicting your migraines today. Our AI learns your patterns 
              and helps you stay one step ahead.
            </p>
            <Link href={hasUserId ? '/dashboard' : '/onboarding'}>
              <Button size="lg" className="min-w-[250px]">
                {hasUserId ? 'Go to Dashboard' : 'Start Your Journey'}
                <ChevronRight className="w-5 h-5" />
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-slate-800">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neural-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-display font-bold text-lg">MigraineMamba</span>
          </div>
          <p className="text-slate-500 text-sm">
            Built with PyTorch Mamba • Next.js • FastAPI
          </p>
          <div className="flex items-center gap-6 text-slate-500 text-sm">
            <Link href="/privacy" className="hover:text-white transition-colors">
              Privacy
            </Link>
            <Link href="/terms" className="hover:text-white transition-colors">
              Terms
            </Link>
            <Link href="/contact" className="hover:text-white transition-colors">
              Contact
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}