'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import {
  Brain,
  Home,
  ClipboardList,
  BarChart3,
  History,
  Settings,
  LogOut,
  Menu,
  X,
} from 'lucide-react';
import { useState } from 'react';

const navItems = [
  { href: '/dashboard', label: 'Dashboard', icon: Home },
  { href: '/daily-log', label: 'Daily Log', icon: ClipboardList },
  { href: '/insights', label: 'Insights', icon: BarChart3 },
  { href: '/history', label: 'History', icon: History },
];

export function Navigation() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="hidden md:flex fixed left-0 top-0 h-screen w-20 lg:w-64 bg-slate-900/80 backdrop-blur-xl border-r border-slate-800 flex-col z-50">
        {/* Logo */}
        <div className="p-4 lg:p-6 border-b border-slate-800">
          <Link href="/dashboard" className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neural-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="hidden lg:block font-display font-bold text-xl gradient-text">
              MigraineMamba
            </span>
          </Link>
        </div>

        {/* Nav Items */}
        <div className="flex-1 py-6 px-3">
          <ul className="space-y-2">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              const Icon = item.icon;

              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={cn(
                      'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group relative',
                      isActive
                        ? 'bg-neural-500/20 text-neural-400'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                    )}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="activeNav"
                        className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-neural-500 rounded-r-full"
                      />
                    )}
                    <Icon className={cn('w-5 h-5 transition-colors', isActive && 'text-neural-400')} />
                    <span className="hidden lg:block font-medium">{item.label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Bottom Actions */}
        <div className="p-4 border-t border-slate-800 space-y-2">
          <Link
            href="/settings"
            className="flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800/50 transition-all duration-300"
          >
            <Settings className="w-5 h-5" />
            <span className="hidden lg:block font-medium">Settings</span>
          </Link>
          <button
            onClick={() => {
              localStorage.clear();
              window.location.href = '/';
            }}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-all duration-300"
          >
            <LogOut className="w-5 h-5" />
            <span className="hidden lg:block font-medium">Logout</span>
          </button>
        </div>
      </nav>

      {/* Mobile Navigation */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-slate-900/95 backdrop-blur-xl border-t border-slate-800 z-50 pb-safe">
        <div className="flex items-center justify-around py-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex flex-col items-center gap-1 px-4 py-2 rounded-lg transition-all',
                  isActive ? 'text-neural-400' : 'text-slate-500'
                )}
              >
                <Icon className="w-5 h-5" />
                <span className="text-xs font-medium">{item.label}</span>
                {isActive && (
                  <motion.div
                    layoutId="activeMobileNav"
                    className="absolute bottom-0 w-12 h-0.5 bg-neural-500 rounded-full"
                  />
                )}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Mobile Header */}
      <header className="md:hidden fixed top-0 left-0 right-0 bg-slate-900/95 backdrop-blur-xl border-b border-slate-800 z-50">
        <div className="flex items-center justify-between px-4 py-3">
          <Link href="/dashboard" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neural-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-display font-bold text-lg gradient-text">
              MigraineMamba
            </span>
          </Link>
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 right-0 bg-slate-900 border-b border-slate-800 py-4 px-4"
          >
            <div className="space-y-2">
              <Link
                href="/settings"
                onClick={() => setIsMobileMenuOpen(false)}
                className="flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800/50 transition-all"
              >
                <Settings className="w-5 h-5" />
                <span className="font-medium">Settings</span>
              </Link>
              <button
                onClick={() => {
                  localStorage.clear();
                  window.location.href = '/';
                }}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-all"
              >
                <LogOut className="w-5 h-5" />
                <span className="font-medium">Logout</span>
              </button>
            </div>
          </motion.div>
        )}
      </header>
    </>
  );
}