'use client';

import { cn } from '@/lib/utils';
import { motion, HTMLMotionProps } from 'framer-motion';
import { forwardRef, ReactNode } from 'react';

// Button Component
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', isLoading, leftIcon, rightIcon, children, disabled, ...props }, ref) => {
    const variants = {
      primary: 'bg-gradient-to-r from-neural-600 to-neural-500 hover:from-neural-500 hover:to-neural-400 text-white shadow-lg shadow-neural-500/25 hover:shadow-neural-500/40',
      secondary: 'bg-slate-700/50 hover:bg-slate-700 text-slate-200 border border-slate-600/50',
      ghost: 'hover:bg-slate-800/50 text-slate-300 hover:text-white',
      danger: 'bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-500/25',
    };

    const sizes = {
      sm: 'py-2 px-4 text-sm',
      md: 'py-3 px-6',
      lg: 'py-4 px-8 text-lg',
    };

    return (
      <button
        ref={ref}
        className={cn(
          'font-semibold rounded-xl transition-all duration-300 inline-flex items-center justify-center gap-2',
          'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100',
          'hover:scale-[1.02] active:scale-[0.98]',
          variants[variant],
          sizes[size],
          className
        )}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading ? (
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        ) : leftIcon}
        {children}
        {!isLoading && rightIcon}
      </button>
    );
  }
);
Button.displayName = 'Button';

// Card Component
interface CardProps extends HTMLMotionProps<'div'> {
  variant?: 'default' | 'glass' | 'elevated';
  hover?: boolean;
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'glass', hover = false, children, ...props }, ref) => {
    const variants = {
      default: 'bg-slate-800 border-slate-700',
      glass: 'bg-slate-800/50 backdrop-blur-xl border-slate-700/50',
      elevated: 'bg-slate-800/80 backdrop-blur-xl border-slate-700/50 shadow-xl',
    };

    return (
      <motion.div
        ref={ref}
        className={cn(
          'rounded-2xl border',
          variants[variant],
          hover && 'transition-all duration-300 hover:border-neural-500/50 hover:shadow-lg hover:shadow-neural-500/10 hover:-translate-y-1',
          className
        )}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);
Card.displayName = 'Card';

// Progress Ring Component
interface ProgressRingProps {
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  bgColor?: string;
  children?: ReactNode;
}

export function ProgressRing({
  progress,
  size = 120,
  strokeWidth = 8,
  color = '#0c8ce9',
  bgColor = '#334155',
  children,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={bgColor}
          strokeWidth={strokeWidth}
          fill="none"
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: 'easeOut' }}
          style={{
            strokeDasharray: circumference,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        {children}
      </div>
    </div>
  );
}

// Badge Component
interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'sm' | 'md';
  children: ReactNode;
  className?: string;
}

export function Badge({ variant = 'default', size = 'md', children, className }: BadgeProps) {
  const variants = {
    default: 'bg-slate-700 text-slate-300',
    success: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    danger: 'bg-red-500/20 text-red-400 border-red-500/30',
    info: 'bg-neural-500/20 text-neural-400 border-neural-500/30',
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center font-medium rounded-full border',
        variants[variant],
        sizes[size],
        className
      )}
    >
      {children}
    </span>
  );
}

// Input Component
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
  leftIcon?: ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, hint, leftIcon, ...props }, ref) => {
    return (
      <div className="space-y-2">
        {label && (
          <label className="block text-sm font-medium text-slate-300">
            {label}
          </label>
        )}
        <div className="relative">
          {leftIcon && (
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">
              {leftIcon}
            </div>
          )}
          <input
            ref={ref}
            className={cn(
              'w-full bg-slate-800/50 border border-slate-700 rounded-xl px-4 py-3',
              'text-white placeholder-slate-500',
              'focus:border-neural-500 focus:ring-2 focus:ring-neural-500/20 outline-none',
              'transition-all duration-300',
              leftIcon && 'pl-12',
              error && 'border-red-500 focus:border-red-500 focus:ring-red-500/20',
              className
            )}
            {...props}
          />
        </div>
        {error && <p className="text-sm text-red-400">{error}</p>}
        {hint && !error && <p className="text-sm text-slate-500">{hint}</p>}
      </div>
    );
  }
);
Input.displayName = 'Input';

// Slider Component
interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  valuePrefix?: string;
  valueSuffix?: string;
  className?: string;
}

export function Slider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  showValue = true,
  valuePrefix = '',
  valueSuffix = '',
  className,
}: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn('space-y-2', className)}>
      {(label || showValue) && (
        <div className="flex justify-between items-center">
          {label && <span className="text-sm font-medium text-slate-300">{label}</span>}
          {showValue && (
            <span className="text-lg font-semibold text-neural-400">
              {valuePrefix}{value}{valueSuffix}
            </span>
          )}
        </div>
      )}
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full h-2 bg-slate-700 rounded-full appearance-none cursor-pointer"
          style={{
            background: `linear-gradient(to right, #0c8ce9 0%, #0c8ce9 ${percentage}%, #334155 ${percentage}%, #334155 100%)`,
          }}
        />
      </div>
      <div className="flex justify-between text-xs text-slate-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

// Toggle Switch Component
interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
}

export function Toggle({ checked, onChange, label, description, disabled }: ToggleProps) {
  return (
    <label className={cn('flex items-center gap-4 cursor-pointer', disabled && 'opacity-50 cursor-not-allowed')}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={cn(
          'relative inline-flex h-7 w-12 items-center rounded-full transition-colors duration-300',
          checked ? 'bg-neural-500' : 'bg-slate-700'
        )}
      >
        <span
          className={cn(
            'inline-block h-5 w-5 transform rounded-full bg-white shadow-lg transition-transform duration-300',
            checked ? 'translate-x-6' : 'translate-x-1'
          )}
        />
      </button>
      {(label || description) && (
        <div className="flex flex-col">
          {label && <span className="text-sm font-medium text-white">{label}</span>}
          {description && <span className="text-xs text-slate-500">{description}</span>}
        </div>
      )}
    </label>
  );
}

// Skeleton Loading Component
interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
}

export function Skeleton({ className, variant = 'text' }: SkeletonProps) {
  const variants = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  return (
    <div
      className={cn(
        'animate-pulse bg-slate-700/50',
        variants[variant],
        className
      )}
    />
  );
}

// Loading Spinner Component
export function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
  };

  return (
    <svg
      className={cn('animate-spin text-neural-500', sizes[size])}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

// Empty State Component
interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      {icon && <div className="text-slate-600 mb-4">{icon}</div>}
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      {description && <p className="text-slate-400 mb-6 max-w-sm">{description}</p>}
      {action}
    </div>
  );
}