'use client';

import Link from 'next/link';
import { cn } from '@/lib/utils';

interface LogoProps {
  className?: string;
  showText?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export function Logo({ className, showText = true, size = 'md' }: LogoProps) {
  const sizes = {
    sm: { icon: 24, text: 'text-lg' },
    md: { icon: 32, text: 'text-xl' },
    lg: { icon: 40, text: 'text-2xl' },
  };

  const { icon, text } = sizes[size];

  return (
    <Link
      href="/"
      className={cn('flex items-center gap-2 group', className)}
    >
      {/* Logo Icon - Stylized "P" with three flowing lines */}
      <svg
        width={icon}
        height={icon}
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-transform duration-300 group-hover:scale-105"
      >
        {/* Background circle with gradient */}
        <defs>
          <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#2563eb" />
            <stop offset="50%" stopColor="#06b6d4" />
            <stop offset="100%" stopColor="#8b5cf6" />
          </linearGradient>
          <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#60a5fa" />
            <stop offset="100%" stopColor="#22d3ee" />
          </linearGradient>
        </defs>

        {/* Main circle */}
        <circle
          cx="20"
          cy="20"
          r="18"
          fill="url(#logoGradient)"
          opacity="0.9"
        />

        {/* Stylized "P" */}
        <path
          d="M14 10 L14 30 M14 10 L24 10 Q30 10 30 16 Q30 22 24 22 L14 22"
          stroke="white"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Three flowing lines representing the 3 stages */}
        <path
          d="M17 26 Q22 26 26 28"
          stroke="url(#flowGradient)"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
          opacity="0.8"
        />
        <path
          d="M17 29 Q22 29 26 31"
          stroke="url(#flowGradient)"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
          opacity="0.6"
        />
        <path
          d="M17 32 Q22 32 26 34"
          stroke="url(#flowGradient)"
          strokeWidth="1"
          strokeLinecap="round"
          fill="none"
          opacity="0.4"
        />
      </svg>

      {showText && (
        <span className={cn('font-bold tracking-tight', text)}>
          <span className="text-white">Protein</span>
          <span className="gradient-text">ProcessIO</span>
        </span>
      )}
    </Link>
  );
}
