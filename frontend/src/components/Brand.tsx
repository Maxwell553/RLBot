import { Link } from 'react-router-dom'

function Mark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" className="shrink-0">
      <rect width="64" height="64" rx="16" fill="#07110e" />
      <path
        d="M14 44 V26 l8 10 8-14 8 16 6-8 V44"
        fill="none"
        stroke="#b9f6cf"
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="50" cy="18" r="4" fill="#ff7c66" />
    </svg>
  )
}

export function Brand({ light = false, compact = false }: { light?: boolean; compact?: boolean }) {
  return (
    <Link to="/" className="group inline-flex items-center gap-2.5" aria-label="MarketTrainer home">
      <span className={light ? 'rounded-[9px] ring-1 ring-mint/25' : 'rounded-[9px] ring-1 ring-pine/15'}>
        <Mark />
      </span>
      {!compact && (
        <span className={`text-[15px] font-semibold tracking-[-0.02em] ${light ? 'text-cream' : 'text-ink'}`}>
          MarketTrainer
        </span>
      )}
    </Link>
  )
}
