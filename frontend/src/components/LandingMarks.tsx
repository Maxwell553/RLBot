/** Custom product marks for the landing page — not stock icon packs. */

type MarkProps = { className?: string }

export function MandateMark({ className }: MarkProps) {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className={className}>
      <rect x="6" y="8" width="36" height="32" rx="4" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 16h16M12 22h24M12 28h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="34" cy="30" r="7" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="1.5" />
      <path d="M31.5 30.2l1.8 1.8 3.6-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function TrainMark({ className }: MarkProps) {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className={className}>
      <path
        d="M8 34 V18 c0-2 1.5-4 4-4 h10 c2 0 3 1.2 4 2.5S29 20 31 20h5c2.5 0 4 2 4 4v10"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path d="M8 34h32" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      {[12, 20, 28, 36].map((x, i) => (
        <rect
          key={x}
          x={x - 2}
          y={34 - (10 + i * 3)}
          width="4"
          height={10 + i * 3}
          rx="1"
          fill="currentColor"
          fillOpacity={0.2 + i * 0.12}
        />
      ))}
      <circle cx="38" cy="14" r="3" fill="currentColor" />
    </svg>
  )
}

export function EvidenceMark({ className }: MarkProps) {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className={className}>
      <path d="M14 10h14l8 8v20a2 2 0 0 1-2 2H14a2 2 0 0 1-2-2V12a2 2 0 0 1 2-2z" stroke="currentColor" strokeWidth="1.6" />
      <path d="M28 10v8h8" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M17 28c3-5 5-7 8-7s4 4 7 9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="25" cy="21" r="1.6" fill="currentColor" />
      <path d="M17 35h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.45" />
    </svg>
  )
}

export function FlowRibbon({ className }: MarkProps) {
  return (
    <svg viewBox="0 0 120 120" fill="none" aria-hidden="true" className={className}>
      <circle cx="60" cy="60" r="52" stroke="currentColor" strokeOpacity="0.12" strokeWidth="1" />
      <circle cx="60" cy="60" r="36" stroke="currentColor" strokeOpacity="0.18" strokeWidth="1" strokeDasharray="3 5" />
      <path
        d="M28 72 C40 48, 52 44, 64 56 S88 78, 98 42"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.85"
      />
      <circle cx="28" cy="72" r="3.5" fill="currentColor" fillOpacity="0.35" />
      <circle cx="64" cy="56" r="3.5" fill="currentColor" fillOpacity="0.55" />
      <circle cx="98" cy="42" r="4.5" fill="currentColor" />
      <path d="M20 88h28M20 94h18" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

export function ProvenanceSeal({ className }: MarkProps) {
  return (
    <svg viewBox="0 0 160 64" fill="none" aria-hidden="true" className={className}>
      <rect x="1" y="1" width="158" height="62" rx="12" stroke="currentColor" strokeOpacity="0.2" />
      <path d="M18 40 V24 l8 8 8-12 8 14 6-8 V40" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="52" cy="18" r="3" fill="#ff7c66" />
      <text x="68" y="26" fill="currentColor" fontFamily="DM Mono, monospace" fontSize="8" letterSpacing="1.5" opacity="0.55">
        CONFIG HASH
      </text>
      <text x="68" y="44" fill="currentColor" fontFamily="DM Mono, monospace" fontSize="11" letterSpacing="0.5">
        a7f3…9c21
      </text>
    </svg>
  )
}
