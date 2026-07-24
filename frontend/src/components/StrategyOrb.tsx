import { motion, useReducedMotion } from 'framer-motion'

const WEIGHTS = [
  { label: 'EQ', value: 0.22, angle: -38 },
  { label: 'FX', value: 0.14, angle: 18 },
  { label: 'CM', value: 0.18, angle: 72 },
  { label: 'FI', value: 0.16, angle: 148 },
  { label: 'EM', value: 0.12, angle: 210 },
  { label: '$', value: 0.18, angle: 288 },
]

function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

export function StrategyOrb({ compact = false }: { compact?: boolean }) {
  const reduceMotion = useReducedMotion()

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 18 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 1.05, ease: [0.16, 1, 0.3, 1] }}
      className={`relative ${compact ? 'h-[260px] w-[260px]' : 'aspect-square h-auto w-[min(440px,86vw)]'}`}
      aria-hidden="true"
    >
      {/* Soft atmospheric wash behind the orb — product light, not neon glow stacks */}
      <div className="absolute inset-[-12%] rounded-full bg-[radial-gradient(circle_at_50%_45%,rgba(185,246,207,0.18),transparent_62%)]" />

      {/* Outer tick ring */}
      <motion.svg
        viewBox="0 0 100 100"
        className="absolute inset-0 h-full w-full"
        animate={reduceMotion ? undefined : { rotate: 360 }}
        transition={reduceMotion ? undefined : { duration: 48, repeat: Infinity, ease: 'linear' }}
      >
        {Array.from({ length: 48 }).map((_, i) => {
          const a = (i / 48) * 360
          const outer = polar(50, 50, 48.5, a)
          const inner = polar(50, 50, i % 6 === 0 ? 45.5 : 47, a)
          return (
            <line
              key={i}
              x1={inner.x}
              y1={inner.y}
              x2={outer.x}
              y2={outer.y}
              stroke={i % 6 === 0 ? 'rgba(185,246,207,0.55)' : 'rgba(185,246,207,0.18)'}
              strokeWidth={i % 6 === 0 ? 0.45 : 0.25}
            />
          )
        })}
      </motion.svg>

      {/* Core sphere */}
      <div className="absolute inset-[8%] overflow-hidden rounded-full">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_32%_28%,#e8fff1_0%,#9fd4b4_22%,#3d7a64_52%,#12352c_78%,#07110e_100%)]" />
        <div className="absolute inset-0 bg-[conic-gradient(from_210deg_at_50%_50%,transparent_0deg,rgba(185,246,207,0.12)_80deg,transparent_140deg,rgba(255,255,255,0.08)_220deg,transparent_300deg)]" />
        <div className="absolute -left-[10%] -top-[20%] h-[55%] w-[70%] rounded-full bg-white/20 blur-2xl" />
        <div className="absolute inset-0 shadow-[inset_-28px_-34px_48px_rgba(0,0,0,0.35),inset_16px_14px_28px_rgba(255,255,255,0.12)]" />
      </div>

      {/* Equity path etched on the sphere */}
      <svg className="absolute inset-[18%] h-[64%] w-[64%] overflow-visible" viewBox="0 0 100 100">
        <defs>
          <linearGradient id="orb-path" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#d7ffe5" stopOpacity="0.2" />
            <stop offset="45%" stopColor="#d7ffe5" stopOpacity="0.95" />
            <stop offset="100%" stopColor="#fff8ef" stopOpacity="0.85" />
          </linearGradient>
        </defs>
        <motion.path
          d="M6 68 C16 64, 22 78, 32 58 S48 30, 58 44 S72 62, 94 22"
          fill="none"
          stroke="url(#orb-path)"
          strokeWidth="1.6"
          strokeLinecap="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1.7, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
        />
        {/* Area fill under curve */}
        <motion.path
          d="M6 68 C16 64, 22 78, 32 58 S48 30, 58 44 S72 62, 94 22 L94 86 L6 86 Z"
          fill="rgba(185,246,207,0.08)"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.8 }}
        />
        {[6, 32, 58, 94].map((x, index) => (
          <motion.circle
            key={x}
            cx={x}
            cy={[68, 58, 44, 22][index]}
            r={index === 3 ? 2.2 : 1.5}
            fill={index === 3 ? '#fff8ef' : '#b9f6cf'}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.85 + index * 0.12 }}
          />
        ))}
      </svg>

      {/* Allocation arcs — the product idea: long-only simplex over assets + cash */}
      <svg className="absolute inset-[4%] h-[92%] w-[92%]" viewBox="0 0 100 100">
        {WEIGHTS.map((w, index) => {
          const start = w.angle
          const sweep = w.value * 86
          const r = 43
          const a0 = polar(50, 50, r, start)
          const a1 = polar(50, 50, r, start + sweep)
          const large = sweep > 180 ? 1 : 0
          const label = polar(50, 50, 37.5, start + sweep / 2)
          return (
            <g key={w.label}>
              <motion.path
                d={`M ${a0.x} ${a0.y} A ${r} ${r} 0 ${large} 1 ${a1.x} ${a1.y}`}
                fill="none"
                stroke={index % 2 === 0 ? 'rgba(185,246,207,0.7)' : 'rgba(255,248,239,0.45)'}
                strokeWidth="1.8"
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ delay: 0.55 + index * 0.07, duration: 0.9 }}
              />
              {!compact && (
                <motion.text
                  x={label.x}
                  y={label.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="rgba(244,243,236,0.72)"
                  fontSize="3.2"
                  fontFamily="DM Mono, monospace"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.1 + index * 0.05 }}
                >
                  {w.label}
                </motion.text>
              )}
            </g>
          )
        })}
      </svg>

      {/* Center readout — plain-language product idea, not RL jargon */}
      <motion.div
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.75, duration: 0.55 }}
        className="absolute inset-[34%] flex flex-col items-center justify-center rounded-full border border-white/10 bg-ink/45 px-3 text-center backdrop-blur-md"
      >
        <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-mint/80">Daily</span>
        <span className="font-display mt-0.5 text-[22px] leading-none tracking-[-0.04em] text-cream sm:text-[26px]">
          allocation
        </span>
        <span className="mt-1.5 max-w-[9ch] text-[10px] leading-tight text-cream/55">
          across markets, with cash
        </span>
      </motion.div>

      {/* Slow counter-orbit marker */}
      <motion.div
        animate={reduceMotion ? undefined : { rotate: -360 }}
        transition={reduceMotion ? undefined : { duration: 26, repeat: Infinity, ease: 'linear' }}
        className="absolute inset-[2%] rounded-full"
      >
        <span className="absolute left-1/2 top-0 h-2 w-2 -translate-x-1/2 rounded-full bg-coral shadow-[0_0_0_3px_rgba(255,124,102,0.18)]" />
      </motion.div>
    </motion.div>
  )
}
