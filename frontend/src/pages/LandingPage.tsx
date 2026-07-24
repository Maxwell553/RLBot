import { motion, useReducedMotion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Brand } from '../components/Brand'
import {
  EvidenceMark,
  FlowRibbon,
  MandateMark,
  ProvenanceSeal,
  TrainMark,
} from '../components/LandingMarks'
import { StrategyOrb } from '../components/StrategyOrb'
import { MOTION, reveal } from '../lib/motion'
import { OPS_UI_ENABLED } from '../lib/runtime'

const steps = [
  {
    mark: MandateMark,
    n: '01',
    title: 'Encode the mandate',
    body: 'Assets, allocation caps, costs, capital, lag assumptions, and model budget become a reproducible configuration.',
  },
  {
    mark: TrainMark,
    n: '02',
    title: 'Train with discipline',
    body: 'Walk-forward windows and strict chronological holdouts preserve the separation between model selection and OOS evidence.',
  },
  {
    mark: EvidenceMark,
    n: '03',
    title: 'Review the evidence',
    body: 'Compare risk, return, turnover, exposure, and provenance before a model is considered for promotion.',
  },
]

const delivery = [
  ['Mandate', 'Select instruments, exclusions, concentration limits, and risk preferences.'],
  ['Quote', 'Review eligibility, scope, expected delivery range, price, terms, and methodology disclosures.'],
  ['Controlled build', 'A verified payment locks the mandate version before a vetted research plan enters the queue.'],
  ['Approved release', 'Receive comparable OOS evidence only after provenance checks and release review.'],
] as const

export function LandingPage() {
  const reduceMotion = useReducedMotion()

  return (
    <main className="overflow-hidden bg-cream">
      <section className="noise relative isolate min-h-[100svh] bg-ink text-cream">
        {/* Atmospheric plane — full-bleed product field */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_70%_40%,rgba(45,101,83,0.45),transparent_55%),radial-gradient(ellipse_50%_40%_at_15%_80%,rgba(185,246,207,0.08),transparent_50%),linear-gradient(180deg,#07110e_0%,#0c1c17_55%,#102b23_100%)]" />
          <div className="fine-grid absolute inset-0 opacity-60 [mask-image:linear-gradient(90deg,transparent,black_30%,black_75%,transparent)]" />
          <motion.div
            aria-hidden="true"
            className="absolute -right-[15%] top-[8%] h-[70vmin] w-[70vmin] rounded-full border border-mint/[0.07]"
            animate={reduceMotion ? undefined : { scale: [1, 1.04, 1] }}
            transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
          />
          <div className="absolute -left-[20%] bottom-[-10%] h-[50vmin] w-[50vmin] rounded-full border border-mint/[0.05]" />
        </div>

        <header className="relative z-20 mx-auto flex max-w-[1240px] items-center justify-between px-6 py-6 lg:px-10">
          <Brand light />
          <nav className="hidden items-center gap-8 text-xs font-medium text-cream/55 md:flex" aria-label="Page">
            <a href="#platform" className="hover:text-cream">Platform</a>
            <a href="#process" className="hover:text-cream">Process</a>
            <a href="#governance" className="hover:text-cream">Governance</a>
          </nav>
          <Link
            to="/portal"
            className="inline-flex h-10 items-center gap-2 rounded-2xl border border-cream/15 bg-cream/[0.04] px-4 text-xs font-semibold text-cream/85 hover:border-mint/35 hover:bg-mint/10 hover:text-mint"
          >
            Open investor portal <ArrowRight size={14} aria-hidden="true" />
          </Link>
        </header>

        <div className="relative z-10 mx-auto grid min-h-[calc(100svh-88px)] max-w-[1240px] items-center gap-12 px-6 pb-16 pt-6 lg:grid-cols-[1.05fr_0.95fr] lg:gap-8 lg:px-10 lg:pb-20">
          <div>
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: MOTION.entrance, ease: MOTION.ease }}
              className="font-display text-[clamp(2.75rem,7vw,5.5rem)] leading-[0.92] tracking-[-0.045em] text-cream"
            >
              MarketTrainer
            </motion.p>
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: MOTION.entrance, delay: 0.08, ease: MOTION.ease }}
              className="font-display mt-5 max-w-[18ch] text-[clamp(2rem,4.8vw,3.75rem)] leading-[1.02] tracking-[-0.035em]"
            >
              Your mandate.
              <span className="block text-mint">Your model.</span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: MOTION.entrance, delay: 0.16, ease: MOTION.ease }}
              className="mt-7 max-w-md text-[15px] leading-7 text-cream/65"
            >
              Reinforcement-learning portfolios shaped by your assets, constraints, costs, and
              governance—not a generic strategy off the shelf.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: MOTION.entrance, delay: 0.24, ease: MOTION.ease }}
              className="mt-9 flex flex-wrap items-center gap-3"
            >
              <Link
                to="/portal/mandates/new"
                className="inline-flex h-12 items-center gap-2 rounded-2xl bg-mint px-6 text-sm font-semibold text-ink hover:bg-white"
              >
                Design a mandate <ArrowRight size={16} aria-hidden="true" />
              </Link>
              <a
                href="#process"
                className="inline-flex h-12 items-center rounded-2xl border border-cream/15 px-6 text-sm font-semibold text-cream/70 hover:border-cream/30 hover:text-cream"
              >
                See the delivery flow
              </a>
            </motion.div>
          </div>

          <div className="relative flex items-center justify-center lg:justify-end">
            <StrategyOrb />
          </div>
        </div>
      </section>

      <section id="platform" className="relative mx-auto max-w-[1240px] px-6 py-24 lg:px-10 lg:py-32">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-[radial-gradient(ellipse_at_top,rgba(16,43,35,0.06),transparent_70%)]" />
        <motion.div {...reveal} className="grid gap-8 lg:grid-cols-2">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-pine/65">Built around the mandate</p>
            <h2 className="font-display mt-4 max-w-xl text-5xl leading-[1.02] tracking-[-0.035em] sm:text-6xl">
              A strategy specification, not a black box.
            </h2>
          </div>
          <p className="max-w-lg self-end text-sm leading-7 text-ink/50">
            Configure an investable universe and realistic friction model, define risk controls,
            then receive customer-facing build updates and release-approved evidence through a governed portal.
          </p>
        </motion.div>

        <div className="mt-16 grid gap-5 md:grid-cols-3">
          {steps.map((item, index) => (
            <motion.article
              key={item.title}
              {...reveal}
              transition={{ ...reveal.transition, delay: index * 0.08 }}
              className="relative overflow-hidden rounded-[28px] border border-line bg-paper p-7"
            >
              <div
                aria-hidden="true"
                className="absolute -right-8 -top-10 h-28 w-28 rounded-full bg-[radial-gradient(circle,rgba(185,246,207,0.35),transparent_70%)]"
              />
              <div className="relative flex items-start justify-between">
                <span className="grid h-14 w-14 place-items-center rounded-2xl bg-pine text-mint">
                  <item.mark className="h-8 w-8" />
                </span>
                <span className="font-mono text-[11px] text-ink/40">{item.n}</span>
              </div>
              <h3 className="relative mt-10 text-lg font-semibold tracking-[-0.025em]">{item.title}</h3>
              <p className="relative mt-3 text-sm leading-6 text-ink/60">{item.body}</p>
            </motion.article>
          ))}
        </div>
      </section>

      <section id="process" className="bg-[#e7ebe2] px-6 py-24 lg:px-10 lg:py-28">
        <motion.div
          {...reveal}
          className="mx-auto grid max-w-[1160px] overflow-hidden rounded-[36px] border border-pine/10 bg-pine text-cream shadow-[0_40px_80px_rgba(7,17,14,0.12)] lg:grid-cols-[0.85fr_1.15fr]"
        >
          <div className="relative overflow-hidden p-8 sm:p-12">
            <FlowRibbon className="h-28 w-28 text-mint" />
            <h2 className="font-display mt-10 text-4xl leading-[1.05] tracking-[-0.04em] sm:text-5xl">
              From rules to a research-ready configuration.
            </h2>
            <p className="mt-6 max-w-sm text-sm leading-7 text-cream/65">
              The investor specifies supported product choices; the server owns configuration
              materialization, validation, and research authorization.
            </p>
            <div
              aria-hidden="true"
              className="pointer-events-none absolute -bottom-24 -right-16 h-72 w-72 rounded-full border-[28px] border-mint/[0.06]"
            />
          </div>
          <div className="bg-paper p-7 text-ink sm:p-12">
            <ol className="relative">
              <div
                aria-hidden="true"
                className="absolute bottom-3 left-[11px] top-3 w-px bg-gradient-to-b from-pine/25 via-pine/15 to-transparent"
              />
              {delivery.map(([title, body], index) => (
                <li key={title} className="relative flex gap-5 py-6 first:pt-0 last:pb-0">
                  <span className="relative z-[1] mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full border border-pine/20 bg-paper font-mono text-[10px] text-pine">
                    {index + 1}
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold tracking-[-0.01em]">{title}</h3>
                    <p className="mt-2 text-sm leading-6 text-ink/60">{body}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </motion.div>
      </section>

      <section id="governance" className="px-6 py-24 lg:px-10 lg:py-32">
        <motion.div {...reveal} className="mx-auto flex max-w-[980px] flex-col items-center text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-pine/65">
            Research infrastructure, productized
          </p>
          <h2 className="font-display mt-5 max-w-3xl text-5xl leading-[1.02] tracking-[-0.04em] sm:text-7xl">
            Institutional intent, translated into code.
          </h2>
          <p className="mt-7 max-w-xl text-sm leading-7 text-ink/60">
            Every mandate becomes a versioned, hash-stamped configuration; every result carries its
            provenance. Governance is the product, not an afterthought.
          </p>
          <ProvenanceSeal className="mt-10 h-16 w-full max-w-sm text-pine" />
          <Link
            to="/portal"
            className="mt-10 inline-flex h-12 items-center gap-2 rounded-2xl bg-pine px-6 text-sm font-semibold text-cream hover:bg-[#173c31]"
          >
            Open investor portal <ArrowRight size={16} aria-hidden="true" />
          </Link>
        </motion.div>
      </section>

      <footer className="border-t border-line px-6 py-8 lg:px-10">
        <div className="mx-auto flex max-w-[1240px] flex-col justify-between gap-4 text-xs text-ink/55 sm:flex-row">
          <span>© 2026 MarketTrainer</span>
          <span className="flex flex-wrap items-center gap-4">
            <span>Research tooling only · Not investment advice</span>
            {OPS_UI_ENABLED && (
              <Link to="/ops" className="font-semibold text-pine hover:text-ink">Research Operations</Link>
            )}
          </span>
        </div>
      </footer>
    </main>
  )
}
