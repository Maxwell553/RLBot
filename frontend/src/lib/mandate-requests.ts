export type MandateInstrument = {
  name: string
  ticker: string
  group: string
}

export type MandateSubmission = {
  name: string
  instruments: MandateInstrument[]
  maxWeight: number
  riskPreference: string
  /** Approximate portfolio notional used to size slippage / trading-cost assumptions. */
  approximateTradingCapital: number
}

export type EligibilityResult = {
  ticker: string
  symbolFound: boolean
  historyBars: number
  firstDate: string | null
  lastDate: string | null
  approvedPolicy: boolean
  panelCompatible: boolean
  sufficientHistory: boolean
  eligible: boolean
}

export type WorkflowEvent = {
  actorId: string
  actorRole: string
  eventType: string
  fromState: string | null
  toState: string | null
  detail: Record<string, unknown>
  createdAt: string
}

export type WorkflowMandate = {
  id: string
  organizationId: string
  ownerId: string
  name: string
  state: 'draft' | 'preflight_passed' | 'quote_issued' | 'checkout' | 'payment_verified' | 'queued' | 'training' | 'validation' | 'governed_oos_evaluation' | 'released' | 'cancelled'
  version: number
  versionId: string
  immutable: boolean
  assignedOperator: string | null
  quoteAmount: number | null
  paymentState: string
  instruments: MandateInstrument[]
  configuration: {
    maxWeight: number
    riskPreference: string
    approximateTradingCapital?: number
    longOnly: boolean
    cashAllowed: boolean
    decisionFrequency: string
  }
  eligibility: EligibilityResult[]
  runPlan: {
    cohortId: string
    windows: number[]
    seeds: number[]
    totalJobs: number
    status: string
  } | null
  release: {
    artifactUrl?: string
    reportId?: string
  } | null
  allowedActions: string[]
  createdAt: string
  updatedAt: string
  auditLog: WorkflowEvent[]
}

export function createMandateSubmission(input: MandateSubmission): MandateSubmission {
  return {
    name: input.name.trim(),
    instruments: input.instruments.map((instrument) => ({
      name: instrument.name.trim(),
      ticker: instrument.ticker.trim().toUpperCase(),
      group: instrument.group,
    })),
    maxWeight: input.maxWeight,
    riskPreference: input.riskPreference,
    approximateTradingCapital: Math.round(input.approximateTradingCapital),
  }
}

/** Investor-facing size bands → representative notional for cost modeling. */
export const TRADING_SIZE_OPTIONS = [
  { label: 'About $250,000', value: 250_000 },
  { label: 'About $1 million', value: 1_000_000 },
  { label: 'About $5 million', value: 5_000_000 },
  { label: 'About $25 million', value: 25_000_000 },
  { label: 'About $100 million', value: 100_000_000 },
] as const

export function formatTradingCapital(amount: number | undefined): string {
  if (amount == null || !(amount > 0)) return 'Not specified'
  if (amount >= 1_000_000) {
    const millions = amount / 1_000_000
    return `~$${millions % 1 === 0 ? millions.toFixed(0) : millions.toFixed(1)}M`
  }
  return `~$${amount.toLocaleString()}`
}
