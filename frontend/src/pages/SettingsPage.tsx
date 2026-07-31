import { CheckCircle2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Badge, Button, Card, Field, Input, Select, Toggle } from '../components/ui'
import { OFFLINE_MODE } from '../lib/api'

type WorkspaceSettings = {
  workspaceName: string
  organizationType: string
  baseCurrency: string
  requireApproval: boolean
  emailUpdates: boolean
}

const DEFAULTS: WorkspaceSettings = {
  workspaceName: 'My workspace',
  organizationType: 'Asset manager',
  baseCurrency: 'USD',
  requireApproval: true,
  emailUpdates: false,
}

const STORAGE_KEY = 'markettrainer.workspace-settings'

function loadSettings(): WorkspaceSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULTS
    return { ...DEFAULTS, ...(JSON.parse(raw) as Partial<WorkspaceSettings>) }
  } catch {
    return DEFAULTS
  }
}

export function SettingsPage() {
  const [settings, setSettings] = useState<WorkspaceSettings>(loadSettings)
  const [savedAt, setSavedAt] = useState<number | null>(null)

  useEffect(() => {
    if (savedAt == null) return
    const timer = window.setTimeout(() => setSavedAt(null), 2000)
    return () => window.clearTimeout(timer)
  }, [savedAt])

  const update = <K extends keyof WorkspaceSettings>(key: K, value: WorkspaceSettings[K]) =>
    setSettings((current) => ({ ...current, [key]: value }))

  const save = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
    setSavedAt(Date.now())
  }

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header>
        <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Workspace</p>
        <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">Settings</h1>
        <p className="mt-2 max-w-xl text-xs leading-5 text-ink/60">
          These preferences are stored in this browser only. Team accounts, roles, and server-side
          policy enforcement require the future authenticated backend and are intentionally not
          simulated here.
        </p>
      </header>

      <div className="mt-8 max-w-2xl space-y-5">
        <Card className="p-6">
          <h2 className="text-sm font-semibold">Workspace profile</h2>
          <p className="mt-1 text-[11px] text-ink/60">Used for display and exported-file naming.</p>
          <div className="mt-6 grid gap-5 sm:grid-cols-2">
            <Field label="Workspace name">
              <Input value={settings.workspaceName} onChange={(event) => update('workspaceName', event.target.value)} />
            </Field>
            <Field label="Organization type">
              <Select value={settings.organizationType} onChange={(event) => update('organizationType', event.target.value)}>
                <option>Asset manager</option>
                <option>Family office</option>
                <option>Quant fund</option>
                <option>Research firm</option>
              </Select>
            </Field>
            <Field label="Default base currency">
              <Select value={settings.baseCurrency} onChange={(event) => update('baseCurrency', event.target.value)}>
                <option>USD</option>
                <option>EUR</option>
                <option>GBP</option>
                <option>JPY</option>
              </Select>
            </Field>
          </div>
        </Card>

        <Card className="p-6">
          <h2 className="text-sm font-semibold">Workflow preferences</h2>
          <p className="mt-1 text-[11px] text-ink/60">Advisory flags recorded with exported mandates.</p>
          <div className="mt-6 space-y-4">
            <Toggle
              checked={settings.requireApproval}
              onChange={(value) => update('requireApproval', value)}
              label="Operator approval before launch"
              description="Recorded in exported mandate metadata so the operator workflow can enforce it."
            />
            <Toggle
              checked={settings.emailUpdates}
              onChange={(value) => update('emailUpdates', value)}
              label="Notification preference"
              description="Stored for the future backend; no emails are sent from the browser."
            />
          </div>
          <div className="mt-6 flex items-center justify-end gap-3">
            {savedAt != null && (
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700" role="status">
                <CheckCircle2 size={14} aria-hidden="true" /> Saved to this browser
              </span>
            )}
            <Button onClick={save}>Save preferences</Button>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold">Research API</h2>
            {OFFLINE_MODE ? <Badge tone="warning">Offline</Badge> : <Badge tone="success">Connected</Badge>}
          </div>
          <p className="mt-3 text-[12px] leading-6 text-ink/70">
            {OFFLINE_MODE ? (
              <>
                The research API provides live run data, provenance, and engine-backed preflight. Configure{' '}
                <code className="rounded bg-ink/5 px-1.5 py-0.5 font-mono text-[11px]">VITE_API_URL</code>{' '}
                in <code className="rounded bg-ink/5 px-1.5 py-0.5 font-mono text-[11px]">frontend/.env.local</code>{' '}
                to connect.
              </>
            ) : (
              <>
                Run data is served by the research API
                {import.meta.env.VITE_API_URL ? (
                  <>
                    {' '}
                    at{' '}
                    <a
                      className="font-semibold text-pine underline"
                      href={`${import.meta.env.VITE_API_URL}/docs`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {import.meta.env.VITE_API_URL}/docs
                    </a>
                  </>
                ) : (
                  <>
                    {' '}
                    via the Vite proxy (
                    <a
                      className="font-semibold text-pine underline"
                      href="http://127.0.0.1:8787/docs"
                      target="_blank"
                      rel="noreferrer"
                    >
                      http://127.0.0.1:8787/docs
                    </a>
                    ).
                  </>
                )}
              </>
            )}
          </p>
        </Card>
      </div>
    </div>
  )
}
