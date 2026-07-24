import { spawn } from 'node:child_process'
import { accessSync, constants } from 'node:fs'
import net from 'node:net'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const frontendDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = path.resolve(frontendDir, '..')
const children = []
/** Must match scripts/frontend_api.py /api/health → oos_aggregation. */
const REQUIRED_OOS_AGGREGATION = 'backtest_summaries'

function portOpen(port) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: '127.0.0.1', port })
    socket.setTimeout(350)
    socket.once('connect', () => {
      socket.destroy()
      resolve(true)
    })
    const unavailable = () => {
      socket.destroy()
      resolve(false)
    }
    socket.once('error', unavailable)
    socket.once('timeout', unavailable)
  })
}

function freePort(port) {
  return new Promise((resolve) => {
    const killer = spawn(
      'bash',
      ['-lc', `pids=$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null); [ -n "$pids" ] && kill $pids; exit 0`],
      { stdio: 'ignore' },
    )
    killer.once('exit', () => resolve())
  })
}

async function fetchJson(url, timeoutMs = 800) {
  const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

async function researchApiIsCurrent() {
  try {
    const health = await fetchJson('http://127.0.0.1:8787/api/health')
    return health?.status === 'ok' && health?.oos_aggregation === REQUIRED_OOS_AGGREGATION
  } catch {
    return false
  }
}

async function workflowApiIsHealthy() {
  try {
    // Unauthenticated root may 404; session requires a token. Port accept is enough
    // plus a cheap OPTIONS/GET that returns any HTTP response from uvicorn.
    return await portOpen(8790)
  } catch {
    return false
  }
}

async function waitFor(predicate, label, attempts = 40, delayMs = 250) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (await predicate()) return true
    await new Promise((resolve) => setTimeout(resolve, delayMs))
  }
  console.error(`[dev] Timed out waiting for ${label}`)
  return false
}

function pythonExecutable() {
  const candidates = [
    process.env.VIRTUAL_ENV && path.join(process.env.VIRTUAL_ENV, 'bin', 'python'),
    path.join(repoRoot, '.venv', 'bin', 'python'),
    'python3',
  ].filter(Boolean)
  for (const candidate of candidates) {
    if (candidate === 'python3') return candidate
    try {
      accessSync(candidate, constants.X_OK)
      return candidate
    } catch {
      // Try the next interpreter.
    }
  }
  return 'python3'
}

function start(name, command, args, env = process.env) {
  console.log(`[dev] Starting ${name}…`)
  const child = spawn(command, args, {
    cwd: repoRoot,
    env,
    stdio: 'inherit',
  })
  child.once('exit', (code, signal) => {
    if (code && code !== 0) {
      console.error(`[dev] ${name} exited with code ${code}${signal ? ` (${signal})` : ''}`)
    }
  })
  children.push(child)
  return child
}

const python = pythonExecutable()

if (await researchApiIsCurrent()) {
  console.log('[dev] Research API ready on http://127.0.0.1:8787')
} else {
  if (await portOpen(8787)) {
    console.log('[dev] Research API on :8787 is missing or stale; restarting')
    await freePort(8787)
    await new Promise((resolve) => setTimeout(resolve, 400))
  }
  start('research API', python, ['scripts/frontend_api.py', '--port', '8787'])
  await waitFor(researchApiIsCurrent, 'research API health')
}

if (await workflowApiIsHealthy()) {
  console.log('[dev] Workflow API ready on http://127.0.0.1:8790')
} else {
  const workflowEnv = {
    ...process.env,
    MARKETTRAINER_WORKFLOW_TOKENS: process.env.MARKETTRAINER_WORKFLOW_TOKENS || JSON.stringify({
      'investor-local': { user_id: 'investor_1', org_id: 'org_1', role: 'investor' },
      'operator-local': { user_id: 'operator_1', org_id: 'internal', role: 'operator' },
    }),
    MARKETTRAINER_PAYMENT_WEBHOOK_SECRET:
      process.env.MARKETTRAINER_PAYMENT_WEBHOOK_SECRET || 'local-webhook-secret',
  }
  start('workflow API', python, ['scripts/workflow_api.py', '--port', '8790'], workflowEnv)
  await waitFor(workflowApiIsHealthy, 'workflow API')
}

const vite = spawn(path.join(frontendDir, 'node_modules', '.bin', 'vite'), process.argv.slice(2), {
  cwd: frontendDir,
  env: process.env,
  stdio: 'inherit',
})
children.push(vite)

function shutdown(signal) {
  for (const child of children) {
    if (!child.killed) child.kill(signal)
  }
}

process.once('SIGINT', () => {
  shutdown('SIGINT')
  process.exit(130)
})
process.once('SIGTERM', () => {
  shutdown('SIGTERM')
  process.exit(143)
})

vite.once('exit', (code) => {
  shutdown('SIGTERM')
  process.exit(code ?? 0)
})
