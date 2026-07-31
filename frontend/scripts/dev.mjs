import { spawn } from 'node:child_process'
import {
  accessSync,
  constants,
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readlinkSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs'
import { createHash } from 'node:crypto'
import net from 'node:net'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const frontendDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = path.resolve(frontendDir, '..')
const children = []
/** Must match scripts/frontend_api.py /api/health → oos_aggregation. */
const REQUIRED_OOS_AGGREGATION = 'backtest_summaries'
/** Full FastAPI hangs on iCloud Desktop while training; lite is the default. */
const PREFER_FULL_API = process.env.MARKETTRAINER_FULL_API === '1'
/** Page data is static JSON under public/data — API is optional for ops GETs. */
const SKIP_RESEARCH_API = process.env.MARKETTRAINER_SKIP_RESEARCH_API === '1'
/**
 * Fresh npm install under /tmp (not a copy of iCloud node_modules — that hangs
 * while materializing placeholders). Vite resolves packages from here.
 */
const LOCAL_FRONTEND = process.env.MARKETTRAINER_LOCAL_FRONTEND || '/tmp/markettrainer-frontend'
const LOCAL_NM = process.env.MARKETTRAINER_NODE_MODULES || path.join(LOCAL_FRONTEND, 'node_modules')
const VITE_CACHE = process.env.VITE_CACHE_DIR || '/tmp/markettrainer-vite-cache'

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
      [
        '-lc',
        `pids=$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null); [ -n "$pids" ] && kill $pids 2>/dev/null; sleep 0.2; pids=$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null); [ -n "$pids" ] && kill -9 $pids 2>/dev/null; exit 0`,
      ],
      { stdio: 'ignore' },
    )
    killer.once('exit', () => resolve())
  })
}

/** Kill by exact script basename so lite/full do not cross-kill each other. */
function pkillScript(scriptBasename) {
  return new Promise((resolve) => {
    const killer = spawn(
      'bash',
      ['-lc', `pkill -f "[.]py ${scriptBasename}" 2>/dev/null; pkill -f "scripts/${scriptBasename}" 2>/dev/null; exit 0`],
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
    const health = await fetchJson('http://127.0.0.1:8787/api/health', 2_000)
    return health?.status === 'ok' && health?.oos_aggregation === REQUIRED_OOS_AGGREGATION
  } catch {
    return false
  }
}

async function workflowApiIsHealthy() {
  try {
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

function lockfileFingerprint() {
  const lockPath = path.join(frontendDir, 'package-lock.json')
  const pkgPath = path.join(frontendDir, 'package.json')
  const hash = createHash('sha256')
  for (const file of [lockPath, pkgPath]) {
    try {
      hash.update(readFileSync(file))
    } catch {
      hash.update(file)
    }
  }
  return hash.digest('hex').slice(0, 24)
}

function runCommand(command, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: 'inherit', ...opts })
    child.once('error', reject)
    child.once('exit', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`${command} exited with code ${code}`))
    })
  })
}

/**
 * Install deps into /tmp from the registry (npm ci). Never copy iCloud
 * node_modules — APFS clone still stalls on cloud placeholders for minutes.
 */
async function ensureLocalNodeModules() {
  if (process.env.MARKETTRAINER_SKIP_NM_CLONE === '1') {
    return path.join(frontendDir, 'node_modules')
  }
  const stampPath = path.join(LOCAL_FRONTEND, '.markettrainer-nm-stamp')
  const fingerprint = lockfileFingerprint()
  let current = ''
  try {
    current = readFileSync(stampPath, 'utf8').trim()
  } catch {
    // missing stamp → refresh
  }
  if (current === fingerprint && existsSync(path.join(LOCAL_NM, 'vite', 'package.json'))) {
    console.log(`[dev] Using /tmp node_modules at ${LOCAL_NM}`)
    return LOCAL_NM
  }

  console.log(`[dev] Installing frontend deps into ${LOCAL_FRONTEND} (once per lockfile change)…`)
  mkdirSync(LOCAL_FRONTEND, { recursive: true })
  copyFileSync(path.join(frontendDir, 'package.json'), path.join(LOCAL_FRONTEND, 'package.json'))
  copyFileSync(path.join(frontendDir, 'package-lock.json'), path.join(LOCAL_FRONTEND, 'package-lock.json'))
  rmSync(LOCAL_NM, { recursive: true, force: true })
  await runCommand('npm', ['ci', '--no-fund', '--no-audit'], { cwd: LOCAL_FRONTEND, env: process.env })
  writeFileSync(stampPath, `${fingerprint}\n`)
  console.log('[dev] /tmp node_modules ready')
  return LOCAL_NM
}

/**
 * Point frontend/node_modules at the /tmp install so Vite's optimizer serves
 * prebundled deps (not raw CJS from a custom resolve plugin).
 */
function ensureNodeModulesSymlink(targetNm) {
  if (process.env.MARKETTRAINER_SKIP_NM_CLONE === '1') return
  const linkPath = path.join(frontendDir, 'node_modules')
  const absTarget = path.resolve(targetNm)
  try {
    if (lstatSync(linkPath).isSymbolicLink()) {
      const current = path.resolve(path.dirname(linkPath), readlinkSync(linkPath))
      if (current === absTarget) {
        console.log('[dev] frontend/node_modules → /tmp (symlink ok)')
        return
      }
      rmSync(linkPath, { force: true })
    } else if (existsSync(linkPath)) {
      const backup = path.join(frontendDir, '.node_modules.icloud')
      if (!existsSync(backup)) {
        console.log('[dev] Moving iCloud node_modules → frontend/.node_modules.icloud')
        renameSync(linkPath, backup)
      } else {
        console.log('[dev] Replacing frontend/node_modules with /tmp symlink')
        rmSync(linkPath, { recursive: true, force: true })
      }
    }
  } catch {
    // missing — create symlink below
  }
  symlinkSync(absTarget, linkPath, 'dir')
  console.log(`[dev] Symlinked frontend/node_modules → ${absTarget}`)
}

/** Lite API is stdlib-only — prefer system python3 so a hung iCloud .venv cannot block it. */
function litePythonExecutable() {
  for (const candidate of ['/usr/bin/python3', 'python3']) {
    try {
      if (candidate.startsWith('/')) accessSync(candidate, constants.X_OK)
      return candidate
    } catch {
      // continue
    }
  }
  return 'python3'
}

/**
 * Spawn a child. Optional ``restart`` only relaunches when the health check fails —
 * never when another process already owns a healthy listener (avoids EADDRINUSE loops).
 */
function start(name, command, args, env = process.env, options = {}) {
  const { restart = false, port = null, healthy = null, maxRestarts = 4 } = options
  const restartsLeft = options._restartsLeft ?? maxRestarts
  console.log(`[dev] Starting ${name}…`)
  const child = spawn(command, args, {
    cwd: repoRoot,
    env,
    stdio: 'inherit',
  })
  child.once('exit', (code, signal) => {
    if (shuttingDown) return
    if (code && code !== 0) {
      console.error(`[dev] ${name} exited with code ${code}${signal ? ` (${signal})` : ''}`)
    }
    if (!restart) return

    setTimeout(() => {
      void (async () => {
        if (shuttingDown) return
        if (typeof healthy === 'function') {
          try {
            if (await healthy()) {
              console.log(`[dev] ${name} still healthy on port — skip restart`)
              return
            }
          } catch {
            // continue to restart
          }
        }
        if (restartsLeft <= 0) {
          console.error(
            `[dev] ${name} failed repeatedly; not restarting again. ` +
              `Free the port and re-run: bash scripts/start_ui.sh`,
          )
          return
        }
        if (port != null) {
          await freePort(port)
          // Wait until the listener is actually gone (avoids EADDRINUSE thrash).
          for (let i = 0; i < 10; i += 1) {
            if (!(await portOpen(port))) break
            await freePort(port)
            await new Promise((resolve) => setTimeout(resolve, 300))
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 500))
        if (shuttingDown) return
        console.error(`[dev] Restarting ${name} (${restartsLeft} left)…`)
        start(name, command, args, env, { ...options, _restartsLeft: restartsLeft - 1 })
      })()
    }, 1_500)
  })
  children.push(child)
  return child
}

let shuttingDown = false
const python = pythonExecutable()
const litePython = litePythonExecutable()

/** Publish /data/*.json before Vite so the first paint never waits on :8787. */
async function publishFrontendData() {
  console.log('[dev] Publishing frontend/public/data snapshots…')
  try {
    // Cache-only stubs (no Runs/ scan) — typically <1s so Vite can start immediately.
    await runCommand(litePython, ['scripts/publish_frontend_data.py', '--with-details'], {
      cwd: repoRoot,
      env: process.env,
    })
  } catch (error) {
    console.error(
      `[dev] Snapshot publish failed (${error instanceof Error ? error.message : error}). ` +
        'Ops pages need frontend/public/data/*.json — re-run: python3 scripts/publish_frontend_data.py',
    )
  }
}

await publishFrontendData()

if (SKIP_RESEARCH_API) {
  console.log('[dev] Skipping research API (MARKETTRAINER_SKIP_RESEARCH_API=1); UI uses /data/*.json')
} else if (await researchApiIsCurrent()) {
  console.log('[dev] Research API already healthy on http://127.0.0.1:8787 — reusing')
} else {
  // Exact basenames only (never pkill the shared prefix "frontend_api").
  await pkillScript('frontend_api_lite.py')
  await pkillScript('frontend_api.py')
  await freePort(8787)
  await new Promise((resolve) => setTimeout(resolve, 400))

  let ok = false
  if (PREFER_FULL_API) {
    start('research API', python, ['scripts/frontend_api.py', '--port', '8787'], process.env, {
      restart: false,
    })
    ok = await waitFor(researchApiIsCurrent, 'research API health', 24, 250)
    if (!ok) {
      console.error('[dev] Full research API did not become healthy; starting lite fallback')
      await pkillScript('frontend_api.py')
      await freePort(8787)
      await new Promise((resolve) => setTimeout(resolve, 400))
    }
  }

  if (!ok) {
    // Fire-and-forget: UI does not wait — static /data is already published.
    start(
      'research API (lite)',
      litePython,
      ['scripts/frontend_api_lite.py', '--port', '8787'],
      process.env,
      {
        restart: true,
        port: 8787,
        healthy: researchApiIsCurrent,
      },
    )
    console.log(
      '[dev] Research API starting in background (optional). Page GETs use /data/*.json.',
    )
  }
}

// Install deps under /tmp and symlink frontend/node_modules → that install.
const localNm = await ensureLocalNodeModules()
ensureNodeModulesSymlink(localNm)
const viteEnv = {
  ...process.env,
  VITE_CACHE_DIR: VITE_CACHE,
}
mkdirSync(VITE_CACHE, { recursive: true })

const viteBin = path.join(frontendDir, 'node_modules', '.bin', 'vite')
// Stale orphan Vite processes (PPID 1) commonly hold :5173 after a crashed
// start_ui — free the port before spawn so we never hit "Port already in use"
// or a hung white page talking to a dead listener.
if (await portOpen(5173)) {
  console.log('[dev] Port 5173 busy — freeing stale Vite…')
  await freePort(5173)
  for (let i = 0; i < 15; i += 1) {
    if (!(await portOpen(5173))) break
    await freePort(5173)
    await new Promise((resolve) => setTimeout(resolve, 200))
  }
  if (await portOpen(5173)) {
    console.error(
      '[dev] Could not free port 5173. Run: lsof -tiTCP:5173 -sTCP:LISTEN | xargs kill -9',
    )
    process.exit(1)
  }
}
console.log('[dev] Starting Vite on http://127.0.0.1:5173 …')
const vite = spawn(viteBin, process.argv.slice(2), {
  cwd: frontendDir,
  env: viteEnv,
  stdio: 'inherit',
})
children.push(vite)

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
  // Fire-and-forget: do not await health — Vite is already up for Forward/Runs.
  start('workflow API', python, ['scripts/workflow_api.py', '--port', '8790'], workflowEnv, {
    restart: false,
  })
  console.log(
    '[dev] Workflow API starting in background (mandates need :8790). Forward/Runs use :8787.',
  )
}

function shutdown(signal) {
  shuttingDown = true
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
