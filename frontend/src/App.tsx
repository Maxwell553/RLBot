import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { OPS_UI_ENABLED } from './lib/runtime'
import { LandingPage } from './pages/LandingPage'

/**
 * Landing is eager so ``/`` does not wait on a route chunk + Suspense flash.
 * Ops/portal pages stay lazy — they pull heavier charts and forms.
 */
const AppShell = lazy(() =>
  import('./layout/AppShell').then((module) => ({ default: module.AppShell })),
)
const PortalShell = lazy(() =>
  import('./layout/PortalShell').then((module) => ({ default: module.PortalShell })),
)
const ConfigurePage = lazy(() =>
  import('./pages/ConfigurePage').then((module) => ({ default: module.ConfigurePage })),
)
const DashboardPage = lazy(() =>
  import('./pages/DashboardPage').then((module) => ({ default: module.DashboardPage })),
)
const DeveloperRequestsPage = lazy(() =>
  import('./pages/DeveloperRequestsPage').then((module) => ({ default: module.DeveloperRequestsPage })),
)
const InvestorMandatePage = lazy(() =>
  import('./pages/InvestorMandatePage').then((module) => ({ default: module.InvestorMandatePage })),
)
const PortalPage = lazy(() =>
  import('./pages/PortalPage').then((module) => ({ default: module.PortalPage })),
)
const PortalBuildsPage = lazy(() =>
  import('./pages/PortalPage').then((module) => ({ default: module.PortalBuildsPage })),
)
const PortalReportsPage = lazy(() =>
  import('./pages/PortalPage').then((module) => ({ default: module.PortalReportsPage })),
)
const ResultsPage = lazy(() =>
  import('./pages/ResultsPage').then((module) => ({ default: module.ResultsPage })),
)
const RunsPage = lazy(() =>
  import('./pages/RunsPage').then((module) => ({ default: module.RunsPage })),
)
const ForwardPage = lazy(() =>
  import('./pages/ForwardPage').then((module) => ({ default: module.ForwardPage })),
)
const SettingsPage = lazy(() =>
  import('./pages/SettingsPage').then((module) => ({ default: module.SettingsPage })),
)

function RouteFallback() {
  return (
    <div className="min-h-screen bg-cream px-8 py-10" aria-busy="true" aria-label="Loading">
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="h-10 w-64 animate-pulse rounded-xl bg-ink/8" />
        <div className="h-4 w-96 max-w-full animate-pulse rounded-lg bg-ink/8" />
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          <div className="h-40 animate-pulse rounded-3xl bg-ink/8" />
          <div className="h-40 animate-pulse rounded-3xl bg-ink/8" />
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <Suspense fallback={<RouteFallback />}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/portal" element={<PortalShell />}>
          <Route index element={<PortalPage />} />
          <Route path="mandates/new" element={<InvestorMandatePage />} />
          <Route path="builds" element={<PortalBuildsPage />} />
          <Route path="reports" element={<PortalReportsPage />} />
        </Route>
        {OPS_UI_ENABLED && (
          <Route path="/ops" element={<AppShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="forward" element={<ForwardPage />} />
            <Route path="requests" element={<DeveloperRequestsPage />} />
            <Route path="mandates/new" element={<ConfigurePage />} />
            <Route path="runs" element={<RunsPage />} />
            <Route path="results" element={<ResultsPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        )}
        {!OPS_UI_ENABLED && <Route path="/ops/*" element={<Navigate to="/portal" replace />} />}
        <Route path="/app" element={<Navigate to="/ops" replace />} />
        <Route path="/app/mandates/new" element={<Navigate to="/ops/mandates/new" replace />} />
        <Route path="/app/runs" element={<Navigate to="/ops/runs" replace />} />
        <Route path="/app/results" element={<Navigate to="/ops/results" replace />} />
        <Route path="/app/settings" element={<Navigate to="/ops/settings" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}
