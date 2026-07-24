import { BarChart3, ClipboardList, FlaskConical, LayoutDashboard, Menu, Settings, SlidersHorizontal, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Brand } from '../components/Brand'
import { ModeSwitch } from '../components/ModeSwitch'
import { cn } from '../lib/utils'

const nav = [
  { to: '/ops', label: 'Operations overview', icon: LayoutDashboard, end: true, preload: () => import('../pages/DashboardPage') },
  { to: '/ops/requests', label: 'Mandate requests', icon: ClipboardList, preload: () => import('../pages/DeveloperRequestsPage') },
  { to: '/ops/mandates/new', label: 'Config builder', icon: SlidersHorizontal, preload: () => import('../pages/ConfigurePage') },
  { to: '/ops/runs', label: 'Runs', icon: FlaskConical, preload: () => import('../pages/RunsPage') },
  { to: '/ops/results', label: 'OOS results', icon: BarChart3, preload: () => import('../pages/ResultsPage') },
]

function SidebarContent({ close }: { close?: () => void }) {
  return (
    <>
      <div className="px-5 py-6"><Brand light /></div>
      <ModeSwitch mode="developer" onNavigate={close} />
      <nav className="mt-7 flex-1 px-3" aria-label="Research Operations">
        <div className="mb-2 px-3 font-mono text-[11px] uppercase tracking-[0.18em] text-cream/70">Research Operations</div>
        {nav.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={close}
            onMouseEnter={item.preload}
            onFocus={item.preload}
            className={({ isActive }) => cn(
              'mb-1 flex h-11 items-center gap-3 rounded-xl px-3 text-xs font-medium',
              isActive ? 'bg-mint text-ink' : 'text-cream/70 hover:bg-white/[.05] hover:text-cream',
            )}
          >
            <item.icon size={16} strokeWidth={1.8} aria-hidden="true" />
            {item.label}
          </NavLink>
        ))}
        <div className="mb-2 mt-7 px-3 font-mono text-[11px] uppercase tracking-[0.18em] text-cream/70">Organization</div>
        <NavLink
          to="/ops/settings"
          onClick={close}
          onMouseEnter={() => import('../pages/SettingsPage')}
          onFocus={() => import('../pages/SettingsPage')}
          className={({ isActive }) => cn(
            'flex h-11 items-center gap-3 rounded-xl px-3 text-xs font-medium',
            isActive ? 'bg-mint text-ink' : 'text-cream/70 hover:bg-white/[.05] hover:text-cream',
          )}
        >
          <Settings size={16} strokeWidth={1.8} aria-hidden="true" /> Settings
        </NavLink>
      </nav>
      <div className="m-4 rounded-2xl border border-mint/10 bg-mint/[.06] p-4">
        <p className="text-xs font-semibold text-cream">Operator workflow</p>
        <p className="mt-1 text-[11px] leading-4 text-cream/60">
          Exported mandates are validated and launched from the research CLI, never from the browser.
        </p>
      </div>
    </>
  )
}

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    // Warm ops page chunks once the shell mounts so tab switches skip cold loads.
    for (const item of nav) void item.preload()
    void import('../pages/SettingsPage')
  }, [])

  useEffect(() => {
    if (!mobileOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMobileOpen(false)
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [mobileOpen])

  return (
    <div className="min-h-screen bg-cream">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[248px] flex-col bg-ink lg:flex">
        <SidebarContent />
      </aside>
      <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-line bg-cream/90 px-5 backdrop-blur-xl lg:hidden">
        <Brand />
        <button
          onClick={() => setMobileOpen(true)}
          className="grid h-10 w-10 place-items-center rounded-full border border-line bg-paper"
          aria-label="Open navigation"
          aria-expanded={mobileOpen}
        >
          <Menu size={18} aria-hidden="true" />
        </button>
      </header>
      {mobileOpen && (
          <>
            <button
              onClick={() => setMobileOpen(false)}
              className="fixed inset-0 z-40 bg-ink/50 lg:hidden"
              aria-label="Close navigation"
            />
            <aside
              role="dialog"
              aria-modal="true"
              aria-label="Navigation"
              className="fixed inset-y-0 left-0 z-50 flex w-[280px] flex-col bg-ink lg:hidden"
            >
              <button
                onClick={() => setMobileOpen(false)}
                className="absolute right-4 top-5 grid h-8 w-8 place-items-center rounded-full bg-white/5 text-cream/70"
                aria-label="Close navigation"
              >
                <X size={16} aria-hidden="true" />
              </button>
              <SidebarContent close={() => setMobileOpen(false)} />
            </aside>
          </>
        )}
      
      <main className="min-h-screen lg:ml-[248px]">
        <div key={location.pathname}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
