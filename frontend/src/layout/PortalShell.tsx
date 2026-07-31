import { FileCheck2, FolderKanban, LayoutDashboard, Menu, ShieldCheck, SlidersHorizontal, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Brand } from '../components/Brand'
import { ModeSwitch } from '../components/ModeSwitch'
import { cn } from '../lib/utils'

const nav = [
  { to: '/portal', label: 'Home', icon: LayoutDashboard, end: true, preload: () => import('../pages/PortalPage') },
  { to: '/portal/mandates/new', label: 'Create mandate', icon: SlidersHorizontal, preload: () => import('../pages/InvestorMandatePage') },
  { to: '/portal/builds', label: 'Build status', icon: FolderKanban, preload: () => import('../pages/PortalPage') },
  { to: '/portal/reports', label: 'Delivered models', icon: FileCheck2, preload: () => import('../pages/PortalPage') },
]

function PortalNavigation({ close }: { close?: () => void }) {
  return (
    <>
      <div className="px-5 py-6"><Brand light /></div>
      <ModeSwitch mode="investor" onNavigate={close} />
      <nav className="mt-7 flex-1 px-3" aria-label="Investor portal">
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
      </nav>
      <div className="m-4 rounded-2xl border border-mint/10 bg-mint/[.06] p-4">
        <ShieldCheck size={16} className="text-mint" aria-hidden="true" />
        <p className="mt-3 text-xs font-semibold text-cream">Governed delivery</p>
        <p className="mt-1 text-[11px] leading-4 text-cream/60">
          Payment, training, OOS evaluation, and report release require verified server-side transitions.
        </p>
      </div>
    </>
  )
}

export function PortalShell() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  // Preload only on hover/focus — keep first paint to the active route.

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
        <PortalNavigation />
      </aside>
      <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-line bg-cream/90 px-5 backdrop-blur-xl lg:hidden">
        <Brand />
        <button
          type="button"
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
              type="button"
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
                type="button"
                onClick={() => setMobileOpen(false)}
                className="absolute right-4 top-5 grid h-8 w-8 place-items-center rounded-full bg-white/5 text-cream/70"
                aria-label="Close navigation"
              >
                <X size={16} aria-hidden="true" />
              </button>
              <PortalNavigation close={() => setMobileOpen(false)} />
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
