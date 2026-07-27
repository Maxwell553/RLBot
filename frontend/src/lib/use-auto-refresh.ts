import { useEffect, useRef } from 'react'

/** Default ops dashboard poll interval (ms). Keeps OOS / training rows current. */
export const DEFAULT_AUTO_REFRESH_MS = 15_000

/**
 * Periodically invoke ``reload`` while the tab is visible.
 * Skips ticks while a prior refresh is still marked ``refreshing``.
 */
export function useAutoRefresh(
  reload: () => void,
  {
    enabled = true,
    intervalMs = DEFAULT_AUTO_REFRESH_MS,
    refreshing = false,
  }: {
    enabled?: boolean
    intervalMs?: number
    refreshing?: boolean
  } = {},
): void {
  const reloadRef = useRef(reload)
  const refreshingRef = useRef(refreshing)
  reloadRef.current = reload
  refreshingRef.current = refreshing

  useEffect(() => {
    if (!enabled || intervalMs <= 0) return undefined
    const id = window.setInterval(() => {
      if (document.visibilityState === 'hidden') return
      if (refreshingRef.current) return
      reloadRef.current()
    }, intervalMs)
    return () => window.clearInterval(id)
  }, [enabled, intervalMs])
}
