import { Code2, UserRound } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { MODE_SWITCH_VISIBLE } from '../lib/runtime'
import { cn } from '../lib/utils'

export function ModeSwitch({
  mode,
  onNavigate,
}: {
  mode: 'investor' | 'developer'
  onNavigate?: () => void
}) {
  const navigate = useNavigate()
  if (!MODE_SWITCH_VISIBLE) return null

  const selectMode = (nextMode: 'investor' | 'developer') => {
    if (nextMode === mode) return
    onNavigate?.()
    navigate(nextMode === 'investor' ? '/portal' : '/ops')
  }

  return (
    <div className="mx-4 mt-2 rounded-2xl border border-white/10 bg-white/[.04] p-1">
      <div className="grid grid-cols-2 gap-1" role="group" aria-label="Application mode">
        <button
          type="button"
          onClick={() => selectMode('investor')}
          aria-pressed={mode === 'investor'}
          className={cn(
            'flex h-9 items-center justify-center gap-1.5 rounded-xl text-[11px] font-semibold transition-colors',
            mode === 'investor' ? 'bg-mint text-ink' : 'text-cream/60 hover:bg-white/[.05] hover:text-cream',
          )}
        >
          <UserRound size={13} aria-hidden="true" />
          Investor
        </button>
        <button
          type="button"
          onClick={() => selectMode('developer')}
          aria-pressed={mode === 'developer'}
          className={cn(
            'flex h-9 items-center justify-center gap-1.5 rounded-xl text-[11px] font-semibold transition-colors',
            mode === 'developer' ? 'bg-mint text-ink' : 'text-cream/60 hover:bg-white/[.05] hover:text-cream',
          )}
        >
          <Code2 size={13} aria-hidden="true" />
          Research Ops
        </button>
      </div>
    </div>
  )
}
