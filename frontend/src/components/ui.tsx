import { X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'
import { cn } from '../lib/utils'

export function Button({
  children,
  variant = 'primary',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode
  variant?: 'primary' | 'secondary' | 'ghost' | 'dark'
}) {
  const variants = {
    primary: 'bg-pine text-cream hover:bg-[#173c31] shadow-[0_10px_25px_rgba(16,43,35,.16)]',
    secondary: 'border border-line bg-paper text-ink hover:border-pine/35 hover:bg-white',
    ghost: 'text-ink/70 hover:bg-ink/5 hover:text-ink',
    dark: 'border border-mint/20 bg-mint text-ink hover:bg-[#cdfbdd]',
  }
  return (
    <button
      className={cn(
        'inline-flex h-11 items-center justify-center gap-2 rounded-full px-5 text-sm font-semibold disabled:pointer-events-none disabled:opacity-40',
        variants[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function Badge({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode
  tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'dark'
  className?: string
}) {
  const tones = {
    neutral: 'border-line bg-white/60 text-ink/70',
    success: 'border-emerald-700/15 bg-emerald-50 text-emerald-800',
    warning: 'border-amber-700/15 bg-amber-50 text-amber-900',
    danger: 'border-red-700/15 bg-red-50 text-red-900',
    dark: 'border-mint/15 bg-mint/10 text-mint',
  }
  return (
    <span className={cn('inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold', tones[tone], className)}>
      {children}
    </span>
  )
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('soft-card rounded-[24px]', className)}>{children}</div>
}

export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: string
  hint?: string
  children: ReactNode
  className?: string
}) {
  return (
    <label className={cn('block', className)}>
      <span className="mb-2 flex items-center justify-between gap-4 text-xs font-semibold text-ink/80">
        {label}
        {hint && <span className="font-normal text-ink/55">{hint}</span>}
      </span>
      {children}
    </label>
  )
}

const controlClass =
  'h-11 w-full rounded-xl border border-line bg-white/75 px-3.5 text-sm text-ink outline-none placeholder:text-ink/60 hover:border-pine/30 focus:border-pine focus:ring-4 focus:ring-pine/5'

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(controlClass, className)} {...props} />
}

export function Select({ className, children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn(controlClass, 'appearance-none', className)} {...props}>
      {children}
    </select>
  )
}

/** Bare switch control with proper switch semantics. */
export function SwitchControl({
  checked,
  onChange,
  label,
  disabled,
  className,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label: string
  disabled?: boolean
  className?: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn('relative h-6 w-11 shrink-0 rounded-full p-0.5 disabled:opacity-55', checked ? 'bg-pine' : 'bg-ink/20', className)}
    >
      <span className={cn('block h-5 w-5 rounded-full bg-white shadow-sm transition-transform', checked && 'translate-x-5')} />
    </button>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label: string
  description?: string
  disabled?: boolean
}) {
  return (
    <div className="flex w-full items-center justify-between gap-5 rounded-2xl border border-line bg-white/60 p-4 text-left">
      <span>
        <span className={cn('block text-sm font-semibold text-ink', disabled && 'text-ink/60')}>{label}</span>
        {description && <span className="mt-1 block text-xs leading-5 text-ink/60">{description}</span>}
      </span>
      <SwitchControl checked={checked} onChange={onChange} label={label} disabled={disabled} />
    </div>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cn('animate-pulse rounded-xl bg-ink/8', className)} />
}

/**
 * Accessible modal: focus moves into the dialog, Tab is trapped, Escape
 * closes, and focus returns to the previously focused element on close.
 */
export function Modal({
  open,
  onClose,
  title,
  description,
  children,
}: {
  open: boolean
  onClose: () => void
  title: string
  description?: string
  children: ReactNode
}) {
  const dialogRef = useRef<HTMLDivElement>(null)
  const previouslyFocused = useRef<HTMLElement | null>(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  useEffect(() => {
    if (!open) return
    previouslyFocused.current = document.activeElement as HTMLElement | null
    const dialog = dialogRef.current
    dialog?.querySelector<HTMLElement>(
      'input:not([disabled]), select:not([disabled]), textarea:not([disabled])',
    )?.focus()

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onCloseRef.current()
        return
      }
      if (event.key !== 'Tab' || !dialog) return
      const focusable = Array.from(
        dialog.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'),
      )
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown, true)
    return () => {
      document.removeEventListener('keydown', onKeyDown, true)
      previouslyFocused.current?.focus()
    }
  }, [open])

  return (
      open && (
        <div className="fixed inset-0 z-50 grid place-items-center p-5">
          <div
            onClick={onClose}
            className="absolute inset-0 bg-ink/55"
            aria-hidden="true"
          />
          <div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-label={title}
            className="relative w-full max-w-md rounded-[26px] bg-paper p-6 shadow-2xl"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-ink">{title}</h3>
                {description && <p className="mt-1 text-xs text-ink/60">{description}</p>}
              </div>
              <button
                onClick={onClose}
                aria-label="Close dialog"
                className="grid h-8 w-8 shrink-0 place-items-center rounded-full text-ink/55 hover:bg-ink/5 hover:text-ink"
              >
                <X size={18} />
              </button>
            </div>
            {children}
          </div>
        </div>
      )
  )
}

/** Standard fetch-state panels so every page has loading / error / empty UI. */
export function ErrorPanel({
  message,
  onRetry,
  title = 'The configured API could not be reached.',
  hint = 'Check your network connection and API configuration, then retry.',
}: {
  message: string
  onRetry: () => void
  title?: string
  hint?: string
}) {
  return (
    <div role="alert" className="rounded-[24px] border border-red-700/15 bg-red-50 p-8 text-center">
      <p className="text-sm font-semibold text-red-900">{title}</p>
      <p className="mx-auto mt-2 max-w-md font-mono text-xs leading-5 text-red-900/70">{message}</p>
      <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-red-900/70">{hint}</p>
      <Button variant="secondary" className="mt-5" onClick={onRetry}>
        Retry
      </Button>
    </div>
  )
}

export function EmptyPanel({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-[24px] border border-dashed border-line bg-paper/60 p-10 text-center">
      <p className="text-sm font-semibold text-ink">{title}</p>
      <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-ink/60">{body}</p>
    </div>
  )
}
