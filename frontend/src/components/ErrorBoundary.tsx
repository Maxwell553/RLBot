import { Component, type ReactNode } from 'react'

type Props = { children: ReactNode }
type State = { error: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div role="alert" className="grid min-h-screen place-items-center bg-cream px-6">
        <div className="w-full max-w-md rounded-3xl border border-line bg-paper p-8 text-center shadow-lg">
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-coral">Application error</p>
          <h1 className="font-display mt-3 text-3xl tracking-tight text-ink">Something went wrong.</h1>
          <p className="mt-3 text-sm leading-6 text-ink/60">
            The interface hit an unexpected error. Reload the page; if the problem persists, check the
            browser console and report the error message.
          </p>
          <pre className="mt-4 overflow-x-auto rounded-xl bg-ink/5 p-3 text-left font-mono text-[11px] text-ink/70">
            {this.state.error.message}
          </pre>
          <button
            onClick={() => window.location.reload()}
            className="mt-6 inline-flex h-11 items-center rounded-full bg-pine px-6 text-sm font-semibold text-cream hover:bg-[#173c31]"
          >
            Reload
          </button>
        </div>
      </div>
    )
  }
}
