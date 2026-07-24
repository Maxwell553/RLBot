import { afterEach, describe, expect, it, vi } from 'vitest'
import { RequestCoordinator, RequestFailure } from './http'

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('RequestCoordinator', () => {
  it('deduplicates GET requests even when consumers use abort signals', async () => {
    let resolveFetch!: (response: Response) => void
    const fetchMock = vi.fn(() => new Promise<Response>((resolve) => {
      resolveFetch = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)
    const coordinator = new RequestCoordinator(1_000, 5_000)
    const firstController = new AbortController()
    const secondController = new AbortController()

    const first = coordinator.request<{ ok: boolean }>('https://example.test/api', {}, firstController.signal)
    const second = coordinator.request<{ ok: boolean }>('https://example.test/api', {}, secondController.signal)
    const firstResult = first.catch((error) => error)
    firstController.abort()
    resolveFetch(new Response(JSON.stringify({ ok: true }), { status: 200 }))

    expect(await firstResult).toMatchObject({ kind: 'aborted' })
    await expect(second).resolves.toEqual({ ok: true })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('fails stalled requests after a bounded timeout', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => undefined)))
    const coordinator = new RequestCoordinator(250, 0)
    const pending = coordinator.request('https://example.test/slow')
    const result = pending.catch((error) => error)

    await vi.advanceTimersByTimeAsync(251)

    expect(await result).toBeInstanceOf(RequestFailure)
    expect(await result).toMatchObject({ kind: 'timeout' })
  })

  it('classifies HTTP and network failures', async () => {
    const coordinator = new RequestCoordinator(1_000, 0)
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 503 })))
    await expect(coordinator.request('https://example.test/unavailable')).rejects.toMatchObject({
      kind: 'http',
      status: 503,
    })

    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }))
    await expect(coordinator.request('https://example.test/cors')).rejects.toMatchObject({
      kind: 'network',
    })
  })
})
