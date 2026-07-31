export type RequestFailureKind = 'timeout' | 'network' | 'http' | 'aborted' | 'invalid-response'

export class RequestFailure extends Error {
  readonly kind: RequestFailureKind
  readonly status?: number

  constructor(
    kind: RequestFailureKind,
    message: string,
    status?: number,
  ) {
    super(message)
    this.name = 'RequestFailure'
    this.kind = kind
    this.status = status
  }
}

type CacheEntry = {
  expiresAt: number
  promise: Promise<unknown>
}

function consumerAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise
  if (signal.aborted) return Promise.reject(new RequestFailure('aborted', 'Request cancelled'))
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(new RequestFailure('aborted', 'Request cancelled'))
    signal.addEventListener('abort', abort, { once: true })
    promise.then(
      (value) => {
        signal.removeEventListener('abort', abort)
        resolve(value)
      },
      (error) => {
        signal.removeEventListener('abort', abort)
        reject(error)
      },
    )
  })
}

export class RequestCoordinator {
  private readonly cache = new Map<string, CacheEntry>()
  private readonly timeoutMs: number
  private readonly cacheMs: number

  constructor(timeoutMs = 10_000, cacheMs = 5_000) {
    this.timeoutMs = timeoutMs
    this.cacheMs = cacheMs
  }

  clear(path?: string) {
    if (path) this.cache.delete(path)
    else this.cache.clear()
  }

  request<T>(
    url: string,
    init: RequestInit = {},
    consumerSignal?: AbortSignal,
    timeoutMs?: number,
  ): Promise<T> {
    const method = (init.method ?? 'GET').toUpperCase()
    const cacheable = method === 'GET'
    const now = Date.now()
    const existing = cacheable ? this.cache.get(url) : undefined
    if (existing && existing.expiresAt > now) {
      return consumerAbort(existing.promise as Promise<T>, consumerSignal)
    }

    const effectiveTimeout = timeoutMs ?? this.timeoutMs
    const controller = new AbortController()
    let timeout: ReturnType<typeof globalThis.setTimeout>
    const timeoutFailure = new Promise<never>((_, reject) => {
      timeout = globalThis.setTimeout(() => {
        controller.abort('timeout')
        reject(new RequestFailure('timeout', `Request timed out after ${effectiveTimeout / 1000} seconds`))
      }, effectiveTimeout)
    })
    const networkRequest = fetch(url, { ...init, signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new RequestFailure('http', `HTTP ${response.status} from ${new URL(url, 'http://local.invalid').pathname}`, response.status)
        }
        try {
          return await response.json() as T
        } catch {
          throw new RequestFailure('invalid-response', `Invalid JSON from ${new URL(url, 'http://local.invalid').pathname}`)
        }
      })
    const promise = Promise.race([networkRequest, timeoutFailure])
      .catch((error: unknown) => {
        if (error instanceof RequestFailure) throw error
        if (controller.signal.aborted) {
          throw new RequestFailure('timeout', `Request timed out after ${effectiveTimeout / 1000} seconds`)
        }
        throw new RequestFailure(
          'network',
          error instanceof Error ? `Network or CORS failure: ${error.message}` : 'Network or CORS failure',
        )
      })
      .finally(() => globalThis.clearTimeout(timeout!))

    if (cacheable) {
      this.cache.set(url, { expiresAt: now + this.cacheMs, promise })
      promise.catch(() => {
        if (this.cache.get(url)?.promise === promise) this.cache.delete(url)
      })
    }
    return consumerAbort(promise, consumerSignal)
  }
}
