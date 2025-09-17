const computeBackendBase = (): string => {
  const env = (import.meta as any).env?.VITE_BACKEND_URL as string | undefined
  if (env && env.trim()) {
    return env.replace(/\/$/, '')
  }
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol
    const hostname = window.location.hostname
    const defaultPort = 8000
    return `${protocol}//${hostname}:${defaultPort}`
  }
  return ''
}

export const BACKEND_BASE = computeBackendBase()
