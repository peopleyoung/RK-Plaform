import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, ApiError } from './client'
import type { Artifact, Dataset, Job, ModelProfile, ServiceEndpoint, WorkerNode } from '../types'

interface PlatformState {
  profiles: ModelProfile[]
  datasets: Dataset[]
  jobs: Job[]
  workers: WorkerNode[]
  artifacts: Artifact[]
  serviceEndpoints: ServiceEndpoint[]
  loading: boolean
  error: string
  authRequired: boolean
  authenticate: (token: string) => Promise<void>
  refresh: () => Promise<void>
}

const PlatformContext = createContext<PlatformState | null>(null)

export function PlatformProvider({ children }: { children: ReactNode }) {
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [workers, setWorkers] = useState<WorkerNode[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [serviceEndpoints, setServiceEndpoints] = useState<ServiceEndpoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [authRequired, setAuthRequired] = useState(false)
  const refreshSequence = useRef(0)

  const refresh = useCallback(async () => {
    const sequence = ++refreshSequence.current
    try {
      const [nextProfiles, nextDatasets, nextJobs, nextWorkers, nextArtifacts, nextServiceEndpoints] = await Promise.all([
        api.modelProfiles(), api.datasets(), api.jobs(), api.workers(), api.artifacts(), api.serviceEndpoints(),
      ])
      if (sequence !== refreshSequence.current) return
      setProfiles(nextProfiles)
      setDatasets(nextDatasets)
      setJobs(nextJobs)
      setWorkers(nextWorkers)
      setArtifacts(nextArtifacts)
      setServiceEndpoints(nextServiceEndpoints)
      setError('')
      setAuthRequired(false)
    } catch (reason) {
      if (sequence !== refreshSequence.current) return
      if (reason instanceof ApiError && reason.status === 401) setAuthRequired(true)
      setError(reason instanceof Error ? reason.message : '平台数据加载失败')
    } finally {
      if (sequence === refreshSequence.current) setLoading(false)
    }
  }, [])

  const authenticate = useCallback(async (token: string) => {
    sessionStorage.setItem('rknode.adminToken', token)
    setAuthRequired(false)
    setLoading(true)
    await refresh()
  }, [refresh])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void refresh()
    }, 3000)
    return () => window.clearInterval(timer)
  }, [refresh])

  const value = useMemo(() => ({ profiles, datasets, jobs, workers, artifacts, serviceEndpoints, loading, error, authRequired, authenticate, refresh }), [profiles, datasets, jobs, workers, artifacts, serviceEndpoints, loading, error, authRequired, authenticate, refresh])
  return <PlatformContext.Provider value={value}>{children}</PlatformContext.Provider>
}

export function usePlatform() {
  const value = useContext(PlatformContext)
  if (!value) throw new Error('usePlatform must be used inside PlatformProvider')
  return value
}
