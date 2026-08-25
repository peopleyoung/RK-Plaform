import type {
  Artifact,
  ConversionJobInput,
  Dataset, DatasetFormat,
  Job,
  JobEvent,
  ModelProfile,
  ModelRelease,
  NodeGroup,
  InferenceNode,
  InferenceNodeCreated,
  InferenceSummary,
  InferenceTask,
  InferencePlaybackSession,
  MediaGateway,
  MediaGatewayInput,
  Deployment,
  ServiceEndpoint,
  ServiceEndpointCreated,
  ServiceEndpointEnrollment,
  ServiceEndpointInput,
  ServiceEndpointTestResult,
  TaskType,
  TrainingJobInput,
  WorkerNode,
} from '../types'

declare global {
  interface Window {
    __RKNODE_CONFIG__?: { apiBaseUrl?: string; mediaWorkerEnabled?: boolean }
  }
}

const runtimeApiBaseUrl = typeof window === 'undefined'
  ? undefined
  : window.__RKNODE_CONFIG__?.apiBaseUrl
const apiBaseUrl = (runtimeApiBaseUrl ?? import.meta.env.VITE_API_BASE_URL ?? '/api/v1').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details: Record<string, unknown> = {}) {
    super(message)
  }
}

function adminToken() {
  return sessionStorage.getItem('rknode.adminToken') ?? (import.meta.env.DEV ? 'dev-admin-token' : '')
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = adminToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  headers.set('Accept', 'application/json')
  const response = await fetch(`${apiBaseUrl}${path}`, { ...init, headers })
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { error?: { code?: string; message?: string; details?: Record<string, unknown> } } | null
    throw new ApiError(response.status, payload?.error?.code ?? 'http_error', payload?.error?.message ?? `请求失败 (${response.status})`, payload?.error?.details)
  }
  return await response.json() as T
}

async function requestEmpty(path: string, init: RequestInit): Promise<void> {
  const token = adminToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${apiBaseUrl}${path}`, { ...init, headers })
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { error?: { code?: string; message?: string; details?: Record<string, unknown> } } | null
    throw new ApiError(response.status, payload?.error?.code ?? 'http_error', payload?.error?.message ?? `请求失败 (${response.status})`, payload?.error?.details)
  }
}

export const api = {
  modelProfiles: async () => (await request<{ schemaVersion: number; profiles: ModelProfile[] }>('/model-profiles')).profiles,
  datasets: () => request<Dataset[]>('/datasets'),
  jobs: (type?: Job['type']) => request<Job[]>(`/jobs${type ? `?type=${type}` : ''}`),
  job: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),
  jobEvents: (id: string, afterId = 0, limit = 500) => request<JobEvent[]>(
    `/jobs/${encodeURIComponent(id)}/events?afterId=${afterId}&limit=${limit}`,
  ),
  workers: () => request<WorkerNode[]>('/workers'),
  deleteWorker: (id: string) => requestEmpty(`/workers/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  artifacts: (kind?: string) => request<Artifact[]>(`/artifacts${kind ? `?kind=${encodeURIComponent(kind)}` : ''}`),
  serviceEndpoints: () => request<ServiceEndpoint[]>('/service-endpoints'),
  createTrainingJob: (payload: TrainingJobInput) => request<Job>('/training-jobs', { method: 'POST', body: JSON.stringify(payload) }),
  createConversionJob: (payload: ConversionJobInput) => request<Job>('/conversion-jobs', { method: 'POST', body: JSON.stringify(payload) }),
  retryJob: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}/retry`, { method: 'POST' }),
  createServiceEndpoint: (payload: ServiceEndpointInput) => request<ServiceEndpointCreated>('/service-endpoints', { method: 'POST', body: JSON.stringify(payload) }),
  updateServiceEndpoint: (id: string, payload: ServiceEndpointInput) => request<ServiceEndpoint>(`/service-endpoints/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(payload) }),
  testServiceEndpoint: (payload: ServiceEndpointInput) => request<ServiceEndpointTestResult>('/service-endpoints/test', { method: 'POST', body: JSON.stringify(payload) }),
  testServiceEndpointUpdate: (id: string, payload: ServiceEndpointInput) => request<ServiceEndpointTestResult>(`/service-endpoints/${encodeURIComponent(id)}/test`, { method: 'POST', body: JSON.stringify(payload) }),
  probeServiceEndpoint: (id: string) => request<ServiceEndpoint>(`/service-endpoints/${encodeURIComponent(id)}/probe`, { method: 'POST' }),
  reissueServiceEndpointEnrollment: (id: string) => request<ServiceEndpointEnrollment>(`/service-endpoints/${encodeURIComponent(id)}/enrollment-token`, { method: 'POST' }),
  deleteDataset: (id: string) => requestEmpty(`/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  deleteJob: (id: string) => requestEmpty(`/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  deleteServiceEndpoint: (id: string) => requestEmpty(`/service-endpoints/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  modelReleases: (page = 1, pageSize = 50) => request<{ items: ModelRelease[]; page: number; pageSize: number; total: number }>(`/model-releases?page=${page}&pageSize=${pageSize}`),
  createModelRelease: (payload: { name: string; version: string; conversionJobId: string; description?: string }) => request<ModelRelease>('/model-releases', { method: 'POST', body: JSON.stringify(payload) }),
  publishModelRelease: (id: string) => request<ModelRelease>(`/model-releases/${encodeURIComponent(id)}/publish`, { method: 'POST' }),
  deprecateModelRelease: (id: string) => request<ModelRelease>(`/model-releases/${encodeURIComponent(id)}/deprecate`, { method: 'POST' }),
  deleteModelRelease: (id: string) => requestEmpty(`/model-releases/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  nodeGroups: () => request<NodeGroup[]>('/node-groups'),
  createNodeGroup: (payload: { name: string; description?: string; labels?: string[] }) => request<NodeGroup>('/node-groups', { method: 'POST', body: JSON.stringify(payload) }),
  updateNodeGroup: (id: string, payload: { name: string; description?: string; labels?: string[] }) => request<NodeGroup>(`/node-groups/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(payload) }),
  deleteNodeGroup: (id: string) => requestEmpty(`/node-groups/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  inferenceNodes: (page = 1, pageSize = 50) => request<{ items: InferenceNode[]; page: number; pageSize: number; total: number }>(`/inference-nodes?page=${page}&pageSize=${pageSize}`),
  inferenceSummary: () => request<InferenceSummary>('/inference-summary'),
  createInferenceNode: (payload: { name: string; groupId?: string; labels?: string[]; maxModelInstances?: number }) => request<InferenceNodeCreated>('/inference-nodes', { method: 'POST', body: JSON.stringify(payload) }),
  approveInferenceNode: (id: string) => request<InferenceNode>(`/inference-nodes/${encodeURIComponent(id)}/approve`, { method: 'POST' }),
  retireInferenceNode: (id: string) => request<InferenceNode>(`/inference-nodes/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  deleteRetiredInferenceNode: (id: string) => requestEmpty(`/inference-nodes/${encodeURIComponent(id)}/record`, { method: 'DELETE' }),
  inferenceTasks: (page = 1, pageSize = 50) => request<{ items: InferenceTask[]; page: number; pageSize: number; total: number }>(`/inference-tasks?page=${page}&pageSize=${pageSize}`),
  createInferenceTask: (payload: { name: string; releaseId: string; nodeId?: string; groupId?: string; inputUri: string; interval?: number; thresholds?: Record<string, number>; output?: Record<string, unknown>; media?: Record<string, unknown>; analytics?: Record<string, unknown>; npuCoreMask?: InferenceTask['npuCoreMask']; npuCorePolicy?: InferenceTask['npuCorePolicy'] }) => request<InferenceTask>('/inference-tasks', { method: 'POST', body: JSON.stringify(payload) }),
  updateInferenceTask: (id: string, payload: { name: string; releaseId: string; nodeId?: string; groupId?: string; inputUri: string; interval?: number; thresholds?: Record<string, number>; output?: Record<string, unknown>; media?: Record<string, unknown>; analytics?: Record<string, unknown>; npuCoreMask?: InferenceTask['npuCoreMask']; npuCorePolicy?: InferenceTask['npuCorePolicy'] }) => request<InferenceTask>(`/inference-tasks/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(payload) }),
  stopInferenceTask: (id: string) => request<InferenceTask>(`/inference-tasks/${encodeURIComponent(id)}/stop`, { method: 'POST' }),
  restartInferenceTask: (id: string) => request<InferenceTask>(`/inference-tasks/${encodeURIComponent(id)}/restart`, { method: 'POST' }),
  inferencePlaybackSession: (id: string) => request<InferencePlaybackSession>(
    `/inference-tasks/${encodeURIComponent(id)}/playback-session`,
    { method: 'POST' },
  ),
  mediaGateways: () => request<MediaGateway[]>('/media-gateways'),
  createMediaGateway: (payload: MediaGatewayInput) => request<MediaGateway>('/media-gateways', { method: 'POST', body: JSON.stringify(payload) }),
  updateMediaGateway: (id: string, payload: MediaGatewayInput) => request<MediaGateway>(`/media-gateways/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(payload) }),
  probeMediaGateway: (id: string) => request<MediaGateway>(`/media-gateways/${encodeURIComponent(id)}/probe`, { method: 'POST' }),
  deleteMediaGateway: (id: string) => requestEmpty(`/media-gateways/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  retireInferenceTask: (id: string) => request<InferenceTask>(`/inference-tasks/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  deployments: (page = 1, pageSize = 50) => request<{ items: Deployment[]; page: number; pageSize: number; total: number }>(`/deployments?page=${page}&pageSize=${pageSize}`),
  deployment: (id: string) => request<Deployment>(`/deployments/${encodeURIComponent(id)}`),
  deploymentEvents: (id: string, afterId = 0) => request<Array<{ id: number; type: string; level: string; message: string; createdAt: string; data: Record<string, unknown> }>>(`/deployments/${encodeURIComponent(id)}/events?afterId=${afterId}&limit=500`),
  createDeployment: (payload: { name: string; releaseId: string; taskIds: string[]; strategy: 'canary' | 'rolling' | 'all_at_once'; batchSize?: number }) => request<Deployment>('/deployments', { method: 'POST', body: JSON.stringify(payload) }),
  retryDeployment: (id: string) => request<Deployment>(`/deployments/${encodeURIComponent(id)}/retry`, { method: 'POST' }),
  rollbackDeployment: (id: string) => request<Deployment>(`/deployments/${encodeURIComponent(id)}/rollback`, { method: 'POST' }),
  deleteDeployment: (id: string) => requestEmpty(`/deployments/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  uploadDataset(metadata: { name: string; description: string; version: string; taskType: TaskType; datasetFormat: DatasetFormat }, file: File, onProgress: (value: number) => void) {
    return new Promise<Dataset>((resolve, reject) => {
      const body = new FormData()
      body.set('metadata', JSON.stringify(metadata))
      body.set('file', file)
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${apiBaseUrl}/datasets`)
      const token = adminToken()
      if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100))
      }
      xhr.onerror = () => reject(new ApiError(0, 'connection_error', '无法连接平台 API'))
      xhr.onload = () => {
        const payload = JSON.parse(xhr.responseText || '{}') as Dataset & { error?: { code?: string; message?: string; details?: Record<string, unknown> } }
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload)
        else reject(new ApiError(xhr.status, payload.error?.code ?? 'upload_failed', payload.error?.message ?? '上传失败', payload.error?.details))
      }
      xhr.send(body)
    })
  },
  async downloadArtifact(artifact: Artifact) {
    const token = adminToken()
    const response = await fetch(`${apiBaseUrl}/artifacts/${encodeURIComponent(artifact.id)}/download`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!response.ok) throw new ApiError(response.status, 'download_failed', '产物下载失败')
    const url = URL.createObjectURL(await response.blob())
    const link = document.createElement('a')
    link.href = url
    link.download = artifact.filename
    link.click()
    URL.revokeObjectURL(url)
  },
}
