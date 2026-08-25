export type RouteKey = 'overview' | 'datasets' | 'training' | 'conversion' | 'nodes' | 'inference' | 'monitoring' | 'settings'
export type StatusTone = 'success' | 'warning' | 'danger' | 'neutral' | 'info'
export type TaskType = 'object_detection' | 'semantic_segmentation' | 'ocr_detection' | 'ocr_recognition'
export type DatasetFormat = 'auto' | 'yolo' | 'coco_detection' | 'voc_detection' | 'mask_pairs' | 'coco_segmentation' | 'voc_segmentation' | 'ppocr_detection' | 'ppocr_recognition'
export type Precision = 'int8' | 'fp16'
export type JobStatus = 'queued' | 'claimed' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type ModelReleaseStatus = 'qualified' | 'published' | 'deprecated' | 'revoked'
export type InferenceNodeLifecycle = 'pending_registration' | 'awaiting_approval' | 'active' | 'maintenance' | 'retired'
export type InferenceNodeConnectivity = 'online' | 'offline'
export type InferenceNodeHealth = 'unknown' | 'validating' | 'healthy' | 'degraded' | 'unhealthy'
export type InferenceTaskStatus = 'draft' | 'stopped' | 'deploying' | 'running' | 'degraded' | 'failed' | 'retired'
export type DeploymentStatus = 'queued' | 'rolling' | 'succeeded' | 'paused' | 'failed' | 'rolling_back' | 'rolled_back' | 'cancelled'
export type DeploymentTargetState = 'pending' | 'downloading' | 'verifying' | 'staged' | 'draining' | 'activating' | 'warming' | 'healthy' | 'failed' | 'rolled_back'

export interface Resolution {
  width: number
  height: number
}

export interface ResolutionRule {
  minWidth: number
  maxWidth: number
  minHeight: number
  maxHeight: number
  widthMultiple: number
  heightMultiple: number
}

export interface ModelProfile {
  id: string
  family: string
  label: string
  taskType: TaskType
  framework: string
  variants: string[]
  precisions: Precision[]
  defaultResolution: Resolution
  resolutionRule: ResolutionRule
  input: {
    name: string
    layout: 'NCHW' | 'NHWC'
    channels: number
    dtype: string
    colorSpace: string
    resizePolicy: string
  }
  preprocessing: { mean: number[]; std: number[] }
  outputContract: string
}

export interface Dataset {
  id: string
  name: string
  description: string
  version: string
  taskType: TaskType
  datasetFormat: DatasetFormat
  classes: string[]
  status: 'uploaded' | 'validating' | 'ready' | 'failed'
  filename: string
  sizeBytes: number
  sha256: string
  createdAt: string
  updatedAt: string
  errorMessage: string | null
}

export interface Job {
  id: string
  type: 'training' | 'conversion'
  name: string
  status: JobStatus
  profileId: string
  datasetId: string | null
  workerId: string | null
  progress: number
  stage: string
  spec: Record<string, unknown>
  result: Record<string, unknown> | null
  retryCount: number
  maxRetries: number
  errorCode: string | null
  errorMessage: string | null
  createdAt: string
  updatedAt: string
  startedAt: string | null
  completedAt: string | null
}

export interface JobEvent {
  id: number
  type: string
  level: 'debug' | 'info' | 'warning' | 'error' | string
  message: string
  data: Record<string, unknown>
  createdAt: string
}

export interface WorkerNode {
  id: string
  name: string
  kind: 'trainer' | 'converter'
  status: 'online' | 'busy' | 'offline'
  capabilities: string[]
  accelerator: 'cpu' | 'cuda' | 'rk3588'
  maxConcurrency: number
  activeJobs: number
  version: string
  metadata: Record<string, unknown>
  lastSeenAt: string
  createdAt: string
}

export interface Artifact {
  id: string
  jobId: string | null
  kind: string
  filename: string
  mediaType: string
  sizeBytes: number
  sha256: string
  manifest: Record<string, unknown> | null
  createdAt: string
}

export type ServiceEndpointEnrollmentStatus = 'pending' | 'claimed' | 'enrolled'

export interface ServiceEndpoint {
  id: string
  name: string
  kind: 'trainer' | 'converter' | 'inference'
  mode: 'pull' | 'direct'
  endpoint: string
  scheme: 'http' | 'https'
  host: string
  port: number
  accelerator: 'cpu' | 'cuda' | 'rk3588'
  capabilities: string[]
  enabled: boolean
  tokenConfigured: boolean
  enrollmentStatus: ServiceEndpointEnrollmentStatus
  enrollmentExpiresAt: string | null
  enrollmentClaimedAt: string | null
  enrolledAt: string | null
  probeStatus: 'unprobed' | 'online' | 'offline' | 'error' | string
  lastProbeAt: string | null
  lastError: string | null
  remoteMetadata: Record<string, unknown>
  inferenceNodeId: string | null
  createdAt: string
  updatedAt: string
}

export interface ServiceEndpointCreated extends ServiceEndpoint {
  enrollmentToken: string | null
}

export interface ServiceEndpointEnrollment {
  endpointId: string
  enrollmentStatus: ServiceEndpointEnrollmentStatus
  enrollmentToken: string
  enrollmentExpiresAt: string
}

export interface ServiceEndpointInput {
  name: string
  kind: 'trainer' | 'converter' | 'inference'
  mode: 'pull' | 'direct'
  endpoint?: string
  scheme: 'http' | 'https'
  host: string
  port: number
  accelerator: 'cpu' | 'cuda' | 'rk3588'
  capabilities: string[]
  enabled: boolean
  token?: string
}

export interface ServiceEndpointTestResult {
  ok: boolean
  endpoint: string
  message: string
  remote: Record<string, unknown>
}

export interface TrainingJobInput {
  name: string
  datasetId: string
  profileId: string
  variant: string
  resolution: Resolution
  hyperparameters: {
    epochs: number
    batchSize: number
    optimizer: 'auto' | 'AdamW' | 'SGD'
    learningRate?: number
    pretrained: boolean
    seed: number
  }
  accelerator: 'cpu' | 'cuda'
}

export interface ConversionJobInput {
  name: string
  sourceArtifactId: string
  precision: Precision
  calibrationDatasetId?: string
}

export interface ModelRelease {
  id: string
  name: string
  version: string
  description: string
  status: ModelReleaseStatus
  profileId: string
  variant: string
  taskType: TaskType
  precision: Precision
  adapter: string
  rknnArtifactId: string
  validationArtifactId: string | null
  sourceTrainingJobId: string | null
  sourceConversionJobId: string
  datasetId: string | null
  manifest: Record<string, unknown>
  createdAt: string
  publishedAt: string | null
}

export interface NodeGroup {
  id: string
  name: string
  description: string
  labels: string[]
  createdAt: string
  updatedAt: string
}

export interface InferenceNode {
  id: string
  name: string
  groupId: string | null
  labels: string[]
  lifecycle: InferenceNodeLifecycle
  connectivity: InferenceNodeConnectivity
  health: InferenceNodeHealth
  deploymentStatus: string
  maxModelInstances: number
  hardwareId: string | null
  runtimeVersion: string | null
  driverVersion: string | null
  pipelineVersion: string | null
  adapters: string[]
  metadata: Record<string, unknown>
  desiredRevision: number
  actualRevision: number
  selfTestPassed: boolean
  lastSeenAt: string | null
  createdAt: string
  updatedAt: string
}

export interface InferenceNodeCreated extends InferenceNode {
  registrationToken: string
  registrationExpiresAt: string
}

export interface InferenceSummary {
  onlineNodes: number
  totalNodes: number
  publishedReleases: number
  runningTasks: number
  activeDeployments: number
}

export interface InferenceGraphNode {
  id: string
  operator: string
  config: Record<string, unknown>
}

export interface InferenceGraphEdge {
  source: string
  sourcePort: string
  target: string
  targetPort: string
}

export interface InferenceGraph {
  schemaVersion: number
  catalogVersion: string
  nodes: InferenceGraphNode[]
  edges: InferenceGraphEdge[]
}

export interface InferenceGraphLayout {
  positions: Record<string, { x: number; y: number }>
}

export interface InferenceOperator {
  operatorId: string
  runtimeNode: string
  category: 'capture' | 'inference' | 'processing' | 'output'
  title: string
  description: string
  inputPorts: string[]
  outputPorts: string[]
  minInstances: number
  maxInstances: number
  defaults: Record<string, unknown>
  dependencies: string[]
  supportedAdapters: string[]
  requiredFeatures: string[]
  configurableFields: string[]
  readOnlyFields: string[]
}

export interface InferenceOperatorCatalog {
  schemaVersion: number
  catalogVersion: string
  operators: InferenceOperator[]
}

export interface InferenceGraphValidationIssue {
  code: string
  message: string
  path: string
  severity: string
  details: Record<string, unknown>
}

export interface InferenceGraphValidation {
  valid: boolean
  normalizedGraph: InferenceGraph | null
  graphHash: string | null
  issues: InferenceGraphValidationIssue[]
  releaseIds: string[]
  requiredFeatures: string[]
  requiredAdapters: string[]
  requiredContexts: number
  compatibleNodeIds: string[]
}

export interface InferenceTask {
  id: string
  name: string
  status: InferenceTaskStatus
  nodeId: string
  groupId: string | null
  inputUri: string
  graph: InferenceGraph
  layout: InferenceGraphLayout
  graphRevisionId: string
  graphHash: string
  npuCoreMask: NpuCoreMask
  npuCorePolicy: NpuCorePolicy
  previewCapability: PreviewCapability
  configRevision: number
  errorMessage: string | null
  createdAt: string
  updatedAt: string
}

export type PreviewCapabilityState = 'available' | 'unsupported' | 'migration_required' | 'gateway_offline'

export interface PreviewCapability {
  state: PreviewCapabilityState
  reason: string | null
}

export interface InferencePlaybackSession {
  streamUrl: string
  expiresAt: string
  taskId: string
  revision: number
  gatewayId: string
  app: string
  streamName: string
  codec: 'h264' | 'h265' | 'unknown'
  reconnectMs: number
}

export type MediaGatewayStatus = 'disabled' | 'probing' | 'online' | 'error'

export interface MediaGateway {
  id: string
  name: string
  builtin: boolean
  enabled: boolean
  publishHost: string
  rtspPort: number
  playbackHost: string
  wsPort: number
  apiHost: string
  apiPort: number
  app: string
  status: MediaGatewayStatus
  apiSecretConfigured: boolean
  hookIdentityConfigured: boolean
  lastProbeAt: string | null
  lastHookAt: string | null
  lastError: string | null
  createdAt: string | null
  updatedAt: string | null
}

export interface MediaGatewayInput {
  name: string
  enabled: boolean
  publishHost: string
  rtspPort: number
  playbackHost: string
  wsPort: number
  apiHost: string
  apiPort: number
  app: string
  apiSecret?: string
  hookIdentity?: string
}

export type NpuCoreMask = 'auto' | 'core0' | 'core1' | 'core2' | 'core0_1' | 'core0_1_2'
export type NpuCorePolicy = 'shared' | 'exclusive'

export interface DeploymentTarget {
  id: string
  deploymentId: string
  nodeId: string
  taskId: string
  graphRevisionId: string
  graphHash: string
  sequence: number
  desiredRevision: number
  state: DeploymentTargetState
  progress: number
  stage: string
  errorCode: string | null
  errorMessage: string | null
  startedAt: string | null
  completedAt: string | null
  updatedAt: string
}

export interface Deployment {
  id: string
  name: string
  status: DeploymentStatus
  strategy: string
  batchSize: number
  targets: DeploymentTarget[]
  createdAt: string
  updatedAt: string
  completedAt: string | null
}
