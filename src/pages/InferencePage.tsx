import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Check, CloudDownload, Container, Eye, GitBranch, HardDrive, Layers3, MonitorPlay, Pencil, Play, RadioTower, RefreshCw, RotateCcw, ServerCog, ShieldCheck, StopCircle, Trash2, UploadCloud } from 'lucide-react'
import { usePlatform } from '../api/PlatformContext'
import { api } from '../api/client'
import { loadAllPages } from '../api/pagination'
import { formatTime } from '../api/presentation'
import { AddButton, Button, ConfirmDialog, EmptyState, Modal, PageHeader, ProgressBar, StatusBadge, TablePagination } from '../components'
import { InferenceStreamPlayer } from '../components/InferenceStreamPlayer'
import {
  InferenceBusinessFields,
  emptyInferenceAnalytics,
  inferenceAnalyticsError,
  inferenceAnalyticsSummary,
  normalizeInferenceAnalytics,
  type InferenceAnalyticsConfig,
} from '../components/InferenceBusinessFields'
import { buildInferenceTaskMedia } from './inferenceTaskPayload'
import type { Deployment, InferenceNode, InferenceTask, MediaGateway, ModelRelease, NodeGroup } from '../types'

type Tab = 'nodes' | 'releases' | 'tasks' | 'deployments'
type Dialog = 'group' | 'release' | 'task' | 'deployment' | null
interface PaginationState { total: number; page: number; pageSize: number; onPageChange: (page: number) => void; onPageSizeChange: (pageSize: number) => void }
interface InferenceSummary { onlineNodes: number; totalNodes: number; publishedReleases: number; runningTasks: number; activeDeployments: number }

const INFERENCE_STATUS_REFRESH_MS = 3_000
const MEDIA_STREAM_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/

function isHttpUrl(value: string): boolean {
  try {
    const url = new URL(value)
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password
  } catch {
    return false
  }
}

const lifecycleLabels: Record<InferenceNode['lifecycle'], string> = {
  pending_registration: '待注册', awaiting_approval: '待审批', active: '已启用', maintenance: '维护中', retired: '已退役',
}
const releaseLabels: Record<ModelRelease['status'], string> = { qualified: '待发布', published: '已发布', deprecated: '已弃用', revoked: '已撤销' }
const taskLabels: Record<InferenceTask['status'], string> = { draft: '草稿', stopped: '已停止', deploying: '部署中', running: '运行中', degraded: '降级', failed: '失败', retired: '已退役' }
const npuCoreLabels: Record<InferenceTask['npuCoreMask'], string> = { auto: '自动', core0: '核心 0', core1: '核心 1', core2: '核心 2', core0_1: '核心 0 + 1', core0_1_2: '全部核心' }
const deploymentLabels: Record<Deployment['status'], string> = { queued: '排队中', rolling: '滚动中', succeeded: '已完成', paused: '已暂停', failed: '失败', rolling_back: '回滚中', rolled_back: '已回滚', cancelled: '已取消' }
const mediaFeatureLabels: Record<string, string> = {
  rkmpp_decode: 'MPP 硬解码',
  bytetrack: 'ByteTrack',
  kafka: 'Kafka',
  zlm_sei: 'ZLM SEI',
  analytics_area: '区域分析',
  analytics_line: '越线分析',
  event_snapshot: '事件抓拍',
  event_record: '事件录像',
  secondary_infer: '二级推理',
}

function nodeMediaFeatures(node: InferenceNode): string[] {
  const value = node.metadata.features
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
}

function tone(status: string): 'success' | 'warning' | 'danger' | 'neutral' | 'info' {
  if (['active', 'online', 'healthy', 'published', 'running', 'succeeded'].includes(status)) return 'success'
  if (['awaiting_approval', 'pending_registration', 'deploying', 'rolling', 'queued', 'qualified', 'degraded', 'paused', 'maintenance'].includes(status)) return 'warning'
  if (['failed', 'unhealthy', 'retired', 'revoked'].includes(status)) return 'danger'
  return 'neutral'
}

export function InferencePage({ notify }: { notify: (message: string) => void }) {
  const { jobs, loading: platformLoading, error: platformError, refresh } = usePlatform()
  const [tab, setTab] = useState<Tab>('nodes')
  const [dialog, setDialog] = useState<Dialog>(null)
  const [saving, setSaving] = useState(false)
  const [taskActionId, setTaskActionId] = useState('')
  const [editingGroupId, setEditingGroupId] = useState('')
  const [groupName, setGroupName] = useState('生产板卡组')
  const [groupDescription, setGroupDescription] = useState('')
  const [groupLabels, setGroupLabels] = useState('production')
  const [releaseName, setReleaseName] = useState('DeepLabV3+ 生产版本')
  const [releaseVersion, setReleaseVersion] = useState('v1.0.0')
  const [releaseJobId, setReleaseJobId] = useState('')
  const [taskName, setTaskName] = useState('产线摄像头分割')
  const [taskReleaseId, setTaskReleaseId] = useState('')
  const [taskNodeId, setTaskNodeId] = useState('')
  const [taskInput, setTaskInput] = useState('rtsp://camera/line-a')
  const [editingTaskId, setEditingTaskId] = useState('')
  const [taskOutputType, setTaskOutputType] = useState<'jsonl' | 'http'>('jsonl')
  const [taskOutputUrl, setTaskOutputUrl] = useState('https://inference-consumer.local/api/results')
  const [taskOutputAuthEnv, setTaskOutputAuthEnv] = useState('RKNODE_RESULT_SINK_TOKEN')
  const [taskConnectTimeout, setTaskConnectTimeout] = useState(1000)
  const [taskRequestTimeout, setTaskRequestTimeout] = useState(3000)
  const [taskNpuCoreMask, setTaskNpuCoreMask] = useState<InferenceTask['npuCoreMask']>('auto')
  const [taskNpuCorePolicy, setTaskNpuCorePolicy] = useState<InferenceTask['npuCorePolicy']>('shared')
  const [taskDecoder, setTaskDecoder] = useState<'opencv' | 'rkmpp'>('opencv')
  const [taskTracking, setTaskTracking] = useState(false)
  const [taskTrackBuffer, setTaskTrackBuffer] = useState(30)
  const [taskKafka, setTaskKafka] = useState(false)
  const [taskKafkaBrokers, setTaskKafkaBrokers] = useState('')
  const [taskKafkaTopic, setTaskKafkaTopic] = useState('sei_msg')
  const [taskKafkaKey, setTaskKafkaKey] = useState('')
  const [taskZlmSei, setTaskZlmSei] = useState(false)
  const [taskZlmGatewayId, setTaskZlmGatewayId] = useState('')
  const [taskZlmStreamName, setTaskZlmStreamName] = useState('')
  const [taskAnalytics, setTaskAnalytics] = useState<InferenceAnalyticsConfig>(emptyInferenceAnalytics)
  const [deploymentName, setDeploymentName] = useState('生产线灰度发布')
  const [deploymentReleaseId, setDeploymentReleaseId] = useState('')
  const [deploymentStrategy, setDeploymentStrategy] = useState<'canary' | 'rolling' | 'all_at_once'>('canary')
  const [selectedTaskIds, setSelectedTaskIds] = useState<string[]>([])
  const [detailDeployment, setDetailDeployment] = useState<Deployment | null>(null)
  const [previewTask, setPreviewTask] = useState<InferenceTask | null>(null)
  const [deleteNodeTarget, setDeleteNodeTarget] = useState<InferenceNode | null>(null)
  const [deleteReleaseTarget, setDeleteReleaseTarget] = useState<ModelRelease | null>(null)
  const [deleteDeploymentTarget, setDeleteDeploymentTarget] = useState<Deployment | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [modelReleases, setModelReleases] = useState<ModelRelease[]>([])
  const [mediaGateways, setMediaGateways] = useState<MediaGateway[]>([])
  const [nodeGroups, setNodeGroups] = useState<NodeGroup[]>([])
  const [inferenceNodes, setInferenceNodes] = useState<InferenceNode[]>([])
  const [inferenceTasks, setInferenceTasks] = useState<InferenceTask[]>([])
  const [pageItems, setPageItems] = useState<{ nodes: InferenceNode[]; releases: ModelRelease[]; tasks: InferenceTask[]; deployments: Deployment[] }>({ nodes: [], releases: [], tasks: [], deployments: [] })
  const [total, setTotal] = useState(0)
  const [summary, setSummary] = useState<InferenceSummary>({ onlineNodes: 0, totalNodes: 0, publishedReleases: 0, runningTasks: 0, activeDeployments: 0 })
  const [inferenceLoading, setInferenceLoading] = useState(true)
  const [lookupError, setLookupError] = useState('')
  const [pageError, setPageError] = useState('')
  const [summaryError, setSummaryError] = useState('')
  const pageRequestId = useRef(0)

  const loadLookups = useCallback(async () => {
    try {
      const [groups, releases, nodes, tasks, gateways] = await Promise.all([
        api.nodeGroups(),
        loadAllPages(api.modelReleases),
        loadAllPages(api.inferenceNodes),
        loadAllPages(api.inferenceTasks),
        api.mediaGateways(),
      ])
      setNodeGroups(groups)
      setModelReleases(releases)
      setInferenceNodes(nodes)
      setInferenceTasks(tasks)
      setMediaGateways(gateways)
      setLookupError('')
    } catch (reason) {
      setLookupError(reason instanceof Error ? reason.message : '推理资源索引加载失败')
    }
  }, [])

  const loadSummary = useCallback(async () => {
    try {
      setSummary(await api.inferenceSummary())
      setSummaryError('')
    } catch (reason) {
      setSummaryError(reason instanceof Error ? reason.message : '推理资源概况加载失败')
    }
  }, [])

  const loadCurrentPage = useCallback(async (background = false) => {
    const requestId = pageRequestId.current + 1
    pageRequestId.current = requestId
    if (!background) setInferenceLoading(true)
    try {
      const result = tab === 'nodes'
        ? await api.inferenceNodes(page, pageSize)
        : tab === 'releases'
          ? await api.modelReleases(page, pageSize)
          : tab === 'tasks'
            ? await api.inferenceTasks(page, pageSize)
            : await api.deployments(page, pageSize)
      if (requestId !== pageRequestId.current) return
      setPageItems((current) => ({ ...current, [tab]: result.items }))
      setTotal(result.total)
      setPageError('')
    } catch (reason) {
      if (requestId === pageRequestId.current) setPageError(reason instanceof Error ? reason.message : '推理列表加载失败')
    } finally {
      if (requestId === pageRequestId.current) setInferenceLoading(false)
    }
  }, [page, pageSize, tab])

  const reloadInference = useCallback(async () => {
    await Promise.all([loadLookups(), loadSummary(), loadCurrentPage()])
  }, [loadCurrentPage, loadLookups, loadSummary])

  const succeededConversions = useMemo(() => jobs.filter((job) => job.type === 'conversion' && job.status === 'succeeded'), [jobs])
  const publishedReleases = modelReleases.filter((item) => item.status === 'published')
  const activeNodes = inferenceNodes.filter((item) => item.lifecycle === 'active' && item.health === 'healthy')
  const onlineMediaGateways = mediaGateways.filter((item) => item.enabled && item.status === 'online')
  const selectedMediaGateway = mediaGateways.find((item) => item.id === taskZlmGatewayId)
  const deploymentTasks = inferenceTasks.filter((item) => item.status !== 'retired' && item.releaseId === deploymentReleaseId)
  const selectedTaskRelease = modelReleases.find((item) => item.id === taskReleaseId)
  const trackingSupported = selectedTaskRelease?.adapter.startsWith('yolo_') ?? false
  const taskOutputInvalid = taskOutputType === 'http' && (
    !isHttpUrl(taskOutputUrl.trim())
    || (taskOutputAuthEnv.trim() !== '' && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(taskOutputAuthEnv.trim()))
    || taskConnectTimeout < 100
    || taskRequestTimeout < taskConnectTimeout
    || taskRequestTimeout > 60000
  )
  const taskMediaError = taskDecoder === 'rkmpp' && !taskInput.trim().startsWith('rtsp://')
    ? 'RKMPP 和 RTSP + SEI 仅支持 rtsp:// 输入源。'
    : taskZlmSei && taskDecoder !== 'rkmpp'
      ? 'RTSP + SEI 必须使用 RKMPP 硬解码。'
    : taskTracking && (!trackingSupported || taskTrackBuffer < 1 || taskTrackBuffer > 10000)
      ? '当前跟踪配置与模型或缓冲区范围不兼容。'
      : taskKafka && (!taskKafkaBrokers.trim() || !taskKafkaTopic.trim())
        ? '启用 Kafka 时必须填写 Broker 和 Topic。'
        : taskZlmSei && (!selectedMediaGateway || !selectedMediaGateway.enabled || selectedMediaGateway.status !== 'online')
          ? 'RTSP + SEI 需要选择一个在线媒体网关。'
          : taskZlmSei && !MEDIA_STREAM_PATTERN.test(taskZlmStreamName.trim())
            ? '流名称需为 1 到 64 位字母、数字、点、下划线或连字符，且以字母或数字开头。'
            : ''
  const taskMediaInvalid = Boolean(taskMediaError)
  const taskAnalyticsError = inferenceAnalyticsError(
    taskAnalytics,
    trackingSupported,
    taskTracking,
    taskDecoder,
  )
  const visibleItems = pageItems[tab]
  const pagination: PaginationState = { total, page, pageSize, onPageChange: setPage, onPageSizeChange: setPageSize }
  const loading = platformLoading || inferenceLoading
  const error = pageError || lookupError || summaryError || platformError

  useEffect(() => setPage(1), [tab, pageSize])
  useEffect(() => { if (!trackingSupported) setTaskTracking(false) }, [trackingSupported])
  useEffect(() => {
    const lastPage = Math.max(1, Math.ceil(total / pageSize))
    if (page > lastPage) setPage(lastPage)
  }, [page, pageSize, total])
  useEffect(() => { void loadLookups() }, [loadLookups])
  useEffect(() => {
    void loadCurrentPage()
    const refreshVisiblePage = () => {
      if (document.visibilityState === 'visible') void loadCurrentPage(true)
    }
    const timer = window.setInterval(refreshVisiblePage, INFERENCE_STATUS_REFRESH_MS)
    document.addEventListener('visibilitychange', refreshVisiblePage)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', refreshVisiblePage)
    }
  }, [loadCurrentPage])
  useEffect(() => {
    void loadSummary()
    const timer = window.setInterval(() => { if (document.visibilityState === 'visible') void loadSummary() }, 10_000)
    return () => window.clearInterval(timer)
  }, [loadSummary])

  const close = () => { if (!saving) setDialog(null) }
  const openRelease = () => {
    setReleaseJobId(succeededConversions[0]?.id ?? '')
    setDialog('release')
  }
  const openTask = () => {
    setEditingTaskId('')
    setTaskReleaseId(publishedReleases[0]?.id ?? '')
    setTaskNodeId(activeNodes[0]?.id ?? '')
    setTaskOutputType('jsonl')
    setTaskOutputUrl('https://inference-consumer.local/api/results')
    setTaskOutputAuthEnv('RKNODE_RESULT_SINK_TOKEN')
    setTaskConnectTimeout(1000)
    setTaskRequestTimeout(3000)
    setTaskNpuCoreMask('auto')
    setTaskNpuCorePolicy('shared')
    setTaskDecoder('opencv')
    setTaskTracking(false)
    setTaskTrackBuffer(30)
    setTaskKafka(false)
    setTaskKafkaBrokers('')
    setTaskKafkaTopic('sei_msg')
    setTaskKafkaKey('')
    setTaskZlmSei(false)
    setTaskZlmGatewayId(onlineMediaGateways[0]?.id ?? '')
    setTaskZlmStreamName('')
    setTaskAnalytics(emptyInferenceAnalytics())
    setDialog('task')
  }

  const editTask = (task: InferenceTask) => {
    setEditingTaskId(task.id)
    setTaskName(task.name)
    setTaskReleaseId(task.releaseId)
    setTaskNodeId(task.nodeId)
    setTaskInput(task.inputUri)
    const output = task.output ?? { type: 'jsonl' }
    setTaskOutputType(output.type === 'http' ? 'http' : 'jsonl')
    setTaskOutputUrl(typeof output.url === 'string' ? output.url : 'https://inference-consumer.local/api/results')
    setTaskOutputAuthEnv(typeof output.authorizationEnv === 'string' ? output.authorizationEnv : 'RKNODE_RESULT_SINK_TOKEN')
    setTaskConnectTimeout(typeof output.connectTimeoutMs === 'number' ? output.connectTimeoutMs : 1000)
    setTaskRequestTimeout(typeof output.requestTimeoutMs === 'number' ? output.requestTimeoutMs : 3000)
    setTaskNpuCoreMask(task.npuCoreMask ?? 'auto')
    setTaskNpuCorePolicy(task.npuCorePolicy ?? 'shared')
    const media = task.media ?? {}
    const tracking = typeof media.tracking === 'object' && media.tracking !== null ? media.tracking as Record<string, unknown> : {}
    const kafka = typeof media.kafka === 'object' && media.kafka !== null ? media.kafka as Record<string, unknown> : {}
    const zlmSei = typeof media.zlmSei === 'object' && media.zlmSei !== null ? media.zlmSei as Record<string, unknown> : {}
    setTaskDecoder(media.decoder === 'rkmpp' ? 'rkmpp' : 'opencv')
    setTaskTracking(tracking.enabled === true)
    setTaskTrackBuffer(typeof tracking.trackBuffer === 'number' ? tracking.trackBuffer : 30)
    setTaskKafka(kafka.enabled === true)
    setTaskKafkaBrokers(typeof kafka.brokers === 'string' ? kafka.brokers : '')
    setTaskKafkaTopic(typeof kafka.topic === 'string' ? kafka.topic : 'sei_msg')
    setTaskKafkaKey(typeof kafka.key === 'string' ? kafka.key : '')
    setTaskZlmSei(zlmSei.enabled === true)
    setTaskZlmGatewayId(typeof zlmSei.gatewayId === 'string' ? zlmSei.gatewayId : '')
    setTaskZlmStreamName(typeof zlmSei.streamName === 'string' ? zlmSei.streamName : '')
    setTaskAnalytics(normalizeInferenceAnalytics(task.analytics))
    setDialog('task')
  }
  const openDeployment = () => {
    const releaseId = publishedReleases[0]?.id ?? ''
    setDeploymentReleaseId(releaseId)
    setSelectedTaskIds(inferenceTasks.filter((item) => item.releaseId === releaseId && item.status === 'stopped').map((item) => item.id))
    setDialog('deployment')
  }

  const save = async () => {
    setSaving(true)
    try {
      if (dialog === 'group') {
        const payload = { name: groupName.trim(), description: groupDescription.trim(), labels: groupLabels.split(',').map((item) => item.trim()).filter(Boolean) }
        if (editingGroupId) await api.updateNodeGroup(editingGroupId, payload)
        else await api.createNodeGroup(payload)
        await reloadInference()
        setEditingGroupId('')
        setGroupName('生产板卡组')
        setGroupDescription('')
        setGroupLabels('production')
        notify(editingGroupId ? '节点组已更新' : '节点组已创建')
        return
      }
      if (dialog === 'release') {
        await api.createModelRelease({ name: releaseName.trim(), version: releaseVersion.trim(), conversionJobId: releaseJobId })
        await reloadInference(); setDialog(null); notify('模型版本已登记，发布前仍需确认验证报告')
        return
      }
      if (dialog === 'task') {
        const output = taskOutputType === 'http'
          ? { type: 'http', url: taskOutputUrl.trim(), authorizationEnv: taskOutputAuthEnv.trim(), connectTimeoutMs: taskConnectTimeout, requestTimeoutMs: taskRequestTimeout }
          : { type: 'jsonl' }
        const payload = {
          name: taskName.trim(),
          releaseId: taskReleaseId,
          nodeId: taskNodeId,
          inputUri: taskInput.trim(),
          output,
          media: buildInferenceTaskMedia({
            decoder: taskDecoder,
            trackingEnabled: taskTracking,
            trackBuffer: taskTrackBuffer,
            kafkaEnabled: taskKafka,
            kafkaBrokers: taskKafkaBrokers,
            kafkaTopic: taskKafkaTopic,
            kafkaKey: taskKafkaKey,
            zlmSeiEnabled: taskZlmSei,
            zlmGatewayId: taskZlmGatewayId,
            zlmStreamName: taskZlmStreamName,
          }),
          analytics: taskAnalytics,
          npuCoreMask: taskNpuCoreMask,
          npuCorePolicy: taskNpuCorePolicy,
        }
        if (editingTaskId) {
          await api.updateInferenceTask(editingTaskId, payload)
          await reloadInference()
          setDialog(null)
          notify('推理任务已更新，请重新创建部署批次使配置生效')
        } else {
          const created = await api.createInferenceTask(payload)
          try {
            const started = await api.restartInferenceTask(created.id)
            await reloadInference()
            setDialog(null)
            notify(`推理任务「${created.name}」正在启动，配置修订 ${started.configRevision} 已下发`)
          } catch (reason) {
            await reloadInference()
            setDialog(null)
            const message = reason instanceof Error ? reason.message : '启动失败'
            notify(`推理任务已创建，但启动失败：${message}`)
          }
        }
        return
      }
      if (dialog === 'deployment') {
        await api.createDeployment({ name: deploymentName.trim(), releaseId: deploymentReleaseId, taskIds: selectedTaskIds, strategy: deploymentStrategy, batchSize: 1 })
        await reloadInference(); setDialog(null); notify('部署批次已创建，板端代理会按版本清单拉取并校验模型')
      }
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '操作失败')
    } finally {
      setSaving(false)
    }
  }

  const approve = async (node: InferenceNode) => {
    try { await api.approveInferenceNode(node.id); await reloadInference(); notify(`节点「${node.name}」已启用`) } catch (reason) { notify(reason instanceof Error ? reason.message : '节点审批失败') }
  }
  const publish = async (release: ModelRelease) => {
    try { await api.publishModelRelease(release.id); await reloadInference(); notify(`模型版本「${release.name} ${release.version}」已发布`) } catch (reason) { notify(reason instanceof Error ? reason.message : '模型发布失败') }
  }
  const stopTask = async (task: InferenceTask) => {
    if (taskActionId) return
    setTaskActionId(task.id)
    try { await api.stopInferenceTask(task.id); await reloadInference(); notify(`推理任务「${task.name}」已停止`) } catch (reason) { notify(reason instanceof Error ? reason.message : '停止任务失败') } finally { setTaskActionId('') }
  }
  const restartTask = async (task: InferenceTask) => {
    if (taskActionId) return
    setTaskActionId(task.id)
    try {
      const restarted = await api.restartInferenceTask(task.id)
      await reloadInference()
      notify(`推理任务「${task.name}」正在重启，配置修订 ${restarted.configRevision} 已下发`)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '重启任务失败')
    } finally {
      setTaskActionId('')
    }
  }
  const rollback = async (deployment: Deployment) => {
    if (!window.confirm(`确认回滚部署「${deployment.name}」？`)) return
    try { await api.rollbackDeployment(deployment.id); await reloadInference(); notify(`部署「${deployment.name}」已进入回滚`) } catch (reason) { notify(reason instanceof Error ? reason.message : '回滚失败') }
  }
  const retryDeployment = async (deployment: Deployment) => {
    try { await api.retryDeployment(deployment.id); await reloadInference(); notify(`部署「${deployment.name}」已重新进入队列`) } catch (reason) { notify(reason instanceof Error ? reason.message : '部署重试失败') }
  }
  const retireNode = async (node: InferenceNode) => {
    if (!window.confirm(`确认退役板卡「${node.name}」？退役后将不能接收新任务。`)) return
    try { await api.retireInferenceNode(node.id); await reloadInference(); notify(`节点「${node.name}」已退役`) } catch (reason) { notify(reason instanceof Error ? reason.message : '节点退役失败') }
  }
  const deleteRetiredNode = async () => {
    if (!deleteNodeTarget) return
    setDeleting(true)
    try {
      await api.deleteRetiredInferenceNode(deleteNodeTarget.id)
      await reloadInference()
      notify(`退役板卡「${deleteNodeTarget.name}」已永久删除`)
      setDeleteNodeTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '退役板卡删除失败')
    } finally {
      setDeleting(false)
    }
  }
  const editGroup = (group: NodeGroup) => {
    setEditingGroupId(group.id)
    setGroupName(group.name)
    setGroupDescription(group.description)
    setGroupLabels(group.labels.join(', '))
  }
  const deleteGroup = async (group: NodeGroup) => {
    if (!window.confirm(`确认删除节点组「${group.name}」？`)) return
    try { await api.deleteNodeGroup(group.id); await reloadInference(); notify(`节点组「${group.name}」已删除`) } catch (reason) { notify(reason instanceof Error ? reason.message : '节点组删除失败') }
  }
  const deprecateRelease = async (release: ModelRelease) => {
    if (!window.confirm(`确认弃用模型版本「${release.name} ${release.version}」？`)) return
    try { await api.deprecateModelRelease(release.id); await reloadInference(); notify(`模型版本「${release.name} ${release.version}」已弃用`) } catch (reason) { notify(reason instanceof Error ? reason.message : '模型弃用失败') }
  }
  const deleteDeprecatedRelease = async () => {
    if (!deleteReleaseTarget) return
    setDeleting(true)
    try {
      await api.deleteModelRelease(deleteReleaseTarget.id)
      await reloadInference()
      notify(`模型版本「${deleteReleaseTarget.name} ${deleteReleaseTarget.version}」已永久删除`)
      setDeleteReleaseTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '模型版本删除失败')
    } finally {
      setDeleting(false)
    }
  }
  const retireTask = async (task: InferenceTask) => {
    if (!window.confirm(`确认退役推理任务「${task.name}」？`)) return
    try { await api.retireInferenceTask(task.id); await reloadInference(); notify(`推理任务「${task.name}」已退役`) } catch (reason) { notify(reason instanceof Error ? reason.message : '任务退役失败') }
  }
  const deleteDeployment = async () => {
    if (!deleteDeploymentTarget) return
    setDeleting(true)
    try {
      await api.deleteDeployment(deleteDeploymentTarget.id)
      await reloadInference()
      notify(`部署批次「${deleteDeploymentTarget.name}」已删除`)
      setDeleteDeploymentTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '部署批次删除失败')
    } finally {
      setDeleting(false)
    }
  }
  const tabs: Array<[Tab, string]> = [['nodes', 'RK3588 板卡'], ['releases', '模型版本'], ['tasks', '推理任务'], ['deployments', '部署批次']]
  return <div className="page-stack">
    <PageHeader title="推理下发" description="平台向 RK3588 节点主动下发期望版本，并记录校验、切换、健康和回滚过程。" actions={<><Button variant="secondary" icon={<RefreshCw size={16} />} onClick={() => void Promise.all([refresh(), reloadInference()]).then(() => notify('推理资源已刷新'))}>刷新</Button>{tab === 'nodes' && <Button variant="secondary" icon={<Layers3 size={16} />} onClick={() => { setEditingGroupId(''); setDialog('group') }}>节点组</Button>}{tab === 'nodes' && <Button icon={<ServerCog size={16} />} onClick={() => { window.location.hash = '/settings' }}>节点管理</Button>}{tab === 'releases' && <AddButton onClick={openRelease}>登记模型版本</AddButton>}{tab === 'tasks' && <AddButton onClick={openTask}>新建推理任务</AddButton>}{tab === 'deployments' && <AddButton onClick={openDeployment}>创建部署批次</AddButton>}</>} />
    {error && <div className="api-banner danger">{error}<Button variant="quiet" onClick={() => void Promise.all([refresh(), reloadInference()])}>重试</Button></div>}
    <section className="inference-summary"><div><ServerCog size={18} /><span>板卡在线<strong>{summary.onlineNodes} / {summary.totalNodes}</strong></span></div><div><ShieldCheck size={18} /><span>已发布版本<strong>{summary.publishedReleases}</strong></span></div><div><GitBranch size={18} /><span>运行中任务<strong>{summary.runningTasks}</strong></span></div><div><UploadCloud size={18} /><span>进行中部署<strong>{summary.activeDeployments}</strong></span></div></section>
    <div className="toolbar tab-toolbar"><div className="tabs" role="tablist">{tabs.map(([value, label]) => <button key={value} className={tab === value ? 'active' : ''} onClick={() => setTab(value)} role="tab">{label}</button>)}</div></div>
    {loading && !visibleItems.length ? <EmptyState title="正在加载推理资源" message="正在读取当前页和节点概况。" /> : <>
      {tab === 'nodes' && <NodesTable nodes={visibleItems as InferenceNode[]} groups={nodeGroups} pagination={pagination} onApprove={approve} onRetire={retireNode} onDelete={setDeleteNodeTarget} />}
      {tab === 'releases' && <ReleasesTable releases={visibleItems as ModelRelease[]} jobs={jobs} pagination={pagination} onPublish={publish} onDeprecate={deprecateRelease} onDelete={setDeleteReleaseTarget} />}
      {tab === 'tasks' && <TasksTable tasks={visibleItems as InferenceTask[]} releases={modelReleases} nodes={inferenceNodes} pagination={pagination} busyTaskId={taskActionId} onPreview={setPreviewTask} onStop={stopTask} onRestart={restartTask} onEdit={editTask} onRetire={retireTask} />}
      {tab === 'deployments' && <DeploymentsTable deployments={visibleItems as Deployment[]} releases={modelReleases} nodes={inferenceNodes} pagination={pagination} onRetry={retryDeployment} onRollback={rollback} onDetail={setDetailDeployment} onDelete={setDeleteDeploymentTarget} />}
    </>}
    <Modal open={dialog === 'group'} title="管理节点组" description="节点组用于把同一模型按灰度或滚动策略下发到多块 RK3588。" onClose={close} footer={<><Button variant="secondary" onClick={close}>关闭</Button><Button onClick={() => void save()} disabled={saving || !groupName.trim()}>{saving ? '正在保存…' : editingGroupId ? '保存修改' : '新增节点组'}</Button></>}><div className="form-sections"><section className="form-section"><h3><Layers3 size={17} />节点组信息</h3><div className="form-grid two-columns"><label className="field"><span>组名称 <b>*</b></span><input value={groupName} onChange={(event) => setGroupName(event.target.value)} /></label><label className="field"><span>标签</span><input value={groupLabels} onChange={(event) => setGroupLabels(event.target.value)} placeholder="production, line-a" /><small>多个标签用英文逗号分隔。</small></label><label className="field full-span"><span>说明</span><input value={groupDescription} onChange={(event) => setGroupDescription(event.target.value)} /></label></div></section><section className="form-section"><h3><ServerCog size={17} />已有节点组</h3><div className="selection-list">{nodeGroups.length ? nodeGroups.map((group) => <div key={group.id} className="selection-row"><span><strong>{group.name}</strong><small>{group.description || group.labels.join(', ') || '未填写说明'}</small></span><div className="row-actions"><button className="icon-button ghost" title="编辑节点组" aria-label={`编辑节点组 ${group.name}`} onClick={() => editGroup(group)}><Pencil size={16} /></button><button className="icon-button ghost danger-action" title="删除节点组" aria-label={`删除节点组 ${group.name}`} onClick={() => void deleteGroup(group)}><Trash2 size={16} /></button></div></div>) : <EmptyState title="暂无节点组" message="填写上方信息后创建第一个节点组。" />}</div></section></div></Modal>
    <Modal open={dialog === 'release'} title="登记模型版本" description="只允许已完成并通过部署校验的转换任务进入模型资产库。" onClose={close} footer={<><Button variant="secondary" onClick={close}>取消</Button><Button onClick={() => void save()} disabled={saving || !releaseJobId}>{saving ? '正在登记…' : '登记版本'}</Button></>}><div className="form-sections"><section className="form-section"><h3><HardDrive size={17} />版本信息</h3><div className="form-grid two-columns"><label className="field"><span>版本名称 <b>*</b></span><input value={releaseName} onChange={(event) => setReleaseName(event.target.value)} /></label><label className="field"><span>版本号 <b>*</b></span><input value={releaseVersion} onChange={(event) => setReleaseVersion(event.target.value)} /></label><label className="field full-span"><span>来源转换任务 <b>*</b></span><select value={releaseJobId} onChange={(event) => setReleaseJobId(event.target.value)}><option value="">请选择已完成转换任务</option>{succeededConversions.map((job) => <option key={job.id} value={job.id}>{job.name} · {job.profileId}</option>)}</select></label></div></section></div></Modal>
    <Modal
      open={dialog === 'task'}
      title={editingTaskId ? '编辑推理任务' : '新建推理任务'}
      description="任务绑定模型版本、目标板卡、NPU 核心、媒体链路和业务结果出口；修改后需重新创建部署批次使新配置下发。"
      onClose={close}
      footer={<>
        <Button variant="secondary" onClick={close}>取消</Button>
        <Button
          onClick={() => void save()}
          disabled={saving || !taskName.trim() || !taskReleaseId || !taskNodeId || !taskInput.trim() || taskOutputInvalid || taskMediaInvalid || (taskNpuCorePolicy === 'exclusive' && taskNpuCoreMask === 'auto')}
        >
          {saving ? '正在保存…' : editingTaskId ? '保存并升级' : '创建任务'}
        </Button>
      </>}
    >
      <div className="form-sections">
        <section className="form-section">
          <h3><Container size={17} />任务配置</h3>
          <div className="form-grid two-columns">
            <label className="field"><span>任务名称 <b>*</b></span><input value={taskName} onChange={(event) => setTaskName(event.target.value)} /></label>
            <label className="field"><span>模型版本 <b>*</b></span><select value={taskReleaseId} onChange={(event) => setTaskReleaseId(event.target.value)}><option value="">请选择已发布版本</option>{publishedReleases.map((release) => <option key={release.id} value={release.id}>{release.name} · {release.version}</option>)}</select></label>
            <label className="field"><span>目标板卡 <b>*</b></span><select value={taskNodeId} onChange={(event) => setTaskNodeId(event.target.value)}><option value="">请选择健康板卡</option>{activeNodes.map((node) => <option key={node.id} value={node.id}>{node.name}</option>)}</select></label>
            <label className="field"><span>输入源 URI <b>*</b></span><input value={taskInput} onChange={(event) => setTaskInput(event.target.value)} placeholder="rtsp://... 或文件目录" /></label>
          </div>
        </section>
        <section className="form-section">
          <h3><RadioTower size={17} />媒体链路</h3>
          <div className="form-grid two-columns">
            <label className="field">
              <span>解码方式</span>
              <select value={taskDecoder} disabled={taskZlmSei} onChange={(event) => setTaskDecoder(event.target.value as 'opencv' | 'rkmpp')}>
                <option value="opencv">OpenCV 通用解码</option>
                <option value="rkmpp">RKMPP 硬解码</option>
              </select>
              <small>{taskZlmSei ? 'RTSP + SEI 已自动锁定 RKMPP，以保留原始编码流。' : 'RKMPP 仅支持 RTSP H.264/H.265 输入。'}</small>
            </label>
            <label className="toggle-card">
              <input
                type="checkbox"
                checked={taskZlmSei}
                disabled={!taskZlmSei && onlineMediaGateways.length === 0}
                onChange={(event) => {
                  const enabled = event.target.checked
                  setTaskZlmSei(enabled)
                  if (enabled) {
                    setTaskDecoder('rkmpp')
                    setTaskZlmGatewayId((current) => current || onlineMediaGateways[0]?.id || '')
                  }
                }}
              />
              <span><strong>RTSP + SEI 实时预览</strong><small>转发原始码流，由播放端解析 SEI 并绘制结果。</small></span>
            </label>
            {taskZlmSei && <>
              <label className="field">
                <span>媒体网关 <b>*</b></span>
                <select value={taskZlmGatewayId} onChange={(event) => setTaskZlmGatewayId(event.target.value)}>
                  <option value="">请选择在线媒体网关</option>
                  {mediaGateways.map((gateway) => <option key={gateway.id} value={gateway.id} disabled={!gateway.enabled || gateway.status !== 'online'}>{gateway.name} · {gateway.status === 'online' ? '在线' : '不可用'}</option>)}
                </select>
              </label>
              <label className="field">
                <span>发布流名称 <b>*</b></span>
                <input maxLength={64} value={taskZlmStreamName} onChange={(event) => setTaskZlmStreamName(event.target.value)} placeholder="line-a-camera-01" />
                <small>同一媒体网关内必须唯一。</small>
              </label>
            </>}
          </div>
          {!onlineMediaGateways.length && !taskZlmSei && <div className="field-hint">当前没有在线媒体网关，请先在系统设置中完成媒体网关连接。</div>}
          {taskMediaError && <div className="api-banner danger" role="alert">{taskMediaError}</div>}
        </section>
        <section className="form-section">
          <h3><ServerCog size={17} />NPU 核心调度</h3>
          <div className="form-grid two-columns">
            <label className="field"><span>使用核心</span><select value={taskNpuCoreMask} onChange={(event) => setTaskNpuCoreMask(event.target.value as InferenceTask['npuCoreMask'])}><option value="auto">自动调度（推荐）</option><option value="core0">核心 0</option><option value="core1">核心 1</option><option value="core2">核心 2</option><option value="core0_1">核心 0 + 1</option><option value="core0_1_2">全部核心</option></select></label>
            <label className="field"><span>核心策略</span><select value={taskNpuCorePolicy} onChange={(event) => setTaskNpuCorePolicy(event.target.value as InferenceTask['npuCorePolicy'])}><option value="shared">共享（允许其他任务使用）</option><option value="exclusive">独占（禁止核心重叠）</option></select><small>{taskNpuCorePolicy === 'exclusive' ? '独占策略必须选择明确核心，平台会在部署前检查冲突。' : '共享策略适合默认自动调度和同模型多路视频复用。'}</small></label>
          </div>
        </section>
        <section className="form-section">
          <h3><UploadCloud size={17} />结果出口</h3>
          <div className="form-grid two-columns">
            <label className="field"><span>输出方式</span><select value={taskOutputType} onChange={(event) => setTaskOutputType(event.target.value as 'jsonl' | 'http')}><option value="jsonl">板端 JSONL 文件</option><option value="http">HTTP 业务接口</option></select></label>
            {taskOutputType === 'http' && <label className="field"><span>业务接口 URL <b>*</b></span><input value={taskOutputUrl} onChange={(event) => setTaskOutputUrl(event.target.value)} placeholder="https://service.example/results" /></label>}
            {taskOutputType === 'http' && <label className="field"><span>Bearer 令牌环境变量</span><input value={taskOutputAuthEnv} onChange={(event) => setTaskOutputAuthEnv(event.target.value)} placeholder="RKNODE_RESULT_SINK_TOKEN" /></label>}
            {taskOutputType === 'http' && <label className="field"><span>连接 / 请求超时（毫秒）</span><div className="inline-fields"><input type="number" min="100" max="60000" value={taskConnectTimeout} onChange={(event) => setTaskConnectTimeout(Number(event.target.value))} /><input type="number" min="100" max="60000" value={taskRequestTimeout} onChange={(event) => setTaskRequestTimeout(Number(event.target.value))} /></div><small>请求超时必须不小于连接超时，最大 60000 毫秒。</small></label>}
          </div>
        </section>
      </div>
    </Modal>
    <Modal open={dialog === 'deployment'} title="创建部署批次" description="平台按策略推进目标板卡，板端代理逐阶段回报下载、校验、排空、热身和健康状态。" onClose={close} footer={<><Button variant="secondary" onClick={close}>取消</Button><Button onClick={() => void save()} disabled={saving || !deploymentReleaseId || !selectedTaskIds.length}>{saving ? '正在创建…' : '开始部署'}</Button></>}><div className="form-sections"><section className="form-section"><h3><CloudDownload size={17} />发布策略</h3><div className="form-grid two-columns"><label className="field"><span>批次名称 <b>*</b></span><input value={deploymentName} onChange={(event) => setDeploymentName(event.target.value)} /></label><label className="field"><span>模型版本 <b>*</b></span><select value={deploymentReleaseId} onChange={(event) => { const value = event.target.value; setDeploymentReleaseId(value); setSelectedTaskIds(inferenceTasks.filter((item) => item.releaseId === value && item.status === 'stopped').map((item) => item.id)) }}><option value="">请选择已发布版本</option>{publishedReleases.map((release) => <option key={release.id} value={release.id}>{release.name} · {release.version}</option>)}</select></label><label className="field"><span>推进策略</span><select value={deploymentStrategy} onChange={(event) => setDeploymentStrategy(event.target.value as typeof deploymentStrategy)}><option value="canary">先一台金丝雀</option><option value="rolling">滚动批次</option><option value="all_at_once">全部同时</option></select></label></div></section><section className="form-section"><h3><GitBranch size={17} />选择推理任务</h3><div className="selection-list">{deploymentTasks.length ? deploymentTasks.map((task) => <label key={task.id} className="selection-row"><input type="checkbox" checked={selectedTaskIds.includes(task.id)} onChange={(event) => setSelectedTaskIds(event.target.checked ? [...selectedTaskIds, task.id] : selectedTaskIds.filter((id) => id !== task.id))} /><span><strong>{task.name}</strong><small>{task.inputUri}</small></span><StatusBadge tone={tone(task.status)}>{taskLabels[task.status]}</StatusBadge></label>) : <EmptyState title="暂无可部署任务" message="先为该模型版本创建已停止的推理任务。" />}</div></section></div></Modal>
    {detailDeployment && <DeploymentDetailModal deployment={detailDeployment} nodes={inferenceNodes} tasks={inferenceTasks} onClose={() => setDetailDeployment(null)} />}
    {previewTask && <Modal open title={`实时预览 · ${previewTask.name}`} description="播放原始 RTSP 转发流，并在浏览器端解析 SEI 结果叠加。" width="large" onClose={() => setPreviewTask(null)}><InferenceStreamPlayer task={previewTask} /></Modal>}
    <ConfirmDialog open={Boolean(deleteNodeTarget)} title="永久删除退役板卡" description={deleteNodeTarget ? `确定永久删除「${deleteNodeTarget.name}」吗？系统会同时清理其推理服务配置和已退役任务；如仍有未退役任务或部署历史，删除将被拒绝。` : ''} confirmLabel="永久删除" busy={deleting} onClose={() => !deleting && setDeleteNodeTarget(null)} onConfirm={() => void deleteRetiredNode()} />
    <ConfirmDialog open={Boolean(deleteReleaseTarget)} title="永久删除弃用模型版本" description={deleteReleaseTarget ? `确定永久删除「${deleteReleaseTarget.name} ${deleteReleaseTarget.version}」吗？如仍被推理任务或部署历史引用，系统将拒绝删除。模型产物和来源任务不会被删除。` : ''} confirmLabel="永久删除" busy={deleting} onClose={() => !deleting && setDeleteReleaseTarget(null)} onConfirm={() => void deleteDeprecatedRelease()} />
    <ConfirmDialog open={Boolean(deleteDeploymentTarget)} title="删除部署批次" description={deleteDeploymentTarget ? `确定删除部署批次「${deleteDeploymentTarget.name}」及其目标状态和事件日志吗？推理任务与模型版本不会被删除。` : ''} confirmLabel="删除部署批次" busy={deleting} onClose={() => !deleting && setDeleteDeploymentTarget(null)} onConfirm={() => void deleteDeployment()} />
  </div>
}

function NodesTable({ nodes, groups, pagination, onApprove, onRetire, onDelete }: { nodes: InferenceNode[]; groups: NodeGroup[]; pagination: PaginationState; onApprove: (node: InferenceNode) => void; onRetire: (node: InferenceNode) => void; onDelete: (node: InferenceNode) => void }) {
  return <section className="panel table-panel"><div className="table-meta"><span>共 <strong>{pagination.total}</strong> 块板卡</span><span>当前页 {nodes.length} 块</span></div>{nodes.length ? <><div className="table-scroll"><table className="data-table inference-table"><thead><tr><th>板卡</th><th>节点组</th><th>生命周期</th><th>连接 / 健康</th><th>适配器 / 媒体能力</th><th>期望 / 实际版本</th><th>最近心跳</th><th aria-label="操作" /></tr></thead><tbody>{nodes.map((node) => { const features = nodeMediaFeatures(node); return <tr key={node.id}><td><strong>{node.name}</strong><small>{node.hardwareId ?? '等待板端注册'}</small></td><td>{groups.find((group) => group.id === node.groupId)?.name ?? '未分组'}</td><td><StatusBadge tone={tone(node.lifecycle)}>{lifecycleLabels[node.lifecycle]}</StatusBadge></td><td><StatusBadge tone={tone(node.health)}>{node.connectivity === 'online' ? '在线' : '离线'} · {node.health}</StatusBadge></td><td><div className="capability-list">{node.adapters.map((adapter) => <span key={adapter}>{adapter}</span>)}</div><small className="node-media-features">媒体：{features.length ? features.map((feature) => mediaFeatureLabels[feature] ?? feature).join(' · ') : '未上报（旧节点）'}</small></td><td><strong>{node.desiredRevision} / {node.actualRevision}</strong><small>{node.deploymentStatus}</small></td><td className="muted-cell">{node.lastSeenAt ? formatTime(node.lastSeenAt) : '尚未心跳'}</td><td>{node.lifecycle === 'awaiting_approval' ? <Button variant="secondary" onClick={() => onApprove(node)} disabled={!node.selfTestPassed} icon={<Check size={15} />}>审批启用</Button> : node.lifecycle === 'active' ? <div className="row-actions"><StatusBadge tone="success">可接收部署</StatusBadge><button className="icon-button ghost danger-action" title="退役板卡" aria-label={`退役板卡 ${node.name}`} onClick={() => onRetire(node)}><Trash2 size={16} /></button></div> : node.lifecycle === 'retired' ? <button className="icon-button ghost danger-action" title="永久删除退役板卡" aria-label={`永久删除退役板卡 ${node.name}`} onClick={() => onDelete(node)}><Trash2 size={16} /></button> : null}</td></tr>})}</tbody></table></div><TablePagination {...pagination} /></> : <EmptyState title="暂无 RK3588 板卡" message="先在系统设置启动并添加推理节点，再创建推理任务。" />}</section>
}

function ReleasesTable({ releases, jobs, pagination, onPublish, onDeprecate, onDelete }: { releases: ModelRelease[]; jobs: { id: string; name: string }[]; pagination: PaginationState; onPublish: (release: ModelRelease) => void; onDeprecate: (release: ModelRelease) => void; onDelete: (release: ModelRelease) => void }) {
  return <section className="panel table-panel"><div className="table-meta"><span>共 <strong>{pagination.total}</strong> 个模型版本</span><span>当前页 {releases.length} 个</span></div>{releases.length ? <><div className="table-scroll"><table className="data-table inference-table"><thead><tr><th>模型版本</th><th>任务 / 变体</th><th>精度 / 适配器</th><th>来源转换任务</th><th>状态</th><th>登记时间</th><th aria-label="操作" /></tr></thead><tbody>{releases.map((release) => <tr key={release.id}><td><strong>{release.name}</strong><small>{release.version}</small></td><td><strong>{release.profileId}</strong><small>{release.variant}</small></td><td><span className="precision-badge int8">{release.precision.toUpperCase()}</span><small>{release.adapter}</small></td><td className="muted-cell">{jobs.find((job) => job.id === release.sourceConversionJobId)?.name ?? '来源任务不可用'}</td><td><StatusBadge tone={tone(release.status)}>{releaseLabels[release.status]}</StatusBadge></td><td className="muted-cell">{formatTime(release.createdAt)}</td><td><div className="row-actions">{release.status === 'qualified' && <Button variant="secondary" onClick={() => onPublish(release)} icon={<UploadCloud size={15} />}>发布</Button>}{release.status === 'published' && <button className="icon-button ghost danger-action" title="弃用模型版本" aria-label={`弃用模型版本 ${release.name}`} onClick={() => onDeprecate(release)}><Trash2 size={16} /></button>}{release.status === 'deprecated' && <button className="icon-button ghost danger-action" title="永久删除弃用模型版本" aria-label={`永久删除模型版本 ${release.name}`} onClick={() => onDelete(release)}><Trash2 size={16} /></button>}</div></td></tr>)}</tbody></table></div><TablePagination {...pagination} /></> : <EmptyState title="暂无模型版本" message="转换任务完成并通过校验后，可登记为可下发版本。" />}</section>
}

function TasksTable({ tasks, releases, nodes, pagination, busyTaskId, onPreview, onStop, onRestart, onEdit, onRetire }: { tasks: InferenceTask[]; releases: ModelRelease[]; nodes: InferenceNode[]; pagination: PaginationState; busyTaskId: string; onPreview: (task: InferenceTask) => void; onStop: (task: InferenceTask) => void; onRestart: (task: InferenceTask) => void; onEdit: (task: InferenceTask) => void; onRetire: (task: InferenceTask) => void }) {
  return <section className="panel table-panel">
    <div className="table-meta"><span>共 <strong>{pagination.total}</strong> 个推理任务</span><span>当前页 {tasks.length} 个</span></div>
    {tasks.length ? <>
      <div className="table-scroll">
        <table className="data-table inference-table">
          <thead><tr><th>任务</th><th>模型版本</th><th>目标板卡</th><th>NPU 核心</th><th>输入源</th><th>状态</th><th aria-label="操作" /></tr></thead>
          <tbody>{tasks.map((task) => <tr key={task.id}>
            <td><strong>{task.name}</strong></td>
            <td>{releases.find((item) => item.id === task.releaseId)?.name ?? '模型版本不可用'}</td>
            <td>{nodes.find((item) => item.id === task.nodeId)?.name ?? '板卡不可用'}</td>
            <td><span className="type-label">{npuCoreLabels[task.npuCoreMask]}<small>{task.npuCorePolicy === 'exclusive' ? '独占' : '共享'}</small></span></td>
            <td><code className="service-endpoint">{task.inputUri}</code></td>
            <td><StatusBadge tone={tone(task.status)}>{taskLabels[task.status]}</StatusBadge></td>
            <td><div className="row-actions">
              {['running', 'deploying', 'degraded'].includes(task.status) && <button className="icon-button ghost" title="查看实时预览" aria-label={`查看实时预览 ${task.name}`} onClick={() => onPreview(task)}><MonitorPlay size={16} /></button>}
              {['stopped', 'failed', 'draft'].includes(task.status) && <button className="icon-button ghost" title="编辑或升级推理任务" aria-label={`编辑推理任务 ${task.name}`} onClick={() => onEdit(task)}><Pencil size={16} /></button>}
              {['stopped', 'failed'].includes(task.status) && <Button variant="secondary" disabled={Boolean(busyTaskId)} onClick={() => onRestart(task)} icon={<Play size={15} />}>{busyTaskId === task.id ? '重启中…' : '重启'}</Button>}
              {['running', 'deploying', 'degraded'].includes(task.status) && <Button variant="quiet" disabled={Boolean(busyTaskId)} onClick={() => onStop(task)} icon={<StopCircle size={15} />}>{busyTaskId === task.id ? '停止中…' : '停止'}</Button>}
              {['stopped', 'failed'].includes(task.status) && <button className="icon-button ghost danger-action" title="退役推理任务" aria-label={`退役推理任务 ${task.name}`} disabled={Boolean(busyTaskId)} onClick={() => onRetire(task)}><Trash2 size={16} /></button>}
            </div></td>
          </tr>)}</tbody>
        </table>
      </div>
      <TablePagination {...pagination} />
    </> : <EmptyState title="暂无推理任务" message="创建任务后选择一个已发布模型版本和健康板卡。" />}
  </section>
}

function DeploymentsTable({ deployments, releases, nodes, pagination, onRetry, onRollback, onDetail, onDelete }: { deployments: Deployment[]; releases: ModelRelease[]; nodes: InferenceNode[]; pagination: PaginationState; onRetry: (deployment: Deployment) => void; onRollback: (deployment: Deployment) => void; onDetail: (deployment: Deployment) => void; onDelete: (deployment: Deployment) => void }) {
  return <section className="panel table-panel"><div className="table-meta"><span>共 <strong>{pagination.total}</strong> 个部署批次</span><span>最近批次优先</span></div>{deployments.length ? <><div className="table-scroll"><table className="data-table inference-table"><thead><tr><th>部署批次</th><th>模型版本</th><th>策略</th><th>目标板卡</th><th>进度</th><th>状态</th><th aria-label="操作" /></tr></thead><tbody>{deployments.map((deployment) => { const healthy = deployment.targets.filter((target) => target.state === 'healthy' || target.state === 'rolled_back').length; const progress = deployment.targets.length ? Math.round(deployment.targets.reduce((total, target) => total + target.progress, 0) / deployment.targets.length) : 0; const active = ['queued', 'rolling', 'rolling_back'].includes(deployment.status); return <tr key={deployment.id}><td><strong>{deployment.name}</strong><small>{formatTime(deployment.createdAt)}</small></td><td>{releases.find((item) => item.id === deployment.releaseId)?.name ?? '模型版本不可用'}</td><td><span className="type-label">{deployment.strategy} · {healthy}/{deployment.targets.length}</span></td><td><div className="capability-list">{deployment.targets.map((target) => <span key={target.id}>{nodes.find((node) => node.id === target.nodeId)?.name ?? '板卡不可用'}</span>)}</div></td><td className="progress-cell"><ProgressBar value={progress} tone={deployment.status === 'failed' ? 'danger' : 'blue'} /></td><td><StatusBadge tone={tone(deployment.status)}>{deploymentLabels[deployment.status]}</StatusBadge></td><td><div className="row-actions"><button className="icon-button ghost" title="查看部署详情" aria-label={`查看部署详情 ${deployment.name}`} onClick={() => onDetail(deployment)}><Eye size={16} /></button>{['failed', 'paused'].includes(deployment.status) && <Button variant="secondary" onClick={() => onRetry(deployment)} icon={<RefreshCw size={15} />}>重试</Button>}{['succeeded', 'failed', 'paused'].includes(deployment.status) && <Button variant="quiet" onClick={() => onRollback(deployment)} icon={<RotateCcw size={15} />}>回滚</Button>}{!active && <button className="icon-button ghost danger-action" title="删除部署批次" aria-label={`删除部署批次 ${deployment.name}`} onClick={() => onDelete(deployment)}><Trash2 size={16} /></button>}</div></td></tr> })}</tbody></table></div><TablePagination {...pagination} /></> : <EmptyState title="暂无部署批次" message="创建部署后，板端代理会主动拉取模型并上报阶段进度。" />}</section>
}

function DeploymentDetailModal({ deployment, nodes, tasks, onClose }: { deployment: Deployment; nodes: InferenceNode[]; tasks: InferenceTask[]; onClose: () => void }) {
  const [currentDeployment, setCurrentDeployment] = useState(deployment)
  const [events, setEvents] = useState<Array<{ id: number; type: string; level: string; message: string; createdAt: string }>>([])
  const eventCursor = useRef(0)
  const [loadError, setLoadError] = useState('')
  useEffect(() => {
    setCurrentDeployment(deployment)
    setEvents([])
    eventCursor.current = 0
  }, [deployment.id, deployment])
  useEffect(() => {
    let active = true
    const poll = async () => {
      try {
        const [latest, nextEvents] = await Promise.all([
          api.deployment(deployment.id),
          api.deploymentEvents(deployment.id, eventCursor.current),
        ])
        if (!active) return
        setCurrentDeployment(latest)
        if (nextEvents.length) {
          eventCursor.current = nextEvents[nextEvents.length - 1].id
          setEvents((current) => [...current, ...nextEvents.map((event) => ({ id: event.id, type: event.type, level: event.level, message: event.message, createdAt: event.createdAt }))])
        }
        setLoadError('')
      } catch (reason) {
        if (active) setLoadError(reason instanceof Error ? reason.message : '部署详情加载失败')
      }
    }
    void poll()
    const live = ['queued', 'rolling', 'rolling_back'].includes(currentDeployment.status)
    const timer = live ? window.setInterval(() => { if (document.visibilityState === 'visible') void poll() }, 2000) : undefined
    return () => { active = false; if (timer) window.clearInterval(timer) }
  }, [currentDeployment.status, deployment.id])
  return <Modal open title={`部署详情 · ${currentDeployment.name}`} description="按目标板卡查看服务端事件和当前阶段。" width="wide" onClose={onClose} footer={<Button variant="secondary" onClick={onClose}>关闭</Button>}><>{loadError && <div className="api-banner danger">{loadError}</div>}<div className="deployment-detail-grid"><section className="form-section"><h3><GitBranch size={17} />目标状态</h3>{currentDeployment.targets.map((target) => <div className="deployment-target-row" key={target.id}><div><strong>{nodes.find((node) => node.id === target.nodeId)?.name ?? '未知板卡'}</strong><small>{tasks.find((task) => task.id === target.taskId)?.name ?? '未知任务'}</small></div><ProgressBar value={target.progress} tone={target.state === 'failed' ? 'danger' : 'blue'} /><StatusBadge tone={tone(target.state)}>{target.state}</StatusBadge></div>)}</section><section className="form-section"><h3><Eye size={17} />事件日志</h3><div className="deployment-events">{events.length ? events.map((event) => <div key={event.id}><small>{formatTime(event.createdAt)} · {event.type}</small><span>{event.message}</span></div>) : <EmptyState title="暂无事件" message="代理上报阶段后会出现在这里。" />}</div></section></div></></Modal>
}
