import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUpRight, Boxes, Database, Download, Play, ServerCog, Sparkles, Zap } from 'lucide-react'
import { api } from '../api/client'
import { loadAllPages } from '../api/pagination'
import { usePlatform } from '../api/PlatformContext'
import { formatTime, jobStatusLabels, jobTone, variantLabel } from '../api/presentation'
import { Button, ChevronAction, EmptyState, MetricCard, PageHeader, ProgressBar, StatusBadge } from '../components'
import { primaryReleaseId } from '../inferenceGraph'
import type { Deployment, InferenceSummary, InferenceTask, Job, ModelRelease, RouteKey, ServiceEndpoint, StatusTone, WorkerNode } from '../types'

const emptyInferenceSummary: InferenceSummary = {
  onlineNodes: 0,
  totalNodes: 0,
  publishedReleases: 0,
  runningTasks: 0,
  activeDeployments: 0,
}

interface OverviewTask {
  id: string
  name: string
  model: string
  type: string
  progress: number
  stage: string
  status: string
  tone: StatusTone
  updatedAt: string
  outcome: 'success' | 'danger' | 'active'
}

const inferenceStatusLabels: Record<InferenceTask['status'], string> = {
  draft: '草稿',
  stopped: '已停止',
  deploying: '部署中',
  running: '运行中',
  degraded: '降级运行',
  failed: '失败',
  retired: '已退役',
}

const deploymentStatusLabels: Record<Deployment['status'], string> = {
  queued: '排队中',
  rolling: '部署中',
  succeeded: '已完成',
  paused: '已暂停',
  failed: '失败',
  rolling_back: '回滚中',
  rolled_back: '已回滚',
  cancelled: '已取消',
}

function accelerator(job: Job) {
  return job.spec.accelerator === 'cuda' ? 'cuda' : 'cpu'
}

function jobModel(job: Job) {
  return typeof job.spec.variant === 'string' ? variantLabel(job.spec.variant) : job.profileId
}

function releaseName(releases: ModelRelease[], releaseId: string) {
  const release = releases.find((item) => item.id === releaseId)
  return release ? `${release.name} · ${release.version}` : '模型版本不可用'
}

function inferenceTone(status: InferenceTask['status']): StatusTone {
  if (status === 'running') return 'success'
  if (status === 'deploying') return 'info'
  if (status === 'degraded' || status === 'failed') return 'danger'
  return 'neutral'
}

function deploymentTone(status: Deployment['status']): StatusTone {
  if (status === 'succeeded') return 'success'
  if (status === 'failed' || status === 'cancelled') return 'danger'
  if (status === 'queued' || status === 'rolling' || status === 'rolling_back') return 'info'
  return 'warning'
}

function endpointCapacity(endpoint: ServiceEndpoint, workers: WorkerNode[], tasks: InferenceTask[]) {
  if (endpoint.kind === 'inference') {
    const active = tasks.filter((task) => task.nodeId === endpoint.inferenceNodeId && ['deploying', 'running', 'degraded'].includes(task.status)).length
    const configured = endpoint.remoteMetadata.maxConcurrency
    return { active, capacity: typeof configured === 'number' && configured > 0 ? configured : Math.max(1, active) }
  }
  const worker = workers.find((item) => item.name === endpoint.name)
  return { active: worker?.activeJobs ?? 0, capacity: Math.max(1, worker?.maxConcurrency ?? 1) }
}

function endpointOnline(endpoint: ServiceEndpoint, workers: WorkerNode[]) {
  if (!endpoint.enabled) return false
  if (endpoint.mode === 'direct') return endpoint.probeStatus === 'online'
  return workers.some((worker) => worker.name === endpoint.name && worker.status !== 'offline')
}

function deploymentProgress(deployment: Deployment) {
  if (!deployment.targets.length) return deployment.status === 'succeeded' ? 100 : 0
  return Math.round(deployment.targets.reduce((sum, target) => sum + target.progress, 0) / deployment.targets.length)
}

function taskProgress(task: InferenceTask, deployments: Deployment[]) {
  if (task.status === 'running' || task.status === 'degraded') return 100
  const targets = deployments.flatMap((deployment) => deployment.targets).filter((target) => target.taskId === task.id)
  return targets.length ? Math.max(...targets.map((target) => target.progress)) : 0
}

export function OverviewPage({ onNavigate, onCreateTraining }: { onNavigate: (route: RouteKey) => void; onCreateTraining: () => void; notify: (message: string) => void }) {
  const { datasets, jobs, workers, artifacts, serviceEndpoints, loading, error, refresh } = usePlatform()
  const [inferenceSummary, setInferenceSummary] = useState<InferenceSummary>(emptyInferenceSummary)
  const [inferenceTasks, setInferenceTasks] = useState<InferenceTask[]>([])
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [modelReleases, setModelReleases] = useState<ModelRelease[]>([])
  const [inferenceError, setInferenceError] = useState('')
  const inferenceRefreshSequence = useRef(0)

  const refreshInferenceOverview = useCallback(async () => {
    const sequence = ++inferenceRefreshSequence.current
    try {
      const [nextSummary, nextTasks, nextDeployments, nextReleases] = await Promise.all([
        api.inferenceSummary(),
        api.inferenceTasks(1, 50),
        api.deployments(1, 50),
        loadAllPages(api.modelReleases),
      ])
      if (sequence !== inferenceRefreshSequence.current) return
      setInferenceSummary(nextSummary)
      setInferenceTasks(nextTasks.items)
      setDeployments(nextDeployments.items)
      setModelReleases(nextReleases)
      setInferenceError('')
    } catch (reason) {
      if (sequence !== inferenceRefreshSequence.current) return
      setInferenceError(reason instanceof Error ? reason.message : '推理概况加载失败')
    }
  }, [])

  useEffect(() => {
    void refresh()
    void refreshInferenceOverview()
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void refreshInferenceOverview()
    }, 10000)
    return () => {
      window.clearInterval(timer)
      inferenceRefreshSequence.current += 1
    }
  }, [refresh, refreshInferenceOverview])

  const activeJobs = jobs.filter((job) => ['queued', 'claimed', 'running'].includes(job.status))
  const activeTraining = activeJobs.filter((job) => job.type === 'training')
  const activeConversions = activeJobs.filter((job) => job.type === 'conversion')
  const visibleInferenceTasks = inferenceTasks.filter((task) => task.status !== 'retired')
  const activeInferenceTasks = visibleInferenceTasks.filter((task) => ['deploying', 'running', 'degraded'].includes(task.status))
  const activeDeployments = deployments.filter((deployment) => ['queued', 'rolling', 'rolling_back'].includes(deployment.status))
  const onlineServices = serviceEndpoints.filter((endpoint) => endpointOnline(endpoint, workers))
  const activeTaskCount = activeJobs.length + activeInferenceTasks.length + activeDeployments.length

  const jobActivities: OverviewTask[] = jobs.map((job) => ({
    id: job.id,
    name: job.name,
    model: jobModel(job),
    type: job.type === 'training' ? '模型训练' : 'RKNN 转换',
    progress: job.progress,
    stage: job.stage,
    status: jobStatusLabels[job.status],
    tone: jobTone(job.status),
    updatedAt: job.updatedAt,
    outcome: job.status === 'succeeded' ? 'success' : job.status === 'failed' || job.status === 'cancelled' ? 'danger' : 'active',
  }))
  const inferenceActivities: OverviewTask[] = visibleInferenceTasks.map((task) => ({
    id: task.id,
    name: task.name,
    model: releaseName(modelReleases, primaryReleaseId(task)),
    type: '推理任务',
    progress: taskProgress(task, deployments),
    stage: inferenceStatusLabels[task.status],
    status: inferenceStatusLabels[task.status],
    tone: inferenceTone(task.status),
    updatedAt: task.updatedAt,
    outcome: task.status === 'running' ? 'success' : task.status === 'failed' || task.status === 'degraded' ? 'danger' : 'active',
  }))
  const deploymentActivities: OverviewTask[] = deployments.map((deployment) => ({
    id: deployment.id,
    name: deployment.name,
    model: `${deployment.targets.length} 个图修订`,
    type: '模型部署',
    progress: deploymentProgress(deployment),
    stage: deployment.targets.find((target) => !['healthy', 'rolled_back'].includes(target.state))?.stage ?? deploymentStatusLabels[deployment.status],
    status: deploymentStatusLabels[deployment.status],
    tone: deploymentTone(deployment.status),
    updatedAt: deployment.updatedAt,
    outcome: deployment.status === 'succeeded' ? 'success' : deployment.status === 'failed' || deployment.status === 'cancelled' ? 'danger' : 'active',
  }))
  const activeIds = new Set([...activeJobs, ...activeInferenceTasks, ...activeDeployments].map((item) => item.id))
  const activities = [...jobActivities, ...inferenceActivities, ...deploymentActivities]
  const currentTasks = activities.filter((item) => activeIds.has(item.id)).sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)).slice(0, 6)
  const recent = [...activities].sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)).slice(0, 5)
  const lastSevenDays = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(); date.setHours(0, 0, 0, 0); date.setDate(date.getDate() - (6 - index))
    const end = new Date(date); end.setDate(end.getDate() + 1)
    const dayJobs = jobs.filter((job) => job.type === 'training' && Date.parse(job.createdAt) >= date.getTime() && Date.parse(job.createdAt) < end.getTime())
    return { label: new Intl.DateTimeFormat('zh-CN', { weekday: 'short' }).format(date), cuda: dayJobs.filter((job) => accelerator(job) === 'cuda').length, cpu: dayJobs.filter((job) => accelerator(job) === 'cpu').length }
  })
  const maxDaily = Math.max(1, ...lastSevenDays.flatMap((item) => [item.cuda, item.cpu]))

  return <div className="page-stack">
    <PageHeader eyebrow={new Intl.DateTimeFormat('zh-CN', { dateStyle: 'full' }).format(new Date())} title="平台总览" description={`${onlineServices.length} / ${serviceEndpoints.length} 个服务节点在线，${activeTaskCount} 个任务正在执行或等待调度。`} actions={<Button icon={<Play size={17} fill="currentColor" />} onClick={onCreateTraining}>新建训练任务</Button>} />
    {(error || inferenceError) && <div className="api-banner danger">{error || inferenceError}</div>}
    <section className="metric-grid" aria-label="平台指标"><MetricCard label="可用数据集" value={String(datasets.filter((item) => item.status === 'ready').length)} detail={`${datasets.length} 个记录`} icon={<Database size={20} />} tone="teal" /><MetricCard label="训练队列" value={String(activeTraining.length)} detail={`CUDA ${activeTraining.filter((job) => accelerator(job) === 'cuda').length} · CPU ${activeTraining.filter((job) => accelerator(job) === 'cpu').length}`} icon={<Zap size={20} />} tone="amber" /><MetricCard label="转换队列" value={String(activeConversions.length)} detail={`${artifacts.filter((item) => item.kind === 'rknn').length} 个 RKNN 产物`} icon={<Boxes size={20} />} tone="blue" /><MetricCard label="运行中推理" value={String(inferenceSummary.runningTasks)} detail={`板卡 ${inferenceSummary.onlineNodes} / ${inferenceSummary.totalNodes} 在线 · 部署中 ${inferenceSummary.activeDeployments}`} icon={<ServerCog size={20} />} tone="coral" /></section>
    <div className="overview-grid"><section className="panel span-two"><div className="panel-heading"><div><h3>训练任务提交量</h3><p>最近 7 天 CUDA / CPU 任务数</p></div></div><div className="runtime-chart" aria-label="最近七天训练任务数柱状图">{lastSevenDays.map((item) => <div className="bar-group" key={item.label}><div className="bar-stack"><span className="chart-bar cpu" style={{ height: `${Math.max(item.cpu ? 12 : 2, item.cpu / maxDaily * 100)}%` }} /><span className="chart-bar gpu" style={{ height: `${Math.max(item.cuda ? 12 : 2, item.cuda / maxDaily * 100)}%` }} /></div><small>{item.label}</small></div>)}<div className="chart-axis-line axis-1" /><div className="chart-axis-line axis-2" /><div className="chart-axis-line axis-3" /></div><div className="chart-legend"><span><i className="legend-dot gpu" />CUDA 任务</span><span><i className="legend-dot cpu" />CPU 任务</span><strong>合计 {lastSevenDays.reduce((sum, item) => sum + item.cuda + item.cpu, 0)}</strong></div></section><section className="panel"><div className="panel-heading"><div><h3>服务节点占用</h3><p>系统设置中的训练、转换和推理节点</p></div><ChevronAction label="全部节点" onClick={() => onNavigate('nodes')} /></div><div className="node-list compact-list">{serviceEndpoints.length ? serviceEndpoints.slice(0, 5).map((endpoint) => { const capacity = endpointCapacity(endpoint, workers, visibleInferenceTasks); const value = Math.round(capacity.active / capacity.capacity * 100); const symbol = endpoint.kind === 'inference' ? 'RK' : endpoint.kind === 'converter' ? 'CV' : 'TR'; return <div className="node-load-row" key={endpoint.id}><div className="node-load-head"><span className={endpoint.kind === 'inference' || endpoint.kind === 'converter' ? 'node-symbol rk' : 'node-symbol gpu'}>{symbol}</span><div><strong>{endpoint.name}</strong><small>{capacity.active} / {capacity.capacity} 个任务 · {endpointOnline(endpoint, workers) ? '在线' : '离线'}</small></div><span>{value}%</span></div><ProgressBar value={value} showValue={false} compact tone={endpoint.kind === 'trainer' ? 'teal' : 'blue'} /></div> }) : <EmptyState title="暂无服务节点" message="在系统设置中添加节点后显示。" />}</div></section></div>
    <div className="overview-grid lower-grid"><section className="panel span-two"><div className="panel-heading"><div><h3>当前任务</h3><p>训练、转换、推理和部署的实时状态</p></div><ChevronAction label="推理下发" onClick={() => onNavigate('inference')} /></div>{loading && !activities.length ? <EmptyState title="正在加载任务" message="正在连接调度服务。" /> : currentTasks.length ? <div className="table-scroll"><table className="data-table compact-table"><thead><tr><th>任务</th><th>模型</th><th>任务类型</th><th>进度</th><th>状态</th></tr></thead><tbody>{currentTasks.map((task) => <tr key={`${task.type}-${task.id}`}><td><strong>{task.name}</strong></td><td>{task.model}</td><td className="muted-cell">{task.type}</td><td><ProgressBar value={task.progress} compact tone={task.type === '模型训练' ? 'teal' : 'blue'} /></td><td><StatusBadge tone={task.tone}>{task.status}</StatusBadge></td></tr>)}</tbody></table></div> : <EmptyState title="暂无进行中任务" message="提交训练、转换、推理或部署任务后显示。" />}</section><section className="panel"><div className="panel-heading"><div><h3>最近状态</h3><p>全部业务任务最新五条</p></div></div><div className="activity-list">{recent.length ? recent.map((task) => <div className="activity-row" key={`${task.type}-${task.id}`}><span className={`activity-icon ${task.outcome === 'success' ? 'success' : task.outcome === 'danger' ? 'danger' : ''}`}>{task.outcome === 'success' ? <Sparkles size={15} /> : task.outcome === 'danger' ? <ArrowUpRight size={15} /> : <Download size={15} />}</span><div><strong>{task.name}</strong><p>{task.type} · {task.stage} · {task.status}</p><small>{formatTime(task.updatedAt)}</small></div></div>) : <EmptyState title="暂无任务动态" message="提交任务后显示状态。" />}</div></section></div>
  </div>
}
