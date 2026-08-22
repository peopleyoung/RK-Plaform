import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Check, Circle, Download, ListChecks, Radio, SquareTerminal } from 'lucide-react'
import { api } from '../api/client'
import { projectLogLines } from '../api/jobTelemetry'
import { formatTime, jobStatusLabels, jobTone } from '../api/presentation'
import type { Artifact, Job, JobEvent } from '../types'
import { Button, EmptyState, Modal, ProgressBar, StatusBadge } from '../components'

const ACTIVE_STATUSES = new Set<Job['status']>(['queued', 'claimed', 'running'])

const stageLabels: Record<string, string> = {
  queued: '等待调度',
  download_onnx: '下载 ONNX 产物',
  verify_onnx: '校验 ONNX 文件',
  download_calibration: '下载校准数据集',
  prepare_calibration: '准备校准样本',
  validate_onnx: '校验部署契约',
  verify_integrity: '验证 ONNX 完整性',
  optimize_graph: '优化 RK3588 部署图',
  initialize_toolkit: '初始化 RKNN Toolkit2',
  configure_rknn: '配置 RKNN 转换参数',
  load_onnx: '加载 ONNX 模型',
  build_rknn: '构建 RKNN 模型',
  export_rknn: '导出 RKNN 模型',
  initialize_runtime: '初始化 RK3588 运行时',
  validate_inference: '执行板端验证推理',
  benchmark_runtime: '执行稳定性能测试',
  audit_performance: '分析算子执行位置',
  upload_rknn: '上传 RKNN 产物',
  upload_report: '上传验证报告',
  upload_log: '上传转换日志',
  completed: '转换完成',
  failed: '转换失败',
}

interface ConversionStage {
  eventId: number
  stage: string
  message: string
  progress: number
  createdAt: string
}

interface ConversionValidation {
  deploymentReady?: boolean
  performanceReady?: boolean
  toolkitVersion?: string
  outputShapes?: number[][]
  benchmark?: { averageMs?: number; fps?: number }
  performance?: { cpuFallbackDetected?: boolean }
}

interface ConversionMonitorProps {
  job: Job
  workerName?: string
  reportArtifact?: Artifact
  logArtifact?: Artifact
  onClose: () => void
  onDownload: (artifact: Artifact) => Promise<void>
}

export function ConversionMonitor({
  job,
  workerName,
  reportArtifact,
  logArtifact,
  onClose,
  onDownload,
}: ConversionMonitorProps) {
  const [currentJob, setCurrentJob] = useState(job)
  const [events, setEvents] = useState<JobEvent[]>([])
  const [loadError, setLoadError] = useState('')
  const [followEvents, setFollowEvents] = useState(true)
  const eventLogRef = useRef<HTMLDivElement>(null)
  const stages = useMemo(() => projectConversionStages(events), [events])
  const logs = useMemo(() => projectLogLines(events), [events])
  const validation = conversionValidation(currentJob)
  const precision = conversionPrecision(currentJob)

  useEffect(() => {
    let active = true
    let timer = 0
    let cursor = 0
    setCurrentJob(job)
    setEvents([])
    setLoadError('')

    const poll = async () => {
      try {
        const latest = await api.job(job.id)
        const incoming: JobEvent[] = []
        for (let page = 0; page < 10; page += 1) {
          const batch = await api.jobEvents(job.id, cursor, 500)
          incoming.push(...batch)
          if (batch.length) cursor = batch[batch.length - 1].id
          if (batch.length < 500) break
        }
        if (!active) return
        setCurrentJob(latest)
        if (incoming.length) setEvents((previous) => [...previous, ...incoming].slice(-10000))
        setLoadError('')
        if (ACTIVE_STATUSES.has(latest.status)) timer = window.setTimeout(poll, 2000)
      } catch (reason) {
        if (!active) return
        setLoadError(reason instanceof Error ? reason.message : '转换进度数据加载失败')
        timer = window.setTimeout(poll, 4000)
      }
    }

    void poll()
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [job])

  useEffect(() => {
    if (!followEvents || !eventLogRef.current) return
    eventLogRef.current.scrollTop = eventLogRef.current.scrollHeight
  }, [followEvents, logs])

  return (
    <Modal
      open
      width="wide"
      title={`转换进度 · ${currentJob.name}`}
      description={`最近更新 ${formatTime(currentJob.updatedAt)}`}
      onClose={onClose}
      footer={<>
        {reportArtifact && <Button variant="secondary" icon={<Download size={16} />} onClick={() => void onDownload(reportArtifact)}>下载验证报告</Button>}
        {logArtifact && <Button variant="secondary" icon={<Download size={16} />} onClick={() => void onDownload(logArtifact)}>下载转换日志</Button>}
        <div className="footer-spacer" />
        <Button variant="secondary" onClick={onClose}>关闭</Button>
      </>}
    >
      <div className="training-monitor conversion-monitor">
        {loadError && <div className="api-banner danger">{loadError}</div>}
        {currentJob.errorMessage && <div className="api-banner danger">{currentJob.errorMessage}</div>}
        <section className="monitor-status-band" aria-label="转换实时状态">
          <div><span className="monitor-status-icon"><Activity size={18} /></span><p><small>任务状态</small><strong><StatusBadge tone={jobTone(currentJob.status)}>{jobStatusLabels[currentJob.status]}</StatusBadge></strong></p></div>
          <div><small>当前阶段</small><strong>{stageLabel(currentJob.stage)}</strong></div>
          <div><small>转换精度</small><strong>{precision.toUpperCase()}</strong></div>
          <div><small>执行节点</small><strong>{workerName ?? (currentJob.workerId ? '已分配转换节点' : '等待调度')}</strong></div>
        </section>
        <div className="monitor-progress"><ProgressBar value={currentJob.progress} tone={currentJob.status === 'failed' ? 'danger' : 'blue'} /></div>
        {validation && <ValidationSummary validation={validation} />}
        <div className="conversion-monitor-grid">
          <section className="monitor-section conversion-stage-section" aria-label="转换阶段明细">
            <header><div><ListChecks size={16} /><span><strong>转换阶段</strong><small>{stages.length} 个已上报阶段</small></span></div>{ACTIVE_STATUSES.has(currentJob.status) && <span className="live-indicator"><Radio size={13} />实时</span>}</header>
            {stages.length ? <div className="conversion-stage-list">{stages.map((stage) => {
              const complete = stage.progress < currentJob.progress || currentJob.status === 'succeeded'
              const active = stage.stage === currentJob.stage && ACTIVE_STATUSES.has(currentJob.status)
              return <div className={`conversion-stage-row${active ? ' active' : ''}`} key={stage.eventId}>
                <span className="conversion-stage-marker">{complete ? <Check size={14} /> : <Circle size={11} />}</span>
                <div><strong>{stageLabel(stage.stage)}</strong><p>{stage.message}</p><time>{new Date(stage.createdAt).toLocaleTimeString('zh-CN', { hour12: false })}</time></div>
                <b>{stage.progress}%</b>
              </div>
            })}</div> : <EmptyState title="暂无阶段数据" message={ACTIVE_STATUSES.has(currentJob.status) ? '等待转换节点上报。' : '该历史任务没有阶段事件。'} />}
          </section>
          <section className="monitor-section log-section" aria-label="转换事件日志">
            <header><div><SquareTerminal size={16} /><span><strong>转换事件</strong><small>{logs.length} 条</small></span></div><label className="log-follow"><input type="checkbox" checked={followEvents} onChange={(event) => setFollowEvents(event.target.checked)} />自动滚动</label></header>
            {logs.length ? <div className="training-log" role="log" aria-live="polite" ref={eventLogRef}>{logs.map((line, index) => <div className={`training-log-line ${line.level}`} key={`${line.eventId}-${index}`}><time>{new Date(line.createdAt).toLocaleTimeString('zh-CN', { hour12: false })}</time><span>{stageLabel(line.stage)}</span><code>{line.message}</code></div>)}</div> : <EmptyState title="暂无转换事件" message={ACTIVE_STATUSES.has(currentJob.status) ? '等待转换节点上报。' : '该历史任务没有事件记录。'} />}
          </section>
        </div>
      </div>
    </Modal>
  )
}

function projectConversionStages(events: JobEvent[]): ConversionStage[] {
  const byStage = new Map<string, ConversionStage>()
  for (const event of events) {
    const stage = typeof event.data.stage === 'string' ? event.data.stage : ''
    const progress = typeof event.data.progress === 'number' ? event.data.progress : null
    if (!stage || progress === null) continue
    byStage.set(stage, {
      eventId: event.id,
      stage,
      message: event.message,
      progress,
      createdAt: event.createdAt,
    })
  }
  return [...byStage.values()].sort((left, right) => left.progress - right.progress || left.eventId - right.eventId)
}

function conversionPrecision(job: Job): string {
  const precision = job.spec.precision
  return typeof precision === 'string' ? precision : '--'
}

function conversionValidation(job: Job): ConversionValidation | null {
  const result = job.result
  if (!result || typeof result !== 'object') return null
  const candidate = result.validation
  return candidate && typeof candidate === 'object' ? candidate as ConversionValidation : null
}

function stageLabel(stage: string): string {
  return stageLabels[stage] ?? stage.replaceAll('_', ' ')
}

function ValidationSummary({ validation }: { validation: ConversionValidation }) {
  const outputShape = validation.outputShapes?.[0]?.join(' × ') ?? '--'
  const averageMs = validation.benchmark?.averageMs
  const fps = validation.benchmark?.fps
  const fallback = validation.performance?.cpuFallbackDetected
  return <section className="conversion-validation-strip" aria-label="RK3588 验证结果">
    <div><small>部署验证</small><strong>{validation.deploymentReady ? '已通过' : '未通过'}</strong></div>
    <div><small>RKNN Toolkit</small><strong>{validation.toolkitVersion ?? '--'}</strong></div>
    <div><small>输出形状</small><strong>{outputShape}</strong></div>
    <div><small>平均延迟 / FPS</small><strong>{typeof averageMs === 'number' ? `${averageMs.toFixed(2)} ms` : '--'} / {typeof fps === 'number' ? fps.toFixed(2) : '--'}</strong></div>
    <div><small>计算算子 CPU 回退</small><strong>{fallback === false ? '未发现' : fallback === true ? '已发现' : '--'}</strong></div>
  </section>
}
