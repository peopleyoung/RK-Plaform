import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Download, Radio, SquareTerminal } from 'lucide-react'
import { api } from '../api/client'
import {
  metricLabel,
  metricNames,
  projectLogLines,
  projectMetricSamples,
  type TrainingMetricSample,
} from '../api/jobTelemetry'
import { formatTime, jobStatusLabels, jobTone } from '../api/presentation'
import type { Artifact, Job, JobEvent } from '../types'
import { Button, EmptyState, Modal, ProgressBar, StatusBadge } from '../components'

const ACTIVE_STATUSES = new Set<Job['status']>(['queued', 'claimed', 'running'])

interface TrainingMonitorProps {
  job: Job
  logArtifact?: Artifact
  onClose: () => void
  onDownload: (artifact: Artifact) => Promise<void>
}

export function TrainingMonitor({ job, logArtifact, onClose, onDownload }: TrainingMonitorProps) {
  const [currentJob, setCurrentJob] = useState(job)
  const [events, setEvents] = useState<JobEvent[]>([])
  const [loadError, setLoadError] = useState('')
  const [followLogs, setFollowLogs] = useState(true)
  const logRef = useRef<HTMLDivElement>(null)
  const metrics = useMemo(() => projectMetricSamples(events), [events])
  const logs = useMemo(() => projectLogLines(events), [events])
  const latestSample = metrics[metrics.length - 1]

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
        if (incoming.length) {
          setEvents((previous) => [...previous, ...incoming].slice(-10000))
        }
        setLoadError('')
        if (ACTIVE_STATUSES.has(latest.status)) timer = window.setTimeout(poll, 2000)
      } catch (reason) {
        if (!active) return
        setLoadError(reason instanceof Error ? reason.message : '训练监控数据加载失败')
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
    if (!followLogs || !logRef.current) return
    logRef.current.scrollTop = logRef.current.scrollHeight
  }, [followLogs, logs])

  const epoch = latestSample?.epoch
  const totalEpochs = latestSample?.totalEpochs
  return (
    <Modal
      open
      width="wide"
      title={`训练监控 · ${currentJob.name}`}
      description={`${currentJob.id} · ${formatTime(currentJob.updatedAt)}`}
      onClose={onClose}
      footer={<>
        {logArtifact && <Button variant="secondary" icon={<Download size={16} />} onClick={() => void onDownload(logArtifact)}>下载完整日志</Button>}
        <div className="footer-spacer" />
        <Button variant="secondary" onClick={onClose}>关闭</Button>
      </>}
    >
      <div className="training-monitor">
        {loadError && <div className="api-banner danger">{loadError}</div>}
        <section className="monitor-status-band" aria-label="训练实时状态">
          <div><span className="monitor-status-icon"><Activity size={18} /></span><p><small>任务状态</small><strong><StatusBadge tone={jobTone(currentJob.status)}>{jobStatusLabels[currentJob.status]}</StatusBadge></strong></p></div>
          <div><small>当前阶段</small><strong>{currentJob.stage}</strong></div>
          <div><small>训练轮次</small><strong>{epoch ? `${epoch}${totalEpochs ? ` / ${totalEpochs}` : ''}` : '--'}</strong></div>
          <div><small>执行节点</small><strong>{currentJob.workerId ?? '等待调度'}</strong></div>
        </section>
        <div className="monitor-progress"><ProgressBar value={currentJob.progress} tone={currentJob.status === 'failed' ? 'danger' : 'teal'} /></div>
        <div className="training-monitor-grid">
          <section className="monitor-section metric-section" aria-label="训练指标曲线">
            <header><div><Activity size={16} /><span><strong>训练指标</strong><small>{metrics.length} 个采样点</small></span></div>{ACTIVE_STATUSES.has(currentJob.status) && <span className="live-indicator"><Radio size={13} />实时</span>}</header>
            <MetricsChart samples={metrics} />
          </section>
          <section className="monitor-section log-section" aria-label="实时训练日志">
            <header><div><SquareTerminal size={16} /><span><strong>训练日志</strong><small>{logs.length} 行</small></span></div><label className="log-follow"><input type="checkbox" checked={followLogs} onChange={(event) => setFollowLogs(event.target.checked)} />自动滚动</label></header>
            {logs.length ? <div className="training-log" role="log" aria-live="polite" ref={logRef}>{logs.map((line, index) => <div className={`training-log-line ${line.level}`} key={`${line.eventId}-${index}`}><time>{new Date(line.createdAt).toLocaleTimeString('zh-CN', { hour12: false })}</time><span>{line.stage}</span><code>{line.message}</code></div>)}</div> : <EmptyState title="暂无训练日志" message={ACTIVE_STATUSES.has(currentJob.status) ? '等待训练节点上报。' : '该历史任务没有实时日志记录。'} />}
          </section>
        </div>
      </div>
    </Modal>
  )
}

function MetricsChart({ samples }: { samples: TrainingMetricSample[] }) {
  const names = useMemo(() => metricNames(samples), [samples])
  const [selected, setSelected] = useState('')

  useEffect(() => {
    if (!names.length) setSelected('')
    else if (!names.includes(selected)) setSelected(names[0])
  }, [names, selected])

  if (!selected) {
    return <EmptyState title="暂无指标数据" message="指标将在训练轮次或评估步骤完成后出现。" />
  }
  const points = samples
    .map((sample, index) => ({ sample, index, value: sample.metrics[selected] }))
    .filter((point): point is { sample: TrainingMetricSample; index: number; value: number } => typeof point.value === 'number')
  if (!points.length) return null

  const width = 720
  const height = 250
  const plot = { left: 54, right: 18, top: 18, bottom: 34 }
  const values = points.map((point) => point.value)
  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)
  const padding = rawMin === rawMax ? Math.max(Math.abs(rawMin) * 0.1, 0.1) : (rawMax - rawMin) * 0.1
  const minimum = rawMin - padding
  const maximum = rawMax + padding
  const x = (index: number) => plot.left + index / Math.max(1, points.length - 1) * (width - plot.left - plot.right)
  const y = (value: number) => plot.top + (maximum - value) / (maximum - minimum) * (height - plot.top - plot.bottom)
  const path = points.map((point, index) => `${index ? 'L' : 'M'} ${x(index).toFixed(2)} ${y(point.value).toFixed(2)}`).join(' ')
  const latest = points[points.length - 1]
  const firstLabel = sampleAxisLabel(points[0].sample, 1)
  const lastLabel = sampleAxisLabel(latest.sample, points.length)

  return <div className="metric-chart-wrap">
    <div className="metric-chart-toolbar"><label><span>指标</span><select aria-label="指标曲线" value={selected} onChange={(event) => setSelected(event.target.value)}>{names.map((name) => <option value={name} key={name}>{metricLabel(name)}</option>)}</select></label><div><small>最新值</small><strong>{formatMetric(latest.value)}</strong></div></div>
    <svg className="metric-line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${metricLabel(selected)}变化曲线`}>
      {[0, 1, 2, 3, 4].map((line) => {
        const position = plot.top + line / 4 * (height - plot.top - plot.bottom)
        const value = maximum - line / 4 * (maximum - minimum)
        return <g key={line}><line x1={plot.left} x2={width - plot.right} y1={position} y2={position} /><text x={plot.left - 8} y={position + 3}>{formatMetric(value)}</text></g>
      })}
      <path className="metric-line" d={path} />
      {points.length <= 40 && points.map((point, index) => <circle className="metric-point" key={point.sample.eventId} cx={x(index)} cy={y(point.value)} r="3"><title>{`${sampleAxisLabel(point.sample, index + 1)} · ${formatMetric(point.value)}`}</title></circle>)}
      <text className="axis-label start" x={plot.left} y={height - 10}>{firstLabel}</text>
      <text className="axis-label end" x={width - plot.right} y={height - 10}>{lastLabel}</text>
    </svg>
  </div>
}

function sampleAxisLabel(sample: TrainingMetricSample, fallback: number): string {
  if (sample.epoch !== null) return `Epoch ${sample.epoch}`
  if (sample.step !== null) return `Step ${sample.step}`
  return `#${fallback}`
}

function formatMetric(value: number): string {
  const absolute = Math.abs(value)
  if (absolute >= 100) return value.toFixed(1)
  if (absolute >= 1) return value.toFixed(3)
  if (absolute === 0) return '0'
  return value.toPrecision(4)
}
