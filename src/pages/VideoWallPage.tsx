import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Grid2X2, LayoutGrid, Radio, RefreshCw, Search, Video, VideoOff, X } from 'lucide-react'
import { api } from '../api/client'
import { loadAllPages } from '../api/pagination'
import { EmptyState, PageHeader, StatusBadge } from '../components'
import { InferenceStreamPlayer } from '../components/InferenceStreamPlayer'
import type { InferenceTask, StatusTone } from '../types'

type WallLayout = 4 | 6
type PlayableTask = InferenceTask & { status: 'running' | 'deploying' | 'degraded' }

interface WallSettings {
  layout: WallLayout
  assignments: Array<string | null>
}

const STORAGE_KEY = 'rknode.videoWall.v1'
const PLAYABLE_STATUSES = new Set<InferenceTask['status']>(['running', 'deploying', 'degraded'])
const taskLabels: Record<InferenceTask['status'], string> = {
  draft: '草稿',
  stopped: '已停止',
  deploying: '部署中',
  running: '运行中',
  degraded: '降级',
  failed: '失败',
  retired: '已退役',
}

function blankAssignments(): Array<string | null> {
  return Array.from({ length: 6 }, () => null)
}

function readSettings(): WallSettings {
  try {
    const value = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}') as Record<string, unknown>
    const saved = Array.isArray(value.assignments) ? value.assignments.slice(0, 6) : []
    const assignments = blankAssignments()
    saved.forEach((item, index) => {
      if (typeof item === 'string' && item.trim()) assignments[index] = item
    })
    return { layout: value.layout === 6 ? 6 : 4, assignments }
  } catch {
    return { layout: 4, assignments: blankAssignments() }
  }
}

function taskTone(status: InferenceTask['status']): StatusTone {
  if (status === 'running') return 'success'
  if (status === 'deploying' || status === 'degraded') return 'warning'
  if (status === 'failed' || status === 'retired') return 'danger'
  return 'neutral'
}

function isPlayable(task: InferenceTask | undefined): task is PlayableTask {
  return Boolean(task && PLAYABLE_STATUSES.has(task.status) && task.previewCapability.state === 'available')
}

export function VideoWallPage() {
  const [initialSettings] = useState(readSettings)
  const [layout, setLayout] = useState<WallLayout>(initialSettings.layout)
  const [assignments, setAssignments] = useState(initialSettings.assignments)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [tasks, setTasks] = useState<InferenceTask[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState('')
  const requestId = useRef(0)

  const loadTasks = useCallback(async (showBusy = false) => {
    const currentRequest = requestId.current + 1
    requestId.current = currentRequest
    if (showBusy) setRefreshing(true)
    try {
      const next = await loadAllPages(api.inferenceTasks)
      if (currentRequest !== requestId.current) return
      setTasks(next)
      setError('')
    } catch (reason) {
      if (currentRequest !== requestId.current) return
      setError(reason instanceof Error ? reason.message : '视频流列表加载失败')
    } finally {
      if (currentRequest === requestId.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }, [])

  useEffect(() => {
    void loadTasks()
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void loadTasks()
    }, 15_000)
    return () => window.clearInterval(timer)
  }, [loadTasks])

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ layout, assignments }))
  }, [assignments, layout])

  const taskById = useMemo(() => new Map(tasks.map((task) => [task.id, task])), [tasks])
  const visibleTasks = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase()
    return tasks
      .filter((task) => PLAYABLE_STATUSES.has(task.status) && task.previewCapability.state === 'available')
      .filter((task) => !normalized || `${task.name} ${task.inputUri}`.toLocaleLowerCase().includes(normalized))
      .sort((left, right) => {
        const statusOrder = { running: 0, degraded: 1, deploying: 2 } as const
        const statusDifference = statusOrder[left.status as keyof typeof statusOrder] - statusOrder[right.status as keyof typeof statusOrder]
        return statusDifference || left.name.localeCompare(right.name, 'zh-CN')
      })
  }, [query, tasks])

  const selectLayout = (next: WallLayout) => {
    setLayout(next)
    setSelectedIndex((current) => Math.min(current, next - 1))
  }

  const assignTask = (taskId: string) => {
    setAssignments((current) => current.map((value, index) => index === selectedIndex ? taskId : value))
  }

  const clearSelected = () => {
    setAssignments((current) => current.map((value, index) => index === selectedIndex ? null : value))
  }

  const selectedTaskId = assignments[selectedIndex]

  return <div className="page-stack video-wall-page">
    <PageHeader
      eyebrow="LIVE MONITORING"
      title="视频监控"
      description="集中查看 RK3588 推理任务的实时标注画面。"
      actions={<>
        <div className="segmented-control wall-layout-control" aria-label="画面布局">
          <button className={layout === 4 ? 'active' : ''} aria-pressed={layout === 4} onClick={() => selectLayout(4)}><Grid2X2 size={15} />四宫格</button>
          <button className={layout === 6 ? 'active' : ''} aria-pressed={layout === 6} onClick={() => selectLayout(6)}><LayoutGrid size={15} />六宫格</button>
        </div>
        <button className="icon-button ghost" title="刷新视频流" aria-label="刷新视频流" disabled={refreshing} onClick={() => void loadTasks(true)}><RefreshCw className={refreshing ? 'spin' : ''} size={17} /></button>
      </>}
    />

    {error && <div className="wall-error" role="alert"><VideoOff size={16} /><span>{error}</span><button onClick={() => void loadTasks(true)}>重试</button></div>}

    <div className="video-wall-layout">
      <section className={`video-wall-grid layout-${layout}`} aria-label={`${layout === 4 ? '四' : '六'}宫格实时画面`}>
        {assignments.slice(0, layout).map((taskId, index) => <WallStreamTile
          key={index}
          index={index}
          selected={selectedIndex === index}
          taskId={taskId}
          task={taskId ? taskById.get(taskId) : undefined}
          loading={loading}
          onSelect={() => setSelectedIndex(index)}
          onClear={() => setAssignments((current) => current.map((value, itemIndex) => itemIndex === index ? null : value))}
        />)}
      </section>

      <aside className="stream-sidebar" aria-label="视频流列表">
        <header className="stream-sidebar-head">
          <div><span><Radio size={15} />视频流</span><small>{visibleTasks.length} 路可播放</small></div>
          <button className="icon-button ghost" title="清空当前格子" aria-label={`清空画面 ${selectedIndex + 1}`} disabled={!selectedTaskId} onClick={clearSelected}><X size={16} /></button>
        </header>
        <label className="search-box stream-search"><Search size={15} /><input aria-label="搜索视频流" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索任务或输入源" /></label>
        <div className="stream-list">
          {loading && !tasks.length ? <EmptyState title="正在加载视频流" message="正在读取推理任务。" /> : visibleTasks.length ? visibleTasks.map((task) => {
            const assigned = selectedTaskId === task.id
            return <button
              key={task.id}
              className={assigned ? 'stream-option assigned' : 'stream-option'}
              aria-label={`将 ${task.name} 播放到画面 ${selectedIndex + 1}`}
              aria-pressed={assigned}
              onClick={() => assignTask(task.id)}
            >
              <span className={`stream-signal ${task.status}`} />
              <span className="stream-option-copy"><strong>{task.name}</strong><small>{task.inputUri}</small></span>
              <span className={`stream-option-state ${task.status}`}>{taskLabels[task.status]}</span>
            </button>
          }) : <EmptyState title="暂无可播放视频流" message={query ? '没有匹配的运行任务。' : '运行推理任务后会出现在这里。'} />}
        </div>
        <footer className="stream-sidebar-foot"><span>当前格子</span><strong>画面 {selectedIndex + 1}</strong></footer>
      </aside>
    </div>
  </div>
}

function WallStreamTile({ index, selected, taskId, task, loading, onSelect, onClear }: {
  index: number
  selected: boolean
  taskId: string | null
  task: InferenceTask | undefined
  loading: boolean
  onSelect: () => void
  onClear: () => void
}) {
  const playable = isPlayable(task)
  const waitingForTask = Boolean(taskId && !task && loading)
  const unavailableLabel = task ? taskLabels[task.status] : taskId ? '任务不可用' : '未分配视频流'
  const displayState = playable ? '可播放' : unavailableLabel
  const displayError = task?.previewCapability.reason ?? task?.errorMessage ?? ''
  const tone: StatusTone = !task ? 'neutral' : playable ? 'success' : taskTone(task.status)

  return <article className={selected ? 'video-wall-tile selected' : 'video-wall-tile'}>
    <header className="wall-tile-head">
      <button className="wall-tile-selector" aria-label={`选择画面 ${index + 1}`} aria-pressed={selected} onClick={onSelect}>
        <span className="wall-slot">{String(index + 1).padStart(2, '0')}</span>
        <span className="wall-task-copy"><strong>{task?.name ?? (taskId ? '任务不可用' : `画面 ${index + 1}`)}</strong><small>{selected ? '已选中' : '监控画面'}</small></span>
      </button>
      <StatusBadge tone={tone}>{displayState}</StatusBadge>
    </header>

    <div className="wall-video-button" onClick={onSelect}>
      {playable && task ? <InferenceStreamPlayer task={task} compact /> : <span className="wall-video-overlay">
        {taskId ? <VideoOff size={26} /> : <Video size={26} />}
        <strong>{waitingForTask ? '正在读取任务' : displayState}</strong>
        {displayError && <small>{displayError}</small>}
      </span>}
    </div>

    <footer className="wall-tile-foot">
      <span title={task?.inputUri}>{task?.inputUri ?? (taskId ? '该任务已不存在或不可访问' : '未绑定视频流')}</span>
      {taskId && <button className="icon-button ghost" title="清空画面" aria-label={`清空画面 ${index + 1}`} onClick={onClear}><X size={14} /></button>}
    </footer>
  </article>
}
