import { useEffect, useMemo } from 'react'
import { LoaderCircle, Radio, RefreshCw, VideoOff } from 'lucide-react'

import { normalizeInferenceAnalytics } from './InferenceBusinessFields'
import { StatusBadge } from '../components'
import { taskAnalytics } from '../inferenceGraph'
import { useInferenceStreamPlayer, type InferencePlayerState } from '../media/useInferenceStreamPlayer'
import type { InferenceTask, StatusTone } from '../types'

const stateLabels: Record<InferencePlayerState, string> = {
  unsupported: '不支持实时画面',
  waiting_publish: '等待视频发布',
  connecting: '正在连接',
  live: '实时画面',
  metadata_degraded: '视频正常，结果中断',
  reconnecting: '正在重连',
  unauthorized: '播放授权失败',
  codec_unsupported: '浏览器不支持该编码',
  stopped: '已停止',
}

const diagnosticLabels: Record<string, string> = {
  mixed_content: 'HTTPS 页面不能连接未加密 WebSocket，请配置 WSS 和证书。',
  browser_unsupported: '当前浏览器不支持低延迟 FLV 播放。',
  media_migration_required: '该任务仍使用旧媒体配置，请先选择媒体网关和流名称。',
  zlm_sei_disabled: '任务未启用 RTSP + SEI 发布。',
  input_not_rtsp: '只有 RTSP 输入支持实时画面。',
  decoder_not_rkmpp: '实时画面要求使用 RKMPP 解码。',
  media_gateway_offline: '媒体网关离线或 Hook 尚未通过校验。',
  h265_unsupported: '当前浏览器不支持原始 H.265 播放，平台不会自动转码。',
  metadata_stale: '超过 1 秒未收到有效 SEI，叠加层已清除。',
  stream_not_published: '节点尚未向媒体网关发布该视频流。',
}

export function InferenceStreamPlayer({ task, compact = false }: {
  task: InferenceTask
  compact?: boolean
}) {
  const analytics = useMemo(() => normalizeInferenceAnalytics(taskAnalytics(task)), [task])
  const options = useMemo(() => ({
    taskId: task.id,
    revision: task.configRevision,
    capability: task.previewCapability,
    areas: analytics.areas,
    lines: analytics.lines,
  }), [analytics.areas, analytics.lines, task.configRevision, task.id, task.previewCapability])
  const player = useInferenceStreamPlayer(options)

  useEffect(() => {
    const canvas = player.canvasRef.current
    if (!canvas) return
    const resize = () => {
      const bounds = canvas.getBoundingClientRect()
      const ratio = Math.max(1, window.devicePixelRatio || 1)
      const width = Math.max(1, Math.round(bounds.width * ratio))
      const height = Math.max(1, Math.round(bounds.height * ratio))
      if (canvas.width !== width) canvas.width = width
      if (canvas.height !== height) canvas.height = height
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [player.canvasRef])

  const tone = stateTone(player.state)
  const diagnostic = player.diagnostic ? diagnosticLabels[player.diagnostic] ?? boundedDiagnostic(player.diagnostic) : ''
  const showBlockingOverlay = !['live', 'metadata_degraded'].includes(player.state)

  return <div className={compact ? 'inference-stream-player compact' : 'inference-stream-player'}>
    <div className="inference-stream-surface" aria-label={`推理实时画面 ${task.name}`}>
      <video ref={player.videoRef} muted playsInline />
      <canvas ref={player.canvasRef} aria-hidden="true" />
      {showBlockingOverlay && <div className="stream-state-overlay">
        {['connecting', 'reconnecting'].includes(player.state)
          ? <LoaderCircle className="spin" size={compact ? 23 : 30} />
          : <VideoOff size={compact ? 23 : 30} />}
        <strong>{stateLabels[player.state]}</strong>
        {diagnostic && <small>{diagnostic}</small>}
      </div>}
      {player.state === 'metadata_degraded' && <div className="stream-metadata-warning" role="status">{stateLabels[player.state]}</div>}
    </div>
    <footer className="inference-stream-meta">
      <span><Radio size={13} /><StatusBadge tone={tone}>{stateLabels[player.state]}</StatusBadge></span>
      <span className="stream-transport">WS-FLV · 原始编码 · SEI</span>
      {!['unsupported', 'unauthorized', 'codec_unsupported', 'stopped'].includes(player.state)
        && <button className="icon-button ghost" title="重新连接" aria-label={`重新连接 ${task.name}`} onClick={player.retry}><RefreshCw size={14} /></button>}
    </footer>
  </div>
}

function stateTone(state: InferencePlayerState): StatusTone {
  if (state === 'live') return 'success'
  if (['connecting', 'reconnecting', 'waiting_publish', 'metadata_degraded'].includes(state)) return 'warning'
  if (['unauthorized', 'codec_unsupported'].includes(state)) return 'danger'
  return 'neutral'
}

function boundedDiagnostic(value: string): string {
  return /^[A-Za-z0-9_.-]{1,80}$/.test(value) ? value : '播放状态异常'
}
