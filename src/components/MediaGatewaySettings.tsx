import { useCallback, useEffect, useState } from 'react'
import { Activity, Edit3, Plus, Power, RadioTower, Trash2 } from 'lucide-react'

import { api } from '../api/client'
import { formatTime } from '../api/presentation'
import { Button, ConfirmDialog, EmptyState, Modal, StatusBadge } from '../components'
import type { MediaGateway, MediaGatewayInput, StatusTone } from '../types'

const emptyGateway: MediaGatewayInput = {
  name: '',
  enabled: true,
  publishHost: '',
  rtspPort: 8554,
  playbackHost: '',
  wsPort: 8081,
  apiHost: '',
  apiPort: 80,
  app: 'live',
  apiSecret: '',
  hookIdentity: '',
}

export function MediaGatewaySettings({ notify }: { notify: (message: string) => void }) {
  const [gateways, setGateways] = useState<MediaGateway[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState<MediaGateway | null>(null)
  const [form, setForm] = useState<MediaGatewayInput>(emptyGateway)
  const [open, setOpen] = useState(false)
  const [busyId, setBusyId] = useState('')
  const [saving, setSaving] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<MediaGateway | null>(null)

  const load = useCallback(async () => {
    try {
      setGateways(await api.mediaGateways())
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '媒体网关加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const openCreate = () => {
    setEditing(null)
    setForm(emptyGateway)
    setOpen(true)
  }

  const openEdit = (gateway: MediaGateway) => {
    setEditing(gateway)
    setForm({
      name: gateway.name,
      enabled: gateway.enabled,
      publishHost: gateway.publishHost,
      rtspPort: gateway.rtspPort,
      playbackHost: gateway.playbackHost,
      wsPort: gateway.wsPort,
      apiHost: gateway.apiHost,
      apiPort: gateway.apiPort,
      app: gateway.app,
      apiSecret: '',
      hookIdentity: '',
    })
    setOpen(true)
  }

  const save = async () => {
    const validation = validate(form, Boolean(editing))
    if (validation) return notify(validation)
    setSaving(true)
    try {
      const payload = normalized(form)
      if (editing) await api.updateMediaGateway(editing.id, payload)
      else await api.createMediaGateway(payload)
      await load()
      setOpen(false)
      notify(editing ? '媒体网关配置已更新，请重新探测' : '媒体网关已创建，请完成探测')
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '媒体网关保存失败')
    } finally {
      setSaving(false)
    }
  }

  const probe = async (gateway: MediaGateway) => {
    setBusyId(gateway.id)
    try {
      const result = await api.probeMediaGateway(gateway.id)
      await load()
      notify(result.status === 'online' ? `媒体网关「${gateway.name}」已在线` : result.lastError ?? '媒体网关探测未通过')
    } catch (reason) {
      await load()
      notify(reason instanceof Error ? reason.message : '媒体网关探测失败')
    } finally {
      setBusyId('')
    }
  }

  const toggle = async (gateway: MediaGateway) => {
    setBusyId(gateway.id)
    try {
      await api.updateMediaGateway(gateway.id, normalized({
        name: gateway.name,
        enabled: !gateway.enabled,
        publishHost: gateway.publishHost,
        rtspPort: gateway.rtspPort,
        playbackHost: gateway.playbackHost,
        wsPort: gateway.wsPort,
        apiHost: gateway.apiHost,
        apiPort: gateway.apiPort,
        app: gateway.app,
      }))
      await load()
      notify(`媒体网关「${gateway.name}」已${gateway.enabled ? '停用' : '启用'}`)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '媒体网关状态更新失败')
    } finally {
      setBusyId('')
    }
  }

  const remove = async () => {
    if (!deleteTarget) return
    setBusyId(deleteTarget.id)
    try {
      await api.deleteMediaGateway(deleteTarget.id)
      await load()
      notify(`媒体网关「${deleteTarget.name}」已删除`)
      setDeleteTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '媒体网关删除失败')
    } finally {
      setBusyId('')
    }
  }

  return <section className="panel table-panel media-gateway-panel">
    <header className="panel-heading gateway-heading">
      <div><RadioTower size={18} /><span><strong>媒体网关</strong><small>RTSP + SEI 发布与浏览器直连播放</small></span></div>
      <Button variant="secondary" icon={<Plus size={15} />} onClick={openCreate}>新增网关</Button>
    </header>
    {error && <div className="api-banner danger">{error}<Button variant="quiet" onClick={() => void load()}>重试</Button></div>}
    {loading && !gateways.length ? <EmptyState title="正在加载媒体网关" message="正在读取媒体控制平面。" /> : gateways.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>网关</th><th>节点发布</th><th>浏览器播放</th><th>平台控制</th><th>密钥</th><th>状态</th><th>最近探测</th><th aria-label="操作" /></tr></thead><tbody>{gateways.map((gateway) => <tr key={gateway.id}>
      <td><strong>{gateway.name}</strong><small>{gateway.builtin ? '内置服务' : '外部服务'} · app/{gateway.app}</small></td>
      <td><code className="service-endpoint">{gateway.publishHost}:{gateway.rtspPort}</code></td>
      <td><code className="service-endpoint">{gateway.playbackHost}:{gateway.wsPort}</code></td>
      <td><code className="service-endpoint">{gateway.apiHost}:{gateway.apiPort}</code></td>
      <td><span className="gateway-secret-state">API {gateway.apiSecretConfigured ? '已配置' : '未配置'}<br />Hook {gateway.hookIdentityConfigured ? '已配置' : '未配置'}</span></td>
      <td><StatusBadge tone={gatewayTone(gateway)}>{gatewayLabel(gateway)}</StatusBadge>{gateway.lastError && <small className="node-error" title={gateway.lastError}>{gateway.lastError}</small>}</td>
      <td className="muted-cell">{gateway.lastProbeAt ? formatTime(gateway.lastProbeAt) : '尚未探测'}</td>
      <td><div className="row-actions"><button className="icon-button ghost" title="探测网关" aria-label={`探测媒体网关 ${gateway.name}`} disabled={busyId === gateway.id || !gateway.enabled} onClick={() => void probe(gateway)}><Activity size={16} /></button><button className="icon-button ghost" title={gateway.enabled ? '停用网关' : '启用网关'} aria-label={`${gateway.enabled ? '停用' : '启用'}媒体网关 ${gateway.name}`} disabled={busyId === gateway.id} onClick={() => void toggle(gateway)}><Power size={16} /></button><button className="icon-button ghost" title="编辑网关" aria-label={`编辑媒体网关 ${gateway.name}`} onClick={() => openEdit(gateway)}><Edit3 size={16} /></button>{!gateway.builtin && <button className="icon-button ghost danger-action" title="删除网关" aria-label={`删除媒体网关 ${gateway.name}`} onClick={() => setDeleteTarget(gateway)}><Trash2 size={16} /></button>}</div></td>
    </tr>)}</tbody></table></div> : <EmptyState title="暂无媒体网关" message="配置网关后，推理任务才能选择 RTSP + SEI 实时播放。" />}

    <Modal open={open} width="large" title={editing ? '编辑媒体网关' : '新增媒体网关'} description="三个地址独立配置，不从节点发布地址推断浏览器可达地址。" onClose={() => !saving && setOpen(false)} footer={<><Button variant="secondary" onClick={() => setOpen(false)} disabled={saving}>取消</Button><Button onClick={() => void save()} disabled={saving}>{saving ? '保存中…' : '保存网关'}</Button></>}>
      <div className="form-sections">
        <section className="form-section"><h3><RadioTower size={17} />网关标识</h3><div className="form-grid three-columns"><label className="field"><span>名称 <b>*</b></span><input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label><label className="field"><span>应用名</span><input value={form.app} maxLength={64} onChange={(event) => setForm({ ...form, app: event.target.value })} /></label><label className="toggle-card"><input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} /><span><strong>启用网关</strong><small>探测通过后可供任务选择</small></span></label></div></section>
        <GatewayAddress title="节点发布地址" host={form.publishHost} port={form.rtspPort} protocol="RTSP" onHost={(publishHost) => setForm({ ...form, publishHost })} onPort={(rtspPort) => setForm({ ...form, rtspPort })} />
        <GatewayAddress title="浏览器播放地址" host={form.playbackHost} port={form.wsPort} protocol="WS-FLV" onHost={(playbackHost) => setForm({ ...form, playbackHost })} onPort={(wsPort) => setForm({ ...form, wsPort })} />
        <GatewayAddress title="平台控制地址" host={form.apiHost} port={form.apiPort} protocol="HTTP API" onHost={(apiHost) => setForm({ ...form, apiHost })} onPort={(apiPort) => setForm({ ...form, apiPort })} />
        <section className="form-section"><h3><Power size={17} />写入型密钥</h3><div className="form-grid two-columns"><label className="field"><span>API Secret {editing?.apiSecretConfigured && <small>已配置，留空保持不变</small>}</span><input type="password" autoComplete="new-password" value={form.apiSecret ?? ''} onChange={(event) => setForm({ ...form, apiSecret: event.target.value })} /></label><label className="field"><span>Hook Identity {editing?.hookIdentityConfigured && <small>已配置，留空保持不变</small>}</span><input type="password" autoComplete="new-password" value={form.hookIdentity ?? ''} onChange={(event) => setForm({ ...form, hookIdentity: event.target.value })} /></label></div></section>
      </div>
    </Modal>
    <ConfirmDialog open={Boolean(deleteTarget)} title="删除媒体网关" description={deleteTarget ? `确定删除「${deleteTarget.name}」吗？存在任务绑定时平台会拒绝删除。` : ''} confirmLabel="删除网关" busy={Boolean(deleteTarget && busyId === deleteTarget.id)} onClose={() => setDeleteTarget(null)} onConfirm={() => void remove()} />
  </section>
}

function GatewayAddress({ title, host, port, protocol, onHost, onPort }: { title: string; host: string; port: number; protocol: string; onHost: (value: string) => void; onPort: (value: number) => void }) {
  return <section className="form-section"><h3><Activity size={17} />{title}</h3><div className="form-grid two-columns"><label className="field"><span>宿主机 IP / 域名 <b>*</b></span><input value={host} placeholder="192.168.1.10" onChange={(event) => onHost(event.target.value)} /></label><label className="field"><span>{protocol} 端口 <b>*</b></span><input type="number" min={1} max={65535} value={port} onChange={(event) => onPort(Number(event.target.value))} /></label></div></section>
}

function normalized(value: MediaGatewayInput): MediaGatewayInput {
  const result: MediaGatewayInput = { ...value, name: value.name.trim(), app: value.app.trim(), publishHost: value.publishHost.trim(), playbackHost: value.playbackHost.trim(), apiHost: value.apiHost.trim() }
  if (!result.apiSecret?.trim()) delete result.apiSecret
  if (!result.hookIdentity?.trim()) delete result.hookIdentity
  return result
}

function validate(value: MediaGatewayInput, editing: boolean): string {
  if (!value.name.trim() || !value.publishHost.trim() || !value.playbackHost.trim() || !value.apiHost.trim()) return '请完整填写网关名称和三个服务地址'
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(value.app.trim())) return '应用名格式无效'
  if (![value.rtspPort, value.wsPort, value.apiPort].every((port) => Number.isInteger(port) && port >= 1 && port <= 65535)) return '服务端口必须在 1 到 65535 之间'
  if (!editing && (!value.apiSecret?.trim() || !value.hookIdentity?.trim())) return '首次创建必须填写 API Secret 和 Hook Identity'
  return ''
}

function gatewayTone(gateway: MediaGateway): StatusTone {
  if (gateway.status === 'online') return 'success'
  if (gateway.status === 'probing') return 'warning'
  if (gateway.status === 'error') return 'danger'
  return 'neutral'
}

function gatewayLabel(gateway: MediaGateway): string {
  if (!gateway.enabled || gateway.status === 'disabled') return '已停用'
  return { probing: '探测中', online: '在线', error: '异常' }[gateway.status] ?? gateway.status
}
