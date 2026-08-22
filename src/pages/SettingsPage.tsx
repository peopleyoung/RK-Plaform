import { useEffect, useState } from 'react'
import { Activity, BrainCircuit, Check, Container, Copy, Cpu, Download, Edit3, KeyRound, Network, RefreshCw, ServerCog, Trash2, Zap } from 'lucide-react'
import { usePlatform } from '../api/PlatformContext'
import { api } from '../api/client'
import { formatTime } from '../api/presentation'
import { AddButton, Button, ConfirmDialog, EmptyState, Modal, PageHeader, StatusBadge } from '../components'
import { MediaGatewaySettings } from '../components/MediaGatewaySettings'
import type { ServiceEndpoint, ServiceEndpointCreated, ServiceEndpointEnrollment, ServiceEndpointEnrollmentStatus, ServiceEndpointInput } from '../types'

const inferenceCapabilities = [
  { id: 'yolo_dfl_split_v1', label: 'YOLO DFL 检测' },
  { id: 'deeplab_logits_v1', label: 'DeepLab 语义分割' },
  { id: 'ppocr_db_det_v1', label: 'PPOCR 文本检测' },
  { id: 'ppocr_ctc_rec_v1', label: 'PPOCR 文本识别' },
]

const emptyForm: ServiceEndpointInput = {
  name: '',
  kind: 'trainer',
  mode: 'direct',
  scheme: 'http',
  host: '',
  port: 10081,
  accelerator: 'cpu',
  capabilities: ['yolo-detect', 'deeplabv3plus'],
  enabled: true,
}

type DeploymentCredential = ServiceEndpointCreated | ServiceEndpointEnrollment

function kindLabel(endpoint: Pick<ServiceEndpoint, 'kind' | 'accelerator'>) {
  if (endpoint.kind === 'converter') return 'RK3588 转换'
  if (endpoint.kind === 'inference') return 'RK3588 推理'
  return endpoint.accelerator === 'cuda' ? 'CUDA 训练' : 'CPU 训练'
}

function probeTone(status: string) {
  if (status === 'online') return 'success' as const
  if (status === 'offline' || status === 'error') return 'danger' as const
  return 'neutral' as const
}

function probeLabel(status: string) {
  return { online: '在线', offline: '离线', error: '配置错误', unprobed: '未探测' }[status] ?? status
}

function enrollmentStatus(endpoint: ServiceEndpoint): ServiceEndpointEnrollmentStatus {
  return endpoint.enrollmentStatus ?? (endpoint.tokenConfigured ? 'enrolled' : 'pending')
}

function endpointStatus(endpoint: ServiceEndpoint) {
  if (!endpoint.enabled) return { label: '已停用', tone: 'neutral' as const }
  if (endpoint.mode === 'pull') return { label: '旧版兼容', tone: 'neutral' as const }
  const status = enrollmentStatus(endpoint)
  if (status === 'pending') {
    const expired = endpoint.enrollmentExpiresAt
      ? Date.parse(endpoint.enrollmentExpiresAt) < Date.now()
      : true
    return { label: expired ? '注册错误' : '待部署', tone: 'warning' as const }
  }
  if (status === 'claimed') {
    return { label: '已领取/待探测', tone: 'warning' as const }
  }
  return { label: probeLabel(endpoint.probeStatus), tone: probeTone(endpoint.probeStatus) }
}

function credentialEndpointId(credential: DeploymentCredential) {
  return 'endpointId' in credential ? credential.endpointId : credential.id
}

export function SettingsPage({ notify }: { notify: (message: string) => void }) {
  const { profiles, serviceEndpoints, loading, error, refresh } = usePlatform()
  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState<ServiceEndpoint | null>(null)
  const [form, setForm] = useState<ServiceEndpointInput>(emptyForm)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [probingId, setProbingId] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<ServiceEndpoint | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [credential, setCredential] = useState<DeploymentCredential | null>(null)
  const [reissueTarget, setReissueTarget] = useState<ServiceEndpoint | null>(null)
  const [reissuing, setReissuing] = useState(false)

  useEffect(() => () => setCredential(null), [])

  const openCreate = () => {
    setEditing(null)
    setForm({ ...emptyForm, capabilities: ['yolo-detect', 'deeplabv3plus'] })
    setOpen(true)
  }

  const openEdit = (endpoint: ServiceEndpoint) => {
    setEditing(endpoint)
    setForm({
      name: endpoint.name,
      kind: endpoint.kind,
      mode: endpoint.mode,
      endpoint: endpoint.endpoint,
      scheme: endpoint.scheme,
      host: endpoint.host,
      port: endpoint.port,
      accelerator: endpoint.accelerator,
      capabilities: endpoint.capabilities,
      enabled: endpoint.enabled,
    })
    setOpen(true)
  }

  const closeForm = () => {
    setOpen(false)
    setEditing(null)
    setForm(emptyForm)
  }

  const setKind = (kind: ServiceEndpointInput['kind']) => {
    if (kind === 'inference') {
      setForm({ ...form, kind, mode: 'direct', accelerator: 'rk3588', port: 10082, capabilities: inferenceCapabilities.map((item) => item.id) })
      return
    }
    if (kind === 'converter') {
      setForm({ ...form, kind, mode: 'direct', accelerator: 'rk3588', port: 10081, capabilities: profiles.map((profile) => profile.id) })
      return
    }
    setForm({ ...form, kind, accelerator: 'cpu', port: 10081, capabilities: ['yolo-detect', 'deeplabv3plus'] })
  }

  const options = form.kind === 'inference'
    ? inferenceCapabilities
    : profiles.map((profile) => ({ id: profile.id, label: profile.label }))

  const toggleCapability = (capability: string) => {
    const capabilities = form.capabilities.includes(capability)
      ? form.capabilities.filter((item) => item !== capability)
      : [...form.capabilities, capability]
    setForm({ ...form, capabilities })
  }

  const normalizedPayload = (): ServiceEndpointInput => ({
    ...form,
    name: form.name.trim(),
    host: form.host.trim(),
    endpoint: form.mode === 'pull' ? form.endpoint?.trim() : '',
    token: undefined,
  })

  const validate = () => {
    if (!form.name.trim() || !form.capabilities.length) return '请填写节点名称并至少选择一项能力'
    if (form.mode === 'direct' && !form.host.trim()) return '请填写节点宿主机 IP 或域名'
    if (form.mode === 'pull' && !form.endpoint?.trim()) return '兼容模式需要填写完整服务接口'
    return ''
  }

  const testConnection = async () => {
    if (!editing || enrollmentStatus(editing) !== 'enrolled') return
    const validationError = validate()
    if (validationError) return notify(validationError)
    setTesting(true)
    try {
      await api.testServiceEndpointUpdate(editing.id, normalizedPayload())
      notify('节点连接和能力校验通过')
      await refresh()
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '节点连接测试失败')
    } finally {
      setTesting(false)
    }
  }

  const save = async () => {
    const validationError = validate()
    if (validationError) return notify(validationError)
    setSaving(true)
    try {
      const payload = normalizedPayload()
      if (editing) {
        await api.updateServiceEndpoint(editing.id, payload)
        notify('节点配置已更新')
      } else {
        const created = await api.createServiceEndpoint(payload)
        if (created.enrollmentToken) {
          setCredential(created)
          notify('节点已登记，请保存部署凭据')
        } else {
          notify('旧版兼容节点已保存')
        }
      }
      await refresh()
      closeForm()
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '保存节点失败')
    } finally {
      setSaving(false)
    }
  }

  const probe = async (endpoint: ServiceEndpoint) => {
    setProbingId(endpoint.id)
    try {
      await api.probeServiceEndpoint(endpoint.id)
      await refresh()
      notify(`节点「${endpoint.name}」探测完成`)
    } catch (reason) {
      await refresh()
      notify(reason instanceof Error ? reason.message : '节点探测失败')
    } finally {
      setProbingId('')
    }
  }

  const reissue = async () => {
    if (!reissueTarget) return
    setReissuing(true)
    try {
      const issued = await api.reissueServiceEndpointEnrollment(reissueTarget.id)
      setReissueTarget(null)
      setCredential(issued)
      await refresh()
      notify('新的部署凭据已生成，旧注册码已失效')
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '重新签发失败')
    } finally {
      setReissuing(false)
    }
  }

  const remove = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await api.deleteServiceEndpoint(deleteTarget.id)
      await refresh()
      notify(`节点「${deleteTarget.name}」已删除`)
      setDeleteTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '删除节点失败')
    } finally {
      setDeleting(false)
    }
  }

  const copyValue = async (value: string, label: string) => {
    try {
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(value)
      } else {
        const textarea = document.createElement('textarea')
        textarea.value = value
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand('copy')
        textarea.remove()
      }
      notify(`${label}已复制`)
    } catch {
      notify(`${label}复制失败`)
    }
  }

  const downloadCredential = () => {
    if (!credential?.enrollmentToken) return
    const endpointId = credentialEndpointId(credential)
    const content = [
      `RKNODE_ENDPOINT_ID=${endpointId}`,
      'RKNODE_PLATFORM_URL=http://CENTRAL_SERVER_IP:8000',
      'RKNODE_ENROLLMENT_TOKEN_PATH=./secrets/node-enrollment-token',
      '',
      `enrollment_token=${credential.enrollmentToken}`,
      `expires_at=${credential.enrollmentExpiresAt}`,
    ].join('\n')
    const url = URL.createObjectURL(new Blob([`${content}\n`], { type: 'text/plain;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `rknode-${endpointId.replace(/[^a-zA-Z0-9_-]/g, '-')}-enrollment.txt`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    notify('部署凭据已下载')
  }

  const schedulableCount = serviceEndpoints.filter((endpoint) => (
    endpoint.enabled
    && enrollmentStatus(endpoint) === 'enrolled'
    && endpoint.probeStatus === 'online'
  )).length

  return <div className="page-stack">
    <PageHeader title="系统设置" description="先登记节点宿主机地址，再使用平台生成的一次性凭据部署训练、转换或推理服务。" actions={<AddButton onClick={openCreate}>新增节点</AddButton>} />
    {error && <div className="api-banner danger">{error}<Button variant="quiet" onClick={() => void refresh()}>重试</Button></div>}
    <section className="settings-principles">
      <div><span className="summary-icon teal"><Network size={18} /></span><p><strong>统一接入</strong><small>三类节点均先在平台注册宿主机地址和端口。</small></p></div>
      <div><span className="summary-icon amber"><KeyRound size={18} /></span><p><strong>一次凭据</strong><small>注册码只展示一次，节点领取后持久化独立 Token。</small></p></div>
      <div><span className="summary-icon blue"><Activity size={18} /></span><p><strong>探测激活</strong><small>领取凭据后仍需首次健康探测通过才进入调度。</small></p></div>
    </section>
    <MediaGatewaySettings notify={notify} />
    <section className="panel table-panel">
      <div className="table-meta"><span>共 <strong>{serviceEndpoints.length}</strong> 个节点</span><span>{schedulableCount} 个可调度</span></div>
      {loading && !serviceEndpoints.length ? <EmptyState title="正在加载节点" message="正在连接平台 API。" /> : serviceEndpoints.length ? <div className="table-scroll"><table className="data-table service-table"><thead><tr><th>节点</th><th>地址</th><th>类型</th><th>能力</th><th>状态</th><th>注册</th><th>最近探测</th><th aria-label="操作" /></tr></thead><tbody>{serviceEndpoints.map((endpoint) => {
        const status = endpointStatus(endpoint)
        const enrollment = enrollmentStatus(endpoint)
        const isLegacyEnrollment = enrollment === 'enrolled' && !endpoint.enrollmentClaimedAt
        const canReissue = endpoint.mode === 'direct' && (enrollment === 'pending' || enrollment === 'claimed' || isLegacyEnrollment)
        const canProbe = endpoint.mode === 'direct' && enrollment === 'enrolled'
        return <tr key={endpoint.id}>
          <td><strong>{endpoint.name}</strong><small>{endpoint.mode === 'direct' ? '直连模式' : '旧版兼容'}</small></td>
          <td><code className="service-endpoint">{endpoint.endpoint}</code>{endpoint.lastError && <small className="node-error" title={endpoint.lastError}>{endpoint.lastError}</small>}</td>
          <td><span className="node-inline">{endpoint.kind === 'converter' ? <Container size={15} /> : endpoint.kind === 'inference' ? <BrainCircuit size={15} /> : endpoint.accelerator === 'cuda' ? <Zap size={15} /> : <Cpu size={15} />}{kindLabel(endpoint)}</span></td>
          <td><div className="capability-list">{endpoint.capabilities.map((capability) => <span key={capability}>{capability}</span>)}</div></td>
          <td><StatusBadge tone={status.tone}>{status.label}</StatusBadge></td>
          <td><StatusBadge tone={endpoint.mode === 'pull' ? 'neutral' : enrollment === 'enrolled' ? 'success' : 'warning'}>{endpoint.mode === 'pull' ? '旧版兼容' : enrollment === 'pending' ? '待领取' : enrollment === 'claimed' ? '已领取' : '已注册'}</StatusBadge></td>
          <td className="muted-cell">{endpoint.lastProbeAt ? formatTime(endpoint.lastProbeAt) : '尚未探测'}</td>
          <td><div className="row-actions">{canReissue && <button className="icon-button ghost" aria-label={`${isLegacyEnrollment ? '迁移统一接入' : '重新签发注册码'} ${endpoint.name}`} title={isLegacyEnrollment ? '迁移统一接入' : '重新签发注册码'} onClick={() => setReissueTarget(endpoint)}><RefreshCw size={16} /></button>}<button className="icon-button ghost" aria-label={`探测节点 ${endpoint.name}`} title={canProbe ? '立即探测' : '节点注册后可探测'} disabled={probingId === endpoint.id || !canProbe} onClick={() => void probe(endpoint)}><Activity size={16} /></button><button className="icon-button ghost" aria-label={`编辑节点 ${endpoint.name}`} title="编辑节点" onClick={() => openEdit(endpoint)}><Edit3 size={16} /></button><button className="icon-button ghost danger-action" aria-label={`删除节点 ${endpoint.name}`} title="删除节点" onClick={() => setDeleteTarget(endpoint)}><Trash2 size={16} /></button></div></td>
        </tr>
      })}</tbody></table></div> : <EmptyState title="暂无节点" message="先登记节点宿主机 IP 和服务端口，再使用一次性凭据部署节点。" />}
    </section>

    <Modal open={open} title={editing ? '编辑节点' : '新增节点'} description={editing ? '修改节点地址、能力和调度状态。' : '平台保存节点身份后生成短期一次性部署凭据。'} width="large" onClose={() => !saving && !testing && closeForm()} footer={<><Button variant="secondary" onClick={closeForm} disabled={saving || testing}>取消</Button>{editing && enrollmentStatus(editing) === 'enrolled' && <Button variant="secondary" icon={<Activity size={15} />} onClick={() => void testConnection()} disabled={saving || testing}>{testing ? '测试中…' : '测试连接'}</Button>}<Button onClick={() => void save()} disabled={saving || testing}>{saving ? '保存中…' : editing ? '保存修改' : form.mode === 'direct' ? '保存并生成注册码' : '保存兼容节点'}</Button></>}>
      <div className="form-sections">
        <section className="form-section"><h3><ServerCog size={17} />节点标识</h3><div className="form-grid three-columns"><label className="field"><span>节点名称 <b>*</b></span><input value={form.name} disabled={Boolean(editing)} placeholder="rk3588-inference-01" onChange={(event) => setForm({ ...form, name: event.target.value })} /></label><label className="field"><span>节点类型</span><select value={form.kind} disabled={Boolean(editing)} onChange={(event) => setKind(event.target.value as ServiceEndpointInput['kind'])}><option value="trainer">模型训练</option><option value="converter">模型转换</option><option value="inference">模型推理</option></select></label><label className="field"><span>接入模式</span><select value={form.mode} disabled={form.kind === 'inference'} onChange={(event) => setForm({ ...form, mode: event.target.value as ServiceEndpointInput['mode'] })}><option value="direct">直连调度</option><option value="pull">旧版兼容</option></select></label></div></section>
        {form.mode === 'direct' ? <section className="form-section"><h3><Network size={17} />节点服务地址</h3><div className="form-grid three-columns"><label className="field"><span>协议</span><select value={form.scheme} onChange={(event) => setForm({ ...form, scheme: event.target.value as 'http' | 'https' })}><option value="http">HTTP</option><option value="https">HTTPS</option></select></label><label className="field"><span>节点宿主机 IP / 域名 <b>*</b></span><input value={form.host} placeholder="172.30.82.12" onChange={(event) => setForm({ ...form, host: event.target.value })} /></label><label className="field"><span>服务端口 <b>*</b></span><input type="number" min={1} max={65535} value={form.port} onChange={(event) => setForm({ ...form, port: Number(event.target.value) })} /></label></div><p className="field-help">此地址供中央平台访问节点；节点端 RKNODE_PLATFORM_URL 填写节点可访问的中央平台地址。</p></section> : <section className="form-section"><h3><Network size={17} />兼容接口</h3><label className="field"><span>完整服务接口 <b>*</b></span><input value={form.endpoint ?? ''} placeholder="http://trainer.example:9000" onChange={(event) => setForm({ ...form, endpoint: event.target.value })} /></label></section>}
        <section className="form-section"><h3><KeyRound size={17} />运行与调度</h3><div className="form-grid two-columns"><label className="field"><span>加速器</span><select value={form.accelerator} disabled={form.kind !== 'trainer'} onChange={(event) => setForm({ ...form, accelerator: event.target.value as ServiceEndpointInput['accelerator'] })}><option value="cpu">CPU</option><option value="cuda">NVIDIA CUDA</option>{form.kind !== 'trainer' && <option value="rk3588">RK3588</option>}</select></label><label className="toggle-card"><input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} /><span><strong>允许调度</strong><small>完成注册和探测后下发任务或 revision</small></span></label></div></section>
        <section className="form-section"><h3><Check size={17} />{form.kind === 'inference' ? '推理适配器' : '模型能力'}</h3><div className="capability-choice-grid">{options.map((option) => <label key={option.id} className={form.capabilities.includes(option.id) ? 'capability-choice active' : 'capability-choice'}><input type="checkbox" checked={form.capabilities.includes(option.id)} onChange={() => toggleCapability(option.id)} /><span><strong>{option.label}</strong><small>{option.id}</small></span><Check size={16} /></label>)}</div></section>
      </div>
    </Modal>

    <Modal open={Boolean(credential)} title="节点部署凭据" description="注册码仅在本窗口显示。关闭前请复制或下载，并写入节点宿主机上的 secret 文件。" width="large" dismissible={false} onClose={() => setCredential(null)} footer={<><Button variant="secondary" icon={<Download size={16} />} onClick={downloadCredential}>下载凭据</Button><Button onClick={() => setCredential(null)}>关闭部署凭据</Button></>}>
      {credential && <div className="enrollment-credential">
        <div className="credential-row"><span>Endpoint ID</span><code>{credentialEndpointId(credential)}</code><button className="icon-button ghost" aria-label="复制 Endpoint ID" title="复制 Endpoint ID" onClick={() => void copyValue(credentialEndpointId(credential), 'Endpoint ID')}><Copy size={16} /></button></div>
        <div className="credential-row credential-secret"><span>一次性注册码</span><code>{credential.enrollmentToken}</code><button className="icon-button ghost" aria-label="复制一次性注册码" title="复制一次性注册码" onClick={() => void copyValue(credential.enrollmentToken ?? '', '一次性注册码')}><Copy size={16} /></button></div>
        <div className="credential-meta"><span>有效期至</span><strong>{formatTime(credential.enrollmentExpiresAt)}</strong></div>
        <div className="credential-variables"><code>RKNODE_ENDPOINT_ID={credentialEndpointId(credential)}</code><code>RKNODE_PLATFORM_URL=http://CENTRAL_SERVER_IP:8000</code><code>RKNODE_ENROLLMENT_TOKEN_PATH=./secrets/node-enrollment-token</code></div>
      </div>}
    </Modal>

    <Modal open={Boolean(reissueTarget)} title={reissueTarget && enrollmentStatus(reissueTarget) === 'enrolled' ? '迁移为统一接入' : '重新签发注册码'} description={reissueTarget ? enrollmentStatus(reissueTarget) === 'enrolled' ? `将暂停「${reissueTarget.name}」的调度并生成一次性迁移凭据。` : `将为「${reissueTarget.name}」生成新的短期注册码。` : ''} onClose={() => !reissuing && setReissueTarget(null)} footer={<><Button variant="secondary" onClick={() => setReissueTarget(null)} disabled={reissuing}>取消</Button><Button variant="danger" icon={<RefreshCw size={15} />} onClick={() => void reissue()} disabled={reissuing}>{reissuing ? '签发中…' : reissueTarget && enrollmentStatus(reissueTarget) === 'enrolled' ? '确认迁移' : '确认重新签发'}</Button></>}>
      <p className="confirm-dialog-copy">旧注册码会立即失效；已经领取的长期节点 Token 不会在此处显示。</p>
    </Modal>
    <ConfirmDialog open={Boolean(deleteTarget)} title="删除节点" description={deleteTarget ? `确定删除「${deleteTarget.name}」吗？平台会停止向该地址调度；推理节点存在运行任务时需先停止任务。` : ''} confirmLabel="删除节点" busy={deleting} onClose={() => !deleting && setDeleteTarget(null)} onConfirm={() => void remove()} />
  </div>
}
