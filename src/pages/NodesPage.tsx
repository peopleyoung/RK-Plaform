import { useMemo, useState } from 'react'
import { Activity, BrainCircuit, Container, Cpu, Network, RefreshCw, Search, ServerCog, Settings, Zap } from 'lucide-react'
import { usePlatform } from '../api/PlatformContext'
import { formatTime } from '../api/presentation'
import { Button, EmptyState, PageHeader, StatusBadge, TablePagination } from '../components'
import type { ServiceEndpoint } from '../types'

type NodeKindFilter = 'all' | ServiceEndpoint['kind']
type NodeStatusFilter = 'all' | 'online' | 'offline' | 'disabled'

const kindLabels: Record<ServiceEndpoint['kind'], string> = {
  trainer: '训练节点',
  converter: '转换节点',
  inference: '推理节点',
}

const kindDescriptions: Record<ServiceEndpoint['kind'], string> = {
  trainer: 'CPU / NVIDIA CUDA 模型训练',
  converter: 'RK3588 模型转换与板端校验',
  inference: 'RK3588 模型部署与视频推理',
}

function metadataNumber(endpoint: ServiceEndpoint, key: string) {
  const value = endpoint.remoteMetadata[key]
  return typeof value === 'number' ? value : 0
}

function metadataString(endpoint: ServiceEndpoint, key: string, fallback = '--') {
  const value = endpoint.remoteMetadata[key]
  return typeof value === 'string' && value ? value : fallback
}

function inferenceRuntime(endpoint: ServiceEndpoint) {
  const diagnostics = endpoint.remoteMetadata.diagnostics
  if (!diagnostics || typeof diagnostics !== 'object' || Array.isArray(diagnostics)) return null
  const runtime = (diagnostics as Record<string, unknown>).inference
  return runtime && typeof runtime === 'object' && !Array.isArray(runtime)
    ? runtime as Record<string, unknown>
    : null
}

function kindIcon(endpoint: Pick<ServiceEndpoint, 'kind' | 'accelerator'>) {
  if (endpoint.kind === 'converter') return <Container size={17} />
  if (endpoint.kind === 'inference') return <BrainCircuit size={17} />
  return endpoint.accelerator === 'cuda' ? <Zap size={17} /> : <Cpu size={17} />
}

function statusTone(endpoint: ServiceEndpoint) {
  if (!endpoint.enabled) return 'neutral' as const
  if (endpoint.probeStatus === 'online') return 'success' as const
  if (endpoint.probeStatus === 'unprobed') return 'warning' as const
  return 'danger' as const
}

function statusLabel(endpoint: ServiceEndpoint) {
  if (!endpoint.enabled) return '已停用'
  return { online: '在线', offline: '离线', error: '配置错误', unprobed: '未探测' }[endpoint.probeStatus] ?? endpoint.probeStatus
}

export function NodesPage({ notify }: { notify: (message: string) => void }) {
  const { serviceEndpoints, loading, error, refresh } = usePlatform()
  const [kindFilter, setKindFilter] = useState<NodeKindFilter>('all')
  const [statusFilter, setStatusFilter] = useState<NodeStatusFilter>('all')
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)

  const counts = useMemo(() => ({
    trainer: serviceEndpoints.filter((endpoint) => endpoint.kind === 'trainer').length,
    converter: serviceEndpoints.filter((endpoint) => endpoint.kind === 'converter').length,
    inference: serviceEndpoints.filter((endpoint) => endpoint.kind === 'inference').length,
    schedulable: serviceEndpoints.filter((endpoint) => endpoint.enabled && endpoint.probeStatus === 'online').length,
  }), [serviceEndpoints])

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return serviceEndpoints.filter((endpoint) => {
      if (kindFilter !== 'all' && endpoint.kind !== kindFilter) return false
      if (statusFilter === 'online' && (!endpoint.enabled || endpoint.probeStatus !== 'online')) return false
      if (statusFilter === 'offline' && (!endpoint.enabled || !['offline', 'error', 'unprobed'].includes(endpoint.probeStatus))) return false
      if (statusFilter === 'disabled' && endpoint.enabled) return false
      if (!needle) return true
      return [endpoint.name, endpoint.endpoint, endpoint.accelerator, ...endpoint.capabilities]
        .some((value) => value.toLowerCase().includes(needle))
    })
  }, [kindFilter, query, serviceEndpoints, statusFilter])

  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize))
  const currentPage = Math.min(page, pageCount)
  const visible = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize)
  const selectKind = (kind: NodeKindFilter) => { setKindFilter(kind); setPage(1) }

  return <div className="page-stack">
    <PageHeader title="算力节点" description="统一查看已接入平台的训练、转换和推理服务；节点配置在系统设置中集中维护。" actions={<><Button variant="secondary" icon={<Settings size={17} />} onClick={() => { window.location.hash = '/settings' }}>管理节点</Button><Button variant="secondary" icon={<RefreshCw size={17} />} onClick={() => void refresh().then(() => notify('三类节点状态已刷新'))}>刷新状态</Button></>} />
    {error && <div className="api-banner danger">{error}<Button variant="quiet" onClick={() => void refresh()}>重试</Button></div>}

    <section className="compute-node-summary" aria-label="节点接入概况">
      <button className={kindFilter === 'trainer' ? 'active trainer' : 'trainer'} onClick={() => selectKind('trainer')}><span><Zap size={19} /></span><div><small>训练节点</small><strong>{counts.trainer}</strong><p>CPU / NVIDIA CUDA</p></div></button>
      <button className={kindFilter === 'converter' ? 'active converter' : 'converter'} onClick={() => selectKind('converter')}><span><Container size={19} /></span><div><small>转换节点</small><strong>{counts.converter}</strong><p>RK3588 · RKNN Toolkit2</p></div></button>
      <button className={kindFilter === 'inference' ? 'active inference' : 'inference'} onClick={() => selectKind('inference')}><span><BrainCircuit size={19} /></span><div><small>推理节点</small><strong>{counts.inference}</strong><p>RK3588 · NPU Runtime</p></div></button>
      <button className={kindFilter === 'all' ? 'active schedulable' : 'schedulable'} onClick={() => selectKind('all')}><span><Activity size={19} /></span><div><small>可调度节点</small><strong>{counts.schedulable} / {serviceEndpoints.length}</strong><p>在线并已启用</p></div></button>
    </section>

    <div className="toolbar compute-node-toolbar">
      <label className="search-box"><Search size={16} /><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1) }} placeholder="搜索节点名称、地址或能力" aria-label="搜索算力节点" /></label>
      <div className="tabs compute-kind-tabs" role="tablist" aria-label="节点类型">
        {([['all', '全部'], ['trainer', '训练'], ['converter', '转换'], ['inference', '推理']] as Array<[NodeKindFilter, string]>).map(([kind, label]) => <button key={kind} className={kindFilter === kind ? 'active' : ''} onClick={() => selectKind(kind)} role="tab" aria-selected={kindFilter === kind}>{label}</button>)}
      </div>
      <span className="toolbar-spacer" />
      <label className="filter-control"><Network size={15} /><select value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value as NodeStatusFilter); setPage(1) }} aria-label="连接状态"><option value="all">全部状态</option><option value="online">在线可调度</option><option value="offline">连接异常</option><option value="disabled">已停用</option></select></label>
    </div>

    <section className="panel table-panel">
      <div className="table-meta"><span>共 <strong>{filtered.length}</strong> 个接入节点</span><span>当前展示 {kindFilter === 'all' ? '全部类型' : kindLabels[kindFilter]}</span></div>
      {loading && !serviceEndpoints.length ? <EmptyState title="正在加载节点" message="正在读取三类服务接入状态。" /> : visible.length ? <><div className="table-scroll"><table className="data-table compute-node-table"><thead><tr><th>节点</th><th>节点类型</th><th>服务地址</th><th>算力 / 运行版本</th><th>任务负载</th><th>能力</th><th>连接状态</th><th>最近探测</th></tr></thead><tbody>{visible.map((endpoint) => {
        const activeJobs = metadataNumber(endpoint, 'activeJobs')
        const maxConcurrency = metadataNumber(endpoint, 'maxConcurrency')
        const runtime = inferenceRuntime(endpoint)
        const runtimeVersion = endpoint.kind === 'inference' && typeof runtime?.runtimeVersion === 'string' ? runtime.runtimeVersion : metadataString(endpoint, 'version')
        const actualRevision = endpoint.kind === 'inference' && typeof runtime?.actualRevision === 'number' ? runtime.actualRevision : null
        return <tr key={endpoint.id}>
          <td><strong>{endpoint.name}</strong><small>{endpoint.mode === 'direct' ? '直连调度' : '兼容接入'}</small></td>
          <td><span className={`compute-node-kind ${endpoint.kind}`}>{kindIcon(endpoint)}<span><strong>{kindLabels[endpoint.kind]}</strong><small>{kindDescriptions[endpoint.kind]}</small></span></span></td>
          <td><code className="service-endpoint">{endpoint.endpoint}</code>{endpoint.lastError && <small className="node-error" title={endpoint.lastError}>{endpoint.lastError}</small>}</td>
          <td><strong>{endpoint.accelerator === 'cuda' ? 'NVIDIA CUDA' : endpoint.accelerator.toUpperCase()}</strong><small>{runtimeVersion}{actualRevision !== null ? ` · revision ${actualRevision}` : ''}</small></td>
          <td><strong>{activeJobs} / {maxConcurrency || '--'}</strong><small>{endpoint.kind === 'inference' ? '运行实例 / 上限' : '活动任务 / 并发'}</small></td>
          <td><div className="capability-list">{endpoint.capabilities.map((capability) => <span key={capability}>{capability}</span>)}</div></td>
          <td><StatusBadge tone={statusTone(endpoint)}>{statusLabel(endpoint)}</StatusBadge><small>{endpoint.tokenConfigured ? '凭据已配置' : '凭据未配置'}</small></td>
          <td className="muted-cell">{endpoint.lastProbeAt ? formatTime(endpoint.lastProbeAt) : '尚未探测'}</td>
        </tr>
      })}</tbody></table></div><TablePagination total={filtered.length} page={currentPage} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={(value) => { setPageSize(value); setPage(1) }} /></> : <EmptyState title="暂无匹配节点" message={serviceEndpoints.length ? '调整节点类型、状态或搜索条件。' : '先在系统设置中添加训练、转换或推理节点。'} />}
    </section>
  </div>
}
