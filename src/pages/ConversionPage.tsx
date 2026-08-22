import { useEffect, useMemo, useState } from 'react'
import { Box, Check, ChevronLeft, CircleAlert, Clock3, Container, Cpu, Download, FileCode2, ListTree, PackageCheck, RotateCcw, Search, ShieldCheck, Trash2 } from 'lucide-react'
import { usePlatform } from '../api/PlatformContext'
import { api } from '../api/client'
import { isJobDeletable, jobStatusLabels, jobTone, variantLabel } from '../api/presentation'
import { AddButton, Button, ConfirmDialog, EmptyState, Modal, PageHeader, ProgressBar, StatusBadge, TablePagination } from '../components'
import { ConversionMonitor } from '../components/ConversionMonitor'
import type { Artifact, Dataset, Job, Precision, TaskType } from '../types'

type StatusFilter = 'all' | Job['status']
interface ArtifactManifest {
  profileId: string
  variant: string
  taskType: TaskType
  resolution: { width: number; height: number }
  input: { shape: number[]; layout: string; name: string }
  supportedPrecisions: Precision[]
  outputContract: string
}

function artifactManifest(artifact?: Artifact): ArtifactManifest | null {
  return artifact?.manifest as unknown as ArtifactManifest | null
}

function conversionSpec(job: Job) {
  return job.spec as { precision?: Precision; sourceArtifact?: { filename?: string; id?: string }; manifest?: ArtifactManifest }
}

interface SourceModelLink {
  modelName: string
  datasetName: string
  trainingName: string
}

function sourceModelLink(artifact: Artifact, trainingJobs: Map<string, Job>, datasets: Map<string, Dataset>): SourceModelLink {
  const trainingJob = artifact.jobId ? trainingJobs.get(artifact.jobId) : undefined
  const dataset = trainingJob?.datasetId ? datasets.get(trainingJob.datasetId) : undefined
  const trainingSpec = trainingJob?.spec as { variant?: string; dataset?: { name?: string } } | undefined
  const manifest = artifactManifest(artifact)
  return {
    modelName: variantLabel(manifest?.variant ?? trainingSpec?.variant ?? trainingJob?.profileId ?? '未知模型'),
    datasetName: dataset?.name ?? trainingSpec?.dataset?.name ?? '数据集记录缺失',
    trainingName: trainingJob?.name ?? '训练任务记录缺失',
  }
}

export function ConversionPage({ notify }: { notify: (message: string) => void }) {
  const { profiles, datasets, jobs, workers, artifacts, loading, error, refresh } = usePlatform()
  const conversionJobs = jobs.filter((job) => job.type === 'conversion')
  const sourceArtifacts = artifacts.filter((item) => item.kind === 'onnx' && item.manifest)
  const converterNodes = workers.filter((item) => item.kind === 'converter')
  const sourceModelLinks = useMemo(() => {
    const trainingJobs = new Map(jobs.filter((job) => job.type === 'training').map((job) => [job.id, job]))
    const datasetRecords = new Map(datasets.map((dataset) => [dataset.id, dataset]))
    return new Map(sourceArtifacts.map((artifact) => [artifact.id, sourceModelLink(artifact, trainingJobs, datasetRecords)]))
  }, [artifacts, datasets, jobs])
  const [open, setOpen] = useState(false)
  const [step, setStep] = useState(1)
  const [submitting, setSubmitting] = useState(false)
  const [status, setStatus] = useState<StatusFilter>('all')
  const [query, setQuery] = useState('')
  const [sourceId, setSourceId] = useState('')
  const [name, setName] = useState('RK3588 模型转换')
  const [precision, setPrecision] = useState<Precision>('int8')
  const [calibration, setCalibration] = useState('')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [deleteTarget, setDeleteTarget] = useState<Job | null>(null)
  const [monitorTarget, setMonitorTarget] = useState<Job | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [retryingId, setRetryingId] = useState<string | null>(null)
  const sourceArtifact = sourceArtifacts.find((item) => item.id === sourceId) ?? sourceArtifacts[0]
  const selectedSourceLink = sourceArtifact ? sourceModelLinks.get(sourceArtifact.id) : undefined
  const manifest = artifactManifest(sourceArtifact)
  const profile = profiles.find((item) => item.id === manifest?.profileId)
  const compatibleDatasets = datasets.filter((item) => item.status === 'ready' && item.taskType === manifest?.taskType)

  useEffect(() => {
    if (!sourceArtifact) return
    if (sourceId !== sourceArtifact.id) setSourceId(sourceArtifact.id)
    const supported = artifactManifest(sourceArtifact)?.supportedPrecisions ?? []
    if (!supported.includes(precision)) setPrecision(supported[0] ?? 'fp16')
    if (!compatibleDatasets.some((item) => item.id === calibration)) setCalibration(compatibleDatasets[0]?.id ?? '')
  }, [calibration, compatibleDatasets, precision, sourceArtifact, sourceId])

  const filtered = useMemo(() => conversionJobs.filter((job) => {
    const matchesStatus = status === 'all' || job.status === status
    const term = query.trim().toLowerCase()
    const spec = conversionSpec(job)
    const sourceLink = sourceModelLinks.get(spec.sourceArtifact?.id ?? '')
    const searchable = [job.name, job.id, spec.sourceArtifact?.filename, sourceLink?.modelName, sourceLink?.datasetName, sourceLink?.trainingName].filter(Boolean).join(' ').toLowerCase()
    return matchesStatus && (!term || searchable.includes(term))
  }), [conversionJobs, query, sourceModelLinks, status])
  const visibleJobs = filtered.slice((page - 1) * pageSize, page * pageSize)

  useEffect(() => setPage(1), [query, status, pageSize])
  useEffect(() => {
    const lastPage = Math.max(1, Math.ceil(filtered.length / pageSize))
    if (page > lastPage) setPage(lastPage)
  }, [filtered.length, page, pageSize])

  const chooseSource = (id: string) => {
    const artifact = sourceArtifacts.find((item) => item.id === id)
    const nextManifest = artifactManifest(artifact)
    setSourceId(id)
    setPrecision(nextManifest?.supportedPrecisions[0] ?? 'fp16')
    setCalibration(datasets.find((item) => item.status === 'ready' && item.taskType === nextManifest?.taskType)?.id ?? '')
  }

  const close = () => { if (!submitting) { setOpen(false); setStep(1) } }
  const toReview = () => {
    if (!sourceArtifact || !manifest) return notify('当前没有带部署清单的 ONNX 训练产物')
    if (precision === 'int8' && !calibration) return notify('INT8 转换必须选择同任务类型的校准数据集')
    setStep(2)
  }

  const createJob = async () => {
    if (!sourceArtifact) return
    setSubmitting(true)
    try {
      const created = await api.createConversionJob({ name: name.trim(), sourceArtifactId: sourceArtifact.id, precision, ...(precision === 'int8' ? { calibrationDatasetId: calibration } : {}) })
      await refresh()
      setSubmitting(false)
      close()
      notify(`转换任务 ${created.id} 已提交到 RK3588 队列`)
    } catch (reason) {
      setSubmitting(false)
      notify(reason instanceof Error ? reason.message : '转换任务提交失败')
    }
  }

  const download = async (artifact: Artifact) => {
    try { await api.downloadArtifact(artifact); notify(`正在下载 ${artifact.filename}`) }
    catch (reason) { notify(reason instanceof Error ? reason.message : '下载失败') }
  }

  const deleteJob = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await api.deleteJob(deleteTarget.id)
      await refresh()
      notify(`转换任务「${deleteTarget.name}」已删除`)
      setDeleteTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '删除转换任务失败')
    } finally {
      setDeleting(false)
    }
  }

  const retryJob = async (job: Job) => {
    if (retryingId) return
    setRetryingId(job.id)
    try {
      const created = await api.retryJob(job.id)
      await refresh()
      notify(`重新转换任务 ${created.id} 已进入队列`)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '重新转换任务提交失败')
    } finally {
      setRetryingId(null)
    }
  }

  const statusTabs: Array<[StatusFilter, string]> = [['all', '全部'], ['running', '转换中'], ['queued', '待转换'], ['succeeded', '已完成'], ['failed', '失败']]
  const primaryNode = converterNodes.find((node) => node.status !== 'offline')

  return (
    <div className="page-stack">
      <PageHeader title="RKNN 转换中心" description="将已验证的静态 ONNX 产物转换为 RK3588 RKNN 模型。" actions={<AddButton onClick={() => setOpen(true)}>新建转换任务</AddButton>} />
      {error && <div className="api-banner danger">{error}<Button variant="quiet" onClick={() => void refresh()}>重试</Button></div>}
      <section className="conversion-runtime-strip"><div className="runtime-identity"><span><Cpu size={22} /></span><div><small>目标平台</small><strong>RK3588</strong></div></div><i /><div><small>转换节点</small><strong><span className={primaryNode ? 'health-dot' : 'health-dot offline'} />{primaryNode?.name ?? '未连接'}</strong></div><i /><div><small>工作进程版本</small><strong>{primaryNode?.version ?? '--'}</strong></div><i /><div><small>队列任务</small><strong>{conversionJobs.filter((item) => item.status === 'queued').length} 个</strong></div><span className="docker-badge"><Container size={16} />Docker</span></section>
      <div className="pipeline-grid">{[{ icon: <FileCode2 size={18} />, label: '校验 ONNX', value: '静态图契约', tone: 'blue' }, { icon: <Box size={18} />, label: '图优化与量化', value: 'INT8 / FP16', tone: 'amber' }, { icon: <ShieldCheck size={18} />, label: 'NPU 运行验证', value: 'RK3588', tone: 'teal' }, { icon: <PackageCheck size={18} />, label: '产物归档', value: '.rknn + 报告', tone: 'coral' }].map((item, index) => <div className="pipeline-step" key={item.label}><span className={`summary-icon ${item.tone}`}>{item.icon}</span><div><small>0{index + 1}</small><strong>{item.label}</strong><p>{item.value}</p></div>{index < 3 && <ChevronLeft className="pipeline-arrow" size={18} />}</div>)}</div>
      <div className="toolbar tab-toolbar"><div className="tabs" role="tablist">{statusTabs.map(([value, label]) => <button key={value} className={status === value ? 'active' : ''} onClick={() => setStatus(value)} role="tab">{label}</button>)}</div><div className="toolbar-spacer" /><label className="search-box compact"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索转换任务" /></label></div>
      <section className="panel table-panel">{loading && !conversionJobs.length ? <EmptyState title="正在加载转换任务" message="正在连接调度服务。" /> : filtered.length ? <><div className="table-scroll"><table className="data-table jobs-table"><thead><tr><th>转换任务</th><th>模型</th><th>精度</th><th>转换进度</th><th>输出产物</th><th>状态</th><th aria-label="操作" /></tr></thead><tbody>{visibleJobs.map((job) => {
        const spec = conversionSpec(job); const output = artifacts.find((item) => item.jobId === job.id && item.kind === 'rknn')
        const deletable = isJobDeletable(job.status)
        const sourceLink = sourceModelLinks.get(spec.sourceArtifact?.id ?? '')
        const modelName = sourceLink?.modelName ?? variantLabel(spec.manifest?.variant ?? job.profileId)
        return <tr key={job.id}><td><strong>{job.name}</strong></td><td><strong className="medium-text">{modelName}</strong></td><td><span className={`precision-badge ${spec.precision}`}>{spec.precision?.toUpperCase()}</span></td><td className="progress-cell"><ProgressBar value={job.progress} tone={job.status === 'failed' ? 'danger' : 'blue'} /></td><td>{output ? <button className="artifact-link" onClick={() => void download(output)}><Download size={15} />{output.filename}</button> : <span className="muted-cell">{job.errorMessage ?? '--'}</span>}</td><td><StatusBadge tone={jobTone(job.status)}>{jobStatusLabels[job.status]}</StatusBadge></td><td><div className="row-actions">{job.status === 'failed' && <button className="icon-button ghost" aria-label={`重新转换 ${job.name}`} title={retryingId === job.id ? '正在重新提交' : '重新转换'} disabled={retryingId !== null} onClick={() => void retryJob(job)}><RotateCcw size={17} /></button>}<button className="icon-button ghost" aria-label={`查看转换进度 ${job.name}`} title="查看转换进度详情" onClick={() => setMonitorTarget(job)}><ListTree size={17} /></button><button className="icon-button ghost danger-action" aria-label={`删除转换任务 ${job.name}`} title={deletable ? '删除转换任务' : '任务已被节点领取，暂时不能删除'} onClick={() => deletable ? setDeleteTarget(job) : notify('任务已被节点领取或正在运行，暂时不能删除')}><Trash2 size={17} /></button></div></td></tr>
      })}</tbody></table></div><TablePagination total={filtered.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={setPageSize} /></> : <EmptyState title="没有匹配的转换任务" message="先完成训练导出，再创建转换任务。" />}</section>
      <Modal open={open} title="新建 RKNN 转换任务" description={step === 1 ? '从训练产物选择静态 ONNX，并配置 RK3588 构建精度。' : '确认后由 RK3588 Docker 工作进程执行。'} width="large" onClose={close} footer={step === 1 ? <><Button variant="secondary" onClick={close}>取消</Button><Button onClick={toReview}>下一步：兼容性检查</Button></> : <><Button variant="quiet" icon={<ChevronLeft size={17} />} onClick={() => setStep(1)}>返回修改</Button><div className="footer-spacer" /><Button icon={<Container size={17} />} onClick={() => void createJob()} disabled={submitting}>{submitting ? '正在提交…' : '提交转换任务'}</Button></>}>
        <div className="stepper"><span className="active"><i>1</i>转换配置</span><b /><span className={step === 2 ? 'active' : ''}><i>2</i>兼容性检查</span></div>
        {step === 1 ? <div className="form-sections">
          <section className="form-section"><h3><FileCode2 size={17} />源模型</h3><div className="form-grid two-columns"><label className="field full-width"><span>训练 ONNX 产物 <b>*</b></span><select value={sourceArtifact?.id ?? ''} onChange={(event) => chooseSource(event.target.value)}>{sourceArtifacts.length ? sourceArtifacts.map((item) => { const link = sourceModelLinks.get(item.id); return <option key={item.id} value={item.id}>{link ? `${link.modelName} · ${link.datasetName} · ${link.trainingName}` : item.filename} · {item.filename}</option> }) : <option value="">暂无可转换产物</option>}</select><small>选项已绑定训练模型、训练任务和数据集；提交时仍使用对应 ONNX 产物 ID。</small></label>{selectedSourceLink && <div className="source-link-details full-width" aria-label="源模型关联信息"><span><small>训练模型</small><strong title={selectedSourceLink.modelName}>{selectedSourceLink.modelName}</strong></span><span><small>训练数据集</small><strong title={selectedSourceLink.datasetName}>{selectedSourceLink.datasetName}</strong></span><span><small>训练任务</small><strong title={selectedSourceLink.trainingName}>{selectedSourceLink.trainingName}</strong></span><span><small>ONNX 文件</small><strong title={sourceArtifact?.filename}>{sourceArtifact?.filename}</strong></span></div>}<label className="field"><span>任务名称</span><input value={name} onChange={(event) => setName(event.target.value)} /></label><label className="field"><span>模型变体</span><input value={variantLabel(manifest?.variant ?? '--')} readOnly /></label></div></section>
          <section className="form-section"><h3><Box size={17} />构建参数</h3><div className="form-grid two-columns"><label className="field"><span>输入形状</span><input value={manifest ? `[${manifest.input.shape.join(', ')}]` : '--'} readOnly /><small>来自训练导出清单，转换阶段不可改写</small></label><label className="field"><span>输出契约</span><input value={manifest?.outputContract ?? '--'} readOnly /></label></div><div className="field form-row-gap"><span>转换精度</span><div className="precision-options">{(['int8', 'fp16'] as const).map((item) => { const supported = manifest?.supportedPrecisions.includes(item) ?? false; return <button key={item} disabled={!supported} className={precision === item ? 'precision-option active' : 'precision-option'} onClick={() => supported && setPrecision(item)}><span>{item.toUpperCase()}</span><small>{item === 'int8' ? '需要代表性校准集' : '非量化构建'}</small>{precision === item && <Check size={16} />}{!supported && <em>不支持</em>}</button> })}</div></div>{precision === 'int8' && <label className="field form-row-gap"><span>校准数据集 <b>*</b></span><select value={calibration} onChange={(event) => setCalibration(event.target.value)}>{compatibleDatasets.length ? compatibleDatasets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.version}</option>) : <option value="">暂无同任务数据集</option>}</select></label>}<div className="compatibility-note"><CircleAlert size={18} /><div><strong>{profile?.label ?? '部署清单'} · {precision.toUpperCase()}</strong><span>服务端将再次核对输入名、静态形状、opset、输出名与 ONNX 校验值。</span></div></div></section>
          <section className="form-section"><h3><Container size={17} />转换环境</h3><div className="environment-summary"><div><small>执行节点</small><strong>{primaryNode?.name ?? '等待节点'}</strong></div><div><small>目标平台</small><strong>RK3588</strong></div><div><small>工作进程</small><strong>{primaryNode?.version ?? '--'}</strong></div><div><small>并发上限</small><strong>{primaryNode?.maxConcurrency ?? '--'}</strong></div></div></section>
        </div> : <div className="review-layout"><div className="review-hero"><span><ShieldCheck size={24} /></span><div><strong>静态契约可提交</strong><p>最终部署状态以 RK3588 上的构建、初始化和推理结果为准。</p></div></div><dl className="review-list"><div><dt>训练模型</dt><dd>{selectedSourceLink?.modelName ?? variantLabel(manifest?.variant ?? '--')}</dd></div><div><dt>训练数据集</dt><dd>{selectedSourceLink?.datasetName ?? '--'}</dd></div><div><dt>训练任务</dt><dd>{selectedSourceLink?.trainingName ?? '--'}</dd></div><div><dt>ONNX 文件</dt><dd>{sourceArtifact?.filename}</dd></div><div><dt>部署分辨率</dt><dd>{manifest ? `${manifest.resolution.width} × ${manifest.resolution.height}` : '--'}</dd></div><div><dt>精度模式</dt><dd>{precision.toUpperCase()}</dd></div><div><dt>校准数据</dt><dd>{precision === 'int8' ? datasets.find((item) => item.id === calibration)?.name : '不需要'}</dd></div><div><dt>目标节点</dt><dd>{primaryNode?.name ?? '等待 RK3588 节点'}</dd></div></dl><div className="conversion-estimate"><Clock3 size={19} /><div><strong>部署就绪门槛</strong><span>RKNN 导出、板端运行时初始化和一次确定性推理必须全部成功。</span></div><StatusBadge tone={primaryNode ? 'success' : 'warning'}>{primaryNode ? '节点在线' : '等待节点'}</StatusBadge></div></div>}
      </Modal>
      {monitorTarget && <ConversionMonitor
        job={monitorTarget}
        workerName={workers.find((worker) => worker.id === monitorTarget.workerId)?.name}
        reportArtifact={artifacts.find((artifact) => artifact.jobId === monitorTarget.id && artifact.kind === 'validation_report')}
        logArtifact={artifacts.find((artifact) => artifact.jobId === monitorTarget.id && artifact.kind === 'conversion_log')}
        onDownload={download}
        onClose={() => { setMonitorTarget(null); void refresh() }}
      />}
      <ConfirmDialog open={Boolean(deleteTarget)} title="删除转换任务" description={deleteTarget ? `确定删除「${deleteTarget.name}」及其 RKNN、日志和验证报告吗？` : ''} confirmLabel="删除转换任务" busy={deleting} onClose={() => !deleting && setDeleteTarget(null)} onConfirm={() => void deleteJob()} />
    </div>
  )
}
