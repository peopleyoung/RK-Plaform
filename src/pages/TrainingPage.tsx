import { useEffect, useMemo, useState } from 'react'
import { ChartNoAxesCombined, Check, ChevronLeft, Clock3, Cpu, Download, Gauge, Play, RotateCcw, Search, SlidersHorizontal, Sparkles, Trash2, Zap } from 'lucide-react'
import { usePlatform } from '../api/PlatformContext'
import { api } from '../api/client'
import { isJobDeletable, jobStatusLabels, jobTone, taskLabels, variantLabel } from '../api/presentation'
import { AddButton, Button, ConfirmDialog, EmptyState, Modal, PageHeader, ProgressBar, StatusBadge, TablePagination } from '../components'
import { TrainingMonitor } from '../components/TrainingMonitor'
import type { Artifact, Job, ModelProfile } from '../types'

type StatusFilter = 'all' | Job['status']

function jobSpec(job: Job) {
  return job.spec as { variant?: string; accelerator?: string; dataset?: { name?: string; version?: string }; resolution?: { width?: number; height?: number } }
}

function checkpointLabel(artifact: Artifact) {
  const filename = artifact.filename.toLowerCase()
  if (filename.endsWith('.pt')) return 'PT'
  if (filename.endsWith('.pdparams')) return 'Paddle'
  return '权重'
}

export function TrainingPage({ createSignal, notify }: { createSignal: number; notify: (message: string) => void }) {
  const { profiles, datasets, jobs, workers, artifacts, loading, error, refresh } = usePlatform()
  const trainingJobs = jobs.filter((job) => job.type === 'training')
  const [filter, setFilter] = useState<StatusFilter>('all')
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(createSignal > 0)
  const [lastSignal, setLastSignal] = useState(createSignal)
  const [step, setStep] = useState(1)
  const [submitting, setSubmitting] = useState(false)
  const [profileId, setProfileId] = useState('yolo-detect')
  const profile = profiles.find((item) => item.id === profileId) ?? profiles[0]
  const eligibleDatasets = datasets.filter((dataset) => dataset.taskType === profile?.taskType && dataset.status === 'ready')
  const [variant, setVariant] = useState('yolov8n')
  const [datasetId, setDatasetId] = useState('')
  const [name, setName] = useState('自定义模型训练')
  const [accelerator, setAccelerator] = useState<'cpu' | 'cuda'>('cuda')
  const [epochs, setEpochs] = useState(100)
  const [batch, setBatch] = useState(16)
  const [width, setWidth] = useState(640)
  const [height, setHeight] = useState(640)
  const [optimizer, setOptimizer] = useState<'auto' | 'AdamW' | 'SGD'>('auto')
  const [learningRate, setLearningRate] = useState('')
  const [seed, setSeed] = useState(42)
  const [pretrained, setPretrained] = useState(true)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [deleteTarget, setDeleteTarget] = useState<Job | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [retryingId, setRetryingId] = useState<string | null>(null)
  const [monitorTarget, setMonitorTarget] = useState<Job | null>(null)
  const legacyYoloOptimizer = variant.startsWith('yolov6') || variant.startsWith('yolov7')

  useEffect(() => {
    if (createSignal !== lastSignal) { setLastSignal(createSignal); setOpen(true) }
  }, [createSignal, lastSignal])

  useEffect(() => {
    if (!profile) return
    if (!profile.variants.includes(variant)) setVariant(profile.variants[0])
    if (!eligibleDatasets.some((item) => item.id === datasetId)) setDatasetId(eligibleDatasets[0]?.id ?? '')
  }, [datasetId, eligibleDatasets, profile, variant])

  useEffect(() => {
    if (legacyYoloOptimizer && optimizer === 'AdamW') setOptimizer('auto')
  }, [legacyYoloOptimizer, optimizer])

  const filteredJobs = useMemo(() => trainingJobs.filter((job) => {
    const matchesStatus = filter === 'all' || job.status === filter
    const term = query.trim().toLowerCase()
    return matchesStatus && (!term || job.name.toLowerCase().includes(term) || job.id.toLowerCase().includes(term) || (jobSpec(job).variant ?? '').toLowerCase().includes(term))
  }), [filter, query, trainingJobs])
  const artifactsByJob = useMemo(() => {
    const result = new Map<string, Artifact[]>()
    for (const artifact of artifacts) {
      if (!artifact.jobId) continue
      result.set(artifact.jobId, [...(result.get(artifact.jobId) ?? []), artifact])
    }
    return result
  }, [artifacts])
  const visibleJobs = filteredJobs.slice((page - 1) * pageSize, page * pageSize)

  useEffect(() => setPage(1), [filter, query, pageSize])
  useEffect(() => {
    const lastPage = Math.max(1, Math.ceil(filteredJobs.length / pageSize))
    if (page > lastPage) setPage(lastPage)
  }, [filteredJobs.length, page, pageSize])

  const chooseProfile = (next: ModelProfile) => {
    setProfileId(next.id)
    setVariant(next.variants[0])
    setWidth(next.defaultResolution.width)
    setHeight(next.defaultResolution.height)
    setDatasetId(datasets.find((item) => item.taskType === next.taskType && item.status === 'ready')?.id ?? '')
  }

  const resolutionError = () => {
    if (!profile) return '模型配方尚未加载'
    const rule = profile.resolutionRule
    if (width < rule.minWidth || width > rule.maxWidth || width % rule.widthMultiple) return `宽度需在 ${rule.minWidth}–${rule.maxWidth} 内且为 ${rule.widthMultiple} 的倍数`
    if (height < rule.minHeight || height > rule.maxHeight || height % rule.heightMultiple) return `高度需在 ${rule.minHeight}–${rule.maxHeight} 内且为 ${rule.heightMultiple} 的倍数`
    return ''
  }

  const close = () => { if (!submitting) { setOpen(false); setStep(1) } }
  const continueToReview = () => {
    const invalidResolution = resolutionError()
    if (!name.trim() || !datasetId || !variant) return notify('请完成任务名称、数据集和模型配置')
    if (invalidResolution) return notify(invalidResolution)
    if (learningRate && Number(learningRate) <= 0) return notify('初始学习率必须大于 0')
    if (!Number.isInteger(seed) || seed < 0) return notify('随机种子必须是非负整数')
    setStep(2)
  }

  const createJob = async () => {
    if (!profile) return
    setSubmitting(true)
    try {
      const created = await api.createTrainingJob({
        name: name.trim(), datasetId, profileId: profile.id, variant,
        resolution: { width, height },
        hyperparameters: {
          epochs, batchSize: batch, optimizer, pretrained, seed,
          ...(learningRate ? { learningRate: Number(learningRate) } : {}),
        },
        accelerator,
      })
      await refresh()
      setSubmitting(false)
      close()
      notify(`训练任务 ${created.id} 已进入 ${accelerator.toUpperCase()} 队列`)
    } catch (reason) {
      setSubmitting(false)
      notify(reason instanceof Error ? reason.message : '训练任务提交失败')
    }
  }

  const deleteJob = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await api.deleteJob(deleteTarget.id)
      await refresh()
      notify(`训练任务「${deleteTarget.name}」已删除`)
      setDeleteTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '删除训练任务失败')
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
      notify(`重新训练任务 ${created.id} 已进入队列`)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '重新训练任务提交失败')
    } finally {
      setRetryingId(null)
    }
  }

  const download = async (artifact: Artifact) => {
    try {
      await api.downloadArtifact(artifact)
      notify(`正在下载 ${artifact.filename}`)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '下载模型失败')
    }
  }

  const trainerPools = (kind: 'cpu' | 'cuda') => workers.filter((worker) => worker.kind === 'trainer' && worker.accelerator === kind && worker.status !== 'offline' && (!profile || worker.capabilities.includes(profile.id)))
  const statusTabs: Array<[StatusFilter, string]> = [['all', '全部'], ['running', '运行中'], ['queued', '排队中'], ['succeeded', '已完成'], ['failed', '失败']]

  return (
    <div className="page-stack">
      <PageHeader title="训练任务" description="选择数据集和模型配方，将任务调度到 CPU 或 CUDA 资源池。" actions={<AddButton onClick={() => setOpen(true)}>新建训练任务</AddButton>} />
      {error && <div className="api-banner danger">{error}<Button variant="quiet" onClick={() => void refresh()}>重试</Button></div>}
      <section className="summary-band">
        <div><span className="summary-icon amber"><Play size={18} /></span><p><small>正在执行</small><strong>{trainingJobs.filter((item) => ['claimed', 'running'].includes(item.status)).length}</strong></p></div>
        <div><span className="summary-icon blue"><Clock3 size={18} /></span><p><small>排队等待</small><strong>{trainingJobs.filter((item) => item.status === 'queued').length}</strong></p></div>
        <div><span className="summary-icon teal"><Check size={18} /></span><p><small>已完成</small><strong>{trainingJobs.filter((item) => item.status === 'succeeded').length}</strong></p></div>
        <div><span className="summary-icon coral"><Gauge size={18} /></span><p><small>可用训练节点</small><strong>{workers.filter((item) => item.kind === 'trainer' && item.status !== 'offline').length}</strong></p></div>
      </section>
      <div className="toolbar tab-toolbar"><div className="tabs" role="tablist">{statusTabs.map(([value, label]) => <button key={value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)} role="tab">{label}</button>)}</div><div className="toolbar-spacer" /><label className="search-box compact"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索任务" /></label></div>
      <section className="panel table-panel">
        {loading && !trainingJobs.length ? <EmptyState title="正在加载训练任务" message="正在连接调度服务。" /> : filteredJobs.length ? <><div className="table-scroll"><table className="data-table jobs-table training-jobs-table"><thead><tr><th>训练任务</th><th>模型</th><th>数据集</th><th>资源池 / 节点</th><th>训练进度</th><th>当前阶段</th><th>模型产物</th><th>状态</th><th aria-label="操作" /></tr></thead><tbody>{visibleJobs.map((job) => {
          const spec = jobSpec(job); const worker = workers.find((item) => item.id === job.workerId)
          const jobArtifacts = artifactsByJob.get(job.id) ?? []
          const onnx = jobArtifacts.find((artifact) => artifact.kind === 'onnx')
          const checkpoint = jobArtifacts.find((artifact) => artifact.kind === 'training_checkpoint')
          const deletable = isJobDeletable(job.status)
          const datasetName = spec.dataset?.name ?? datasets.find((item) => item.id === job.datasetId)?.name ?? '数据集记录缺失'
          return <tr key={job.id}><td><strong>{job.name}</strong></td><td><strong className="medium-text">{variantLabel(spec.variant ?? job.profileId)}</strong></td><td><strong className="medium-text">{datasetName}</strong></td><td><span className="node-inline">{spec.accelerator === 'cuda' ? <Zap size={15} /> : <Cpu size={15} />}{worker?.name ?? `${String(spec.accelerator).toUpperCase()} 队列`}</span></td><td className="progress-cell"><ProgressBar value={job.progress} tone={job.status === 'failed' ? 'danger' : 'teal'} /></td><td><strong>{job.stage}</strong><small>{job.errorMessage ?? `${spec.resolution?.width ?? '--'} × ${spec.resolution?.height ?? '--'}`}</small></td><td><div className="training-artifacts">{onnx && <button className="artifact-link" title={onnx.filename} aria-label={`下载 ONNX 模型 ${job.name}`} onClick={() => void download(onnx)}><Download size={14} />ONNX</button>}{checkpoint && <button className="artifact-link" title={checkpoint.filename} aria-label={`下载训练权重 ${job.name}`} onClick={() => void download(checkpoint)}><Download size={14} />{checkpointLabel(checkpoint)}</button>}{!onnx && !checkpoint && <span className="muted-cell">--</span>}</div></td><td><StatusBadge tone={jobTone(job.status)}>{jobStatusLabels[job.status]}</StatusBadge></td><td><div className="row-actions">{job.status === 'failed' && <button className="icon-button ghost" aria-label={`重新训练 ${job.name}`} title={retryingId === job.id ? '正在重新提交' : '重新训练'} disabled={retryingId !== null} onClick={() => void retryJob(job)}><RotateCcw size={17} /></button>}<button className="icon-button ghost" aria-label={`查看训练监控 ${job.name}`} title="查看训练日志与指标" onClick={() => setMonitorTarget(job)}><ChartNoAxesCombined size={17} /></button><button className="icon-button ghost danger-action" aria-label={`删除训练任务 ${job.name}`} title={deletable ? '删除训练任务' : '任务已被节点领取，暂时不能删除'} onClick={() => deletable ? setDeleteTarget(job) : notify('任务已被节点领取或正在运行，暂时不能删除')}><Trash2 size={17} /></button></div></td></tr>
        })}</tbody></table></div><TablePagination total={filteredJobs.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={setPageSize} /></> : <EmptyState title="没有匹配的训练任务" message="新建任务或调整筛选条件。" />}
      </section>
      <Modal open={open} title="新建训练任务" description={step === 1 ? '配置模型、数据和执行资源。' : '确认配置后任务将进入调度队列。'} width="large" onClose={close} footer={step === 1 ? <><Button variant="secondary" onClick={close}>取消</Button><Button onClick={continueToReview}>下一步：确认配置</Button></> : <><Button variant="quiet" icon={<ChevronLeft size={17} />} onClick={() => setStep(1)}>返回修改</Button><div className="footer-spacer" /><Button icon={<Play size={16} fill="currentColor" />} onClick={() => void createJob()} disabled={submitting}>{submitting ? '正在提交…' : '提交训练任务'}</Button></>}>
        <div className="stepper"><span className="active"><i>1</i>任务配置</span><b /><span className={step === 2 ? 'active' : ''}><i>2</i>确认提交</span></div>
        {step === 1 ? <div className="form-sections">
          <section className="form-section"><h3><Sparkles size={17} />基础信息</h3><div className="form-grid two-columns"><label className="field"><span>任务名称 <b>*</b></span><input value={name} onChange={(event) => setName(event.target.value)} /></label><label className="field"><span>训练数据集 <b>*</b></span><select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>{eligibleDatasets.length ? eligibleDatasets.map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.name} · {dataset.version}</option>) : <option value="">暂无匹配数据集</option>}</select></label></div></section>
          <section className="form-section"><h3><Zap size={17} />模型配方</h3><div className="model-choice-grid">{profiles.map((item) => <button key={item.id} className={profileId === item.id ? 'model-choice active' : 'model-choice'} onClick={() => chooseProfile(item)}><span>{item.family === 'YOLO' ? 'YO' : item.family === 'PPOCR' ? 'PP' : 'DL'}</span><div><strong>{item.label}</strong><small>{taskLabels[item.taskType]}</small></div>{profileId === item.id && <Check size={16} />}</button>)}</div>{profile && <><div className="form-grid three-columns form-row-gap"><label className="field"><span>{profile.id === 'deeplabv3plus' ? '主干网络 / 部署版本' : '模型变体'}</span><select value={variant} onChange={(event) => setVariant(event.target.value)}>{profile.variants.map((item) => <option value={item} key={item}>{variantLabel(item)}</option>)}</select></label><label className="field"><span>输入宽度</span><input type="number" min={profile.resolutionRule.minWidth} max={profile.resolutionRule.maxWidth} step={profile.resolutionRule.widthMultiple} value={width} onChange={(event) => setWidth(Number(event.target.value))} /></label><label className="field"><span>输入高度</span><input type="number" min={profile.resolutionRule.minHeight} max={profile.resolutionRule.maxHeight} step={profile.resolutionRule.heightMultiple} value={height} onChange={(event) => setHeight(Number(event.target.value))} /></label><label className="field"><span>训练轮次</span><input type="number" min="1" max="10000" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label><label className="field"><span>批大小</span><input type="number" min="1" max="1024" value={batch} onChange={(event) => setBatch(Number(event.target.value))} /></label><label className="field"><span>优化器</span><select value={optimizer} onChange={(event) => { const value = event.target.value as 'auto' | 'AdamW' | 'SGD'; setOptimizer(value); if (value === 'auto') setLearningRate('') }}><option value="auto">自动</option>{!legacyYoloOptimizer && <option value="AdamW">AdamW</option>}<option value="SGD">SGD</option></select></label><label className="field"><span>初始学习率</span><input type="number" min="0" step="any" value={learningRate} placeholder="使用配方默认值" onChange={(event) => setLearningRate(event.target.value)} disabled={optimizer === 'auto'} /></label><label className="field"><span>随机种子</span><input type="number" min="0" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label></div><div className="resolution-rule">宽 {profile.resolutionRule.minWidth}–{profile.resolutionRule.maxWidth} / {profile.resolutionRule.widthMultiple} 倍数 · 高 {profile.resolutionRule.minHeight}–{profile.resolutionRule.maxHeight} / {profile.resolutionRule.heightMultiple} 倍数 · 固定 batch 1 导出</div><label className="toggle-row"><input type="checkbox" checked={pretrained} onChange={(event) => setPretrained(event.target.checked)} /><span>使用官方预训练权重</span></label></>}</section>
          <section className="form-section"><h3><Cpu size={17} />执行资源池</h3><div className="target-choice-grid">{([{ id: 'cuda', title: 'NVIDIA CUDA', icon: <Zap size={19} /> }, { id: 'cpu', title: '纯 CPU', icon: <Cpu size={19} /> }] as const).map((pool) => { const available = trainerPools(pool.id).length; return <button key={pool.id} className={accelerator === pool.id ? 'target-choice active' : 'target-choice'} onClick={() => setAccelerator(pool.id)}>{pool.icon}<div><strong>{pool.title}</strong><span>{available} 个兼容节点在线</span><small>不可用时保持排队</small></div>{accelerator === pool.id && <Check size={16} />}</button> })}</div></section>
        </div> : <div className="review-layout"><div className="review-hero"><span><Check size={24} /></span><div><strong>配置检查通过</strong><p>数据集任务类型、分辨率边界和模型变体一致。</p></div></div><dl className="review-list"><div><dt>任务名称</dt><dd>{name}</dd></div><div><dt>模型</dt><dd>{variantLabel(variant)}</dd></div><div><dt>数据集</dt><dd>{datasets.find((item) => item.id === datasetId)?.name}</dd></div><div><dt>部署分辨率</dt><dd>{width} × {height}</dd></div><div><dt>训练参数</dt><dd>{epochs} epochs · batch {batch} · {optimizer} · seed {seed}</dd></div><div><dt>资源池</dt><dd>{accelerator.toUpperCase()}</dd></div></dl><div className="format-note"><SlidersHorizontal size={18} /><div><strong>导出契约</strong><span>训练完成后自动导出固定分辨率 ONNX，并在上传前核对模型图与部署清单。</span></div></div></div>}
      </Modal>
      {monitorTarget && <TrainingMonitor job={monitorTarget} logArtifact={(artifactsByJob.get(monitorTarget.id) ?? []).find((artifact) => artifact.kind === 'training_log')} onDownload={download} onClose={() => { setMonitorTarget(null); void refresh() }} />}
      <ConfirmDialog open={Boolean(deleteTarget)} title="删除训练任务" description={deleteTarget ? `确定删除「${deleteTarget.name}」及其训练产物吗？如有活动转换任务依赖其 ONNX，系统会拒绝删除。` : ''} confirmLabel="删除训练任务" busy={deleting} onClose={() => !deleting && setDeleteTarget(null)} onConfirm={() => void deleteJob()} />
    </div>
  )
}
