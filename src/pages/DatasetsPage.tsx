import { useEffect, useMemo, useRef, useState } from 'react'
import { Archive, Database, FileArchive, Filter, Search, Trash2, UploadCloud } from 'lucide-react'
import { usePlatform } from '../api/PlatformContext'
import { api } from '../api/client'
import { datasetFormatLabels, datasetFormatsByTask, datasetTone, defaultDatasetFormat, formatBytes, formatTime, taskLabels } from '../api/presentation'
import { AddButton, Button, ConfirmDialog, EmptyState, Modal, PageHeader, ProgressBar, StatusBadge, TablePagination } from '../components'
import type { Dataset, DatasetFormat, TaskType } from '../types'

const statusLabels = { uploaded: '已上传', validating: '校验中', ready: '可训练', failed: '校验失败' } as const

export function DatasetsPage({ notify }: { notify: (message: string) => void }) {
  const { datasets, loading, error, refresh } = usePlatform()
  const [query, setQuery] = useState('')
  const [type, setType] = useState<'all' | TaskType>('all')
  const [uploadOpen, setUploadOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [version, setVersion] = useState('v1')
  const [taskType, setTaskType] = useState<TaskType>('object_detection')
  const [datasetFormat, setDatasetFormat] = useState<DatasetFormat>('yolo')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null)
  const [deleting, setDeleting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const filtered = useMemo(() => datasets.filter((dataset) => {
    const matchesType = type === 'all' || dataset.taskType === type
    const term = query.trim().toLowerCase()
    return matchesType && (!term || dataset.name.toLowerCase().includes(term))
  }), [datasets, query, type])
  const visibleDatasets = filtered.slice((page - 1) * pageSize, page * pageSize)

  useEffect(() => setPage(1), [query, type, pageSize])
  useEffect(() => {
    const lastPage = Math.max(1, Math.ceil(filtered.length / pageSize))
    if (page > lastPage) setPage(lastPage)
  }, [filtered.length, page, pageSize])

  const resetForm = () => {
    setName('')
    setDescription('')
    setVersion('v1')
    setFile(null)
    setDatasetFormat(defaultDatasetFormat[taskType])
    setProgress(0)
    setUploading(false)
  }

  const closeModal = () => {
    if (uploading) return
    setUploadOpen(false)
    resetForm()
  }

  const submitUpload = async () => {
    if (!name.trim() || !file) {
      notify('请填写数据集名称并选择压缩包')
      return
    }
    setUploading(true)
    try {
      const created = await api.uploadDataset({
        name: name.trim(), description: description.trim(), version, taskType, datasetFormat,
      }, file, setProgress)
      await refresh()
      setUploadOpen(false)
      resetForm()
      notify(`数据集「${created.name}」已上传；类别将在首次训练校验时自动识别`)
    } catch (reason) {
      setUploading(false)
      notify(reason instanceof Error ? reason.message : '数据集上传失败')
    }
  }

  const deleteDataset = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await api.deleteDataset(deleteTarget.id)
      await refresh()
      notify(`数据集「${deleteTarget.name}」已删除`)
      setDeleteTarget(null)
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '删除数据集失败')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="page-stack">
      <PageHeader title="数据资产" description="统一管理训练数据、任务类型和可用版本。" actions={<AddButton onClick={() => setUploadOpen(true)}>上传数据集</AddButton>} />
      {error && <div className="api-banner danger">{error}<Button variant="quiet" onClick={() => void refresh()}>重试</Button></div>}
      <div className="toolbar">
        <label className="search-box"><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索数据集名称" /><kbd>/</kbd></label>
        <div className="toolbar-spacer" />
        <div className="filter-control"><Filter size={16} /><select value={type} onChange={(event) => setType(event.target.value as typeof type)} aria-label="按任务类型筛选"><option value="all">全部</option>{Object.entries(taskLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></div>
      </div>
      <section className="panel table-panel">
        <div className="table-meta"><span>共 <strong>{filtered.length}</strong> 个数据集</span><span>原始归档合计 {formatBytes(filtered.reduce((sum, item) => sum + item.sizeBytes, 0))}</span></div>
        {loading && !datasets.length ? <EmptyState title="正在加载数据集" message="正在连接平台 API。" /> : filtered.length ? (
          <><div className="table-scroll"><table className="data-table dataset-table"><thead><tr><th>数据集</th><th>任务类型</th><th>数据格式</th><th>类别</th><th>原始归档</th><th>校验值</th><th>状态</th><th>最近更新</th><th aria-label="操作" /></tr></thead><tbody>{visibleDatasets.map((dataset) => (
            <tr key={dataset.id}>
              <td><div className="dataset-name"><span className="dataset-thumb mint"><Database size={19} /></span><div><strong>{dataset.name}</strong><small>{dataset.version}</small></div></div></td>
              <td><span className="type-label">{taskLabels[dataset.taskType]}</span></td>
              <td><span className="type-label">{datasetFormatLabels[dataset.datasetFormat ?? 'auto']}</span></td>
              <td>{dataset.classes.length ? <span title={dataset.classes.join(', ')}>{dataset.classes.length}</span> : <span className="muted-cell">待训练识别</span>}</td>
              <td><strong className="medium-text">{dataset.filename}</strong><small>{formatBytes(dataset.sizeBytes)}</small></td>
              <td><code className="checksum">{dataset.sha256.slice(0, 12)}</code></td>
              <td><StatusBadge tone={datasetTone(dataset.status)}>{statusLabels[dataset.status]}</StatusBadge>{dataset.errorMessage && <small>{dataset.errorMessage}</small>}</td>
              <td className="muted-cell">{formatTime(dataset.updatedAt)}</td>
              <td><div className="row-actions"><button className="icon-button ghost danger-action" title="删除数据集" aria-label={`删除数据集 ${dataset.name}`} onClick={() => setDeleteTarget(dataset)}><Trash2 size={17} /></button></div></td>
            </tr>
          ))}</tbody></table></div><TablePagination total={filtered.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={setPageSize} /></>
        ) : <EmptyState title="没有匹配的数据集" message="上传归档或调整筛选条件。" />}
      </section>
      <Modal open={uploadOpen} title="上传数据集" description="归档会进行类型、大小和完整性校验。" width="large" onClose={closeModal} footer={<><Button variant="secondary" onClick={closeModal} disabled={uploading}>取消</Button><Button onClick={() => void submitUpload()} disabled={uploading}>{uploading ? '正在上传…' : '上传并校验'}</Button></>}>
        <div className="form-grid two-columns">
          <label className="field"><span>数据集名称 <b>*</b></span><input value={name} onChange={(event) => setName(event.target.value)} disabled={uploading} /></label>
          <label className="field"><span>版本</span><input value={version} onChange={(event) => setVersion(event.target.value)} disabled={uploading} /></label>
          <label className="field"><span>任务类型 <b>*</b></span><select value={taskType} onChange={(event) => { const next = event.target.value as TaskType; setTaskType(next); setDatasetFormat(defaultDatasetFormat[next]) }} disabled={uploading}>{Object.entries(taskLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className="field"><span>数据格式 <b>*</b></span><select value={datasetFormat} onChange={(event) => setDatasetFormat(event.target.value as DatasetFormat)} disabled={uploading}>{datasetFormatsByTask[taskType].map((value) => <option key={value} value={value}>{datasetFormatLabels[value]}</option>)}</select></label>
          <label className="field full-width"><span>说明</span><input value={description} onChange={(event) => setDescription(event.target.value)} disabled={uploading} /></label>
          <div className="field full-width"><span>数据文件 <b>*</b></span><input ref={inputRef} className="visually-hidden" type="file" accept=".zip,.tar.gz,.tgz" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /><button className={file ? 'drop-zone has-file' : 'drop-zone'} onClick={() => inputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); setFile(event.dataTransfer.files[0] ?? null) }} disabled={uploading}>{file ? <><FileArchive size={28} /><strong>{file.name}</strong><small>{formatBytes(file.size)}</small></> : <><UploadCloud size={30} /><strong>选择或拖放压缩包</strong><small>ZIP、TAR.GZ 或 TGZ</small></>}</button></div>
          {uploading ? <div className="upload-progress full-width"><div><span>上传与完整性校验</span><strong>{progress}%</strong></div><ProgressBar value={progress} showValue={false} tone="blue" /></div> : <div className="format-note full-width"><Archive size={18} /><div><strong>{datasetFormatLabels[datasetFormat]}</strong><span>{datasetFormatHint(datasetFormat)}</span></div></div>}
        </div>
      </Modal>
      <ConfirmDialog open={Boolean(deleteTarget)} title="删除数据集" description={deleteTarget ? `确定删除「${deleteTarget.name}」吗？存在排队或运行任务引用该数据集时，系统会拒绝删除。` : ''} confirmLabel="删除数据集" busy={deleting} onClose={() => !deleting && setDeleteTarget(null)} onConfirm={() => void deleteDataset()} />
    </div>
  )
}

function datasetFormatHint(format: DatasetFormat): string {
  const hints: Record<DatasetFormat, string> = {
    auto: '系统将在训练节点自动识别目录结构。',
    yolo: '压缩包需包含唯一的 data.yaml；类别从 names 自动读取。',
    coco_detection: '压缩包需包含 train/val 的 COCO JSON；类别从 categories 自动读取。',
    voc_detection: '压缩包需包含 Annotations、JPEGImages 和 ImageSets/Main；类别从 XML 自动读取。',
    mask_pairs: '需包含 images/train、masks/train；可用根目录 classes.txt 提供类别名称。',
    coco_segmentation: '压缩包需包含 train/val 的 COCO JSON、图片及 polygon 或 RLE 标注。',
    voc_segmentation: '需包含 JPEGImages、SegmentationClass 和 ImageSets/Segmentation；可提供 classes.txt。',
    ppocr_detection: '压缩包需包含 train.txt、val.txt 和对应图片，标注为 Tab 后 JSON 数组。',
    ppocr_recognition: '压缩包需包含 train.txt、val.txt 和对应图片，标注为 Tab 后识别文本。',
  }
  return hints[format]
}
