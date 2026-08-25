import type { Dataset, DatasetFormat, JobStatus, StatusTone, TaskType, WorkerNode } from '../types'

export const taskLabels: Record<TaskType, string> = {
  object_detection: '目标检测',
  semantic_segmentation: '语义分割',
  ocr_detection: '文字检测',
  ocr_recognition: '文字识别',
}

export const datasetFormatLabels: Record<DatasetFormat, string> = {
  auto: '自动识别',
  yolo: 'YOLO',
  coco_detection: 'COCO 检测',
  voc_detection: 'Pascal VOC 检测',
  mask_pairs: '图像/掩码配对',
  coco_segmentation: 'COCO 分割',
  voc_segmentation: 'Pascal VOC 分割',
  ppocr_detection: 'PPOCR 检测',
  ppocr_recognition: 'PPOCR 识别',
}

export const datasetFormatsByTask: Record<TaskType, DatasetFormat[]> = {
  object_detection: ['yolo', 'coco_detection', 'voc_detection'],
  semantic_segmentation: ['auto', 'mask_pairs', 'coco_segmentation', 'voc_segmentation'],
  ocr_detection: ['ppocr_detection'],
  ocr_recognition: ['ppocr_recognition'],
}

export const defaultDatasetFormat: Record<TaskType, DatasetFormat> = {
  object_detection: 'yolo',
  semantic_segmentation: 'auto',
  ocr_detection: 'ppocr_detection',
  ocr_recognition: 'ppocr_recognition',
}

export const jobStatusLabels: Record<JobStatus, string> = {
  queued: '排队中',
  claimed: '已领取',
  running: '运行中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
}

export function isJobDeletable(status: JobStatus): boolean {
  return ['queued', 'succeeded', 'failed', 'cancelled'].includes(status)
}

export function jobTone(status: JobStatus): StatusTone {
  if (status === 'succeeded') return 'success'
  if (status === 'failed' || status === 'cancelled') return 'danger'
  if (status === 'running' || status === 'claimed') return 'info'
  return 'warning'
}

export function workerTone(status: WorkerNode['status']): StatusTone {
  if (status === 'online') return 'success'
  if (status === 'busy') return 'info'
  return 'danger'
}

export function datasetTone(status: Dataset['status']): StatusTone {
  if (status === 'ready') return 'success'
  if (status === 'failed') return 'danger'
  return 'info'
}

export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / 1024 ** 3).toFixed(1)} GB`
}

export function formatTime(value: string | null) {
  if (!value) return '--'
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

export function variantLabel(value: string) {
  if (value === 'mobilenet_v2_rknn') return 'MobileNetV2（RK3588 适配版）'
  if (value === 'mobilenet_v2') return 'MobileNetV2（原版 DeepLabV3+）'
  if (value === 'resnet50') return 'ResNet50（原版 DeepLabV3+）'
  return value.replace('ppocrv', 'PP-OCRv').replace('_det', ' Det').replace('_rec', ' Rec')
}
