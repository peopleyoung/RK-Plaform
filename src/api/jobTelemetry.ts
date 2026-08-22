import type { JobEvent } from '../types'

export interface TrainingLogLine {
  eventId: number
  level: string
  stage: string
  message: string
  createdAt: string
}

export interface TrainingMetricSample {
  eventId: number
  metrics: Record<string, number>
  step: number | null
  epoch: number | null
  totalEpochs: number | null
  createdAt: string
}

const metricLabels: Record<string, string> = {
  loss: '损失',
  train_loss: '训练损失',
  val_loss: '验证损失',
  box_loss: '边框损失',
  obj_loss: '目标损失',
  cls_loss: '分类损失',
  dfl_loss: 'DFL 损失',
  train_box_loss: '训练边框损失',
  train_cls_loss: '训练分类损失',
  train_dfl_loss: '训练 DFL 损失',
  val_box_loss: '验证边框损失',
  val_cls_loss: '验证分类损失',
  val_dfl_loss: '验证 DFL 损失',
  map50: 'mAP@0.5',
  map50_95: 'mAP@0.5:0.95',
  precision: '精确率',
  recall: '召回率',
  accuracy: '准确率',
  pixel_accuracy: '像素准确率',
  mean_iou: 'mIoU',
  hmean: 'Hmean',
  f1: 'F1',
  lr: '学习率',
  norm_edit_dis: '归一化编辑距离',
}

export const metricPriority = [
  'val_loss', 'train_loss', 'loss', 'map50_95', 'map50', 'mean_iou',
  'pixel_accuracy', 'accuracy', 'hmean', 'precision', 'recall', 'f1', 'lr',
]

export function metricLabel(name: string): string {
  return metricLabels[name] ?? name.replaceAll('_', ' ')
}

export function projectMetricSamples(events: JobEvent[]): TrainingMetricSample[] {
  const samples: TrainingMetricSample[] = []
  for (const event of events) {
    const metrics = numericRecord(event.data.metrics)
    if (!Object.keys(metrics).length) continue
    samples.push({
      eventId: event.id,
      metrics,
      step: optionalNumber(event.data.step),
      epoch: optionalNumber(event.data.epoch),
      totalEpochs: optionalNumber(event.data.totalEpochs),
      createdAt: event.createdAt,
    })
  }
  return samples
}

export function projectLogLines(events: JobEvent[]): TrainingLogLine[] {
  const lines: TrainingLogLine[] = []
  for (const event of events) {
    if (!event.message.trim()) continue
    const stage = typeof event.data.stage === 'string' ? event.data.stage : event.type
    for (const message of event.message.split(/\r?\n/)) {
      if (!message.trim()) continue
      lines.push({
        eventId: event.id,
        level: event.level,
        stage,
        message: message.replace(/^RKNODE_METRIC\s+/, ''),
        createdAt: event.createdAt,
      })
    }
  }
  return lines.slice(-5000)
}

export function metricNames(samples: TrainingMetricSample[]): string[] {
  const names = new Set(samples.flatMap((sample) => Object.keys(sample.metrics)))
  return [...names].sort((left, right) => {
    const leftPriority = metricPriority.indexOf(left)
    const rightPriority = metricPriority.indexOf(right)
    if (leftPriority >= 0 || rightPriority >= 0) {
      return (leftPriority < 0 ? Number.MAX_SAFE_INTEGER : leftPriority)
        - (rightPriority < 0 ? Number.MAX_SAFE_INTEGER : rightPriority)
    }
    return left.localeCompare(right)
  })
}

function numericRecord(value: unknown): Record<string, number> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const result: Record<string, number> = {}
  for (const [name, metric] of Object.entries(value)) {
    if (typeof metric === 'number' && Number.isFinite(metric)) result[name] = metric
  }
  return result
}

function optionalNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
