import type { InferenceGraph, InferenceTask } from './types'

export function inferenceGraphNode(graph: InferenceGraph, operator: string) {
  return graph.nodes.find((node) => node.operator === operator)
}

export function primaryReleaseId(task: InferenceTask): string {
  const value = inferenceGraphNode(task.graph, 'inference.primary')?.config.releaseId
  return typeof value === 'string' ? value : ''
}

export function taskAnalytics(task: InferenceTask): Record<string, unknown> {
  const analytics = inferenceGraphNode(task.graph, 'processing.analytics')?.config ?? {}
  const events = inferenceGraphNode(task.graph, 'processing.events')?.config
  return events ? { ...analytics, events } : analytics
}
