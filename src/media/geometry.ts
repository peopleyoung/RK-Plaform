export interface ContainTransform {
  readonly scale: number
  readonly offsetX: number
  readonly offsetY: number
  readonly width: number
  readonly height: number
}

export function computeContainTransform(
  sourceWidth: number,
  sourceHeight: number,
  containerWidth: number,
  containerHeight: number,
): ContainTransform {
  if (![sourceWidth, sourceHeight, containerWidth, containerHeight].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error('Source and container dimensions must be positive')
  }
  const scale = Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight)
  const width = sourceWidth * scale
  const height = sourceHeight * scale
  return {
    scale,
    offsetX: (containerWidth - width) / 2,
    offsetY: (containerHeight - height) / 2,
    width,
    height,
  }
}

export function mapPoint(transform: ContainTransform, x: number, y: number): { x: number; y: number } {
  return {
    x: transform.offsetX + x * transform.scale,
    y: transform.offsetY + y * transform.scale,
  }
}
