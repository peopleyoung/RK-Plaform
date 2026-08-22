export function pascalColor(classId: number): readonly [number, number, number] {
  if (!Number.isInteger(classId) || classId < 0 || classId > 255) {
    throw new Error('Pascal class ID must be an integer from 0 through 255')
  }
  let value = classId
  let red = 0
  let green = 0
  let blue = 0
  for (let shift = 0; shift < 8; shift += 1) {
    red |= (value & 1) << (7 - shift)
    green |= ((value >> 1) & 1) << (7 - shift)
    blue |= ((value >> 2) & 1) << (7 - shift)
    value >>= 3
  }
  return [red, green, blue]
}
