export interface PageResponse<T> {
  items: T[]
  page: number
  pageSize: number
  total: number
}

export async function loadAllPages<T>(loader: (page: number, pageSize: number) => Promise<PageResponse<T>>): Promise<T[]> {
  const pageSize = 100
  const first = await loader(1, pageSize)
  const items = [...first.items]
  for (let page = 2; items.length < first.total; page += 1) {
    const next = await loader(page, pageSize)
    if (!next.items.length) break
    items.push(...next.items)
  }
  return items
}
