import { useEffect, type ButtonHTMLAttributes, type CSSProperties, type ReactNode } from 'react'
import {
  Boxes,
  ChevronRight,
  CircleUserRound,
  Cpu,
  Database,
  Gauge,
  Layers3,
  MonitorPlay,
  Plus,
  RefreshCw,
  ServerCog,
  Settings,
  X,
  Zap,
} from 'lucide-react'
import { usePlatform } from './api/PlatformContext'
import type { RouteKey, StatusTone } from './types'

const navItems: Array<{ id: RouteKey; label: string; icon: typeof Gauge }> = [
  { id: 'overview', label: '工作台', icon: Gauge },
  { id: 'datasets', label: '数据集', icon: Database },
  { id: 'training', label: '模型训练', icon: Zap },
  { id: 'conversion', label: '模型转换', icon: Boxes },
  { id: 'nodes', label: '算力节点', icon: ServerCog },
  { id: 'inference', label: '推理下发', icon: Cpu },
  { id: 'monitoring', label: '视频监控', icon: MonitorPlay },
  { id: 'settings', label: '系统设置', icon: Settings },
]

const routeTitles: Record<RouteKey, { title: string; subtitle: string }> = {
  overview: { title: '工作台', subtitle: '训练与转换任务运行概况' },
  datasets: { title: '数据集', subtitle: '管理训练数据、版本和标注状态' },
  training: { title: '模型训练', subtitle: '在 CPU 或 NVIDIA GPU 节点调度训练任务' },
  conversion: { title: '模型转换', subtitle: '在 RK3588 Docker 节点生成 RKNN 模型' },
  nodes: { title: '算力节点', subtitle: '查看训练、转换与推理节点的运行状态' },
  inference: { title: '推理下发', subtitle: '管理 RK3588 板卡、模型版本和灰度部署' },
  monitoring: { title: '视频监控', subtitle: '多画面查看 RK3588 实时推理结果' },
  settings: { title: '系统设置', subtitle: '统一配置训练、转换与推理节点' },
}

interface AppShellProps {
  route: RouteKey
  onNavigate: (route: RouteKey) => void
  children: ReactNode
}

export function AppShell({ route, onNavigate, children }: AppShellProps) {
  const { serviceEndpoints, error, refresh } = usePlatform()
  const endpointSummary = (kind: 'trainer' | 'converter' | 'inference') => {
    const endpoints = serviceEndpoints.filter((endpoint) => endpoint.kind === kind)
    return `${endpoints.filter((endpoint) => endpoint.enabled && endpoint.probeStatus === 'online').length} / ${endpoints.length}`
  }
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => onNavigate('overview')} aria-label="返回工作台">
          <span className="brand-mark"><Layers3 size={19} strokeWidth={2.4} /></span>
          <span className="brand-copy"><strong>RKNode</strong><small>模型工作台</small></span>
        </button>

        <nav className="primary-nav" aria-label="主导航">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <button key={item.id} className={route === item.id ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate(item.id)}>
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            )
          })}
        </nav>

        <div className="sidebar-runtime">
          <div className="runtime-head"><span className={error ? 'health-dot offline' : 'health-dot'} />{error ? '平台连接异常' : '调度服务已连接'}</div>
          <div className="runtime-row"><span>训练节点</span><strong>{endpointSummary('trainer')}</strong></div>
          <div className="runtime-row"><span>转换节点</span><strong>{endpointSummary('converter')}</strong></div>
          <div className="runtime-row"><span>推理节点</span><strong>{endpointSummary('inference')}</strong></div>
        </div>

        <div className="sidebar-bottom">
          <button className="user-chip" title="当前用户">
            <CircleUserRound size={26} />
            <span><strong>管理员</strong><small>admin</small></span>
          </button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{routeTitles[route].title}</h1>
            <span>{routeTitles[route].subtitle}</span>
          </div>
          <div className="topbar-actions">
            <span className="environment-pill"><span className={error ? 'health-dot offline' : 'health-dot'} />{error ? '连接异常' : '平台 API'}</span>
            <button className="icon-button ghost" aria-label="刷新数据" title="刷新数据" onClick={() => void refresh()}><RefreshCw size={18} /></button>
          </div>
        </header>
        <div className="mobile-nav" aria-label="移动端导航">
          {navItems.map((item) => {
            const Icon = item.icon
            return <button key={item.id} className={route === item.id ? 'active' : ''} onClick={() => onNavigate(item.id)}><Icon size={16} />{item.label}</button>
          })}
        </div>
        <div className="page-content">{children}</div>
      </main>
    </div>
  )
}

type ButtonVariant = 'primary' | 'secondary' | 'quiet' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  icon?: ReactNode
}

export function Button({ variant = 'primary', icon, className = '', children, ...props }: ButtonProps) {
  return <button className={`button ${variant} ${className}`} {...props}>{icon}{children}</button>
}

interface PageHeaderProps {
  eyebrow?: string
  title: string
  description?: string
  actions?: ReactNode
}

export function PageHeader({ eyebrow, title, description, actions }: PageHeaderProps) {
  return (
    <div className="page-header">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  )
}

interface StatusBadgeProps {
  tone: StatusTone
  children: ReactNode
  dot?: boolean
}

export function StatusBadge({ tone, children, dot = true }: StatusBadgeProps) {
  return <span className={`status-badge ${tone}`}>{dot && <span className="status-dot" />}{children}</span>
}

interface MetricCardProps {
  label: string
  value: string
  detail: ReactNode
  icon: ReactNode
  tone?: 'teal' | 'amber' | 'blue' | 'coral'
}

export function MetricCard({ label, value, detail, icon, tone = 'teal' }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${tone}`}>{icon}</div>
      <div className="metric-body"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
    </div>
  )
}

interface ProgressBarProps {
  value: number
  tone?: 'teal' | 'amber' | 'blue' | 'danger'
  showValue?: boolean
  compact?: boolean
}

export function ProgressBar({ value, tone = 'teal', showValue = true, compact = false }: ProgressBarProps) {
  return (
    <div className={compact ? 'progress-wrap compact' : 'progress-wrap'}>
      <div className="progress-track"><span className={`progress-fill ${tone}`} style={{ '--progress': `${value}%` } as CSSProperties} /></div>
      {showValue && <span className="progress-value">{value}%</span>}
    </div>
  )
}

interface ModalProps {
  open: boolean
  title: string
  description?: string
  width?: 'medium' | 'large' | 'wide'
  dismissible?: boolean
  onClose: () => void
  children: ReactNode
  footer?: ReactNode
}

export function Modal({ open, title, description, width = 'medium', dismissible = true, onClose, children, footer }: ModalProps) {
  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => event.key === 'Escape' && dismissible && onClose()
    document.addEventListener('keydown', onKeyDown)
    document.body.classList.add('modal-open')
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.body.classList.remove('modal-open')
    }
  }, [dismissible, open, onClose])

  if (!open) return null
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => dismissible && event.target === event.currentTarget && onClose()}>
      <section className={`modal ${width}`} role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <header className="modal-header">
          <div><h2 id="modal-title">{title}</h2>{description && <p>{description}</p>}</div>
          {dismissible && <button className="icon-button ghost" onClick={onClose} aria-label="关闭" title="关闭"><X size={19} /></button>}
        </header>
        <div className="modal-content">{children}</div>
        {footer && <footer className="modal-footer">{footer}</footer>}
      </section>
    </div>
  )
}

export function EmptyState({ title, message }: { title: string; message: string }) {
  return <div className="empty-state"><Cpu size={28} /><strong>{title}</strong><span>{message}</span></div>
}

export function ChevronAction({ label, onClick }: { label: string; onClick?: () => void }) {
  return <button className="chevron-action" onClick={onClick}>{label}<ChevronRight size={15} /></button>
}

export function AddButton({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return <Button icon={<Plus size={17} />} onClick={onClick}>{children}</Button>
}

interface TablePaginationProps {
  total: number
  page: number
  pageSize: number
  onPageChange: (page: number) => void
  onPageSizeChange: (pageSize: number) => void
}

export function TablePagination({ total, page, pageSize, onPageChange, onPageSizeChange }: TablePaginationProps) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize))
  const currentPage = Math.min(page, pageCount)
  const start = total ? (currentPage - 1) * pageSize + 1 : 0
  const end = Math.min(total, currentPage * pageSize)
  return <footer className="table-pagination" aria-label="分页控制">
    <span>显示 {start}–{end} 条，共 {total} 条</span>
    <label>每页<select value={pageSize} onChange={(event) => onPageSizeChange(Number(event.target.value))} aria-label="每页数量"><option value={10}>10</option><option value={20}>20</option><option value={50}>50</option></select>条</label>
    <div className="pagination-actions"><button className="icon-button ghost" title="上一页" aria-label="上一页" disabled={currentPage <= 1} onClick={() => onPageChange(currentPage - 1)}><ChevronRight size={17} className="pagination-previous" /></button><strong>{currentPage} / {pageCount}</strong><button className="icon-button ghost" title="下一页" aria-label="下一页" disabled={currentPage >= pageCount} onClick={() => onPageChange(currentPage + 1)}><ChevronRight size={17} /></button></div>
  </footer>
}

interface ConfirmDialogProps {
  open: boolean
  title: string
  description: string
  confirmLabel: string
  busy?: boolean
  onClose: () => void
  onConfirm: () => void
}

export function ConfirmDialog({ open, title, description, confirmLabel, busy = false, onClose, onConfirm }: ConfirmDialogProps) {
  return <Modal open={open} title={title} description={description} onClose={onClose} footer={<><Button variant="secondary" onClick={onClose} disabled={busy}>取消</Button><Button variant="danger" onClick={onConfirm} disabled={busy}>{busy ? '正在删除…' : confirmLabel}</Button></>}><p className="confirm-dialog-copy">此操作会永久移除平台中的关联记录和文件，无法撤销。</p></Modal>
}
