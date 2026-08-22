import { useEffect, useRef, useState } from 'react'
import { CheckCircle2 } from 'lucide-react'
import { PlatformProvider, usePlatform } from './api/PlatformContext'
import { AppShell, Button, Modal } from './components'
import { ConversionPage } from './pages/ConversionPage'
import { DatasetsPage } from './pages/DatasetsPage'
import { NodesPage } from './pages/NodesPage'
import { InferencePage } from './pages/InferencePage'
import { OverviewPage } from './pages/OverviewPage'
import { TrainingPage } from './pages/TrainingPage'
import { SettingsPage } from './pages/SettingsPage'
import { VideoWallPage } from './pages/VideoWallPage'
import type { RouteKey } from './types'

const validRoutes: RouteKey[] = ['overview', 'datasets', 'training', 'conversion', 'nodes', 'inference', 'monitoring', 'settings']

function routeFromHash(): RouteKey {
  const value = window.location.hash.replace('#/', '') as RouteKey
  return validRoutes.includes(value) ? value : 'overview'
}

export function App() {
  const [route, setRoute] = useState<RouteKey>(routeFromHash)
  const [toast, setToast] = useState('')
  const [createTrainingSignal, setCreateTrainingSignal] = useState(0)
  const toastTimer = useRef<number | null>(null)

  useEffect(() => {
    const onHashChange = () => setRoute(routeFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => () => {
    if (toastTimer.current) window.clearTimeout(toastTimer.current)
  }, [])

  const navigate = (next: RouteKey) => {
    window.location.hash = `/${next}`
    setRoute(next)
  }

  const notify = (message: string) => {
    setToast(message)
    if (toastTimer.current) window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(''), 3200)
  }

  const createTraining = () => {
    setCreateTrainingSignal((value) => value + 1)
    navigate('training')
  }

  return (
    <PlatformProvider>
      <AppShell route={route} onNavigate={navigate}>
        {route === 'overview' && <OverviewPage onNavigate={navigate} onCreateTraining={createTraining} notify={notify} />}
        {route === 'datasets' && <DatasetsPage notify={notify} />}
        {route === 'training' && <TrainingPage createSignal={createTrainingSignal} notify={notify} />}
        {route === 'conversion' && <ConversionPage notify={notify} />}
        {route === 'nodes' && <NodesPage notify={notify} />}
        {route === 'inference' && <InferencePage notify={notify} />}
        {route === 'monitoring' && <VideoWallPage />}
        {route === 'settings' && <SettingsPage notify={notify} />}
        <div className={toast ? 'toast show' : 'toast'} role="status" aria-live="polite"><CheckCircle2 size={18} />{toast}</div>
      </AppShell>
      <AuthPrompt />
    </PlatformProvider>
  )
}

function AuthPrompt() {
  const { authRequired, authenticate } = usePlatform()
  const [token, setToken] = useState('')
  const [submitting, setSubmitting] = useState(false)
  return <Modal open={authRequired} title="需要管理员身份" description="当前会话没有平台操作权限，请输入管理员令牌后继续。" dismissible={false} onClose={() => undefined} footer={<Button disabled={!token || submitting} onClick={() => { setSubmitting(true); void authenticate(token).finally(() => setSubmitting(false)) }}>{submitting ? '正在验证…' : '验证并继续'}</Button>}><label className="field"><span>管理员令牌</span><input type="password" autoComplete="current-password" value={token} onChange={(event) => setToken(event.target.value)} /></label></Modal>
}
