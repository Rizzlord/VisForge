import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent as ReactDragEvent,
} from 'react'
import { NodeCanvas } from './editor/NodeCanvas'
import type { EditorSetup } from './editor/createEditor'
import type { NodeCatalogCategory, NodeKind, SerializedWorkflow } from './editor/types'
import { BACKEND_BASE } from './editor/config'
import './App.css'

type WorkflowTab = {
  id: string
  name: string
  data: SerializedWorkflow
}

interface WorkflowStatePayload {
  activeWorkflowId: string | null
  workflows: Array<{
    id: string
    name: string
    data?: SerializedWorkflow
  }>
}

interface BackendLogEntry {
  id: number
  level: string
  logger: string
  message: string
  created: number
}

const LOCAL_WORKFLOWS_KEY = 'visforge.workflows'
const LOCAL_ACTIVE_KEY = 'visforge.activeWorkflowId'
const LOG_POLL_INTERVAL = 2000

function createId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `wf-${Math.random().toString(36).slice(2, 10)}`
}

function readLocalState(): { workflows: WorkflowTab[]; activeId: string | null } {
  if (typeof window === 'undefined') return { workflows: [], activeId: null }
  try {
    const storedWorkflows = window.localStorage.getItem(LOCAL_WORKFLOWS_KEY)
    const storedActive = window.localStorage.getItem(LOCAL_ACTIVE_KEY)
    if (!storedWorkflows) return { workflows: [], activeId: storedActive ?? null }
    const parsed = JSON.parse(storedWorkflows) as WorkflowTab[]
    if (!Array.isArray(parsed)) return { workflows: [], activeId: storedActive ?? null }
    const sanitised = parsed
      .filter((item) => item && typeof item === 'object' && Array.isArray((item as any).data?.nodes))
      .map((item) => ({
        id: typeof item.id === 'string' && item.id.trim() ? item.id.trim() : createId(),
        name: typeof item.name === 'string' && item.name.trim() ? item.name.trim() : 'Workflow',
        data: {
          nodes: Array.isArray(item.data?.nodes) ? item.data.nodes : [],
          connections: Array.isArray(item.data?.connections) ? item.data.connections : [],
        },
      }))
    const activeId = typeof storedActive === 'string' && storedActive.trim() ? storedActive.trim() : null
    return { workflows: sanitised, activeId }
  } catch (error) {
    console.warn('Failed to parse stored workflows', error)
    return { workflows: [], activeId: null }
  }
}

function writeLocalState(workflows: WorkflowTab[], activeId: string | null) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(LOCAL_WORKFLOWS_KEY, JSON.stringify(workflows))
    if (activeId) {
      window.localStorage.setItem(LOCAL_ACTIVE_KEY, activeId)
    } else {
      window.localStorage.removeItem(LOCAL_ACTIVE_KEY)
    }
  } catch (error) {
    console.warn('Failed to persist workflows locally', error)
  }
}

function ensureUniqueName(name: string, existing: Set<string>): string {
  let base = name.trim()
  if (!base) base = 'Workflow'
  let candidate = base
  let counter = 2
  while (existing.has(candidate.toLowerCase())) {
    candidate = `${base} (${counter})`
    counter += 1
  }
  existing.add(candidate.toLowerCase())
  return candidate
}

function normaliseSerializedWorkflow(raw: unknown): SerializedWorkflow | null {
  if (!raw || typeof raw !== 'object') return null
  const payload = 'data' in (raw as Record<string, unknown>) && typeof (raw as any).data === 'object' && (raw as any).data
    ? (raw as any).data
    : raw

  if (!payload || typeof payload !== 'object') return null
  const nodes = Array.isArray((payload as any).nodes) ? (payload as any).nodes : []
  const connections = Array.isArray((payload as any).connections) ? (payload as any).connections : []
  return { nodes, connections }
}

function sanitizeFileName(name: string): string {
  return name.replace(/[^a-z0-9-_]+/gi, '_') || 'workflow'
}

function exportWorkflowPayload(workflow: WorkflowTab) {
  return {
    name: workflow.name,
    data: workflow.data,
  }
}

function downloadWorkflow(workflow: WorkflowTab) {
  try {
    const payload = JSON.stringify(exportWorkflowPayload(workflow), null, 2)
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${sanitizeFileName(workflow.name)}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  } catch (error) {
    console.error('Failed to download workflow', error)
  }
}

function App() {
  const [editorSetup, setEditorSetup] = useState<EditorSetup | null>(null)
  const [workflows, setWorkflows] = useState<WorkflowTab[]>([])
  const [activeWorkflowId, setActiveWorkflowId] = useState<string | null>(null)
  const [isBootstrapping, setBootstrapping] = useState(true)
  const [isPersisting, setPersisting] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const [logs, setLogs] = useState<BackendLogEntry[]>([])

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const mountedRef = useRef(true)
  const bootstrappedRef = useRef(false)
  const logTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const latestLogIdRef = useRef(0)
  const logsBodyRef = useRef<HTMLDivElement | null>(null)
  const [isLogsAutoScroll, setLogsAutoScroll] = useState(true)

  useEffect(() => {
    return () => {
      mountedRef.current = false
      if (logTimerRef.current) {
        clearInterval(logTimerRef.current)
        logTimerRef.current = null
      }
    }
  }, [])

  const catalog: NodeCatalogCategory[] = useMemo(() => editorSetup?.catalog ?? [], [editorSetup])

  const persistState = useCallback(
    async (tabs: WorkflowTab[], activeId: string | null) => {
      writeLocalState(tabs, activeId)
      if (!mountedRef.current) return
      setPersisting(true)
      try {
        const response = await fetch(`${BACKEND_BASE}/workflows/state`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            activeWorkflowId: activeId,
            workflows: tabs.map((wf) => ({
              id: wf.id,
              name: wf.name,
              data: wf.data,
            })),
          }),
        })
        if (!response.ok) {
          const detail = await response.text()
          console.error('Failed to persist workflows', detail)
        }
      } catch (error) {
        console.error('Failed to persist workflows', error)
      } finally {
        if (mountedRef.current) {
          setPersisting(false)
        }
      }
    },
    [],
  )

  useEffect(() => {
    if (!editorSetup) return
    if (bootstrappedRef.current) return
    bootstrappedRef.current = true
    let cancelled = false

    const adoptWorkflows = async (tabs: WorkflowTab[], activeId: string | null, shouldPersist: boolean) => {
      if (cancelled) return
      const resolvedActive = activeId ?? tabs[0]?.id ?? null
      setWorkflows(tabs)
      setActiveWorkflowId(resolvedActive)
      writeLocalState(tabs, resolvedActive)
      if (shouldPersist && tabs.length) {
        await persistState(tabs, resolvedActive)
      }
    }

    const bootstrap = async () => {
      setBootstrapping(true)
      try {
        const response = await fetch(`${BACKEND_BASE}/workflows/state`)
        let payload: WorkflowStatePayload | null = null
        if (response.ok) {
          payload = (await response.json()) as WorkflowStatePayload
        }

        if (payload && Array.isArray(payload.workflows) && payload.workflows.length) {
          const normalised = payload.workflows.map((wf) => ({
            id: wf.id,
            name: wf.name,
            data: normaliseSerializedWorkflow(wf.data) ?? { nodes: [], connections: [] },
          }))
          const activeId = normalised.find((wf) => wf.id === payload?.activeWorkflowId)?.id ?? normalised[0]?.id ?? null
          await adoptWorkflows(normalised, activeId, false)
        } else {
          const local = readLocalState()
          if (local.workflows.length) {
            await adoptWorkflows(local.workflows, local.activeId, true)
          } else {
            const snapshot = await editorSetup.serialize()
            const demo: WorkflowTab = {
              id: createId(),
              name: 'Demo Workflow',
              data: snapshot,
            }
            await adoptWorkflows([demo], demo.id, true)
          }
        }
      } catch (error) {
        console.error('Failed to load workflows', error)
        const local = readLocalState()
        if (local.workflows.length) {
          await adoptWorkflows(local.workflows, local.activeId, true)
        } else {
          const snapshot = await editorSetup.serialize()
          const demo: WorkflowTab = {
            id: createId(),
            name: 'Demo Workflow',
            data: snapshot,
          }
          await adoptWorkflows([demo], demo.id, true)
        }
      } finally {
        if (!cancelled) {
          setBootstrapping(false)
        }
      }
    }

    void bootstrap()

    return () => {
      cancelled = true
    }
  }, [editorSetup, persistState])

  useEffect(() => {
    if (!editorSetup) return
    if (!activeWorkflowId) return
    const active = workflows.find((wf) => wf.id === activeWorkflowId)
    if (!active) return
    void editorSetup.load(active.data)
  }, [editorSetup, activeWorkflowId, workflows])

  const handleAddNode = useCallback(
    async (kind: NodeKind) => {
      if (!editorSetup || isBootstrapping) return
      await editorSetup.addNode(kind)
    },
    [editorSetup, isBootstrapping],
  )

  const handleLibraryDragStart = useCallback((event: ReactDragEvent<HTMLButtonElement>, kind: NodeKind) => {
    event.dataTransfer.setData('application/x-visforge-node', kind)
    event.dataTransfer.setData('text/plain', kind)
    event.dataTransfer.effectAllowed = 'copy'
  }, [])

  const handleLibraryDragEnd = useCallback((event: ReactDragEvent<HTMLButtonElement>) => {
    event.dataTransfer.clearData('application/x-visforge-node')
    event.dataTransfer.clearData('text/plain')
  }, [])

  const handleSelectWorkflow = useCallback(
    async (id: string) => {
      if (isBootstrapping) return
      if (id === activeWorkflowId) return
      setActiveWorkflowId(id)
      await persistState(workflows, id)
    },
    [activeWorkflowId, workflows, persistState, isBootstrapping],
  )

  const handleSave = useCallback(async () => {
    if (!editorSetup || !activeWorkflowId || isBootstrapping) return
    const snapshot = await editorSetup.serialize()
    const nextWorkflows = workflows.map((wf) =>
      wf.id === activeWorkflowId ? { ...wf, data: snapshot } : wf,
    )
    setWorkflows(nextWorkflows)
    await persistState(nextWorkflows, activeWorkflowId)
  }, [editorSetup, activeWorkflowId, workflows, persistState, isBootstrapping])

  const handleSaveAs = useCallback(async () => {
    if (!editorSetup || isBootstrapping) return
    const name = window.prompt('Save workflow as:', 'New Workflow')?.trim()
    if (!name) return
    const snapshot = await editorSetup.serialize()
    const nextWorkflows = [...workflows]
    const uniqueName = ensureUniqueName(name, new Set(nextWorkflows.map((wf) => wf.name.toLowerCase())))
    const newWorkflow: WorkflowTab = {
      id: createId(),
      name: uniqueName,
      data: snapshot,
    }
    nextWorkflows.push(newWorkflow)
    setWorkflows(nextWorkflows)
    setActiveWorkflowId(newWorkflow.id)
    await persistState(nextWorkflows, newWorkflow.id)
    downloadWorkflow(newWorkflow)
  }, [editorSetup, workflows, persistState, isBootstrapping])

  const handleNewWorkflow = useCallback(async () => {
    if (!editorSetup || isBootstrapping) return
    await editorSetup.clear()
    const existingNames = new Set(workflows.map((wf) => wf.name.toLowerCase()))
    const uniqueName = ensureUniqueName(`Workflow ${workflows.length + 1}`, existingNames)
    const fresh: WorkflowTab = {
      id: createId(),
      name: uniqueName,
      data: { nodes: [], connections: [] },
    }
    const nextWorkflows = [...workflows, fresh]
    setWorkflows(nextWorkflows)
    setActiveWorkflowId(fresh.id)
    await persistState(nextWorkflows, fresh.id)
  }, [editorSetup, workflows, persistState, isBootstrapping])

  const handleRenameWorkflow = useCallback(
    (id: string) => {
      if (isBootstrapping) return
      const workflow = workflows.find((wf) => wf.id === id)
      if (!workflow) return
      const name = window.prompt('Rename workflow', workflow.name)?.trim()
      if (!name || name === workflow.name) return
      const existingNames = new Set(workflows.filter((wf) => wf.id !== id).map((wf) => wf.name.toLowerCase()))
      const uniqueName = ensureUniqueName(name, existingNames)
      const nextWorkflows = workflows.map((wf) => (wf.id === id ? { ...wf, name: uniqueName } : wf))
      setWorkflows(nextWorkflows)
      void persistState(nextWorkflows, activeWorkflowId)
    },
    [workflows, activeWorkflowId, persistState, isBootstrapping],
  )

  const handleCloseWorkflow = useCallback(
    async (id: string) => {
      if (isBootstrapping) return
      if (!workflows.length) return
      let nextWorkflows = workflows.filter((wf) => wf.id !== id)
      let nextActive = activeWorkflowId

      if (!nextWorkflows.length) {
        if (editorSetup) {
          await editorSetup.clear()
        }
        const fallback: WorkflowTab = {
          id: createId(),
          name: 'Workflow 1',
          data: { nodes: [], connections: [] },
        }
        nextWorkflows = [fallback]
        nextActive = fallback.id
      } else if (id === activeWorkflowId) {
        nextActive = nextWorkflows[0]?.id ?? null
      }

      setWorkflows(nextWorkflows)
      setActiveWorkflowId(nextActive)
      await persistState(nextWorkflows, nextActive)
    },
    [workflows, activeWorkflowId, editorSetup, persistState, isBootstrapping],
  )

  const importWorkflowFiles = useCallback(
    async (fileList: FileList | File[] | null) => {
      if (!fileList || isBootstrapping) return
      const files = Array.from(fileList)
      if (!files.length) return

      const existingNames = new Set(workflows.map((wf) => wf.name.toLowerCase()))
      const existingIds = new Set(workflows.map((wf) => wf.id))
      const imported: WorkflowTab[] = []
      const failures: string[] = []

      for (const file of files) {
        try {
          const text = await file.text()
          const raw = JSON.parse(text)
          const data = normaliseSerializedWorkflow(raw)
          if (!data) {
            throw new Error('Invalid workflow structure')
          }
          const rawName = typeof raw.name === 'string' && raw.name.trim()
            ? raw.name.trim()
            : file.name.replace(/\.[^.]+$/, '')
          const uniqueName = ensureUniqueName(rawName, existingNames)
          const rawId = typeof raw.id === 'string' && raw.id.trim() ? raw.id.trim() : createId()
          let idCandidate = rawId
          while (existingIds.has(idCandidate)) {
            idCandidate = createId()
          }
          existingIds.add(idCandidate)
          imported.push({ id: idCandidate, name: uniqueName, data })
        } catch (error) {
          failures.push(file.name)
          console.error(`Failed to import workflow "${file.name}"`, error)
        }
      }

      if (!imported.length) {
        return
      }

      const nextWorkflows = [...workflows, ...imported]
      const nextActive = imported[imported.length - 1].id
      setWorkflows(nextWorkflows)
      setActiveWorkflowId(nextActive)
      await persistState(nextWorkflows, nextActive)
    },
    [workflows, persistState, isBootstrapping],
  )

  const handleOpenLoadDialog = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleFileInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const { files } = event.target
      void importWorkflowFiles(files)
      event.target.value = ''
    },
    [importWorkflowFiles],
  )

  const handleToggleLogs = useCallback(() => {
    setShowLogs((prev) => !prev)
  }, [])

  const handleClearLogs = useCallback(() => {
    latestLogIdRef.current = 0
    setLogs([])
  }, [])

  useEffect(() => {
    const handleDragOver = (event: DragEvent) => {
      if (!event.dataTransfer?.types?.includes('Files')) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'copy'
    }

    const handleDrop = (event: DragEvent) => {
      if (!event.dataTransfer?.types?.includes('Files')) return
      event.preventDefault()
      const files = event.dataTransfer.files
      if (files && files.length) {
        void importWorkflowFiles(files)
      }
    }

    window.addEventListener('dragover', handleDragOver)
    window.addEventListener('drop', handleDrop)

    return () => {
      window.removeEventListener('dragover', handleDragOver)
      window.removeEventListener('drop', handleDrop)
    }
  }, [importWorkflowFiles])

  useEffect(() => {
    if (!showLogs) {
      if (logTimerRef.current) {
        clearInterval(logTimerRef.current)
        logTimerRef.current = null
      }
      setLogsAutoScroll(true)
      return
    }

    let isActive = true

    const fetchLogs = async () => {
      try {
        const response = await fetch(`${BACKEND_BASE}/logs?since=${latestLogIdRef.current}`)
        if (!response.ok) return
        const data = (await response.json()) as { logs?: BackendLogEntry[]; latest?: number }
        if (!isActive) return
        const newLogs = Array.isArray(data.logs) ? data.logs : []
        if (newLogs.length) {
          latestLogIdRef.current = data.latest ?? latestLogIdRef.current
          setLogs((prev) => {
            const combined = [...prev, ...newLogs]
            return combined.slice(-1000)
          })
        }
      } catch (error) {
        console.error('Failed to fetch logs', error)
      }
    }

    fetchLogs()
    logTimerRef.current = setInterval(fetchLogs, LOG_POLL_INTERVAL)

    return () => {
      isActive = false
      if (logTimerRef.current) {
        clearInterval(logTimerRef.current)
        logTimerRef.current = null
      }
    }
  }, [showLogs])

  useEffect(() => {
    if (!showLogs) return
    const body = logsBodyRef.current
    if (!body) return

    const handleScroll = () => {
      const distanceFromBottom = body.scrollHeight - (body.scrollTop + body.clientHeight)
      const atBottom = distanceFromBottom <= 16
      setLogsAutoScroll((prev) => (prev === atBottom ? prev : atBottom))
    }

    handleScroll()
    body.addEventListener('scroll', handleScroll)
    return () => {
      body.removeEventListener('scroll', handleScroll)
    }
  }, [showLogs])

  useEffect(() => {
    if (!showLogs || !isLogsAutoScroll) return
    const body = logsBodyRef.current
    if (!body) return
    body.scrollTop = body.scrollHeight
  }, [logs, showLogs, isLogsAutoScroll])

  const formatLogTime = useCallback((timestamp: number) => {
    try {
      return new Date(timestamp * 1000).toLocaleTimeString()
    } catch (error) {
      return ''
    }
  }, [])

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-left">
          <div className="brand">VisForge</div>
          <div className="workflow-tabs">
            {workflows.map((workflow) => (
              <button
                type="button"
                key={workflow.id}
                className={workflow.id === activeWorkflowId ? 'active' : ''}
                onClick={() => void handleSelectWorkflow(workflow.id)}
                onDoubleClick={() => handleRenameWorkflow(workflow.id)}
              >
                <span className="workflow-tab-label">{workflow.name}</span>
                <span
                  className="workflow-tab-close"
                  role="button"
                  tabIndex={0}
                  aria-label={`Close ${workflow.name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    void handleCloseWorkflow(workflow.id)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      event.stopPropagation()
                      void handleCloseWorkflow(workflow.id)
                    }
                  }}
                >
                  ×
                </span>
              </button>
            ))}
            <button
              type="button"
              className="tab-add"
              onClick={() => void handleNewWorkflow()}
              disabled={isBootstrapping || isPersisting}
            >
              +
            </button>
          </div>
        </div>
        <div className="header-actions">
          <button type="button" className={showLogs ? 'active' : ''} onClick={handleToggleLogs}>
            {showLogs ? 'Hide Logs' : 'Logs'}
          </button>
          <button
            type="button"
            onClick={() => void handleOpenLoadDialog()}
            disabled={isBootstrapping || isPersisting}
          >
            Load
          </button>
          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={!editorSetup || !activeWorkflowId || isBootstrapping || isPersisting}
          >
            Save
          </button>
          <button
            type="button"
            onClick={() => void handleSaveAs()}
            disabled={!editorSetup || isBootstrapping || isPersisting}
          >
            Save As
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json"
          multiple
          style={{ display: 'none' }}
          onChange={handleFileInputChange}
        />
      </header>
      <main className="workspace">
        <aside className="library-pane">
          <h2>Node Library</h2>
          {catalog.map((category) => (
            <section key={category.id} className="library-category">
              <h3>{category.label}</h3>
              <div className="library-items">
                {category.entries.map((entry) => (
                  <button
                    type="button"
                    key={entry.kind}
                    onClick={() => void handleAddNode(entry.kind)}
                    draggable
                    onDragStart={(event) => handleLibraryDragStart(event, entry.kind)}
                    onDragEnd={handleLibraryDragEnd}
                    disabled={!editorSetup || isBootstrapping}
                  >
                    <span className="entry-title">{entry.label}</span>
                    {entry.description && <span className="entry-sub">{entry.description}</span>}
                  </button>
                ))}
              </div>
            </section>
          ))}
          {!catalog.length && <p className="library-placeholder">Editor starting…</p>}
        </aside>
        <section className="canvas-pane">
          <NodeCanvas onReady={setEditorSetup} />
        </section>
        <aside className={`logs-pane${showLogs ? ' open' : ''}`} aria-hidden={!showLogs}>
          <div className="logs-pane__header">
            <h2>Generator Logs</h2>
            <div className="logs-pane__actions">
              <button type="button" onClick={handleClearLogs} disabled={!logs.length}>
                Clear
              </button>
            </div>
          </div>
          <div className="logs-pane__body" ref={logsBodyRef}>
            {logs.length ? (
              logs.map((entry) => (
                <div key={entry.id} className={`logs-entry logs-entry--${entry.level.toLowerCase()}`}>
                  <div className="logs-entry__meta">
                    <span className="logs-entry__time">{formatLogTime(entry.created)}</span>
                    <span className="logs-entry__level">{entry.level}</span>
                    <span className="logs-entry__logger">{entry.logger}</span>
                  </div>
                  <div className="logs-entry__message">{entry.message}</div>
                </div>
              ))
            ) : (
              <p className="logs-pane__empty">Logs will appear here while generators run.</p>
            )}
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
