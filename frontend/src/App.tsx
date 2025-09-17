import { useCallback, useEffect, useMemo, useState } from 'react'
import { NodeCanvas } from './editor/NodeCanvas'
import type { EditorSetup } from './editor/createEditor'
import type { NodeCatalogCategory, NodeKind, SerializedWorkflow } from './editor/types'
import './App.css'

type WorkflowTab = {
  id: string
  name: string
  data: SerializedWorkflow
}

const STORAGE_KEY = 'visforge.workflows'

function readStoredWorkflows(): WorkflowTab[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as WorkflowTab[]
    if (!Array.isArray(parsed)) return []
    return parsed
  } catch (error) {
    console.warn('Failed to parse stored workflows', error)
    return []
  }
}

function persistWorkflows(workflows: WorkflowTab[]) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(workflows))
}

const EMPTY_WORKFLOW: SerializedWorkflow = { nodes: [], connections: [] }

function createId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `wf-${Math.random().toString(36).slice(2, 10)}`
}

function App() {
  const [editorSetup, setEditorSetup] = useState<EditorSetup | null>(null)
  const [workflows, setWorkflows] = useState<WorkflowTab[]>(() => readStoredWorkflows())
  const [activeWorkflowId, setActiveWorkflowId] = useState<string | null>(
    () => readStoredWorkflows()[0]?.id ?? null,
  )

  const catalog: NodeCatalogCategory[] = useMemo(() => editorSetup?.catalog ?? [], [editorSetup])

  useEffect(() => {
    if (!editorSetup) return

    if (!workflows.length) {
      void editorSetup.serialize().then((snapshot) => {
        const initial: WorkflowTab = {
          id: createId(),
          name: 'Demo Workflow',
          data: snapshot,
        }
        setWorkflows([initial])
        setActiveWorkflowId(initial.id)
      })
      return
    }

    if (!activeWorkflowId && workflows[0]) {
      setActiveWorkflowId(workflows[0].id)
      return
    }
  }, [editorSetup])

  useEffect(() => {
    if (!editorSetup) return
    if (!activeWorkflowId) return
    const active = workflows.find((wf) => wf.id === activeWorkflowId)
    if (!active) return
    void editorSetup.load(active.data)
  }, [editorSetup, activeWorkflowId, workflows])

  useEffect(() => {
    if (workflows.length) {
      persistWorkflows(workflows)
    }
  }, [workflows])

  const handleAddNode = useCallback(
    async (kind: NodeKind) => {
      if (!editorSetup) return
      await editorSetup.addNode(kind)
    },
    [editorSetup],
  )

  const handleSelectWorkflow = useCallback(
    async (id: string) => {
      setActiveWorkflowId(id)
      if (!editorSetup) return
      const workflow = workflows.find((wf) => wf.id === id)
      if (!workflow) return
      await editorSetup.load(workflow.data)
    },
    [editorSetup, workflows],
  )

  const handleSave = useCallback(async () => {
    if (!editorSetup || !activeWorkflowId) return
    const snapshot = await editorSetup.serialize()
    setWorkflows((prev) =>
      prev.map((wf) => (wf.id === activeWorkflowId ? { ...wf, data: snapshot } : wf)),
    )
  }, [editorSetup, activeWorkflowId])

  const handleSaveAs = useCallback(async () => {
    if (!editorSetup) return
    const name = window.prompt('Save workflow as:', 'New Workflow')?.trim()
    if (!name) return
    const snapshot = await editorSetup.serialize()
    const newWorkflow: WorkflowTab = {
      id: createId(),
      name,
      data: snapshot,
    }
    setWorkflows((prev) => [...prev, newWorkflow])
    setActiveWorkflowId(newWorkflow.id)
  }, [editorSetup])

  const handleNewWorkflow = useCallback(async () => {
    if (!editorSetup) return
    await editorSetup.clear()
    const fresh: WorkflowTab = {
      id: createId(),
      name: `Workflow ${workflows.length + 1}`,
      data: EMPTY_WORKFLOW,
    }
    setWorkflows((prev) => [...prev, fresh])
    setActiveWorkflowId(fresh.id)
  }, [editorSetup, workflows.length])

  const handleRenameWorkflow = useCallback(
    (id: string) => {
      const workflow = workflows.find((wf) => wf.id === id)
      if (!workflow) return
      const name = window.prompt('Rename workflow', workflow.name)?.trim()
      if (!name || name === workflow.name) return
      setWorkflows((prev) => prev.map((wf) => (wf.id === id ? { ...wf, name } : wf)))
    },
    [workflows],
  )

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
                {workflow.name}
              </button>
            ))}
            <button type="button" className="tab-add" onClick={() => void handleNewWorkflow()}>
              +
            </button>
          </div>
        </div>
        <div className="header-actions">
          <button type="button" onClick={() => void handleSave()} disabled={!editorSetup || !activeWorkflowId}>
            Save
          </button>
          <button type="button" onClick={() => void handleSaveAs()} disabled={!editorSetup}>
            Save As
          </button>
        </div>
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
      </main>
    </div>
  )
}

export default App
