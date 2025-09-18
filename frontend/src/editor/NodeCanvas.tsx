import { useEffect } from 'react'
import { useRete } from 'rete-react-plugin'
import { createEditor } from './createEditor'
import type { EditorSetup } from './createEditor'
import type { NodeKind } from './types'

interface NodeCanvasProps {
  onReady?: (setup: EditorSetup | null) => void
}

export function NodeCanvas(props: NodeCanvasProps) {
  const { onReady } = props
  const [containerRef, editor] = useRete<EditorSetup>(createEditor)

  useEffect(() => {
    onReady?.(editor ?? null)
    return () => onReady?.(null)
  }, [editor, onReady])

  useEffect(() => {
    const element = containerRef.current
    if (!element) return

    const handleDragOver = (event: DragEvent) => {
      if (!event.dataTransfer?.types.includes('application/x-visforge-node')) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'copy'
    }

    const handleDrop = async (event: DragEvent) => {
      if (!editor) return
      if (!event.dataTransfer?.types.includes('application/x-visforge-node')) return
      event.preventDefault()

      const kindRaw = event.dataTransfer.getData('application/x-visforge-node')
      if (!kindRaw) return
      const kind = kindRaw as NodeKind
      const isKnown = editor.catalog.some((category) =>
        category.entries.some((entry) => entry.kind === kind),
      )
      if (!isKnown) return

      const position = editor.projectScreenPoint({ x: event.clientX, y: event.clientY })
      await editor.addNode(kind, position)
    }

    element.addEventListener('dragover', handleDragOver)
    element.addEventListener('drop', handleDrop)

    return () => {
      element.removeEventListener('dragover', handleDragOver)
      element.removeEventListener('drop', handleDrop)
    }
  }, [containerRef, editor])

  return <div className="node-canvas" ref={containerRef} />
}
