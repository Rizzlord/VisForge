import { useEffect } from 'react'
import { useRete } from 'rete-react-plugin'
import { createEditor } from './createEditor'
import type { EditorSetup } from './createEditor'

interface NodeCanvasProps {
  onReady?: (setup: EditorSetup | null) => void
}

export function NodeCanvas(props: NodeCanvasProps) {
  const { onReady } = props
  const [ref, editor] = useRete<EditorSetup>(createEditor)

  useEffect(() => {
    onReady?.(editor ?? null)
    return () => onReady?.(null)
  }, [editor, onReady])

  return <div className="node-canvas" ref={ref} />
}
