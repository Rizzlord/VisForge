import { useCallback, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { createRoot } from 'react-dom/client'
import { ClassicPreset, NodeEditor } from 'rete'
import type { GetSchemes } from 'rete'
import { AreaPlugin, AreaExtensions } from 'rete-area-plugin'
import { ConnectionPlugin, Presets as ConnectionPresets } from 'rete-connection-plugin'
import { ReactPlugin, Presets as ReactPresets } from 'rete-react-plugin'
import type { ReactArea2D, RenderEmit } from 'rete-react-plugin'

import {
  ChannelsPreviewControl,
  ChannelsPreviewControlView,
  ImageDisplayControl,
  ImageDisplayControlView,
  ImageUploadControl,
  ImageUploadControlView,
  ModelUploadControl,
  ModelUploadControlView,
  Preview3DControl,
  Preview3DControlView,
  HunyuanGenerationControl,
  HunyuanGenerationControlView,
  TripoGenerationControl,
  TripoGenerationControlView,
  SaveModelControl,
  SaveModelControlView,
  SaveImageControl,
  SaveImageControlView,
} from './controls'
import { useGraphStore } from './store'
import type {
  ChannelKey,
  ChannelValue,
  GraphOutputs,
  ImageValue,
  ModelValue,
  NodeCatalogCategory,
  NodeKind,
  NodeOutputMap,
  NodeOutputValue,
  SerializedConnection,
  SerializedNode,
  SerializedNodeState,
  SerializedWorkflow,
} from './types'
import { combineChannels, separateChannels } from './imageUtils'

export type Schemes = GetSchemes<ClassicPreset.Node, ClassicPreset.Connection<ClassicPreset.Node, ClassicPreset.Node>>
export type AreaExtra = ReactArea2D<Schemes>

const imageSocket = new ClassicPreset.Socket('Image')
const channelSocket = new ClassicPreset.Socket('Channel')
const modelSocket = new ClassicPreset.Socket('Model')

class FoldableNode extends ClassicPreset.Node {
  collapsed = false
  readonly kind: NodeKind
  width?: number
  height?: number

  constructor(label: string, kind: NodeKind) {
    super(label)
    this.kind = kind
  }
}

class LoadImageNode extends FoldableNode {
  readonly uploader: ImageUploadControl

  constructor() {
    super('Load Image', 'loadImage')
    this.addOutput('image', new ClassicPreset.Output(imageSocket, 'Image'))
    this.uploader = new ImageUploadControl(this.id)
    this.addControl('uploader', this.uploader)
  }
}

class LoadModelNode extends FoldableNode {
  readonly loader: ModelUploadControl

  constructor() {
    super('Load Model', 'loadModel')
    this.addOutput('model', new ClassicPreset.Output(modelSocket, 'Model'))
    this.loader = new ModelUploadControl(this.id)
    this.addControl('loader', this.loader)
  }
}

class SeparateChannelsNode extends FoldableNode {
  readonly preview: ChannelsPreviewControl

  constructor() {
    super('Separate Channels', 'separateChannels')
    this.addInput('image', new ClassicPreset.Input(imageSocket, 'Image'))
    this.addOutput('r', new ClassicPreset.Output(channelSocket, 'R'))
    this.addOutput('g', new ClassicPreset.Output(channelSocket, 'G'))
    this.addOutput('b', new ClassicPreset.Output(channelSocket, 'B'))
    this.addOutput('a', new ClassicPreset.Output(channelSocket, 'A'))
    this.preview = new ChannelsPreviewControl(this.id)
    this.addControl('preview', this.preview)
  }
}

class CombineChannelsNode extends FoldableNode {
  readonly preview: ImageDisplayControl

  constructor() {
    super('Combine Channels', 'combineChannels')
    this.addInput('r', new ClassicPreset.Input(channelSocket, 'R'))
    this.addInput('g', new ClassicPreset.Input(channelSocket, 'G'))
    this.addInput('b', new ClassicPreset.Input(channelSocket, 'B'))
    this.addInput('a', new ClassicPreset.Input(channelSocket, 'A'))
    this.addOutput('image', new ClassicPreset.Output(imageSocket, 'Image'))
    this.preview = new ImageDisplayControl(this.id)
    this.addControl('preview', this.preview)
  }
}

class ShowImageNode extends FoldableNode {
  readonly preview: ImageDisplayControl

  constructor() {
    super('Show Image', 'showImage')
    this.addInput('image', new ClassicPreset.Input(imageSocket, 'Image'))
    this.addOutput('image', new ClassicPreset.Output(imageSocket, 'Image'))
    this.preview = new ImageDisplayControl(this.id)
    this.addControl('preview', this.preview)
  }
}

class Preview3DNode extends FoldableNode {
  readonly preview: Preview3DControl

  constructor() {
    super('Preview 3D', 'preview3d')
    this.addInput('model', new ClassicPreset.Input(modelSocket, 'Model'))
    this.preview = new Preview3DControl(this.id)
    this.addControl('preview', this.preview)
  }
}

class GenerateTripoModelNode extends FoldableNode {
  readonly generator: TripoGenerationControl

  constructor() {
    super('Generate Tripo Model', 'generateTripoModel')
    this.addInput('image', new ClassicPreset.Input(imageSocket, 'Image'))
    this.addOutput('model', new ClassicPreset.Output(modelSocket, 'Model'))
    this.generator = new TripoGenerationControl(this.id)
    this.addControl('generate', this.generator)
    this.width = 420
    this.height = 320
  }
}

class GenerateHy21ModelNode extends FoldableNode {
  readonly generator: HunyuanGenerationControl

  constructor() {
    super('Generate Hy 2.1 Model', 'generateHy21Model')
    this.addInput('image', new ClassicPreset.Input(imageSocket, 'Image'))
    this.addOutput('model', new ClassicPreset.Output(modelSocket, 'Model'))
    this.generator = new HunyuanGenerationControl(this.id)
    this.addControl('generate', this.generator)
    this.width = 420
    this.height = 320
  }
}

class SaveModelNode extends FoldableNode {
  readonly saver: SaveModelControl

  constructor() {
    super('Save Model', 'saveModel')
    this.addInput('model', new ClassicPreset.Input(modelSocket, 'Model'))
    this.addOutput('model', new ClassicPreset.Output(modelSocket, 'Model'))
    this.saver = new SaveModelControl(this.id)
    this.addControl('saver', this.saver)
  }
}

class SaveImageNode extends FoldableNode {
  readonly saver: SaveImageControl

  constructor() {
    super('Save Image', 'saveImage')
    this.addInput('image', new ClassicPreset.Input(imageSocket, 'Image'))
    this.addOutput('image', new ClassicPreset.Output(imageSocket, 'Image'))
    this.saver = new SaveImageControl(this.id)
    this.addControl('saver', this.saver)
  }
}

const NODE_FACTORIES: Record<NodeKind, () => FoldableNode> = {
  loadImage: () => new LoadImageNode(),
  loadModel: () => new LoadModelNode(),
  separateChannels: () => new SeparateChannelsNode(),
  combineChannels: () => new CombineChannelsNode(),
  showImage: () => new ShowImageNode(),
  preview3d: () => new Preview3DNode(),
  generateTripoModel: () => new GenerateTripoModelNode(),
  generateHy21Model: () => new GenerateHy21ModelNode(),
  saveModel: () => new SaveModelNode(),
  saveImage: () => new SaveImageNode(),
}

const DEFAULT_NODE_WIDTH = 280
const DEFAULT_NODE_HEIGHT = 220
const MIN_NODE_WIDTH = 200
const MIN_NODE_HEIGHT = 160

const NODE_CATALOG: NodeCatalogCategory[] = [
  {
    id: 'sources',
    label: 'Sources',
    entries: [
      { kind: 'loadImage', label: 'Load Image', description: 'Import an image asset.' },
      { kind: 'loadModel', label: 'Load Model', description: 'Import a 3D model.' },
    ],
  },
  {
    id: 'processing',
    label: 'Processing',
    entries: [
      { kind: 'separateChannels', label: 'Separate Channels', description: 'Split RGBA channels.' },
      { kind: 'combineChannels', label: 'Combine Channels', description: 'Rebuild RGBA from inputs.' },
    ],
  },
  {
    id: 'output',
    label: 'Output',
    entries: [
      { kind: 'showImage', label: 'Show Image', description: 'Preview the final image.' },
      { kind: 'preview3d', label: 'Preview 3D', description: 'Inspect a model in Babylon.js.' },
      { kind: 'saveModel', label: 'Save Model', description: 'Download model as GLB.' },
      { kind: 'saveImage', label: 'Save Image', description: 'Download image as PNG.' },
    ],
  },
  {
    id: '3d-generation',
    label: '3D Generation',
    entries: [
      { kind: 'generateTripoModel', label: 'Generate Tripo Model', description: 'Create 3D geometry from a reference image.' },
      { kind: 'generateHy21Model', label: 'Generate Hy 2.1 Model', description: 'Generate geometry using the Hunyuan3D-2.1 pipeline.' },
    ],
  },
]

export interface EditorSetup {
  destroy(): void
  editor: NodeEditor<Schemes>
  addNode(kind: NodeKind, position?: { x: number; y: number }): Promise<FoldableNode>
  catalog: NodeCatalogCategory[]
  serialize(): Promise<SerializedWorkflow>
  load(workflow: SerializedWorkflow): Promise<void>
  clear(): Promise<void>
}

export async function createEditor(container: HTMLElement): Promise<EditorSetup> {
  const editor = new NodeEditor<Schemes>()
  const area = new AreaPlugin<Schemes, AreaExtra>(container)
  const connection = new ConnectionPlugin<Schemes, AreaExtra>()
  const reactRender = new ReactPlugin<Schemes, AreaExtra>({ createRoot })

  let scheduleEvaluation: () => void = () => {}
  let removeNodeById: (id: string) => Promise<void> = async () => {}

  reactRender.addPreset(
    ReactPresets.classic.setup({
      customize: {
        node() {
          return ({ data, emit }: { data: Schemes['Node']; emit: RenderEmit<Schemes> }) => {
            const nodeInstance = data as FoldableNode
            return (
              <UnrealNode
                node={nodeInstance}
                collapsed={nodeInstance.collapsed}
                emit={emit}
                onToggle={async () => {
                  nodeInstance.collapsed = !nodeInstance.collapsed
                  await area.update('node', nodeInstance.id)
                }}
                onGraphChange={scheduleEvaluation}
                onResize={async (width, height) => {
                  nodeInstance.width = width
                  nodeInstance.height = height
                  await area.resize(nodeInstance.id, width, height)
                  await area.update('node', nodeInstance.id)
                }}
                onRemove={async () => {
                  await removeNodeById(nodeInstance.id)
                }}
              />
            )
          }
        },
        socket() {
          return ({ data }: { data: ClassicPreset.Socket }) => (
            <div className="unreal-socket" title={data?.name ?? ''} />
          )
        },
      },
    }),
  )

  connection.addPreset(ConnectionPresets.classic.setup())

  editor.use(area)
  area.use(reactRender)
  area.use(connection)

  const setOutputs = useGraphStore.getState().setOutputs

  const runEvaluation = async () => {
    try {
      const outputs = await evaluateGraph(editor)
      setOutputs(outputs)
    } catch (error) {
      console.error('Graph evaluation failed', error)
    }
  }

  let pending = false
  scheduleEvaluation = () => {
    if (pending) return
    pending = true
    Promise.resolve().then(() => {
      pending = false
      void runEvaluation()
    })
  }

  editor.addPipe(async (context) => {
    const result = context

    switch (context.type) {
      case 'connectioncreated':
      case 'connectionremoved':
      case 'nodecreated':
      case 'noderemoved':
      case 'cleared':
        scheduleEvaluation()
        break
      default:
        break
    }

    return result
  })

  removeNodeById = async (nodeId: string) => {
    try {
      await editor.removeNode(nodeId)
      scheduleEvaluation()
    } catch (error) {
      console.warn('Failed to remove node', nodeId, error)
    }
  }

  let creationOffset = 0

  const addNode = async (
    kind: NodeKind,
    position?: { x: number; y: number },
    options?: { id?: string; collapsed?: boolean; state?: SerializedNodeState; width?: number; height?: number },
  ): Promise<FoldableNode> => {
    const factory = NODE_FACTORIES[kind]
    const node = factory()

    if (options?.id) {
      node.id = options.id
    }

    if (typeof options?.collapsed === 'boolean') {
      node.collapsed = options.collapsed
    }

    node.width = options?.width ?? node.width ?? DEFAULT_NODE_WIDTH
    node.height = options?.height ?? node.height ?? DEFAULT_NODE_HEIGHT

    applyNodeState(node, options?.state)

    await editor.addNode(node)

    const targetPosition = position ?? {
      x: 140 + creationOffset * 40,
      y: 120 + creationOffset * 30,
    }

    creationOffset = (creationOffset + 1) % 12

    await area.resize(node.id, node.width ?? DEFAULT_NODE_WIDTH, node.height ?? DEFAULT_NODE_HEIGHT)
    await area.translate(node.id, targetPosition)
    await area.update('node', node.id)

    return node
  }

  const serialize = async (): Promise<SerializedWorkflow> => {
    const nodes = editor.getNodes() as FoldableNode[]
    const connections = editor.getConnections() as Schemes['Connection'][]

    const serializedNodes: SerializedNode[] = nodes.map((node) => {
      const view = area.nodeViews.get(node.id)
      const position = view ? { x: view.position.x, y: view.position.y } : { x: 0, y: 0 }

      return {
        id: node.id,
        kind: node.kind,
        position,
        collapsed: node.collapsed,
        width: node.width,
        height: node.height,
        state: captureNodeState(node),
      }
    })

    const serializedConnections: SerializedConnection[] = connections.map((conn) => ({
      id: conn.id,
      source: conn.source,
      sourceOutput: String(conn.sourceOutput),
      target: conn.target,
      targetInput: String(conn.targetInput),
    }))

    return {
      nodes: serializedNodes,
      connections: serializedConnections,
    }
  }

  const load = async (workflow: SerializedWorkflow) => {
    await editor.clear()
    useGraphStore.getState().setOutputs({})

    const nodeMap = new Map<string, FoldableNode>()

    for (const nodeData of workflow.nodes) {
      const node = await addNode(nodeData.kind, nodeData.position, {
        id: nodeData.id,
        collapsed: nodeData.collapsed,
        width: nodeData.width,
        height: nodeData.height,
        state: nodeData.state,
      })
      nodeMap.set(node.id, node)
    }

    for (const link of workflow.connections) {
      const source = nodeMap.get(link.source)
      const target = nodeMap.get(link.target)

      if (!source || !target) continue

      const connectionInstance = new ClassicPreset.Connection(
        source,
        link.sourceOutput as keyof typeof source.outputs,
        target,
        link.targetInput as keyof typeof target.inputs,
      )

      if (link.id) {
        connectionInstance.id = link.id
      }

      try {
        await editor.addConnection(connectionInstance as any)
      } catch (error) {
        console.warn('Failed to restore connection', link, error)
      }
    }

    if (workflow.nodes.length) {
      AreaExtensions.zoomAt(area, editor.getNodes())
    }

    scheduleEvaluation()
  }

  const clear = async () => {
    await editor.clear()
    useGraphStore.getState().setOutputs({})
    scheduleEvaluation()
  }

  // Seed an initial demonstration graph
  await addNode('loadImage', { x: 80, y: 120 })
  await addNode('separateChannels', { x: 360, y: 90 })
  await addNode('combineChannels', { x: 640, y: 90 })
  await addNode('showImage', { x: 920, y: 120 })
  await addNode('loadModel', { x: 80, y: 340 })
  await addNode('preview3d', { x: 360, y: 320 })

  AreaExtensions.zoomAt(area, editor.getNodes())
  await runEvaluation()

  return {
    destroy() {
      area.destroy()
    },
    editor,
    addNode,
    catalog: NODE_CATALOG,
    serialize,
    load,
    clear,
  }
}

function captureNodeState(node: FoldableNode): SerializedNodeState {
  if (node instanceof LoadImageNode) {
    return {
      image: node.uploader.image,
    }
  }

  if (node instanceof LoadModelNode) {
    const model = node.loader.model
    if (!model) return {}
    return {
      model: {
        fileName: model.fileName,
        mimeType: model.mimeType,
        base64: arrayBufferToBase64(model.arrayBuffer),
      },
    }
  }

  if (node instanceof Preview3DNode) {
    return {
      mode: node.preview.mode,
    }
  }

  if (node instanceof GenerateTripoModelNode) {
    return {
      tripo: node.generator.serialize(),
    }
  }

  if (node instanceof GenerateHy21ModelNode) {
    return {
      hunyuan: node.generator.serialize(),
    }
  }

  return {}
}

function applyNodeState(node: FoldableNode, state?: SerializedNodeState) {
  if (!state) return

  if (node instanceof LoadImageNode && state.image) {
    node.uploader.image = state.image
    ;(node.uploader as any).notify?.()
  }

  if (node instanceof LoadModelNode && state.model) {
    const buffer = base64ToArrayBuffer(state.model.base64)
    node.loader.model = {
      kind: 'model',
      arrayBuffer: buffer,
      fileName: state.model.fileName,
      mimeType: state.model.mimeType,
    }
    ;(node.loader as any).notify?.()
  }

  if (node instanceof Preview3DNode && state.mode) {
    node.preview.mode = state.mode
    ;(node.preview as any).notify?.()
  }

  if (node instanceof GenerateTripoModelNode && state.tripo) {
    node.generator.applySerialized(state.tripo)
  }

  if (node instanceof GenerateHy21ModelNode && state.hunyuan) {
    node.generator.applySerialized(state.hunyuan)
  }
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i += 1) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes.buffer
}

function sortEntries<T extends [string, U], U extends { index?: number } | undefined>(entries: T[]): T[] {
  return entries.sort((a, b) => {
    const ai = a[1]?.index ?? 0
    const bi = b[1]?.index ?? 0
    if (ai === bi) return 0
    return ai < bi ? -1 : 1
  })
}

const { RefSocket } = ReactPresets.classic

function UnrealNode(props: {
  node: FoldableNode
  collapsed: boolean
  emit: RenderEmit<Schemes>
  onToggle: () => Promise<void>
  onGraphChange: () => void
  onResize: (width: number, height: number) => Promise<void> | void
  onRemove: () => void
}) {
  const { node, collapsed, emit, onToggle, onGraphChange, onResize, onRemove } = props
  const [size, setSize] = useState(() => ({
    width: node.width ?? DEFAULT_NODE_WIDTH,
    height: node.height ?? DEFAULT_NODE_HEIGHT,
  }))

  const handleResizeStart = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault()
      event.stopPropagation()

      const startX = event.clientX
      const startY = event.clientY
      const startWidth = size.width
      const startHeight = size.height

      const handleMove = (move: PointerEvent) => {
        const nextWidth = Math.max(MIN_NODE_WIDTH, startWidth + (move.clientX - startX))
        const nextHeight = Math.max(MIN_NODE_HEIGHT, startHeight + (move.clientY - startY))
        setSize({ width: nextWidth, height: nextHeight })
      }

      const handleUp = async (up: PointerEvent) => {
        window.removeEventListener('pointermove', handleMove)
        window.removeEventListener('pointerup', handleUp)

        const finalWidth = Math.max(MIN_NODE_WIDTH, startWidth + (up.clientX - startX))
        const finalHeight = Math.max(MIN_NODE_HEIGHT, startHeight + (up.clientY - startY))
        setSize({ width: finalWidth, height: finalHeight })
        await onResize(finalWidth, finalHeight)
      }

      window.addEventListener('pointermove', handleMove)
      window.addEventListener('pointerup', handleUp)
    },
    [onResize, size.height, size.width],
  )

  const inputs = sortEntries(
    Object.entries(node.inputs) as Array<
      [string, ClassicPreset.Input<ClassicPreset.Socket> | undefined]
    >,
  )
  const outputs = sortEntries(
    Object.entries(node.outputs) as Array<
      [string, ClassicPreset.Output<ClassicPreset.Socket> | undefined]
    >,
  )
  const controls = sortEntries(
    Object.entries(node.controls) as Array<[string, ClassicPreset.Control | undefined]>,
  )

  return (
    <div
      className="unreal-node"
      data-collapsed={collapsed}
      data-kind={node.kind}
      style={{ width: `${size.width}px` }}
    >
      <header className="unreal-node__header">
        <button type="button" className="unreal-node__fold" onClick={() => void onToggle()} aria-label="Toggle node">
          {collapsed ? '+' : '–'}
        </button>
        <span className="unreal-node__title">{node.label}</span>
        <button
          type="button"
          className="unreal-node__close"
          onClick={(event) => {
            event.stopPropagation()
            void onRemove()
          }}
          aria-label="Remove node"
        >
          ×
        </button>
      </header>
      {!collapsed && (
        <div className="unreal-node__body" style={{ minHeight: `${Math.max(size.height, MIN_NODE_HEIGHT)}px` }}>
          <div className="unreal-node__inputs">
            {inputs.map(([key, input]) => {
              if (!input) return null
              return (
                <div className="unreal-node__socket-row" key={key}>
                  <RefSocket
                    name="input-socket"
                    side="input"
                    socketKey={key}
                    nodeId={node.id}
                    emit={emit}
                    payload={input.socket}
                  />
                  <span className="unreal-node__socket-label">{input.label ?? key}</span>
                  {input.control && input.showControl && (
                    <div className="unreal-node__inline-control">
                      {renderControlComponent(input.control, onGraphChange)}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          <div className="unreal-node__controls">
            {controls.map(([key, control]) => {
              if (!control) return null
              const content = renderControlComponent(control, onGraphChange)
              if (!content) return null
              const isPreview = control instanceof Preview3DControl
              return (
                <div
                  key={key}
                  className={`unreal-node__control-slot${isPreview ? ' unreal-node__control-slot--fill' : ''}`}
                >
                  {content}
                </div>
              )
            })}
          </div>
          <div className="unreal-node__outputs">
            {outputs.map(([key, output]) => {
              if (!output) return null
              return (
                <div className="unreal-node__socket-row" key={key}>
                  <span className="unreal-node__socket-label">{output.label ?? key}</span>
                  <RefSocket
                    name="output-socket"
                    side="output"
                    socketKey={key}
                    nodeId={node.id}
                    emit={emit}
                    payload={output.socket}
                  />
                </div>
              )
            })}
          </div>
        </div>
      )}
      {!collapsed && (
        <div className="unreal-node__resize-handle" onPointerDown={handleResizeStart} role="presentation" />
      )}
    </div>
  )
}

function renderControlComponent(control: ClassicPreset.Control, onGraphChange: () => void) {
  if (control instanceof ImageUploadControl) {
    return <ImageUploadControlView control={control} onGraphChange={onGraphChange} />
  }

  if (control instanceof ModelUploadControl) {
    return <ModelUploadControlView control={control} onGraphChange={onGraphChange} />
  }

  if (control instanceof ChannelsPreviewControl) {
    return <ChannelsPreviewControlView control={control} onGraphChange={onGraphChange} />
  }

  if (control instanceof ImageDisplayControl) {
    return <ImageDisplayControlView control={control} />
  }

  if (control instanceof Preview3DControl) {
    return <Preview3DControlView control={control} fill />
  }

  if (control instanceof TripoGenerationControl) {
    return <TripoGenerationControlView control={control} onGraphChange={onGraphChange} />
  }

  if (control instanceof HunyuanGenerationControl) {
    return <HunyuanGenerationControlView control={control} onGraphChange={onGraphChange} />
  }

  if (control instanceof SaveModelControl) {
    return <SaveModelControlView control={control} />
  }

  if (control instanceof SaveImageControl) {
    return <SaveImageControlView control={control} />
  }

  return null
}

async function evaluateGraph(editor: NodeEditor<Schemes>): Promise<GraphOutputs> {
  const nodes = editor.getNodes()
  const connections = editor.getConnections() as Schemes['Connection'][]
  const nodeMap = new Map(nodes.map((node) => [node.id, node]))
  const cache = new Map<string, NodeOutputMap>()
  const visiting = new Set<string>()

  const compute = async (nodeId: string): Promise<NodeOutputMap> => {
    if (cache.has(nodeId)) return cache.get(nodeId) as NodeOutputMap
    if (visiting.has(nodeId)) return {}

    visiting.add(nodeId)

    const node = nodeMap.get(nodeId)
    if (!node) {
      visiting.delete(nodeId)
      return {}
    }

    const inbound = connections.filter((connection) => connection.target === nodeId)
    const inputs: Record<string, NodeOutputValue | undefined> = {}

    for (const connection of inbound) {
      const sourceOutputs = await compute(connection.source)
      const value = sourceOutputs[connection.sourceOutput as string]
      inputs[connection.targetInput as string] = value
    }

    const evaluated = await evaluateNode(node as FoldableNode, inputs)
    cache.set(nodeId, evaluated)
    visiting.delete(nodeId)
    return evaluated
  }

  const result: GraphOutputs = {}

  for (const node of nodes) {
    result[node.id] = await compute(node.id)
  }

  return result
}

async function evaluateNode(
  node: FoldableNode,
  inputs: Record<string, NodeOutputValue | undefined>,
): Promise<NodeOutputMap> {
  if (node instanceof LoadImageNode) {
    return node.uploader.image ? { image: node.uploader.image } : {}
  }

  if (node instanceof LoadModelNode) {
    return node.loader.model ? { model: node.loader.model } : {}
  }

  if (node instanceof SeparateChannelsNode) {
    const image = inputs.image as ImageValue | undefined
    if (!image) return {}
    const channels = await separateChannels(image)
    return channels
  }

  if (node instanceof CombineChannelsNode) {
    const gather = async (value: NodeOutputValue | undefined, channel: ChannelKey) => {
      if (!value) return undefined
      if (typeof value === 'object' && value.kind === 'channel') {
        return value as ChannelValue
      }
      if (typeof value === 'object' && value.kind === 'image') {
        const generated = await separateChannels(value)
        return generated[channel]
      }
      return undefined
    }

    const [r, g, b, a] = await Promise.all([
      gather(inputs.r, 'r'),
      gather(inputs.g, 'g'),
      gather(inputs.b, 'b'),
      gather(inputs.a, 'a'),
    ])

    const result = await combineChannels({ r, g, b, a })
    return result ? { image: result } : {}
  }

  if (node instanceof ShowImageNode) {
    const image = inputs.image as ImageValue | undefined
    return image ? { image } : {}
  }

  if (node instanceof Preview3DNode) {
    const model = inputs.model as ModelValue | undefined
    return model ? { model } : {}
  }

  if (node instanceof GenerateTripoModelNode) {
    const control = node.controls.generate as TripoGenerationControl | undefined
    const image = inputs.image as ImageValue | undefined
    control?.setInputImage(image)
    return control?.model ? { model: control.model } : {}
  }

  if (node instanceof GenerateHy21ModelNode) {
    const control = node.controls.generate as HunyuanGenerationControl | undefined
    const image = inputs.image as ImageValue | undefined
    control?.setInputImage(image)
    return control?.model ? { model: control.model } : {}
  }

  if (node instanceof SaveModelNode) {
    const control = node.controls.saver as SaveModelControl | undefined
    const model = inputs.model as ModelValue | undefined
    control?.setModel(model)
    return model ? { model } : {}
  }

  if (node instanceof SaveImageNode) {
    const control = node.controls.saver as SaveImageControl | undefined
    const image = inputs.image as ImageValue | undefined
    control?.setImage(image)
    return image ? { image } : {}
  }

  return {}
}
