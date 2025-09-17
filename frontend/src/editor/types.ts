export type ChannelKey = 'r' | 'g' | 'b' | 'a'

export type ImageValue = {
  kind: 'image'
  dataUrl: string
  width?: number
  height?: number
  fileName?: string
}

export type ChannelValue = {
  kind: 'channel'
  channel: ChannelKey
  dataUrl: string
  width: number
  height: number
}

export type ModelValue = {
  kind: 'model'
  fileName: string
  arrayBuffer: ArrayBuffer
  mimeType: string
}

export type NodeOutputValue = ImageValue | ChannelValue | ModelValue | undefined

export type NodeOutputMap = Record<string, NodeOutputValue>

export type GraphOutputs = Record<string, NodeOutputMap>

export type PreviewMode = 'Base' | 'Wire' | 'Norm'

export type NodeKind =
  | 'loadImage'
  | 'loadModel'
  | 'separateChannels'
  | 'combineChannels'
  | 'showImage'
  | 'preview3d'

export interface NodeCatalogEntry {
  kind: NodeKind
  label: string
  description?: string
}

export interface NodeCatalogCategory {
  id: string
  label: string
  entries: NodeCatalogEntry[]
}

export interface SerializedNodeState {
  image?: ImageValue | null
  model?: {
    fileName: string
    mimeType: string
    base64: string
  } | null
  mode?: PreviewMode
}

export interface SerializedNode {
  id: string
  kind: NodeKind
  position: { x: number; y: number }
  collapsed: boolean
  width?: number
  height?: number
  state?: SerializedNodeState
}

export interface SerializedConnection {
  id?: string | null
  source: string
  sourceOutput: string
  target: string
  targetInput: string
}

export interface SerializedWorkflow {
  name?: string
  nodes: SerializedNode[]
  connections: SerializedConnection[]
}
