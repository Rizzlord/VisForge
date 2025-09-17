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
  | 'generateTripoModel'
  | 'generateHy21Model'
  | 'generateHy21Texture'
  | 'removeBackground'
  | 'saveModel'
  | 'saveImage'

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
  tripo?: TripoSerializedState
  hunyuan?: HunyuanSerializedState
  hunyuanTexture?: HunyuanTextureSerializedState
  removeBg?: RemoveBgSerializedState
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

export interface TripoParams {
  seed: number
  useFloat16: boolean
  extraDepthLevel: number
  numInferenceSteps: number
  cfgScale: number
  simplifyMesh: boolean
  targetFaceNumber: number
  useFlashDecoder: boolean
  denseOctreeResolution: number
  hierarchicalOctreeResolution: number
  flashOctreeResolution: number
  unloadModelAfterGeneration: boolean
}

export interface TripoSerializedState {
  params: TripoParams
  modelBase64?: string
  modelMimeType?: string
  modelFileName?: string
}

export interface HunyuanParams {
  seed: number
  randomizeSeed: boolean
  removeBackground: boolean
  numInferenceSteps: number
  guidanceScale: number
  octreeResolution: number
  numChunks: number
  mcAlgo: 'mc' | 'dmc'
  unloadModelAfterGeneration: boolean
}

export interface HunyuanSerializedState {
  params: HunyuanParams
  modelBase64?: string
  modelMimeType?: string
  modelFileName?: string
}

export interface HunyuanTextureParams {
  seed: number
  randomizeSeed: boolean
  maxViewCount: number
  viewResolution: number
  numInferenceSteps: number
  guidanceScale: number
  targetFaceCount: number
  remeshMesh: boolean
  decimate: boolean
  uvUnwrap: boolean
  unloadModelAfterGeneration: boolean
}

export interface HunyuanTextureSerializedState {
  params: HunyuanTextureParams
  modelBase64?: string
  modelMimeType?: string
  modelFileName?: string
  albedoDataUrl?: string
  albedoFileName?: string
  rmDataUrl?: string
  rmFileName?: string
}

export interface RemoveBgParams {
  mode: 'rgb' | 'rgba'
  transparent: boolean
  color: string
  unloadModel: boolean
}

export interface RemoveBgSerializedState {
  params: RemoveBgParams
  imageDataUrl?: string
  fileName?: string
  width?: number
  height?: number
}
