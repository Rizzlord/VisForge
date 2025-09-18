import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { ClassicPreset } from 'rete'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

import { fileToImageValue, fileToModelValue, base64ToArrayBuffer, arrayBufferToBase64 } from './imageUtils'
import { BACKEND_BASE } from './config'
import { useGraphStore } from './store'
import type {
  ChannelKey,
  ChannelValue,
  GraphOutputs,
  ImageValue,
  ModelValue,
  PreviewMode,
  HunyuanParams,
  HunyuanSerializedState,
  HunyuanTextureParams,
  HunyuanTextureSerializedState,
  RemoveBgParams,
  RemoveBgSerializedState,
  DetailGenParams,
  DetailGenSerializedState,
  TripoParams,
  TripoSerializedState,
} from './types'

const EMPTY_OUTPUTS = Object.freeze({}) as GraphOutputs[string]
const DEFAULT_TRIPO_PARAMS: TripoParams = {
  seed: 681589206,
  useFloat16: true,
  extraDepthLevel: 1,
  numInferenceSteps: 50,
  cfgScale: 15,
  simplifyMesh: false,
  targetFaceNumber: 100000,
  useFlashDecoder: true,
  denseOctreeResolution: 512,
  hierarchicalOctreeResolution: 512,
  flashOctreeResolution: 512,
  unloadModelAfterGeneration: true,
  useRepoVenv: false,
}

const OCTREE_OPTIONS = [256, 512, 1024, 2048]

const DEFAULT_HUNYUAN_PARAMS: HunyuanParams = {
  seed: 1234,
  randomizeSeed: true,
  removeBackground: true,
  numInferenceSteps: 30,
  guidanceScale: 5,
  octreeResolution: 256,
  numChunks: 8000,
  mcAlgo: 'dmc',
  unloadModelAfterGeneration: true,
  useRepoVenv: false,
}

const DEFAULT_REMOVE_BG_PARAMS: RemoveBgParams = {
  mode: 'rgba',
  transparent: true,
  color: '#ffffff',
  unloadModel: true,
  useRepoVenv: false,
}

const HUNYUAN_VIEW_COUNT_OPTIONS = [6, 8] as const

const DEFAULT_HUNYUAN_TEXTURE_PARAMS: HunyuanTextureParams = {
  seed: 1234,
  randomizeSeed: true,
  maxViewCount: 6,
  viewResolution: 512,
  numInferenceSteps: 15,
  guidanceScale: 3,
  targetFaceCount: 40000,
  remeshMesh: true,
  decimate: true,
  uvUnwrap: true,
  textureResolution: 2048,
  unloadModelAfterGeneration: true,
  enableSuperResolution: false,
  useRepoVenv: false,
}

const DEFAULT_DETAILGEN_PARAMS: DetailGenParams = {
  seed: 42,
  numInferenceSteps: 50,
  guidanceScale: 10,
  noiseAug: 0,
  useRepoVenv: false,
}

export class ReactiveControl extends ClassicPreset.Control {
  readonly nodeId: string
  private listeners = new Set<() => void>()

  constructor(nodeId: string) {
    super()
    this.nodeId = nodeId
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  protected notify() {
    this.listeners.forEach((fn) => fn())
  }
}

export class ImageUploadControl extends ReactiveControl {
  image: ImageValue | null = null

  async load(file: File) {
    this.image = await fileToImageValue(file)
    this.notify()
  }

  clear() {
    this.image = null
    this.notify()
  }
}

export class ModelUploadControl extends ReactiveControl {
  model: ModelValue | null = null

  async load(file: File) {
    this.model = await fileToModelValue(file)
    this.notify()
  }

  clear() {
    this.model = null
    this.notify()
  }
}

export class ChannelsPreviewControl extends ReactiveControl {}

export class ImageDisplayControl extends ReactiveControl {
  image: ImageValue | null = null

  setImage(image: ImageValue | undefined) {
    const next = image ?? null
    if (this.image === next) return
    this.image = next
    this.notify()
  }
}

export class Preview3DControl extends ReactiveControl {
  mode: PreviewMode = 'Base'
  model: ModelValue | null = null

  setMode(mode: PreviewMode) {
    if (this.mode === mode) return
    this.mode = mode
    this.notify()
  }

  setModel(model: ModelValue | undefined) {
    const next = model ?? null
    if (this.model === next) return
    this.model = next
    this.notify()
  }
}

export class TripoGenerationControl extends ReactiveControl {
  params: TripoParams = { ...DEFAULT_TRIPO_PARAMS }
  model: ModelValue | null = null
  isGenerating = false
  error: string | null = null
  private inputImage: ImageValue | null = null

  setInputImage(image: ImageValue | undefined) {
    this.inputImage = image ?? null
    this.notify()
  }

  hasInputImage(): boolean {
    return this.inputImage !== null
  }

  updateParam<K extends keyof TripoParams>(key: K, value: TripoParams[K]) {
    this.params = { ...this.params, [key]: value }
    this.notify()
  }

  applySerialized(state?: TripoSerializedState) {
    if (!state) return
    this.params = { ...this.params, ...state.params }
    if (state.modelBase64) {
      const buffer = base64ToArrayBuffer(state.modelBase64)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: state.modelFileName ?? 'tripo-model.glb',
        mimeType: state.modelMimeType ?? 'model/gltf-binary',
      }
    }
    this.notify()
  }

  serialize(): TripoSerializedState {
    const base: TripoSerializedState = { params: { ...this.params } }
    if (this.model) {
      base.modelBase64 = arrayBufferToBase64(this.model.arrayBuffer)
      base.modelFileName = this.model.fileName
      base.modelMimeType = this.model.mimeType
    }
    return base
  }

  async generate(onGraphChange: () => void) {
    if (!this.inputImage) {
      this.error = 'Connect an image input before generating.'
      this.notify()
      return
    }
    this.isGenerating = true
    this.error = null
    this.model = null
    this.notify()

    try {
      const response = await fetch(`${BACKEND_BASE}/triposg/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_data_url: this.inputImage.dataUrl,
          seed: this.params.seed,
          use_float16: this.params.useFloat16,
          extra_depth_level: this.params.extraDepthLevel,
          num_inference_steps: this.params.numInferenceSteps,
          cfg_scale: this.params.cfgScale,
          simplify_mesh: this.params.simplifyMesh,
          target_face_number: this.params.targetFaceNumber,
          use_flash_decoder: this.params.useFlashDecoder,
          dense_octree_resolution: this.params.denseOctreeResolution,
          hierarchical_octree_resolution: this.params.hierarchicalOctreeResolution,
          flash_octree_resolution: this.params.flashOctreeResolution,
          unload_model_after_generation: this.params.unloadModelAfterGeneration,
          use_repo_venv: this.params.useRepoVenv,
        }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Failed to generate model')
      }

      const payload = await response.json()
      if (!payload.glb_base64) {
        throw new Error('Response did not include model data')
      }
      const buffer = base64ToArrayBuffer(payload.glb_base64)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: payload.file_name ?? 'tripo-model.glb',
        mimeType: payload.mime_type ?? 'model/gltf-binary',
      }
      this.isGenerating = false
      this.notify()
      onGraphChange()
    } catch (error) {
      this.isGenerating = false
      this.error = error instanceof Error ? error.message : 'Unknown error'
      this.notify()
    }
  }
}

export class HunyuanGenerationControl extends ReactiveControl {
  params: HunyuanParams = { ...DEFAULT_HUNYUAN_PARAMS }
  model: ModelValue | null = null
  isGenerating = false
  error: string | null = null
  private inputImage: ImageValue | null = null

  setInputImage(image: ImageValue | undefined) {
    this.inputImage = image ?? null
    this.notify()
  }

  hasInputImage(): boolean {
    return this.inputImage !== null
  }

  updateParam<K extends keyof HunyuanParams>(key: K, value: HunyuanParams[K]) {
    this.params = { ...this.params, [key]: value }
    this.notify()
  }

  applySerialized(state?: HunyuanSerializedState) {
    if (!state) return
    this.params = { ...this.params, ...state.params }
    if (state.modelBase64) {
      const buffer = base64ToArrayBuffer(state.modelBase64)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: state.modelFileName ?? 'hunyuan-model.glb',
        mimeType: state.modelMimeType ?? 'model/gltf-binary',
      }
    }
    this.notify()
  }

  serialize(): HunyuanSerializedState {
    const base: HunyuanSerializedState = { params: { ...this.params } }
    if (this.model) {
      base.modelBase64 = arrayBufferToBase64(this.model.arrayBuffer)
      base.modelFileName = this.model.fileName
      base.modelMimeType = this.model.mimeType
    }
    return base
  }

  async generate(onGraphChange: () => void) {
    if (!this.inputImage) {
      this.error = 'Connect an image input before generating.'
      this.notify()
      return
    }

    this.isGenerating = true
    this.error = null
    this.model = null
    this.notify()

    try {
      const response = await fetch(`${BACKEND_BASE}/hunyuan/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_data_url: this.inputImage.dataUrl,
          seed: this.params.seed,
          randomize_seed: this.params.randomizeSeed,
          remove_background: this.params.removeBackground,
          num_inference_steps: this.params.numInferenceSteps,
          guidance_scale: this.params.guidanceScale,
          octree_resolution: this.params.octreeResolution,
          num_chunks: this.params.numChunks,
          mc_algo: this.params.mcAlgo,
          unload_model_after_generation: this.params.unloadModelAfterGeneration,
          use_repo_venv: this.params.useRepoVenv,
        }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Failed to generate model')
      }

      const payload = await response.json()
      if (!payload.glb_base64) {
        throw new Error('Response did not include model data')
      }

      const buffer = base64ToArrayBuffer(payload.glb_base64)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: payload.file_name ?? 'hunyuan-model.glb',
        mimeType: payload.mime_type ?? 'model/gltf-binary',
      }

      const updatedSeed = Number(payload.seed)
      if (Number.isFinite(updatedSeed)) {
        this.params = { ...this.params, seed: updatedSeed }
      }

      this.isGenerating = false
      this.notify()
      onGraphChange()
    } catch (error) {
      this.isGenerating = false
      this.error = error instanceof Error ? error.message : 'Unknown error'
      this.notify()
    }
  }
}

export class HunyuanTextureGenerationControl extends ReactiveControl {
  params: HunyuanTextureParams = { ...DEFAULT_HUNYUAN_TEXTURE_PARAMS }
  model: ModelValue | null = null
  albedo: ImageValue | null = null
  rm: ImageValue | null = null
  isGenerating = false
  error: string | null = null
  private inputImage: ImageValue | null = null
  private inputModel: ModelValue | null = null

  setInputImage(image: ImageValue | undefined) {
    this.inputImage = image ?? null
    this.notify()
  }

  setInputModel(model: ModelValue | undefined) {
    this.inputModel = model ?? null
    this.notify()
  }

  hasRequiredInputs(): boolean {
    return this.inputImage !== null && this.inputModel !== null
  }

  updateParam<K extends keyof HunyuanTextureParams>(key: K, value: HunyuanTextureParams[K]) {
    this.params = { ...this.params, [key]: value }
    this.notify()
  }

  applySerialized(state?: HunyuanTextureSerializedState) {
    if (!state) return
    this.params = { ...DEFAULT_HUNYUAN_TEXTURE_PARAMS, ...state.params }

    if (!HUNYUAN_VIEW_COUNT_OPTIONS.some((value) => value === this.params.maxViewCount)) {
      this.params.maxViewCount = DEFAULT_HUNYUAN_TEXTURE_PARAMS.maxViewCount
    }

    if (state.modelBase64 && state.modelMimeType && state.modelFileName) {
      const buffer = base64ToArrayBuffer(state.modelBase64)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: state.modelFileName,
        mimeType: state.modelMimeType,
      }
    }

    if (state.albedoDataUrl) {
      this.albedo = {
        kind: 'image',
        dataUrl: state.albedoDataUrl,
        fileName: state.albedoFileName,
      }
    }

    if (state.rmDataUrl) {
      this.rm = {
        kind: 'image',
        dataUrl: state.rmDataUrl,
        fileName: state.rmFileName,
      }
    }

    this.notify()
  }

  serialize(): HunyuanTextureSerializedState {
    const base: HunyuanTextureSerializedState = { params: { ...this.params } }

    if (this.model) {
      base.modelBase64 = arrayBufferToBase64(this.model.arrayBuffer)
      base.modelFileName = this.model.fileName
      base.modelMimeType = this.model.mimeType
    }

    if (this.albedo) {
      base.albedoDataUrl = this.albedo.dataUrl
      base.albedoFileName = this.albedo.fileName
    }

    if (this.rm) {
      base.rmDataUrl = this.rm.dataUrl
      base.rmFileName = this.rm.fileName
    }

    return base
  }

  async generate(onGraphChange: () => void) {
    if (!this.hasRequiredInputs() || !this.inputModel || !this.inputImage) {
      this.error = 'Connect both a model and an image before generating textures.'
      this.notify()
      return
    }

    this.isGenerating = true
    this.error = null
    this.notify()

    try {
      const response = await fetch(`${BACKEND_BASE}/hunyuan/texture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_base64: arrayBufferToBase64(this.inputModel.arrayBuffer),
          image_data_url: this.inputImage.dataUrl,
          seed: this.params.seed,
          randomize_seed: this.params.randomizeSeed,
          max_view_count: this.params.maxViewCount,
          view_resolution: this.params.viewResolution,
          num_inference_steps: this.params.numInferenceSteps,
          target_face_count: this.params.targetFaceCount,
          texture_resolution: this.params.textureResolution,
          guidance_scale: this.params.guidanceScale,
          decimate: this.params.decimate,
          uv_unwrap: this.params.uvUnwrap,
          remesh_mesh: this.params.remeshMesh,
          enable_super_resolution: this.params.enableSuperResolution,
          unload_model_after_generation: this.params.unloadModelAfterGeneration,
          use_repo_venv: this.params.useRepoVenv,
        }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Failed to generate textures')
      }

      const payload = await response.json()

      if (!payload.glb_base64) {
        throw new Error('Response did not include model data')
      }

      const buffer = base64ToArrayBuffer(payload.glb_base64 as string)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: payload.file_name ?? 'hunyuan-textured.glb',
        mimeType: payload.mime_type ?? 'model/gltf-binary',
      }

      if (payload.albedo_base64) {
        const mime = payload.albedo_mime_type ?? 'image/jpeg'
        this.albedo = {
          kind: 'image',
          dataUrl: `data:${mime};base64,${payload.albedo_base64}`,
          fileName: payload.albedo_file_name ?? 'hunyuan-albedo.jpg',
          width: typeof payload.albedo_width === 'number' ? payload.albedo_width : undefined,
          height: typeof payload.albedo_height === 'number' ? payload.albedo_height : undefined,
        }
      } else {
        this.albedo = null
      }

      if (payload.rm_base64) {
        const mime = payload.rm_mime_type ?? 'image/png'
        this.rm = {
          kind: 'image',
          dataUrl: `data:${mime};base64,${payload.rm_base64}`,
          fileName: payload.rm_file_name ?? 'hunyuan-metallic-roughness.png',
          width: typeof payload.rm_width === 'number' ? payload.rm_width : undefined,
          height: typeof payload.rm_height === 'number' ? payload.rm_height : undefined,
        }
      } else {
        this.rm = null
      }

      if (typeof payload.seed === 'number') {
        this.params = { ...this.params, seed: payload.seed }
      }

      this.isGenerating = false
      this.notify()
      onGraphChange()
    } catch (error) {
      this.isGenerating = false
      this.error = error instanceof Error ? error.message : 'Unknown error'
      this.notify()
    }
  }
}

export class BackgroundRemovalControl extends ReactiveControl {
  params: RemoveBgParams = { ...DEFAULT_REMOVE_BG_PARAMS }
  image: ImageValue | null = null
  isProcessing = false
  error: string | null = null
  private inputImage: ImageValue | null = null

  setInputImage(image: ImageValue | undefined) {
    this.inputImage = image ?? null
    this.notify()
  }

  hasInputImage(): boolean {
    return this.inputImage !== null
  }

  updateParam<K extends keyof RemoveBgParams>(key: K, value: RemoveBgParams[K]) {
    this.params = { ...this.params, [key]: value }
    this.notify()
  }

  applySerialized(state?: RemoveBgSerializedState) {
    if (!state) return
    this.params = { ...DEFAULT_REMOVE_BG_PARAMS, ...state.params }
    if (state.imageDataUrl) {
      this.image = {
        kind: 'image',
        dataUrl: state.imageDataUrl,
        fileName: state.fileName ?? 'removed-background.png',
        width: state.width,
        height: state.height,
      }
    }
    this.notify()
  }

  serialize(): RemoveBgSerializedState {
    const base: RemoveBgSerializedState = { params: { ...this.params } }
    if (this.image) {
      base.imageDataUrl = this.image.dataUrl
      base.fileName = this.image.fileName
      base.width = this.image.width
      base.height = this.image.height
    }
    return base
  }

  async convert(onGraphChange: () => void) {
    if (!this.inputImage) {
      this.error = 'Connect an image input before converting.'
      this.notify()
      return
    }

    this.isProcessing = true
    this.error = null
    this.notify()

    try {
      const response = await fetch(`${BACKEND_BASE}/image/remove_background`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_data_url: this.inputImage.dataUrl,
          mode: this.params.mode,
          transparent: this.params.transparent,
          color: this.params.transparent ? undefined : this.params.color,
          unload_model: this.params.unloadModel,
          use_repo_venv: this.params.useRepoVenv,
        }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Failed to process image')
      }

      const payload = await response.json()
      if (!payload.image_base64) {
        throw new Error('Response did not include image data')
      }

      const dataUrl = `data:${payload.mime_type ?? 'image/png'};base64,${payload.image_base64}`
      this.image = {
        kind: 'image',
        dataUrl,
        fileName: payload.file_name ?? 'removed-background.png',
        width: typeof payload.width === 'number' ? payload.width : undefined,
        height: typeof payload.height === 'number' ? payload.height : undefined,
      }

      this.isProcessing = false
      this.notify()
      onGraphChange()
    } catch (error) {
      this.isProcessing = false
      this.error = error instanceof Error ? error.message : 'Unknown error'
      this.notify()
    }
  }
}

export class DetailGen3DControl extends ReactiveControl {
  params: DetailGenParams = { ...DEFAULT_DETAILGEN_PARAMS }
  model: ModelValue | null = null
  isGenerating = false
  error: string | null = null
  private inputModel: ModelValue | null = null
  private inputImage: ImageValue | null = null

  setInputModel(model: ModelValue | undefined) {
    const next = model ?? null
    if (this.inputModel === next) return
    this.inputModel = next
    this.notify()
  }

  setInputImage(image: ImageValue | undefined) {
    const next = image ?? null
    if (this.inputImage === next) return
    this.inputImage = next
    this.notify()
  }

  hasRequiredInputs(): boolean {
    return this.inputModel !== null && this.inputImage !== null
  }

  updateParam<K extends keyof DetailGenParams>(key: K, value: DetailGenParams[K]) {
    this.params = { ...this.params, [key]: value }
    this.notify()
  }

  applySerialized(state?: DetailGenSerializedState) {
    if (!state) return
    this.params = { ...this.params, ...state.params }
    if (state.modelBase64) {
      const buffer = base64ToArrayBuffer(state.modelBase64)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: state.modelFileName ?? 'detailgen-refined.glb',
        mimeType: state.modelMimeType ?? 'model/gltf-binary',
      }
    }
    this.notify()
  }

  serialize(): DetailGenSerializedState {
    const base: DetailGenSerializedState = { params: { ...this.params } }
    if (this.model) {
      base.modelBase64 = arrayBufferToBase64(this.model.arrayBuffer)
      base.modelFileName = this.model.fileName
      base.modelMimeType = this.model.mimeType
    }
    return base
  }

  async generate(onGraphChange: () => void) {
    if (!this.hasRequiredInputs() || !this.inputModel || !this.inputImage) {
      this.error = 'Connect both a model and an image before refining.'
      this.notify()
      return
    }

    this.isGenerating = true
    this.error = null
    this.model = null
    this.notify()

    try {
      const response = await fetch(`${BACKEND_BASE}/detailgen/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_base64: arrayBufferToBase64(this.inputModel.arrayBuffer),
          image_data_url: this.inputImage.dataUrl,
          seed: this.params.seed,
          num_inference_steps: this.params.numInferenceSteps,
          guidance_scale: this.params.guidanceScale,
          noise_aug: this.params.noiseAug,
          use_repo_venv: this.params.useRepoVenv,
        }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Failed to refine model')
      }

      const payload = await response.json()
      if (!payload.glb_base64) {
        throw new Error('Response did not include model data')
      }

      const buffer = base64ToArrayBuffer(payload.glb_base64)
      this.model = {
        kind: 'model',
        arrayBuffer: buffer,
        fileName: payload.file_name ?? 'detailgen-refined.glb',
        mimeType: payload.mime_type ?? 'model/gltf-binary',
      }

      this.isGenerating = false
      this.notify()
      onGraphChange()
    } catch (error) {
      this.isGenerating = false
      this.error = error instanceof Error ? error.message : 'Unknown error'
      this.notify()
    }
  }
}

export class SaveModelControl extends ReactiveControl {
  model: ModelValue | null = null

  setModel(model: ModelValue | undefined) {
    this.model = model ?? null
    this.notify()
  }
}

export class SaveImageControl extends ReactiveControl {
  image: ImageValue | null = null

  setImage(image: ImageValue | undefined) {
    this.image = image ?? null
    this.notify()
  }
}

export function ImageUploadControlView(props: {
  control: ImageUploadControl
  onGraphChange: () => void
}) {
  const { control, onGraphChange } = props
  const [, setVersion] = useState(0)

  useEffect(() => control.subscribe(() => setVersion((v) => v + 1)), [control])

  const handleChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    await control.load(file)
    onGraphChange()
  }

  const handleClear = () => {
    control.clear()
    onGraphChange()
  }

  return (
    <div className="control-block">
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        aria-label="Select image file"
      />
      {control.image && (
        <div className="thumbnail transparent-surface">
          <img src={control.image.dataUrl} alt={control.image.fileName ?? 'Loaded'} />
          <div className="meta-row">
            <span>{control.image.fileName}</span>
            <button type="button" onClick={handleClear}>
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export function ModelUploadControlView(props: {
  control: ModelUploadControl
  onGraphChange: () => void
}) {
  const { control, onGraphChange } = props
  const [, setVersion] = useState(0)

  useEffect(() => control.subscribe(() => setVersion((v) => v + 1)), [control])

  const handleChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    await control.load(file)
    onGraphChange()
  }

  const handleClear = () => {
    control.clear()
    onGraphChange()
  }

  return (
    <div className="control-block">
      <input
        type="file"
        accept=".glb,.gltf,.babylon,.obj,.stl"
        onChange={handleChange}
        aria-label="Select model file"
      />
      {control.model && (
        <div className="meta-row">
          <span>{control.model.fileName}</span>
          <button type="button" onClick={handleClear}>
            Clear
          </button>
        </div>
      )}
    </div>
  )
}

export function ChannelsPreviewControlView(props: {
  control: ChannelsPreviewControl
  onGraphChange?: () => void
}) {
  const { control } = props
  const outputs = useGraphOutputs(control.nodeId)
  const channels: Partial<Record<ChannelKey, ChannelValue>> = {
    r: outputs.r as ChannelValue | undefined,
    g: outputs.g as ChannelValue | undefined,
    b: outputs.b as ChannelValue | undefined,
    a: outputs.a as ChannelValue | undefined,
  }

  return (
    <div className="control-block channel-grid">
      {(['r', 'g', 'b', 'a'] as ChannelKey[]).map((key) => (
        <div key={key} className="channel-tile">
          <div className="channel-label">{key.toUpperCase()}</div>
          {channels[key] ? (
            <div className="channel-preview transparent-surface">
              <img src={channels[key]!.dataUrl} alt={`${key} channel`} />
            </div>
          ) : (
            <span className="channel-placeholder">Connect image input</span>
          )}
        </div>
      ))}
    </div>
  )
}

export function ImageDisplayControlView(props: { control: ImageDisplayControl }) {
  const { control } = props
  const outputs = useGraphOutputs(control.nodeId)
  const image = control.image ?? (outputs.image as ImageValue | undefined)

  return (
    <div className="control-block">
      {image ? (
        <div className="preview-frame transparent-surface">
          <img src={image.dataUrl} alt={image.fileName ?? 'Preview'} />
          <div className="meta-row">
            <span>
              {image.width} × {image.height}
            </span>
            {image.fileName && <span>{image.fileName}</span>}
          </div>
        </div>
      ) : (
        <span className="channel-placeholder">Awaiting image input</span>
      )}
    </div>
  )
}

export function Preview3DControlView(props: {
  control: Preview3DControl
  fill?: boolean
}) {
  const { control, fill } = props
  const [, setVersion] = useState(0)

  useEffect(() => control.subscribe(() => setVersion((v) => v + 1)), [control])

  const outputs = useGraphOutputs(control.nodeId)
  const model = control.model ?? (outputs.model as ModelValue | undefined)

  return (
    <div className={`control-block${fill ? ' control-block--fill' : ''}`}>
      <ModeSelector mode={control.mode} onSelect={(m) => control.setMode(m)} />
      <ThreeViewport mode={control.mode} model={model} />
    </div>
  )
}

export function TripoGenerationControlView(props: {
  control: TripoGenerationControl
  onGraphChange: () => void
}) {
  const { control, onGraphChange } = props
  const [, forceUpdate] = useState(0)

  useEffect(() => control.subscribe(() => forceUpdate((v) => v + 1)), [control])

  const params = control.params
  const disableGenerate = control.isGenerating || !control.hasInputImage()

  const handleNumberChange = (key: keyof TripoParams) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = event.target.type === 'number' ? Number(event.target.value) : event.target.value
      control.updateParam(key, value as any)
    }

  const handleCheckbox = (key: keyof TripoParams) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      control.updateParam(key, event.target.checked as any)
    }

  const handleSelect = (key: keyof TripoParams) =>
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      control.updateParam(key, Number(event.target.value) as any)
    }

  return (
    <div className={`control-block${control.isGenerating ? ' generating' : ''}`}>
      {control.error && <div className="control-error">{control.error}</div>}
      {!control.hasInputImage() && <div className="control-hint">Connect an image input to enable generation.</div>}
      <div className="tripo-grid">
        <label>
          Seed
          <input type="number" value={params.seed} onChange={handleNumberChange('seed')} />
        </label>
        <label>
          Steps
          <input type="number" value={params.numInferenceSteps} min={1} max={1000} onChange={handleNumberChange('numInferenceSteps')} />
        </label>
        <label>
          CFG Scale
          <input type="number" value={params.cfgScale} step={0.1} min={0} max={50} onChange={handleNumberChange('cfgScale')} />
        </label>
        <label>
          Extra Depth
          <input type="number" value={params.extraDepthLevel} min={0} max={4} onChange={handleNumberChange('extraDepthLevel')} />
        </label>
        <label>
          Dense Octree
          <select value={params.denseOctreeResolution} onChange={handleSelect('denseOctreeResolution')}>
            {OCTREE_OPTIONS.map((value) => (
              <option key={`dense-${value}`} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          Hierarchical Octree
          <select value={params.hierarchicalOctreeResolution} onChange={handleSelect('hierarchicalOctreeResolution')}>
            {OCTREE_OPTIONS.map((value) => (
              <option key={`hier-${value}`} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          Flash Octree
          <select value={params.flashOctreeResolution} onChange={handleSelect('flashOctreeResolution')}>
            {OCTREE_OPTIONS.map((value) => (
              <option key={`flash-${value}`} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className="checkbox">
          <input type="checkbox" checked={params.useFloat16} onChange={handleCheckbox('useFloat16')} /> Use float16 (CUDA only)
        </label>
        <label className="checkbox">
          <input type="checkbox" checked={params.useFlashDecoder} onChange={handleCheckbox('useFlashDecoder')} /> Flash decoder
        </label>
        <label className="checkbox">
          <input type="checkbox" checked={params.simplifyMesh} onChange={handleCheckbox('simplifyMesh')} /> Simplify mesh
        </label>
        {params.simplifyMesh && (
          <label>
            Target Faces
            <input type="number" value={params.targetFaceNumber} min={500} step={500} onChange={handleNumberChange('targetFaceNumber')} />
          </label>
        )}
      </div>
      <button type="button" onClick={() => void control.generate(onGraphChange)} disabled={disableGenerate}>
        {control.isGenerating ? 'Generating…' : 'Generate'}
      </button>
      <label className="checkbox">
        <input type="checkbox" checked={params.unloadModelAfterGeneration} onChange={handleCheckbox('unloadModelAfterGeneration')} />
        Unload models after generation
      </label>
      <label className="checkbox">
        <input type="checkbox" checked={params.useRepoVenv} onChange={handleCheckbox('useRepoVenv')} />
        Use repo virtual environment
      </label>
      {control.model && <div className="control-hint">Model ready: {control.model.fileName}</div>}
    </div>
  )
}

export function HunyuanGenerationControlView(props: {
  control: HunyuanGenerationControl
  onGraphChange: () => void
}) {
  const { control, onGraphChange } = props
  const [, forceUpdate] = useState(0)

  useEffect(() => control.subscribe(() => forceUpdate((v) => v + 1)), [control])

  const params = control.params
  const disableGenerate = control.isGenerating || !control.hasInputImage()

  const handleNumberChange = (key: keyof HunyuanParams) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = event.target.type === 'number' ? Number(event.target.value) : event.target.value
      control.updateParam(key, value as any)
    }

  const handleCheckbox = (key: keyof HunyuanParams) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      control.updateParam(key, event.target.checked as any)
    }

  const handleSelect = (key: keyof HunyuanParams) =>
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      control.updateParam(key, event.target.value as any)
    }

  return (
    <div className={`control-block${control.isGenerating ? ' generating' : ''}`}>
      {control.error && <div className="control-error">{control.error}</div>}
      {!control.hasInputImage() && <div className="control-hint">Connect an image input to enable generation.</div>}
      <div className="tripo-grid">
        <label>
          Seed
          <input type="number" value={params.seed} min={0} max={0xffffffff} onChange={handleNumberChange('seed')} />
        </label>
        <label className="checkbox">
          <input type="checkbox" checked={params.randomizeSeed} onChange={handleCheckbox('randomizeSeed')} /> Randomize seed
        </label>
        <label>
          Steps
          <input type="number" value={params.numInferenceSteps} min={1} max={200} onChange={handleNumberChange('numInferenceSteps')} />
        </label>
        <label>
          Guidance Scale
          <input type="number" value={params.guidanceScale} min={0} max={50} step={0.1} onChange={handleNumberChange('guidanceScale')} />
        </label>
        <label>
          Octree Resolution
          <input type="number" value={params.octreeResolution} min={16} max={1024} step={16} onChange={handleNumberChange('octreeResolution')} />
        </label>
        <label>
          Num Chunks
          <input type="number" value={params.numChunks} min={1000} step={1000} onChange={handleNumberChange('numChunks')} />
        </label>
        <label>
          Surface Extractor
          <select value={params.mcAlgo} onChange={handleSelect('mcAlgo')}>
            <option value="mc">Marching Cubes (MC)</option>
            <option value="dmc">Differentiable MC (DMC)</option>
          </select>
        </label>
      </div>
      <button type="button" onClick={() => void control.generate(onGraphChange)} disabled={disableGenerate}>
        {control.isGenerating ? 'Generating…' : 'Generate'}
      </button>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={params.unloadModelAfterGeneration}
          onChange={handleCheckbox('unloadModelAfterGeneration')}
        />
        Unload models after generation
      </label>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={params.useRepoVenv}
          onChange={handleCheckbox('useRepoVenv')}
        />
        Use repo virtual environment
      </label>
      {control.model && <div className="control-hint">Model ready: {control.model.fileName}</div>}
    </div>
  )
}

export function HunyuanTextureGenerationControlView(props: {
  control: HunyuanTextureGenerationControl
  onGraphChange: () => void
}) {
  const { control, onGraphChange } = props
  const [, forceUpdate] = useState(0)

  useEffect(() => control.subscribe(() => forceUpdate((v) => v + 1)), [control])

  const params = control.params
  const disableGenerate = control.isGenerating || !control.hasRequiredInputs()

  const handleNumberChange = (key: keyof HunyuanTextureParams) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value)
      if (Number.isFinite(value)) {
        control.updateParam(key, value as any)
      }
    }

  return (
    <div className={`control-block${control.isGenerating ? ' generating' : ''}`}>
      {control.error && <div className="control-error">{control.error}</div>}
      {!control.hasRequiredInputs() && (
        <div className="control-hint">Connect both an image and a model to enable texture generation.</div>
      )}
      <div className="tripo-grid">
        <label>
          Seed
          <input type="number" value={params.seed} onChange={handleNumberChange('seed')} />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.randomizeSeed}
            onChange={(event) => control.updateParam('randomizeSeed', event.target.checked)}
          />
          Randomize seed
        </label>
        <label>
          View Resolution
          <select
            value={params.viewResolution}
            onChange={(event) => control.updateParam('viewResolution', Number(event.target.value))}
          >
            {[256, 384, 512, 768, 1024].map((value) => (
              <option key={value} value={value}>
                {value}px
              </option>
            ))}
          </select>
        </label>
        <label>
          View Count
          <select
            value={params.maxViewCount}
            onChange={(event) => control.updateParam('maxViewCount', Number(event.target.value))}
          >
            {[6, 8].map((value) => (
              <option key={value} value={value}>
                {value} views
              </option>
            ))}
          </select>
        </label>
        <label>
          Steps
          <input
            type="number"
            value={params.numInferenceSteps}
            min={1}
            max={200}
            onChange={handleNumberChange('numInferenceSteps')}
          />
        </label>
        <label>
          Target Face Count
          <input
            type="number"
            value={params.targetFaceCount}
            min={1000}
            step={1000}
            onChange={handleNumberChange('targetFaceCount')}
          />
        </label>
        <label>
          Texture Resolution
          <select
            value={params.textureResolution}
            onChange={(event) => control.updateParam('textureResolution', Number(event.target.value))}
          >
            {[1024, 2048, 4096].map((value) => (
              <option key={value} value={value}>
                {value}px
              </option>
            ))}
          </select>
        </label>
        <label>
          Guidance
          <input
            type="number"
            value={params.guidanceScale}
            min={0}
            max={20}
            step={0.1}
            onChange={(event) => {
              const value = Number(event.target.value)
              if (Number.isFinite(value)) {
                control.updateParam('guidanceScale', value)
              }
            }}
          />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.remeshMesh}
            onChange={(event) => control.updateParam('remeshMesh', event.target.checked)}
          />
          Remesh input model
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.decimate}
            onChange={(event) => control.updateParam('decimate', event.target.checked)}
          />
          Decimate mesh
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.uvUnwrap}
            onChange={(event) => control.updateParam('uvUnwrap', event.target.checked)}
          />
          UV unwrap mesh
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.enableSuperResolution}
            onChange={(event) => control.updateParam('enableSuperResolution', event.target.checked)}
          />
          Run super-resolution pass (higher VRAM)
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.unloadModelAfterGeneration}
            onChange={(event) => control.updateParam('unloadModelAfterGeneration', event.target.checked)}
          />
          Unload models after generation
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.useRepoVenv}
            onChange={(event) => control.updateParam('useRepoVenv', event.target.checked)}
          />
          Use repo virtual environment
        </label>
      </div>
      <button type="button" onClick={() => void control.generate(onGraphChange)} disabled={disableGenerate}>
        {control.isGenerating ? 'Generating…' : 'Generate Texture'}
      </button>
      {control.model && <div className="control-hint">Textured model ready: {control.model.fileName}</div>}
      {control.albedo && <div className="control-hint">Albedo ready: {control.albedo.fileName ?? 'albedo'}</div>}
      {control.rm && <div className="control-hint">Roughness/Metallic ready: {control.rm.fileName ?? 'rm'}</div>}
    </div>
  )
}

export function DetailGen3DControlView(props: {
  control: DetailGen3DControl
  onGraphChange: () => void
}) {
  const { control, onGraphChange } = props
  const [, forceUpdate] = useState(0)

  useEffect(() => control.subscribe(() => forceUpdate((v) => v + 1)), [control])

  const params = control.params
  const disable = control.isGenerating || !control.hasRequiredInputs()

  const handleNumberChange = (key: keyof DetailGenParams, step = 1) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = event.target.type === 'number' ? Number(event.target.value) : event.target.value
      if (typeof value === 'number' && Number.isFinite(value)) {
        const rounded = step === 1 ? value : Math.max(Math.min(value, Number.MAX_SAFE_INTEGER), -Number.MAX_SAFE_INTEGER)
        control.updateParam(key, rounded as any)
      }
    }

  const handleFloatChange = (key: keyof DetailGenParams) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value)
      if (Number.isFinite(value)) {
        control.updateParam(key, value as any)
      }
    }

  return (
    <div className={`control-block${control.isGenerating ? ' generating' : ''}`}>
      {control.error && <div className="control-error">{control.error}</div>}
      {!control.hasRequiredInputs() && <div className="control-hint">Connect both a model and an image before refining.</div>}
      <div className="tripo-grid">
        <label>
          Seed
          <input type="number" value={params.seed} min={0} onChange={handleNumberChange('seed')} />
        </label>
        <label>
          Steps
          <input type="number" value={params.numInferenceSteps} min={1} max={200} onChange={handleNumberChange('numInferenceSteps')} />
        </label>
        <label>
          Guidance
          <input type="number" value={params.guidanceScale} min={0} max={50} step={0.1} onChange={handleFloatChange('guidanceScale')} />
        </label>
        <label>
          Noise Augmentation
          <input type="number" value={params.noiseAug} min={0} max={1} step={0.01} onChange={handleFloatChange('noiseAug')} />
        </label>
        <label className="checkbox">
          <input type="checkbox" checked={params.useRepoVenv} onChange={(event) => control.updateParam('useRepoVenv', event.target.checked)} />
          Use repo virtual environment
        </label>
      </div>
      <button type="button" onClick={() => void control.generate(onGraphChange)} disabled={disable}>
        {control.isGenerating ? 'Refining…' : 'Refine Detail'}
      </button>
      {control.model && <div className="control-hint">Model ready: {control.model.fileName}</div>}
    </div>
  )
}

export function BackgroundRemovalControlView(props: {
  control: BackgroundRemovalControl
  onGraphChange: () => void
}) {
  const { control, onGraphChange } = props
  const [, forceUpdate] = useState(0)

  useEffect(() => control.subscribe(() => forceUpdate((v) => v + 1)), [control])

  const params = control.params
  const disableConvert = control.isProcessing || !control.hasInputImage()

  const handleModeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    control.updateParam('mode', event.target.value as RemoveBgParams['mode'])
  }

  const handleColorChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    control.updateParam('color', event.target.value)
  }

  return (
    <div className={`control-block${control.isProcessing ? ' generating' : ''}`}>
      {control.error && <div className="control-error">{control.error}</div>}
      {!control.hasInputImage() && <div className="control-hint">Connect an image input to enable conversion.</div>}
      <div className="tripo-grid">
        <label>
          Output Mode
          <select value={params.mode} onChange={handleModeChange}>
            <option value="rgb">RGB</option>
            <option value="rgba">RGBA</option>
          </select>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.transparent}
            onChange={(event) => control.updateParam('transparent', event.target.checked)}
          />
          Transparent background
        </label>
        {!params.transparent && (
          <label>
            Background Color
            <input type="color" value={params.color} onChange={handleColorChange} />
          </label>
        )}
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.unloadModel}
            onChange={(event) => control.updateParam('unloadModel', event.target.checked)}
          />
          Unload model after conversion
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={params.useRepoVenv}
            onChange={(event) => control.updateParam('useRepoVenv', event.target.checked)}
          />
          Use repo virtual environment
        </label>
      </div>
      <button type="button" onClick={() => void control.convert(onGraphChange)} disabled={disableConvert}>
        {control.isProcessing ? 'Processing…' : 'Convert'}
      </button>
      {control.image && (
        <div className="control-hint">
          Ready: {control.image.fileName ?? 'processed.png'}
        </div>
      )}
    </div>
  )
}

export function SaveModelControlView(props: { control: SaveModelControl }) {
  const { control } = props
  const [, forceUpdate] = useState(0)
  useEffect(() => control.subscribe(() => forceUpdate((v) => v + 1)), [control])

  const handleDownload = () => {
    if (!control.model) return
    const blob = new Blob([control.model.arrayBuffer], { type: control.model.mimeType })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = control.model.fileName ?? 'model.glb'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="control-block">
      {control.model ? (
        <>
          <div className="control-hint">Model ready: {control.model.fileName ?? 'model.glb'}</div>
          <button type="button" onClick={handleDownload}>
            Download GLB
          </button>
        </>
      ) : (
        <div className="control-hint">Connect a model output to enable download.</div>
      )}
    </div>
  )
}

export function SaveImageControlView(props: { control: SaveImageControl }) {
  const { control } = props
  const [, forceUpdate] = useState(0)
  useEffect(() => control.subscribe(() => forceUpdate((v) => v + 1)), [control])

  const handleDownload = () => {
    if (!control.image) return
    const anchor = document.createElement('a')
    anchor.href = control.image.dataUrl
    anchor.download = control.image.fileName ?? 'image.png'
    anchor.click()
  }

  return (
    <div className="control-block">
      {control.image ? (
        <>
          <div className="preview-frame transparent-surface">
            <img src={control.image.dataUrl} alt="Preview" />
            <div className="meta-row">
              <span>
                {control.image.width} × {control.image.height}
              </span>
              {control.image.fileName && <span>{control.image.fileName}</span>}
            </div>
          </div>
          <button type="button" onClick={handleDownload}>
            Download Image
          </button>
        </>
      ) : (
        <div className="control-hint">Connect an image to enable download.</div>
      )}
    </div>
  )
}

function ModeSelector(props: { mode: PreviewMode; onSelect: (mode: PreviewMode) => void }) {
  const modes: PreviewMode[] = ['Base', 'Wire', 'Norm']
  return (
    <div className="mode-selector">
      {modes.map((mode) => (
        <button
          type="button"
          key={mode}
          className={mode === props.mode ? 'active' : ''}
          onClick={() => props.onSelect(mode)}
        >
          {mode}
        </button>
      ))}
    </div>
  )
}

function ThreeViewport(props: { mode: PreviewMode; model?: ModelValue }) {
  const { mode, model } = props
  const containerRef = useRef<HTMLDivElement | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const groupRef = useRef<THREE.Group | null>(null)
  const frameRef = useRef<number | null>(null)
  const loaderRef = useRef(new GLTFLoader())

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.domElement.classList.add('preview-canvas')
    renderer.domElement.style.touchAction = 'none'
    renderer.domElement.tabIndex = 0
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const scene = new THREE.Scene()
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / Math.max(container.clientHeight, 1), 0.1, 100)
    camera.position.set(3, 2, 5)
    cameraRef.current = camera

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.enablePan = true
    controls.screenSpacePanning = false
    controls.enableRotate = true
    controls.enableZoom = true
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    }
    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    }
    controlsRef.current = controls

    scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 0.9))
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9)
    dirLight.position.set(5, 10, 7.5)
    scene.add(dirLight)

    const stopPointerPropagation = (event: Event) => {
      event.stopPropagation()
    }

    const handleWheel = (event: WheelEvent) => {
      event.stopPropagation()
      event.preventDefault()
    }

    const dom = renderer.domElement
    dom.addEventListener('pointerdown', stopPointerPropagation)
    dom.addEventListener('pointermove', stopPointerPropagation)
    dom.addEventListener('pointerup', stopPointerPropagation)
    dom.addEventListener('pointercancel', stopPointerPropagation)
    dom.addEventListener('contextmenu', stopPointerPropagation)
    dom.addEventListener('dblclick', stopPointerPropagation)
    dom.addEventListener('wheel', handleWheel, { passive: false })

    const handleResize = () => {
      const host = containerRef.current
      if (!host || !rendererRef.current || !cameraRef.current) return
      const { clientWidth, clientHeight } = host
      rendererRef.current.setSize(clientWidth, clientHeight)
      cameraRef.current.aspect = clientWidth / Math.max(clientHeight, 1)
      cameraRef.current.updateProjectionMatrix()
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(container)
    window.addEventListener('resize', handleResize)
    handleResize()

    const animate = () => {
      controls.update()
      renderer.render(scene, camera)
      frameRef.current = requestAnimationFrame(animate)
    }
    frameRef.current = requestAnimationFrame(animate)

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
      resizeObserver.disconnect()
      window.removeEventListener('resize', handleResize)
      if (groupRef.current) {
        scene.remove(groupRef.current)
        disposeObject(groupRef.current)
        groupRef.current = null
      }
      controls.dispose()
      dom.removeEventListener('pointerdown', stopPointerPropagation)
      dom.removeEventListener('pointermove', stopPointerPropagation)
      dom.removeEventListener('pointerup', stopPointerPropagation)
      dom.removeEventListener('pointercancel', stopPointerPropagation)
      dom.removeEventListener('contextmenu', stopPointerPropagation)
      dom.removeEventListener('dblclick', stopPointerPropagation)
      dom.removeEventListener('wheel', handleWheel)
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  useEffect(() => {
    const scene = sceneRef.current
    const camera = cameraRef.current
    const controls = controlsRef.current
    const renderer = rendererRef.current
    if (!scene || !camera || !controls || !renderer) return

    let cancelled = false

    const loadModel = async () => {
      if (groupRef.current) {
        scene.remove(groupRef.current)
        disposeObject(groupRef.current)
        groupRef.current = null
      }

      if (!model) {
        renderer.render(scene, camera)
        return
      }

      const blob = new Blob([model.arrayBuffer], { type: model.mimeType })
      const url = URL.createObjectURL(blob)

      try {
        const gltf = await loaderRef.current.loadAsync(url)
        if (cancelled) {
          disposeObject(gltf.scene)
          return
        }
        groupRef.current = gltf.scene
        scene.add(gltf.scene)
        applyThreeMode(gltf.scene, mode)
        frameObject(camera, controls, gltf.scene)
      } catch (error) {
        console.warn('Failed to import model for preview.', error)
      } finally {
        URL.revokeObjectURL(url)
      }
    }

    loadModel()

    return () => {
      cancelled = true
    }
  }, [model])

  useEffect(() => {
    if (groupRef.current) {
      applyThreeMode(groupRef.current, mode)
    }
  }, [mode])

  return (
    <div className="preview-canvas-wrapper" ref={containerRef}>
      {!model && <div className="control-hint">Connect a model to preview.</div>}
    </div>
  )
}

function applyThreeMode(root: THREE.Object3D | null, mode: PreviewMode) {
  if (!root) return
  root.traverse((child: THREE.Object3D) => {
    if (!(child instanceof THREE.Mesh)) return
    const mesh = child as THREE.Mesh
    const store = mesh.userData as {
      originalMaterial?: THREE.Material | THREE.Material[]
      wireMaterial?: THREE.MeshBasicMaterial
      normalMaterial?: THREE.MeshNormalMaterial
    }

    if (!store.originalMaterial) {
      store.originalMaterial = Array.isArray(mesh.material)
        ? mesh.material.map((mat) => mat)
        : (mesh.material as THREE.Material)
    }
    if (!store.wireMaterial) {
      store.wireMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.85 })
    }
    if (!store.normalMaterial) {
      store.normalMaterial = new THREE.MeshNormalMaterial()
    }

    if (mode === 'Base') {
      mesh.material = store.originalMaterial as THREE.Material | THREE.Material[]
    } else if (mode === 'Wire') {
      mesh.material = store.wireMaterial
    } else {
      mesh.material = store.normalMaterial
    }
  })
}

function disposeObject(object: THREE.Object3D) {
  const materials = new Set<THREE.Material>()
  object.traverse((child: THREE.Object3D) => {
    if (child instanceof THREE.Mesh) {
      const mesh = child as THREE.Mesh
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((mat) => {
          if (mat) materials.add(mat)
        })
      } else if (mesh.material) {
        materials.add(mesh.material as THREE.Material)
      }
      const store = mesh.userData as {
        originalMaterial?: THREE.Material | THREE.Material[]
        wireMaterial?: THREE.Material
        normalMaterial?: THREE.Material
      }
      const { originalMaterial, wireMaterial, normalMaterial } = store
      if (Array.isArray(originalMaterial)) {
        originalMaterial.forEach((mat) => materials.add(mat))
      } else if (originalMaterial) {
        materials.add(originalMaterial)
      }
      if (wireMaterial) materials.add(wireMaterial)
      if (normalMaterial) materials.add(normalMaterial)
      if (mesh.geometry) mesh.geometry.dispose()
      mesh.userData.originalMaterial = undefined
      mesh.userData.wireMaterial = undefined
      mesh.userData.normalMaterial = undefined
    }
  })
  materials.forEach((material) => {
    if (material.dispose) material.dispose()
  })
}

function frameObject(camera: THREE.PerspectiveCamera | null, controls: OrbitControls | null, object: THREE.Object3D | null) {
  if (!camera || !controls || !object) return
  const box = new THREE.Box3().setFromObject(object)
  const size = box.getSize(new THREE.Vector3())
  const center = box.getCenter(new THREE.Vector3())
  const maxDim = Math.max(size.x, size.y, size.z)
  const safeDim = Math.max(maxDim, 0.1)
  const distance = safeDim / Math.max(Math.tan((camera.fov * Math.PI) / 360), 0.01)
  const offset = 1.6
  const direction = new THREE.Vector3(1, 0.8, 1).normalize()
  camera.position.copy(center.clone().add(direction.multiplyScalar(distance * offset)))
  camera.near = Math.max(distance / 100, 0.1)
  camera.far = distance * 100
  camera.updateProjectionMatrix()
  controls.target.copy(center)
  controls.minDistance = Math.max(safeDim * 0.2, 0.1)
  controls.maxDistance = Math.max(safeDim * 20, 10)
  controls.enablePan = true
  controls.update()
}

function useGraphOutputs(nodeId: string) {
  return useGraphStore((state) => state.outputs[nodeId] ?? EMPTY_OUTPUTS)
}
