import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { ClassicPreset } from 'rete'
import { Engine } from '@babylonjs/core/Engines/engine'
import { Scene } from '@babylonjs/core/scene'
import { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera'
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight'
import { Vector3 } from '@babylonjs/core/Maths/math.vector'
import { Color3 } from '@babylonjs/core/Maths/math.color'
import { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh'
import { SceneLoader } from '@babylonjs/core/Loading/sceneLoader'
import { Material } from '@babylonjs/core/Materials/material'
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial'
import { NormalMaterial } from '@babylonjs/materials/normal/normalMaterial'
import '@babylonjs/loaders'

import { fileToImageValue, fileToModelValue, base64ToArrayBuffer, arrayBufferToBase64 } from './imageUtils'
import { useGraphStore } from './store'
import type {
  ChannelKey,
  ChannelValue,
  GraphOutputs,
  ImageValue,
  ModelValue,
  PreviewMode,
  TripoParams,
  TripoSerializedState,
} from './types'

const EMPTY_OUTPUTS = Object.freeze({}) as GraphOutputs[string]
const DEFAULT_BACKEND_BASE = (() => {
  const env = (import.meta as any).env?.VITE_BACKEND_URL as string | undefined
  if (env && env.trim()) {
    return env.replace(/\/$/, '')
  }
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol
    const hostname = window.location.hostname
    const defaultPort = 8000
    return `${protocol}//${hostname}:${defaultPort}`
  }
  return ''
})()
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
}

const OCTREE_OPTIONS = [256, 512, 1024, 2048]

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

export class ImageDisplayControl extends ReactiveControl {}

export class Preview3DControl extends ReactiveControl {
  mode: PreviewMode = 'Base'

  setMode(mode: PreviewMode) {
    if (this.mode === mode) return
    this.mode = mode
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
      const response = await fetch(`${DEFAULT_BACKEND_BASE}/triposg/generate`, {
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
      <label className="control-label">Load Image</label>
      <input type="file" accept="image/*" onChange={handleChange} />
      {control.image && (
        <div className="thumbnail">
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
      <label className="control-label">Load Model</label>
      <input type="file" accept=".glb,.gltf,.babylon,.obj,.stl" onChange={handleChange} />
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
            <img src={channels[key]!.dataUrl} alt={`${key} channel`} />
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
  const image = outputs.image as ImageValue | undefined

  return (
    <div className="control-block">
      <label className="control-label">Image Preview</label>
      {image ? (
        <div className="preview-frame">
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
  const model = outputs.model as ModelValue | undefined

  return (
    <div className={`control-block${fill ? ' control-block--fill' : ''}`}>
      <div className="control-label">Preview 3D</div>
      <ModeSelector mode={control.mode} onSelect={(m) => control.setMode(m)} />
      <BabylonViewport mode={control.mode} model={model} />
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
      <div className="control-label">Generate Tripo Model</div>
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
      {control.model && <div className="control-hint">Model ready: {control.model.fileName}</div>}
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
      <div className="control-label">Save Model</div>
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
      <div className="control-label">Save Image</div>
      {control.image ? (
        <>
          <div className="preview-frame">
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

function BabylonViewport(props: { mode: PreviewMode; model?: ModelValue }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const { mode, model } = props

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const engine = new Engine(canvas, true, { stencil: true })
    const scene = new Scene(engine)

    const camera = new ArcRotateCamera('camera', Math.PI / 2, Math.PI / 3, 6, Vector3.Zero(), scene)
    camera.attachControl(canvas, false)
    const light = new HemisphericLight('light1', new Vector3(0, 1, 0), scene)
    light.intensity = 0.9

    let cleanup = () => {}

    const setupScene = async () => {
      cleanup()
      const meshes = await loadModelOrFallback(scene, model)
      applyMode(meshes, scene, mode)
      cleanup = () => {
        meshes.forEach((mesh) => mesh.dispose())
      }
    }

    setupScene()

    const handlePointerEnter = () => {
      camera.attachControl(canvas, false)
    }

    const handlePointerLeave = () => {
      camera.detachControl()
    }

    const handlePointerDown = (event: PointerEvent) => {
      event.stopPropagation()
      event.preventDefault()
      camera.attachControl(canvas, false)
    }

    const handleWheel = (event: WheelEvent) => {
      event.stopPropagation()
      event.preventDefault()
    }

    const resize = () => engine.resize()
    const observer = new ResizeObserver(() => engine.resize())
    observer.observe(canvas)
    window.addEventListener('resize', resize)
    canvas.addEventListener('mouseenter', handlePointerEnter)
    canvas.addEventListener('mouseleave', handlePointerLeave)
    canvas.addEventListener('pointerdown', handlePointerDown)
    canvas.addEventListener('wheel', handleWheel, { passive: false })
    canvas.tabIndex = 0
    camera.detachControl()

    engine.runRenderLoop(() => {
      applyMode(scene.meshes, scene, mode)
      scene.render()
    })

    return () => {
      canvas.removeEventListener('mouseenter', handlePointerEnter)
      canvas.removeEventListener('mouseleave', handlePointerLeave)
      window.removeEventListener('resize', resize)
      observer.disconnect()
      cleanup()
      scene.dispose()
      engine.dispose()
      canvas.removeEventListener('pointerdown', handlePointerDown)
      canvas.removeEventListener('wheel', handleWheel, { passive: false } as EventListenerOptions)
    }
  }, [mode, model])

  return (
    <div className="preview-canvas-wrapper">
      {!model && <div className="control-hint">Connect a model to preview.</div>}
      <canvas className="preview-canvas" ref={canvasRef} />
    </div>
  )
}

async function loadModelOrFallback(scene: Scene, model?: ModelValue): Promise<AbstractMesh[]> {
  if (!model) {
    return []
  }

  const blob = new Blob([model.arrayBuffer], { type: model.mimeType })
  const url = URL.createObjectURL(blob)
  const pluginExtension = inferPluginExtension(model)

  try {
    const result = await SceneLoader.ImportMeshAsync('', '', url, scene, undefined, pluginExtension)
    return result.meshes.filter((mesh) => mesh instanceof AbstractMesh)
  } catch (error) {
    console.warn('Failed to import model, using fallback mesh.', error)
    return []
  } finally {
    URL.revokeObjectURL(url)
  }
}

function applyMode(meshes: AbstractMesh[], scene: Scene, mode: PreviewMode) {
  meshes
    .filter((mesh) => mesh.material)
    .forEach((mesh) => {
      const baseMaterial = mesh.material as Material

      if (mode === 'Base') {
        if (baseMaterial instanceof NormalMaterial) {
          baseMaterial.dispose()
          mesh.material = new StandardMaterial(`${mesh.name}-base`, scene)
        }
        if (mesh.material instanceof StandardMaterial) {
          mesh.material.wireframe = false
          mesh.material.diffuseColor = new Color3(0.8, 0.8, 0.8)
        }
      }

      if (mode === 'Wire') {
        if (!(mesh.material instanceof StandardMaterial)) {
          mesh.material = new StandardMaterial(`${mesh.name}-wire`, scene)
        }
        ;(mesh.material as StandardMaterial).wireframe = true
        ;(mesh.material as StandardMaterial).diffuseColor = new Color3(0.6, 0.9, 1)
      }

      if (mode === 'Norm') {
        const normalMaterial = new NormalMaterial(`${mesh.name}-normals`, scene)
        mesh.material = normalMaterial
      }
    })
}

function useGraphOutputs(nodeId: string) {
  return useGraphStore((state) => state.outputs[nodeId] ?? EMPTY_OUTPUTS)
}

function inferPluginExtension(model: ModelValue): string | undefined {
  if (model.fileName) {
    const lower = model.fileName.trim().toLowerCase()
    if (lower.endsWith('.glb')) return '.glb'
    if (lower.endsWith('.gltf')) return '.gltf'
    if (lower.endsWith('.obj')) return '.obj'
    if (lower.endsWith('.stl')) return '.stl'
  }

  const mime = model.mimeType.toLowerCase()
  if (mime.includes('glb') || mime === 'model/gltf-binary') return '.glb'
  if (mime.includes('gltf')) return '.gltf'
  if (mime.includes('obj')) return '.obj'
  if (mime.includes('stl')) return '.stl'

  return undefined
}
