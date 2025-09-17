import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { ClassicPreset } from 'rete'
import { Engine } from '@babylonjs/core/Engines/engine'
import { Scene } from '@babylonjs/core/scene'
import { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera'
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight'
import { Vector3 } from '@babylonjs/core/Maths/math.vector'
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder'
import { Color3 } from '@babylonjs/core/Maths/math.color'
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial'
import { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh'
import { SceneLoader } from '@babylonjs/core/Loading/sceneLoader'
import { Material } from '@babylonjs/core/Materials/material'
import { NormalMaterial } from '@babylonjs/materials/normal/normalMaterial'
import '@babylonjs/loaders'

import { fileToImageValue, fileToModelValue } from './imageUtils'
import { useGraphStore } from './store'
import type { ChannelKey, ChannelValue, GraphOutputs, ImageValue, ModelValue, PreviewMode } from './types'

const EMPTY_OUTPUTS = Object.freeze({}) as GraphOutputs[string]

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

export function ChannelsPreviewControlView(props: { control: ChannelsPreviewControl }) {
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
      <canvas className="preview-canvas" ref={canvasRef} />
    </div>
  )
}

async function loadModelOrFallback(scene: Scene, model?: ModelValue): Promise<AbstractMesh[]> {
  if (!model) {
    return [createFallbackMesh(scene)]
  }

  const blob = new Blob([model.arrayBuffer], { type: model.mimeType })
  const url = URL.createObjectURL(blob)
  const extension = extractExtension(model.fileName)

  try {
    const result = await SceneLoader.ImportMeshAsync('', '', url, scene, undefined, extension)
    return result.meshes.filter((mesh) => mesh instanceof AbstractMesh)
  } catch (error) {
    console.warn('Failed to import model, using fallback mesh.', error)
    return [createFallbackMesh(scene)]
  } finally {
    URL.revokeObjectURL(url)
  }
}

function createFallbackMesh(scene: Scene) {
  const mesh = MeshBuilder.CreateSphere('preview-sphere', { diameter: 2, segments: 32 }, scene)
  const material = new StandardMaterial('preview-material', scene)
  material.diffuseColor = new Color3(0.5, 0.7, 1)
  material.specularColor = new Color3(1, 1, 1)
  material.specularPower = 64
  mesh.material = material
  return mesh
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

function extractExtension(filename: string | undefined) {
  if (!filename) return undefined
  const dot = filename.lastIndexOf('.')
  if (dot === -1) return undefined
  return filename.slice(dot).toLowerCase()
}
