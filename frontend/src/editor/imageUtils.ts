import type { ChannelKey, ChannelValue, ImageValue, ModelValue } from './types'

type Size = { width: number; height: number }

export async function fileToImageValue(file: File): Promise<ImageValue> {
  const dataUrl = await readFileAsDataUrl(file)
  const { width, height } = await resolveImageSize(dataUrl)

  return {
    kind: 'image',
    dataUrl,
    width,
    height,
    fileName: file.name,
  }
}

export async function fileToModelValue(file: File): Promise<ModelValue> {
  const arrayBuffer = await file.arrayBuffer()

  return {
    kind: 'model',
    arrayBuffer,
    fileName: file.name,
    mimeType: file.type || 'application/octet-stream',
  }
}

export async function separateChannels(image: ImageValue): Promise<Record<ChannelKey, ChannelValue>> {
  if (!image.dataUrl) {
    throw new Error('Image data is missing')
  }

  const element = await loadImage(image.dataUrl)
  const { width, height } = element

  const ctx = createWorkingContext({ width, height })
  ctx.drawImage(element, 0, 0, width, height)
  const { data } = ctx.getImageData(0, 0, width, height)

  const channels: Record<ChannelKey, ChannelValue> = {
    r: createChannelTexture('r', data, width, height),
    g: createChannelTexture('g', data, width, height),
    b: createChannelTexture('b', data, width, height),
    a: createChannelTexture('a', data, width, height),
  }

  return channels
}

export async function combineChannels(channels: Partial<Record<ChannelKey, ChannelValue>>): Promise<ImageValue | undefined> {
  const sample = channels.r || channels.g || channels.b || channels.a

  if (!sample) return undefined

  const element = await loadImage(sample.dataUrl)
  const { width, height } = element
  const ctx = createWorkingContext({ width, height })

  const output = ctx.createImageData(width, height)
  const rData = await maybeExtractChannel(channels.r, width, height)
  const gData = await maybeExtractChannel(channels.g, width, height)
  const bData = await maybeExtractChannel(channels.b, width, height)
  const aData = await maybeExtractChannel(channels.a, width, height, 255)

  for (let i = 0; i < output.data.length; i += 4) {
    output.data[i] = rData[i]
    output.data[i + 1] = gData[i]
    output.data[i + 2] = bData[i]
    output.data[i + 3] = aData[i]
  }

  ctx.putImageData(output, 0, 0)

  return {
    kind: 'image',
    dataUrl: ctx.canvas.toDataURL(),
    width,
    height,
    fileName: 'combined.png',
  }
}

// Convert a channel to an image (grayscale image showing the channel values)
export async function channelToImage(channel: ChannelValue): Promise<ImageValue> {
  const element = await loadImage(channel.dataUrl)
  const { width, height } = element
  
  return {
    kind: 'image',
    dataUrl: channel.dataUrl,
    width,
    height,
    fileName: `${channel.channel}-channel.png`,
  }
}

// Extract a specific channel from an image and convert it to an image
export async function extractChannelAsImage(image: ImageValue, channelKey: ChannelKey): Promise<ImageValue> {
  if (!image.dataUrl) {
    throw new Error('Image data is missing')
  }

  const element = await loadImage(image.dataUrl)
  const { width, height } = element

  const ctx = createWorkingContext({ width, height })
  ctx.drawImage(element, 0, 0, width, height)
  const { data } = ctx.getImageData(0, 0, width, height)

  const channel = createChannelTexture(channelKey, data, width, height)
  return channelToImage(channel)
}

// Convert image inputs back to channels for processing
export async function imageToChannel(image: ImageValue, channelKey: ChannelKey): Promise<ChannelValue> {
  if (!image.dataUrl) {
    throw new Error('Image data is missing')
  }

  const element = await loadImage(image.dataUrl)
  const { width, height } = element

  const ctx = createWorkingContext({ width, height })
  ctx.drawImage(element, 0, 0, width, height)
  const { data } = ctx.getImageData(0, 0, width, height)

  // For grayscale images (channel representations), use the red channel as the value
  return {
    kind: 'channel',
    channel: channelKey,
    dataUrl: image.dataUrl,
    width,
    height,
  }
}

function createWorkingContext({ width, height }: Size) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })

  if (!ctx) {
    throw new Error('Canvas context is not available')
  }

  return ctx
}

async function maybeExtractChannel(
  channel: ChannelValue | undefined,
  expectedWidth: number,
  expectedHeight: number,
  defaultAlpha = 0,
) {
  if (!channel) {
    const buffer = new Uint8ClampedArray(expectedWidth * expectedHeight * 4)
    for (let i = 0; i < buffer.length; i += 4) {
      buffer[i] = 0
      buffer[i + 1] = 0
      buffer[i + 2] = 0
      buffer[i + 3] = defaultAlpha
    }
    return buffer
  }

  const element = await loadImage(channel.dataUrl)
  const ctx = createWorkingContext({ width: expectedWidth, height: expectedHeight })
  ctx.drawImage(element, 0, 0, expectedWidth, expectedHeight)
  const { data } = ctx.getImageData(0, 0, expectedWidth, expectedHeight)
  return data
}

function createChannelTexture(channel: ChannelKey, source: Uint8ClampedArray, width: number, height: number): ChannelValue {
  const ctx = createWorkingContext({ width, height })
  const output = ctx.createImageData(width, height)

  for (let i = 0; i < source.length; i += 4) {
    const value = channel === 'a' ? source[i + 3] : source[getOffset(channel, i)]
    output.data[i] = value
    output.data[i + 1] = value
    output.data[i + 2] = value
    output.data[i + 3] = channel === 'a' ? value : 255
  }

  ctx.putImageData(output, 0, 0)

  return {
    kind: 'channel',
    channel,
    dataUrl: ctx.canvas.toDataURL(),
    width,
    height,
  }
}

function getOffset(channel: ChannelKey, index: number) {
  switch (channel) {
    case 'r':
      return index
    case 'g':
      return index + 1
    case 'b':
      return index + 2
    case 'a':
      return index + 3
  }
}

async function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error('Unknown file read error'))
    reader.readAsDataURL(file)
  })
}

async function resolveImageSize(dataUrl: string): Promise<Size> {
  const image = await loadImage(dataUrl)
  return { width: image.width, height: image.height }
}

export async function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = src
  })
}

export function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes.buffer
}

export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i += 1) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

// GLB Material extraction utilities
export async function extractMaterialMapsFromGLB(model: ModelValue): Promise<{
  albedo?: ImageValue
  normal?: ImageValue
  roughness?: ImageValue
  metallic?: ImageValue
  ao?: ImageValue
}> {
  try {
    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')
    const THREE = await import('three')
    
    const loader = new GLTFLoader()
    
    // Convert ArrayBuffer to Blob URL for loading
    const blob = new Blob([model.arrayBuffer], { type: model.mimeType })
    const url = URL.createObjectURL(blob)
    
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (gltf: any) => {
          URL.revokeObjectURL(url)
          
          const materials: { [key: string]: ImageValue | undefined } = {
            albedo: undefined,
            normal: undefined,
            roughness: undefined,
            metallic: undefined,
            ao: undefined,
          }
          
          // Extract textures from all materials found
          gltf.scene.traverse((child: any) => {
            if (child.material) {
              const material = child.material
              
              // Handle both single materials and material arrays
              const materialArray = Array.isArray(material) ? material : [material]
              
              for (const mat of materialArray) {
                // Extract different material maps
                if (mat.map && !materials.albedo) {
                  materials.albedo = textureToImageValue(mat.map, 'albedo')
                }
                if (mat.normalMap && !materials.normal) {
                  materials.normal = textureToImageValue(mat.normalMap, 'normal')
                }
                if (mat.roughnessMap && !materials.roughness) {
                  materials.roughness = textureToImageValue(mat.roughnessMap, 'roughness')
                }
                if (mat.metalnessMap && !materials.metallic) {
                  materials.metallic = textureToImageValue(mat.metalnessMap, 'metallic')
                }
                if (mat.aoMap && !materials.ao) {
                  materials.ao = textureToImageValue(mat.aoMap, 'ao')
                }
                
                // If no separate roughness/metallic maps, check for combined metallicRoughnessMap
                if (mat.metallicRoughnessMap && !materials.roughness && !materials.metallic) {
                  const combinedMap = textureToImageValue(mat.metallicRoughnessMap, 'metallic-roughness')
                  if (!materials.roughness) materials.roughness = combinedMap
                  if (!materials.metallic) materials.metallic = combinedMap
                }
                
                // Handle PBR materials that might use different property names
                if (mat.emissiveMap && !materials.albedo) {
                  // Sometimes emissive maps can be used as albedo if no albedo is available
                  materials.albedo = textureToImageValue(mat.emissiveMap, 'albedo')
                }
              }
            }
          })
          
          resolve(materials)
        },
        undefined,
        (error: any) => {
          URL.revokeObjectURL(url)
          reject(error)
        }
      )
    })
  } catch (error) {
    console.error('Failed to extract materials from GLB:', error)
    return {}
  }
}

// Convert Three.js texture to ImageValue
function textureToImageValue(texture: any, mapType: string): ImageValue {
  // Check if texture has valid image data
  if (!texture || !texture.image) {
    // Return a placeholder if texture extraction fails
    return createPlaceholderMaterialMap(mapType)
  }
  
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  
  if (!ctx) {
    // Return a placeholder if canvas context fails
    return createPlaceholderMaterialMap(mapType)
  }
  
  // Set canvas dimensions to match texture
  canvas.width = texture.image.width || 512
  canvas.height = texture.image.height || 512
  
  try {
    // Draw the texture image to canvas
    ctx.drawImage(texture.image, 0, 0, canvas.width, canvas.height)
    
    // Create ImageValue from canvas data
    return {
      kind: 'image',
      dataUrl: canvas.toDataURL(),
      width: canvas.width,
      height: canvas.height,
      fileName: `${mapType}_map.png`,
    }
  } catch (error) {
    console.warn('Failed to extract texture:', error)
    return createPlaceholderMaterialMap(mapType)
  }
}

// Create placeholder material map when extraction fails
function createPlaceholderMaterialMap(mapType: string): ImageValue {
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 512
  const ctx = canvas.getContext('2d')
  
  if (ctx) {
    // Different colors for different map types
    switch (mapType) {
      case 'albedo':
        ctx.fillStyle = '#808080' // Gray for albedo
        break
      case 'normal':
        ctx.fillStyle = '#8080ff' // Blue for normal maps
        break
      case 'roughness':
        ctx.fillStyle = '#ffffff' // White for roughness
        break
      case 'metallic':
        ctx.fillStyle = '#000000' // Black for metallic
        break
      case 'ao':
        ctx.fillStyle = '#ffffff' // White for AO
        break
      default:
        ctx.fillStyle = '#808080'
    }
    ctx.fillRect(0, 0, 512, 512)
    
    // Add text label
    ctx.fillStyle = mapType === 'metallic' ? '#ffffff' : '#000000'
    ctx.font = '24px Arial'
    ctx.textAlign = 'center'
    ctx.fillText(`No ${mapType.toUpperCase()} map`, 256, 256)
  }
  
  return {
    kind: 'image',
    dataUrl: canvas.toDataURL(),
    width: 512,
    height: 512,
    fileName: `placeholder_${mapType}.png`,
  }
}

// Apply material maps to GLB model
export async function applyMaterialMapsToGLB(
  model: ModelValue,
  materials: {
    albedo?: ImageValue
    normal?: ImageValue
    roughness?: ImageValue
    metallic?: ImageValue
    ao?: ImageValue
  }
): Promise<ModelValue> {
  try {
    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')
    const { GLTFExporter } = await import('three/examples/jsm/exporters/GLTFExporter.js')
    const THREE = await import('three')
    
    const loader = new GLTFLoader()
    const exporter = new GLTFExporter()
    
    // Convert ArrayBuffer to Blob URL for loading
    const blob = new Blob([model.arrayBuffer], { type: model.mimeType })
    const url = URL.createObjectURL(blob)
    
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        async (gltf: any) => {
          URL.revokeObjectURL(url)
          
          // Apply materials to all meshes
          const texturePromises: Promise<void>[] = []
          
          gltf.scene.traverse((child: any) => {
            if (child.material) {
              const material = child.material
              
              // Apply albedo map
              if (materials.albedo) {
                const albedoPromise = createTextureFromImage(materials.albedo, THREE.Texture).then((texture) => {
                  if (texture) {
                    material.map = texture
                  }
                })
                texturePromises.push(albedoPromise)
              }
              
              // Apply normal map
              if (materials.normal) {
                const normalPromise = createTextureFromImage(materials.normal, THREE.Texture).then((texture) => {
                  if (texture) {
                    material.normalMap = texture
                    // Set normal map scale if needed
                    if (!material.normalScale) {
                      material.normalScale = new THREE.Vector2(1, 1)
                    }
                  }
                })
                texturePromises.push(normalPromise)
              }
              
              // Apply roughness map
              if (materials.roughness) {
                const roughnessPromise = createTextureFromImage(materials.roughness, THREE.Texture).then((texture) => {
                  if (texture) {
                    material.roughnessMap = texture
                  }
                })
                texturePromises.push(roughnessPromise)
              }
              
              // Apply metallic map
              if (materials.metallic) {
                const metallicPromise = createTextureFromImage(materials.metallic, THREE.Texture).then((texture) => {
                  if (texture) {
                    material.metalnessMap = texture
                  }
                })
                texturePromises.push(metallicPromise)
              }
              
              // Apply AO map
              if (materials.ao) {
                const aoPromise = createTextureFromImage(materials.ao, THREE.Texture).then((texture) => {
                  if (texture) {
                    material.aoMap = texture
                  }
                })
                texturePromises.push(aoPromise)
              }
              
              material.needsUpdate = true
            }
          })
          
          // Wait for all textures to be created and applied
          try {
            await Promise.all(texturePromises)
          } catch (textureError) {
            console.warn('Some textures failed to load:', textureError)
          }
          
          // Export the modified model
          exporter.parse(
            gltf.scene,
            (result: any) => {
              const buffer = result as ArrayBuffer
              resolve({
                kind: 'model',
                arrayBuffer: buffer,
                fileName: model.fileName.replace('.glb', '_textured.glb'),
                mimeType: model.mimeType,
              })
            },
            (error: any) => {
              reject(error)
            },
            { binary: true }
          )
        },
        undefined,
        (error: any) => {
          URL.revokeObjectURL(url)
          reject(error)
        }
      )
    })
  } catch (error) {
    console.error('Failed to apply materials to GLB:', error)
    // Return original model if material application fails
    return model
  }
}

// Create Three.js texture from ImageValue
async function createTextureFromImage(image: ImageValue, TextureClass: any): Promise<any> {
  return new Promise((resolve) => {
    try {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        try {
          const texture = new TextureClass(img)
          texture.needsUpdate = true
          texture.flipY = false // GLB standard
          resolve(texture)
        } catch (error) {
          console.warn('Failed to create texture from image:', error)
          resolve(null)
        }
      }
      img.onerror = () => {
        console.warn('Failed to load image for texture')
        resolve(null)
      }
      img.src = image.dataUrl
    } catch (error) {
      console.warn('Failed to create texture from image:', error)
      resolve(null)
    }
  })
}
