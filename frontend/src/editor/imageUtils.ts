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
