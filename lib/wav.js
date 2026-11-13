/*
MIT License

Copyright (c) 2026 Contributors to the dsh-voice-parlaspeech-hr project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/** Minimal WAV decoder: RIFF PCM16/float32 → mono Float32Array at 16 kHz for ParlaSpeech. */
const TARGET_RATE = 16000

function readTag(view, offset) {
  return String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3))
}

/** Convert interleaved samples of N channels to mono by averaging. */
function toMono(samples, channels) {
  if (channels === 1) return samples
  const frames = Math.floor(samples.length / channels)
  const mono = new Float32Array(frames)
  for (let i = 0; i < frames; i++) {
    let sum = 0
    for (let c = 0; c < channels; c++) sum += samples[i * channels + c]
    mono[i] = sum / channels
  }
  return mono
}

/** Linear-interpolation resampler; adequate quality for speech recognition inputs. */
function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples
  const ratio = fromRate / toRate
  const outLength = Math.max(1, Math.floor(samples.length / ratio))
  const out = new Float32Array(outLength)
  for (let i = 0; i < outLength; i++) {
    const position = i * ratio
    const index = Math.floor(position)
    const fraction = position - index
    const current = samples[index] ?? 0
    const next = samples[index + 1] ?? current
    out[i] = current + (next - current) * fraction
  }
  return out
}

/**
 * Decode WAV bytes to mono Float32 samples at 16 kHz.
 * @param {Uint8Array} bytes - WAV file contents (RIFF).
 * @returns {{ samples: Float32Array, sourceSampleRate: number, durationSeconds: number }}
 */
export function decodeWavMono16k(bytes) {
  if (bytes.length < 44) throw new Error('WAV file is too short')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (readTag(view, 0) !== 'RIFF' || readTag(view, 8) !== 'WAVE') throw new Error('Not a RIFF/WAVE file')
  
  let offset = 12
  let format = -1
  let channels = 0
  let sampleRate = 0
  let dataOffset = -1
  let dataLength = 0
  
  while (offset + 8 <= view.byteLength) {
    const id = readTag(view, offset)
    const size = view.getUint32(offset + 4, true)
    const body = offset + 8
    
    if (id === 'fmt ') {
      format = view.getUint16(body, true)
      channels = view.getUint16(body + 2, true)
      sampleRate = view.getUint32(body + 4, true)
      const bitsPerSample = view.getUint16(body + 14, true)
      
      if (format === 0xfffe && size >= 40) format = view.getUint16(body + 24, true)
      if (format !== 1 && format !== 3) throw new Error(`Unsupported WAV encoding (${format}); expected PCM16 or float32`)
      if (format === 1 && bitsPerSample !== 16) throw new Error(`Unsupported PCM bit depth (${bitsPerSample}); expected 16`)
      if (format === 3 && bitsPerSample !== 32) throw new Error(`Unsupported float bit depth (${bitsPerSample}); expected 32`)
    } else if (id === 'data') {
      dataOffset = body
      dataLength = size === 0xffffffff ? view.byteLength - body : Math.min(size, view.byteLength - body)
    }
    
    offset = body + size + (size % 2)
  }
  
  if (dataOffset < 0) throw new Error('WAV file has no data chunk')
  if (channels < 1 || channels > 8) throw new Error(`Unsupported channel count (${channels})`)
  if (sampleRate < 8000 || sampleRate > 192000) throw new Error(`Unsupported sample rate (${sampleRate})`)
  
  const bytesPerSample = format === 1 ? 2 : 4
  const frames = Math.floor(dataLength / (bytesPerSample * channels))
  let samples = new Float32Array(frames * channels)
  
  if (format === 1) {
    const pcm = new Int16Array(view.buffer, view.byteOffset + dataOffset, frames * channels)
    for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768
  } else {
    samples = new Float32Array(view.buffer, view.byteOffset + dataOffset, frames * channels)
  }
  
  const mono = toMono(samples, channels)
  const normalized = resample(mono, sampleRate, TARGET_RATE)
  
  return { samples: normalized, sourceSampleRate: sampleRate, durationSeconds: frames / sampleRate }
}
