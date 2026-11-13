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

/** Local ParlaSpeech preparation state machine with smart environment detection.
 *
 * All published snapshots conform to the dsh speech contract
 * (packages/experimental/speech-to-text/src/types.ts), which the browser
 * validates with a strict schema on every speech.follow()/speech.catalog()
 * item — one invalid snapshot fails the whole readiness stream and the
 * voice-input provider/language selector disappears:
 *
 *   SpeechPreparationStepKind = 'check' | 'model' | 'vad' | 'verify' | 'load'
 *   phase = 'unprepared' | 'ready' | 'standby' | 'cancelled' | 'downloading'
 *         | 'checking' | 'loading' | 'waking' | 'cancelling' | 'failed'
 *
 * The detailed venv/pip/model-download progress therefore maps onto the
 * single contract step 'model' with phase 'downloading'; fine-grained
 * progress stays in the Host log (logger) lines below.
 */
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { mkdir, access } from 'node:fs/promises'
import { constants as fsConstants } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { decodeWavMono16k } from './wav.js'

/** A request rejected before inference; the loaded worker remains reusable. */
export class SpeechInputError extends Error {}

/** Combine a caller signal with an optional deadline into one abort controller. */
function withDeadline(signal, timeoutMs) {
  const controller = new AbortController()
  const forward = () => { if (!controller.signal.aborted) controller.abort(signal.reason) }
  if (signal) {
    if (signal.aborted) forward()
    else signal.addEventListener('abort', forward, { once: true })
  }
  const timer = setTimeout(() => {
    if (!controller.signal.aborted) controller.abort(new Error('Speech operation timed out'))
  }, timeoutMs)
  timer.unref?.()
  return {
    signal: controller.signal,
    abort: (reason) => { if (!controller.signal.aborted) controller.abort(reason) },
    cleanup: () => { clearTimeout(timer); signal?.removeEventListener('abort', forward) },
  }
}

/** Python packages required by the worker. */
const PYTHON_PACKAGES = ['transformers', 'torch', 'soundfile', 'librosa', 'numpy']

/** Resolve paths to venv python and pip executables (cross-platform). */
function venvPaths(pluginDir) {
  const venvDir = join(pluginDir, 'runtime', 'venv')
  if (process.platform === 'win32') {
    return {
      venvDir,
      python: join(venvDir, 'Scripts', 'python.exe'),
      pip: join(venvDir, 'Scripts', 'pip.exe'),
    }
  }
  return {
    venvDir,
    python: join(venvDir, 'bin', 'python'),
    pip: join(venvDir, 'bin', 'pip'),
  }
}

/** Run a shell command and stream output to a callback; resolves on exit 0. */
function runCommand(command, args, { signal, onLog, cwd } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    const onAbort = () => {
      try { child.kill('SIGTERM') } catch {}
    }
    if (signal) {
      if (signal.aborted) { onAbort(); return }
      signal.addEventListener('abort', onAbort, { once: true })
    }

    let stderr = ''
    child.stdout?.on('data', (chunk) => {
      const text = chunk.toString().trim()
      if (text && onLog) onLog('info', text)
    })
    child.stderr?.on('data', (chunk) => {
      const text = chunk.toString()
      stderr += text
      if (onLog) onLog('info', text.trim())
    })

    child.on('error', (error) => {
      reject(new Error(`Failed to start ${command}: ${error.message}`))
    })

    child.on('close', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`${command} exited with code ${code}: ${stderr.slice(-500)}`))
    })
  })
}

/**
 * Python worker process driven over stdio JSON lines.
 * Requests run strictly in order; a crash terminates the process and the next
 * request transparently spawns a replacement.
 */
class ParlaSpeechWorker {
  #process
  #rl
  #nextId = 1
  #pending = new Map()
  #tail = Promise.resolve()
  #pythonPath
  #runtimePath
  #modelPath

  get alive() { return this.#process !== undefined }

  constructor(pythonPath, runtimePath, modelPath) {
    this.#pythonPath = pythonPath
    this.#runtimePath = runtimePath
    this.#modelPath = modelPath
  }

  #spawn() {
    if (this.#process !== undefined) return

    const process = spawn(this.#pythonPath, [this.#runtimePath, this.#modelPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    })

    // Prevent EPIPE crash when worker exits before we finish writing
    process.stdin.on('error', (err) => {
      if (err.code === 'EPIPE' || err.code === 'ERR_STREAM_DESTROYED') return
      this.#collapse(err)
    })

    this.#rl = createInterface({ input: process.stdout })

    this.#rl.on('line', (line) => {
      if (!line) return
      try {
        const msg = JSON.parse(line)
        const pending = this.#pending.get(msg.id)
        if (pending === undefined) return
        this.#pending.delete(msg.id)
        if (msg.ok) pending.resolve(msg)
        else pending.reject(new Error(msg.error ?? 'ParlaSpeech worker request failed'))
      } catch {
        // Ignore parse errors
      }
    })

    process.stderr.on('data', (chunk) => {
      const text = chunk.toString().trim()
      if (text) console.log(`[parlaspeech:stderr] ${text}`)
    })

    process.on('error', (error) => this.#collapse(error))
    process.on('exit', (code) => {
      if (this.#process === process) this.#process = undefined
      if (this.#pending.size > 0) {
        this.#collapse(new Error(`ParlaSpeech worker exited unexpectedly (code ${code})`))
      }
    })

    this.#process = process
  }

  /** Reject every pending request; the queue drains so later requests can respawn cleanly. */
  #collapse(error) {
    for (const pending of [...this.#pending.values()]) pending.reject(error)
    this.#pending.clear()
  }

  /** Terminate the process; pending and running requests are rejected with the given reason. */
  terminate(reason) {
    const process = this.#process
    if (process === undefined) return
    this.#process = undefined
    this.#collapse(reason)
    try { process.kill('SIGTERM') } catch {}
  }

  /**
   * Queue one request; requests execute strictly in order against one recognizer.
   * @param message - worker payload without an id.
   * @param options - signal, deadline.
   */
  request(message, { signal, deadlineMs }) {
    const run = async () => {
      signal?.throwIfAborted()
      if (this.#process === undefined) this.#spawn()

      const id = this.#nextId++
      return await new Promise((resolve, reject) => {
        let timer
        let settled = false

        const onAbort = () => {
          this.#pending.delete(id)
          reject(signal.reason instanceof Error ? signal.reason : new Error('Speech operation cancelled'))
        }

        const cleanup = () => {
          if (timer !== undefined) clearTimeout(timer)
          signal?.removeEventListener('abort', onAbort)
        }

        const settle = (settleResolve, value) => {
          if (settled) return
          settled = true
          cleanup()
          settleResolve(value)
        }

        const pending = {
          resolve: (value) => settle(resolve, value),
          reject: (error) => settle(reject, error),
        }

        if (deadlineMs !== undefined) {
          timer = setTimeout(() => {
            this.#pending.delete(id)
            this.terminate(new Error('ParlaSpeech worker replaced after deadline overrun'))
            reject(new Error('ParlaSpeech inference timed out'))
          }, deadlineMs)
          timer.unref?.()
        }

        if (signal) {
          if (signal.aborted) { onAbort(); return }
          signal.addEventListener('abort', onAbort, { once: true })
        }

        this.#pending.set(id, pending)
        const payload = JSON.stringify({ id, ...message })
        this.#process.stdin.write(payload + '\n')
      })
    }

    const job = this.#tail.then(run)
    this.#tail = job.then(() => undefined, () => undefined)
    return job
  }

  async dispose() {
    this.terminate(new Error('ParlaSpeech provider disposed'))
    await this.#tail.catch(() => {})
  }
}

/**
 * Host-owned preparation and transcription for ParlaSpeech HR model.
 * Implements the SpeechPreparation contract understood by the voice-input UI.
 *
 * Smart environment detection:
 * 1. User-configured pythonPath (if it has all packages) → use directly
 * 2. venv inside modelDirectory (if modelDirectory is set) → use directly
 * 3. System Python (if it has all packages) → use directly, no venv needed
 * 4. Existing plugin venv (runtime/venv) → reuse
 * 5. None found → create venv + pip install (if venvAutoInstall is true)
 */
export class ParlaSpeechPreparation {
  #ctx
  #config
  #state
  #listeners = new Set()
  #preparing
  #lifetime = new AbortController()
  #worker
  #loadedOnce = false
  #pluginDir
  #venv
  #resolvedPythonPath

  constructor(ctx, config) {
    this.#ctx = ctx
    this.#config = config
    this.#pluginDir = dirname(fileURLToPath(new URL('.', import.meta.url)))
    this.#venv = venvPaths(this.#pluginDir)

    // Contract-conforming step kinds only: venv creation + pip install + the
    // Hugging Face download are all reported as one 'model' step.
    const kinds = ['check', 'model', 'load']
    this.#state = { phase: 'unprepared', steps: kinds.map((kind) => ({ kind, status: 'pending' })) }
  }

  /** @returns the latest Host-owned preparation state. */
  snapshot() { return this.#state }

  /** @param listener - state invalidation callback. @returns subscription disposer. */
  subscribe(listener) {
    this.#listeners.add(listener)
    return () => { this.#listeners.delete(listener) }
  }

  #publish(state) {
    this.#state = state
    for (const listener of this.#listeners) listener()
  }

  #withStep(kind, status) {
    const steps = (this.#state.steps ?? []).map((item) => item.kind === kind
      ? { kind, status, ...(status === 'running' ? { startedAt: Date.now() } : {}) }
      : item)
    return { ...this.#state, steps }
  }

  #log(level, message, error) {
    const logger = this.#ctx.logger
    if (logger && typeof logger[level] === 'function') {
      if (error === undefined) logger[level](message)
      else logger[level](message, error)
    }
  }

  async #exists(path) {
    try {
      await access(path, fsConstants.F_OK)
      return true
    } catch {
      return false
    }
  }

  /** Check if a Python executable has all required packages installed. */
  async #checkPythonPackages(pythonPath) {
    try {
      await runCommand(pythonPath, [
        '-c',
        'import transformers, torch, soundfile, librosa, numpy; print("ok")',
      ], { signal: this.#lifetime.signal })
      return true
    } catch {
      return false
    }
  }

  /** Check if a local model directory contains valid model files. */
  async #localModelReady() {
    const dir = this.#config.modelDirectory
    if (!dir) return false

    try {
      let hasModel = false
      for (const file of ['model.safetensors', 'pytorch_model.bin']) {
        if (await this.#exists(join(dir, file))) {
          hasModel = true
          break
        }
      }
      const hasConfig = await this.#exists(join(dir, 'config.json'))

      if (hasModel && hasConfig) {
        this.#log('info', `✓ Local model found at ${dir}`)
        return true
      }

      if (hasModel && !hasConfig) {
        this.#log('warn', `Local model at ${dir} is missing config.json`)
      }
      return false
    } catch {
      return false
    }
  }

  /**
   * Detect the best Python environment to use, in priority order:
   * 1. User-configured pythonPath
   * 2. venv inside modelDirectory
   * 3. System Python
   * 4. Existing plugin venv
   * 5. None found → needs install
   */
  async #detectPythonEnvironment() {
    // 1. Check user-configured Python path
    if (this.#config.pythonPath) {
      this.#log('info', `Checking configured Python: ${this.#config.pythonPath}`)
      const hasPackages = await this.#checkPythonPackages(this.#config.pythonPath)
      if (hasPackages) {
        this.#log('info', `✓ Configured Python has all packages — using it directly`)
        return { python: this.#config.pythonPath, needsInstall: false, source: 'configured' }
      }
      this.#log('warn', `Configured Python lacks required packages`)
    }

    // 2. Check venv inside modelDirectory
    if (this.#config.modelDirectory) {
      const modelVenvPython = process.platform === 'win32'
        ? join(this.#config.modelDirectory, 'venv', 'Scripts', 'python.exe')
        : join(this.#config.modelDirectory, 'venv', 'bin', 'python')

      if (await this.#exists(modelVenvPython)) {
        this.#log('info', `Checking venv in modelDirectory: ${modelVenvPython}`)
        const hasPackages = await this.#checkPythonPackages(modelVenvPython)
        if (hasPackages) {
          this.#log('info', `✓ modelDirectory venv has all packages — using it directly`)
          return { python: modelVenvPython, needsInstall: false, source: 'model-venv' }
        }
        this.#log('warn', `modelDirectory venv lacks required packages`)
      }
    }

    // 3. Check system Python
    const systemPython = this.#config.systemPythonPath
    this.#log('info', `Checking system Python: ${systemPython}`)
    const systemHasPackages = await this.#checkPythonPackages(systemPython)
    if (systemHasPackages) {
      this.#log('info', `✓ System Python has all packages — using it directly (no venv needed)`)
      return { python: systemPython, needsInstall: false, source: 'system' }
    }

    // 4. Check existing plugin venv
    if (await this.#exists(this.#venv.python)) {
      this.#log('info', `Checking existing plugin venv: ${this.#venv.python}`)
      const venvHasPackages = await this.#checkPythonPackages(this.#venv.python)
      if (venvHasPackages) {
        this.#log('info', `✓ Existing plugin venv has all packages — reusing it`)
        return { python: this.#venv.python, needsInstall: false, source: 'plugin-venv' }
      }
      this.#log('warn', `Existing plugin venv lacks required packages — will reinstall`)
    }

    // 5. Nothing found
    this.#log('info', `No Python environment with required packages found`)
    return { python: null, needsInstall: true, source: 'none' }
  }

  /** Inspect on activation; check if everything is already available. */
  inspect() {
    if (this.#preparing || this.#lifetime.signal.aborted) return

    const deadline = withDeadline(this.#lifetime.signal, 90_000)
    const run = async (signal) => {
      this.#publish({ ...this.#withStep('check', 'running'), phase: 'checking', step: 'check', startedAt: Date.now() })

      const hasLocalModel = await this.#localModelReady()
      const env = await this.#detectPythonEnvironment()
      signal.throwIfAborted()

      if (!env.needsInstall && (hasLocalModel || !this.#config.modelDirectory)) {
        // Everything ready or will be handled on-demand
        this.#resolvedPythonPath = env.python
        this.#publish({
          ...this.#state,
          phase: 'standby',
          steps: (this.#state.steps ?? []).map((item) => ({
            kind: item.kind,
            status: item.kind === 'load' ? 'pending' : 'complete',
          })),
        })
        this.#log('info', `ParlaSpeech HR provider ready (Python: ${env.source}, Model: ${hasLocalModel ? 'local' : 'HuggingFace'})`)
      } else {
        this.#publish({
          ...this.#state,
          phase: 'unprepared',
          steps: (this.#state.steps ?? []).map((item) => ({
            kind: item.kind,
            status: item.kind === 'check' ? 'complete' : 'pending',
          })),
        })
      }
    }

    const task = { signal: deadline.signal, abort: deadline.abort, settled: undefined }
    task.settled = run(deadline.signal).catch((error) => {
      if (!this.#lifetime.signal.aborted) {
        this.#publish({
          phase: 'failed',
          message: error instanceof Error ? error.message : String(error),
          steps: (this.#state.steps ?? []).map((item) =>
            item.status === 'running' ? { kind: item.kind, status: 'failed' } : item),
        })
      }
    }).finally(() => {
      deadline.cleanup()
      if (this.#preparing === task) this.#preparing = undefined
    })

    this.#preparing = task
  }

  /** Start or join the preparation task. */
  prepare() {
    if (this.#lifetime.signal.aborted) return

    const previous = this.#preparing
    if (previous !== undefined) {
      void previous.settled.then(() => {
        if (this.#preparing === undefined && !this.#lifetime.signal.aborted) this.prepare()
      })
      return
    }

    const deadline = withDeadline(this.#lifetime.signal, this.#config.prepareTimeoutMs)
    const task = { signal: deadline.signal, abort: deadline.abort, settled: undefined }

    task.settled = this.#run(task).catch((error) => {
      if (this.#lifetime.signal.aborted) return
      if (task.signal.aborted) {
        // Explicit cancel() — report the contract 'cancelled' phase, not 'failed'
        this.#publish({
          phase: 'cancelled',
          steps: (this.#state.steps ?? []).map((item) => item.status === 'running'
            ? { kind: item.kind, status: 'pending' }
            : item),
        })
        return
      }
      this.#log('warn', 'ParlaSpeech preparation failed', error)
      this.#publish({
        phase: 'failed',
        message: error instanceof Error ? error.message : String(error),
        steps: (this.#state.steps ?? []).map((item) => item.status === 'running'
          ? { kind: item.kind, status: 'failed' }
          : item),
      })
    }).finally(() => {
      deadline.cleanup()
      if (this.#preparing === task) this.#preparing = undefined
    })

    this.#preparing = task
  }

  async #run(task) {
    // 1. check — detect local model and Python environment
    this.#publish({ ...this.#withStep('check', 'running'), phase: 'checking', step: 'check', startedAt: Date.now() })

    const hasLocalModel = await this.#localModelReady()
    const env = await this.#detectPythonEnvironment()

    task.signal.throwIfAborted()
    this.#publish(this.#withStep('check', 'complete'))

    // Determine effective model path
    const effectiveModelPath = hasLocalModel
      ? this.#config.modelDirectory
      : this.#config.modelId

    if (!env.needsInstall) {
      // Found existing environment — skip environment/model setup entirely
      this.#resolvedPythonPath = env.python
      this.#publish(this.#withStep('model', 'complete'))
      this.#log('info',
        `Using existing setup (Python: ${env.source}, Model: ${hasLocalModel ? 'local directory' : 'HuggingFace'})`
        + ` — no installation needed`)
    } else {
      if (this.#config.venvAutoInstall === false) {
        throw new Error(
          `No Python environment with required packages found and auto-install is disabled.`
          + ` Either install packages manually (pip install transformers torch soundfile librosa numpy)`
          + ` or set venvAutoInstall: true in the plugin configuration.`
        )
      }

      // Create venv — contract step 'model', phase 'downloading'
      this.#publish({
        ...this.#withStep('model', 'running'),
        phase: 'downloading',
        step: 'model',
        resource: 'Python virtual environment',
        completedBytes: 0,
      })
      this.#log('info', `Creating Python virtual environment at ${this.#venv.venvDir}`)

      await mkdir(dirname(this.#venv.venvDir), { recursive: true })
      await runCommand(this.#config.systemPythonPath, ['-m', 'venv', this.#venv.venvDir], {
        signal: task.signal,
        onLog: (level, msg) => this.#log(level, `[venv] ${msg}`),
      })

      task.signal.throwIfAborted()
      this.#resolvedPythonPath = this.#venv.python

      // Install packages
      this.#publish({
        ...this.#withStep('model', 'running'),
        phase: 'downloading',
        step: 'model',
        resource: `Python packages (${PYTHON_PACKAGES.join(', ')})`,
        completedBytes: 0,
      })
      this.#log('info', `Installing Python packages: ${PYTHON_PACKAGES.join(', ')}`)

      await runCommand(this.#resolvedPythonPath, ['-m', 'pip', 'install', '--upgrade', 'pip'], {
        signal: task.signal,
        onLog: (level, msg) => this.#log(level, `[pip] ${msg}`),
      })

      await runCommand(this.#resolvedPythonPath, ['-m', 'pip', 'install', ...PYTHON_PACKAGES], {
        signal: task.signal,
        onLog: (level, msg) => this.#log(level, `[pip] ${msg}`),
      })

      task.signal.throwIfAborted()
      this.#publish(this.#withStep('model', 'complete'))
    }

    // 2. load — spawn Python worker and wait for ready signal
    this.#publish({ ...this.#withStep('load', 'running'), phase: 'loading', step: 'load', startedAt: Date.now() })

    if (!this.#worker) {
      this.#worker = new ParlaSpeechWorker(
        this.#resolvedPythonPath,
        this.#config.runtimePath,
        effectiveModelPath,
      )
    }

    // The first 'load' also downloads the model from Hugging Face (~1.2 GB for
    // the default model), so the deadline must cover a slow connection; the
    // whole task is already bounded by prepareTimeoutMs.
    await this.#worker.request({ type: 'load' }, { signal: task.signal, deadlineMs: this.#config.prepareTimeoutMs })
    this.#loadedOnce = true

    task.signal.throwIfAborted()
    this.#publish({ ...this.#withStep('load', 'complete'), phase: 'ready' })
    this.#log('info', `ParlaSpeech HR model loaded (${hasLocalModel ? 'local' : 'HuggingFace'})`)
  }

  /**
   * Recognize one complete recording.
   * @param input - complete WAV bytes and language hint.
   * @param signal - caller cancellation.
   * @returns final text and measured audio/inference durations.
   */
  async transcribe(input, signal) {
    const phase = this.#state.phase
    if (phase !== 'ready' && phase !== 'standby') {
      throw new Error('Prepare the ParlaSpeech provider before recording')
    }
    if (input.audio.byteLength > this.#config.maxAudioBytes) {
      throw new SpeechInputError('Speech audio exceeds the configured byte limit')
    }

    let decoded
    try {
      decoded = decodeWavMono16k(input.audio)
    } catch (error) {
      throw new SpeechInputError(`Invalid speech WAV: ${error instanceof Error ? error.message : String(error)}`)
    }

    signal?.throwIfAborted()

    if (!this.#worker) {
      const hasLocalModel = await this.#localModelReady()
      const effectiveModelPath = hasLocalModel
        ? this.#config.modelDirectory
        : this.#config.modelId

      this.#worker = new ParlaSpeechWorker(
        this.#resolvedPythonPath ?? this.#config.systemPythonPath,
        this.#config.runtimePath,
        effectiveModelPath,
      )
    }

    if (!this.#loadedOnce) {
      this.#publish({ ...this.#withStep('load', 'running'), phase: 'loading', step: 'load', startedAt: Date.now() })
    }

    try {
      const result = await this.#worker.request(
        {
          type: 'transcribe',
          audio: this.#samplesToBase64Wav(decoded.samples, 16000),
          sampleRate: 16000,
          language: input.language,
        },
        { signal, deadlineMs: this.#config.inferenceTimeoutMs },
      )

      this.#loadedOnce = true
      if (this.#state.phase !== 'ready') {
        this.#publish({ ...this.#withStep('load', 'complete'), phase: 'ready' })
      }
      signal?.throwIfAborted()

      return {
        text: result.text ?? '',
        audioSeconds: decoded.durationSeconds,
        inferenceSeconds: (result.inferenceMs ?? 0) / 1000,
      }
    } catch (error) {
      if (signal?.aborted) throw signal.reason instanceof Error ? signal.reason : error

      if (this.#state.phase === 'loading') {
        this.#publish({
          ...this.#state,
          phase: 'standby',
          steps: (this.#state.steps ?? []).map((item) =>
            item.status === 'running' ? { kind: item.kind, status: 'pending' } : item),
        })
      }

      throw error
    }
  }

  /** Convert Float32Array samples to base64-encoded WAV. */
  #samplesToBase64Wav(samples, sampleRate) {
    const numSamples = samples.length
    const buffer = Buffer.alloc(44 + numSamples * 2)
    buffer.write('RIFF', 0)
    buffer.writeUInt32LE(36 + numSamples * 2, 4)
    buffer.write('WAVE', 8)
    buffer.write('fmt ', 12)
    buffer.writeUInt32LE(16, 16)
    buffer.writeUInt16LE(1, 20)
    buffer.writeUInt16LE(1, 22)
    buffer.writeUInt32LE(sampleRate, 24)
    buffer.writeUInt32LE(sampleRate * 2, 28)
    buffer.writeUInt16LE(2, 32)
    buffer.writeUInt16LE(16, 34)
    buffer.write('data', 36)
    buffer.writeUInt32LE(numSamples * 2, 40)
    for (let i = 0; i < numSamples; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]))
      buffer.writeInt16LE(Math.floor(sample * 32767), 44 + i * 2)
    }
    return buffer.toString('base64')
  }

  /** Cancel preparation; completed readiness is retained. @returns after active work settles. */
  async cancel() {
    const task = this.#preparing
    if (task === undefined) return
    task.abort(new Error('Speech preparation cancelled'))
    await task.settled
  }

  /** Stop inference and release the worker process. @returns after resources settle. */
  async dispose() {
    this.#listeners.clear()
    this.#lifetime.abort(new Error('ParlaSpeech provider disposed'))
    await this.#preparing?.settled.catch(() => {})
    await this.#worker?.dispose()
  }
}
