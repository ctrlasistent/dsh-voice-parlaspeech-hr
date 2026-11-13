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

/** Optional local ParlaSpeech HR provider; activation performs no downloads or model loading. */
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ParlaSpeechPreparation } from './preparation.js'
import { installSwitchFix } from './switch-fix.js'

export { Config } from './config.js'
export const name = 'voice-parlaspeech-hr'

const HERE = dirname(fileURLToPath(import.meta.url))
const PLUGIN_ROOT = resolve(HERE, '..')
const DEFAULT_RUNTIME = resolve(PLUGIN_ROOT, 'runtime', 'worker.py')

/**
 * No static `inject`: this plugin must activate on its own so the plugin
 * manager never shows it stuck on "Waiting for dependencies". The official
 * Voice Input bundle can be disabled at install time or toggled later; the
 * recognizer registers into `speechToText` the moment that service exists
 * (retroactive injection) and unregisters again when the bundle is disabled.
 */

/** Normalize config without relying on Config.parse (dsh already validates). */
function normalizeConfig(raw) {
  const source = raw !== null && typeof raw === 'object' ? raw : {}
  const text = (value, fallback) =>
    typeof value === 'string' && value.trim() !== '' ? value.trim() : fallback
  const number = (value, fallback, minimum) => {
    const parsed = typeof value === 'number' ? value : Number(value)
    return Number.isFinite(parsed) && parsed >= minimum ? Math.floor(parsed) : fallback
  }
  return {
    providerId: text(source.providerId, 'parlaspeech-hr'),
    modelId: text(source.modelId, 'classla/wav2vec2-xls-r-parlaspeech-hr'),
    modelDirectory: text(source.modelDirectory, undefined),
    pythonPath: text(source.pythonPath, undefined),
    systemPythonPath: text(source.systemPythonPath, process.platform === 'win32' ? 'python' : 'python3'),
    maxAudioBytes: number(source.maxAudioBytes, 25 * 1024 * 1024, 1),
    inferenceTimeoutMs: number(source.inferenceTimeoutMs, 300_000, 1),
    prepareTimeoutMs: number(source.prepareTimeoutMs, 1_800_000, 1),
    venvAutoInstall: source.venvAutoInstall !== false,
    runtimePath: DEFAULT_RUNTIME,
  }
}

/**
 * Origins offered for an explicit preparation download.
 *
 * IMPORTANT (dsh 0.2.2-rc.1 wire contract): `downloadSources` MUST be an
 * array of origin STRINGS (`readonly string[]`). Older plugin versions sent
 * objects here (`[{ type: 'huggingface', url: … }]`); dsh <= 0.1.7 stripped
 * the unknown field silently, but dsh >= 0.2.2 validates every
 * speech.follow()/speech.catalog() item strictly and one invalid snapshot
 * fails the whole readiness stream — which made the provider/language
 * selector (the whole "hr" menu) disappear from Voice Input.
 */
function downloadSources() {
  return ['https://huggingface.co']
}

/**
 * Register the local recognizer into a scope that owns the speechToText service.
 * @param scope - context whose `speechToText` service receives the registration.
 * @param config - validated runtime configuration.
 * @param preparation - the preparation state machine backing the provider.
 * @returns an async disposer removing the registration and releasing resources.
 */
function register(scope, config, preparation) {
  const restoreSwitchFix = installSwitchFix(scope.speechToText)

  const unregister = scope.speechToText.register({
    info: {
      id: config.providerId,
      name: 'ParlaSpeech HR (local)',
      location: 'host-local',
      languages: ['hr', 'auto'],
      downloadSources: downloadSources(),
      setupEstimate: {
        recommendedDiskBytes: 3 * 1024 * 1024 * 1024, // ~1.2 GB model + ~1.8 GB venv
        expectedMemoryBytes: 2 * 1024 * 1024 * 1024,
        minimumMinutes: 2,
        maximumMinutes: 30,
      },
    },
    preparation,
    transcribe: async (input, signal) => await preparation.transcribe(input, signal),
  })

  preparation.inspect()

  return async () => {
    restoreSwitchFix?.()
    const removing = unregister()
    await preparation.dispose()
    await removing
  }
}

/**
 * Activate the provider. The plugin itself is active immediately; only the
 * registration into the official Voice Input bundle waits for that bundle.
 * @param ctx - Host context; a logger is used, and the speechToText service through the injected scope.
 * @param config - validated runtime configuration.
 */
export function apply(ctx, config) {
  const value = normalizeConfig(config)
  const preparation = new ParlaSpeechPreparation(ctx, value)

  // Retroactive injection: the callback runs once the service exists
  ctx.inject(['speechToText'], (scope) => {
    ctx.logger.info('[dsh-voice-parlaspeech-hr] Voice Input bundle is running; ParlaSpeech HR recognizer registered.')
    return register(scope, value, preparation)
  })

  if (ctx.get('speechToText') === undefined) {
    ctx.logger.warn(
      '[dsh-voice-parlaspeech-hr] The official Voice Input bundle is not active, so the microphone button is missing.'
      + ' Enable it under Plugins (Voice Input); this recognizer registers automatically as soon as it is running.'
    )
  }
}
