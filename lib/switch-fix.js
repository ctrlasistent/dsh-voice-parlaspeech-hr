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


/**
 * Relaxes provider switching in the experimental voice-input selection service.
 *
 * The stock `configure()` refuses a provider switch while the persisted language is not
 * in the target provider's language list (for example Croatian selected on the local
 * Whisper provider, which SenseVoice cannot accept), forcing users through a two-step
 * language reset. This wrapper keeps every other behavior identical: a switch whose
 * target accepts the current language (and every language-only change) runs the
 * original code; only the failing provider-switch case persists the switch together
 * with `language: 'auto'`, which every provider accepts. The disposer restores the
 * original method; cordis reapplies this plugin whenever the service is replaced.
 */

const PATCHED = Symbol('dsh-voice-parlaspeech-hr.switch-fix')

/**
 * @param service - the live speechToText service instance.
 * @returns a disposer restoring the original configure, or undefined when nothing was patched.
 */
export function installSwitchFix(service) {
  if (service === null || typeof service !== 'object') return undefined
  if (typeof service.configure !== 'function' || service[PATCHED]) return undefined
  const original = service.configure
  const patched = async function configure(patch) {
    const relaxable = patch !== null && typeof patch === 'object'
      && patch.providerId !== undefined && patch.language === undefined
    if (relaxable) {
      const registration = service.providers?.get?.(patch.providerId)
      const current = service.config?.language?.get?.()
      const supported = registration !== undefined && current !== undefined
        && registration.provider.info.languages.includes(current)
      if (!supported) {
        const settings = service.ctx?.get?.('settings')
        const entry = service.entryId
        if (registration !== undefined && settings && entry) {
          // The switch itself is valid; only the carried language is not. Every
          // provider accepts automatic detection, so persist both together.
          await settings.update(entry, { defaultProvider: patch.providerId, language: 'auto' })
          return
        }
      }
    }
    return original.call(this, patch)
  }
  service.configure = patched
  service[PATCHED] = true
  return () => {
    if (service.configure === patched) {
      service.configure = original
      delete service[PATCHED]
    }
  }
}
