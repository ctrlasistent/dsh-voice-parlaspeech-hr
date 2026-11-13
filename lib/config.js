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

/** Deployment configuration for the local ParlaSpeech HR recognizer. */
import z from '@deepseek-ai/schemastery'

/**
 * Local runtime and inference settings.
 *
 * Two usage modes:
 *
 * 1. ADVANCED USER (everything pre-installed):
 *    Set `modelDirectory` to a folder containing model.safetensors + config.json
 *    and `pythonPath` to a Python executable that already has transformers, torch,
 *    soundfile, librosa, numpy installed. The plugin uses these directly — no venv
 *    creation, no pip install, no Hugging Face download.
 *
 * 2. NEW USER (nothing installed):
 *    Leave `modelDirectory` and `pythonPath` as empty strings (''), or omit them
 *    entirely from the config, or leave the YAML keys bare (`modelDirectory:` —
 *    parsed as null). All of null / '' / missing are treated as "not set".
 *    The plugin automatically creates a virtual environment in runtime/venv,
 *    installs all Python packages, and downloads the model from Hugging Face
 *    on first use.
 *
 *    IMPORTANT YAML note: a bare key (`modelDirectory:`) is null, NOT an empty
 *    string. The schema normalizes every spelling to '', but prefer writing
 *    `modelDirectory: ''` explicitly — some dsh versions reject null config
 *    values before this schema ever sees them.
 */
export const Config = z.object({
  /** Unique registration id; consumers select this exact id. */
  providerId: z.string().min(1).default('parlaspeech-hr'),

  /** Hugging Face model ID (used when modelDirectory is not set). */
  modelId: z.string().min(1).default('classla/wav2vec2-xls-r-parlaspeech-hr'),

  /**
   * Existing local directory with model files (model.safetensors, config.json, etc.).
   * When set, the plugin uses this directory directly instead of downloading from HF.
   * Empty string '', null (bare YAML key) or a missing key all mean "not set"
   * and trigger the automatic Hugging Face download on first use. The default
   * normalizes null/missing to '' so every spelling behaves identically.
   */
  modelDirectory: z.string().default(''),

  /**
   * Existing Python executable with all required packages already installed
   * (e.g. /home/user/croatian-asr/venv/bin/python).
   * When set, the plugin uses this Python directly — no venv creation, no pip install.
   * Empty string '', null (bare YAML key) or a missing key all mean "not set"
   * and trigger the automatic venv creation + pip install on first use.
   */
  pythonPath: z.string().default(''),

  /** System Python used for venv creation when pythonPath is not set or lacks packages. */
  systemPythonPath: z.string().min(1).default(process.platform === 'win32' ? 'python' : 'python3'),

  /** Maximum decoded audio bytes accepted per recording (~13 min at 16 kHz mono). */
  maxAudioBytes: z.natural().min(1).default(25 * 1024 * 1024),

  /** Deadline for one inference after the worker is ready. */
  inferenceTimeoutMs: z.natural().min(1).default(300_000),

  /** Deadline for preparation (venv creation + pip install + model download). */
  prepareTimeoutMs: z.natural().min(1).default(1_800_000),

  /** Allow automatic venv creation and pip install when no suitable Python is found. */
  venvAutoInstall: z.boolean().default(true),
})
