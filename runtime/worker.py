#!/usr/bin/env python3

"""
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
"""


"""
dsh-voice-parlaspeech-hr runtime: a persistent wav2vec2 worker driven over stdio.

The Host half (lib/index.js) spawns this file once and keeps it alive; every
transcription request is one JSON line in, one JSON line out. Loading the model
costs several seconds, so doing it once per session is the whole point.

Protocol — stdout carries NOTHING but protocol lines, one JSON object per line:
<- {"ready": true, "model": "classla/wav2vec2-xls-r-parlaspeech-hr"}
-> {"id": 1, "type": "load"}
<- {"ok": true, "id": 1, "loadMs": 0}
-> {"id": 2, "type": "transcribe", "audio": "<base64 WAV>", "sampleRate": 16000}
<- {"ok": true, "id": 2, "text": "prepoznati tekst", "inferenceMs": 142}
<- {"ok": false, "id": 2, "error": "..."}

Everything else (library progress bars, warnings, tracebacks) goes to stderr.

argv: <modelPath>
  modelPath is either a Hugging Face model ID (e.g. classla/wav2vec2-xls-r-parlaspeech-hr)
  or a local directory containing model files (model.safetensors, config.json, etc.).

Requires: pip install transformers torch soundfile librosa numpy
"""
import base64
import io
import json
import os
import sys
import time
import traceback
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import pipeline

# Reserve the real stdout for the protocol BEFORE importing heavy libraries
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


SAMPLE_RATE = 16000


def send(payload):
    _PROTOCOL.write(json.dumps(payload) + "\n")
    _PROTOCOL.flush()


def log(message):
    print(f"[parlaspeech] {message}", file=sys.stderr, flush=True)


def is_local_model(model_path):
    """Check if model_path is a local directory with model files."""
    if not os.path.isdir(model_path):
        return False
    return os.path.exists(
        os.path.join(model_path, "model.safetensors")
    ) or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))


def load_engine(model_path):
    """Build the ASR pipeline. model_path can be a Hugging Face ID or a local directory."""
    device = 0 if torch.cuda.is_available() else -1
    local = is_local_model(model_path)

    if local:
        log(f"loading LOCAL model from {model_path} on device {device}")
    else:
        log(f"loading HuggingFace model {model_path} on device {device}")

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        tokenizer=model_path,
        device=device,
    )

    log("model loaded successfully")
    return asr


def transcribe_audio(asr, audio_base64, sample_rate):
    """Decode base64 WAV audio and run transcription."""
    audio_bytes = base64.b64decode(audio_base64)
    buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buffer)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16000 Hz if needed
    if sr != SAMPLE_RATE:
        log(f"resampling from {sr} to {SAMPLE_RATE}")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Run ASR pipeline
    result = asr(
        {
            "raw": audio,
            "sampling_rate": SAMPLE_RATE,
        }
    )

    return result["text"].strip()


def main():
    model_path = (
        sys.argv[1].strip()
        if len(sys.argv) > 1
        else "classla/wav2vec2-xls-r-parlaspeech-hr"
    )

    log(f"starting ParlaSpeech HR worker (model={model_path})")
    asr = load_engine(model_path)

    log("ready")
    send({"ready": True, "model": model_path})

    try:
        for line in sys.stdin:
            line = line.strip()
            if line == "":
                continue

            try:
                request_id = None
                request = json.loads(line)
                request_id = request.get("id")
                msg_type = request.get("type")

                if msg_type == "load":
                    # Model already loaded at startup
                    send({"ok": True, "id": request_id, "loadMs": 0})
                    continue

                if msg_type != "transcribe":
                    raise ValueError(f"unknown message type: {msg_type}")

                audio_base64 = request.get("audio")
                sample_rate = request.get("sampleRate", SAMPLE_RATE)

                if not audio_base64:
                    raise ValueError("missing audio field")

                start_time = time.time()
                text = transcribe_audio(asr, audio_base64, sample_rate)
                inference_ms = int((time.time() - start_time) * 1000)

                send(
                    {
                        "ok": True,
                        "id": request_id,
                        "text": text,
                        "inferenceMs": inference_ms,
                    }
                )

            except Exception as error:
                log(traceback.format_exc())
                send({"ok": False, "id": request_id, "error": str(error)})
    except KeyboardInterrupt:
        # Ctrl+C must be caught so the Host can close the stream gracefully
        # instead of being left hanging on a half-open pipe.
        log("interrupted with Ctrl+C")
        send({"ok": False, "id": None, "error": "interrupted by Ctrl+C"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        sys.exit(1)
