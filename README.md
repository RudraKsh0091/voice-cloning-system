# 🎙️ Voice Cloning System

> Clone any voice from a 4–10 second audio sample and synthesize speech in that voice using state-of-the-art pretrained models.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Demo

> 🤗 [Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/RudraKsh-00/voice-cloning-system)

## Architecture

```
Voice Sample (4-10s)
        ↓
Audio Preprocessor        ← resample, mono, normalize, trim silence
        ↓
Speaker Encoder           ← ECAPA-TDNN (SpeechBrain, VoxCeleb pretrained)
        ↓
Speaker Embedding         ← 192-dim L2-normalized voice fingerprint
        ↓
TTS Synthesizer           ← XTTS-v2 (Coqui, zero-shot voice cloning)
        ↓
Neural Vocoder            ← HiFi-GAN (integrated inside XTTS-v2)
        ↓
Cloned Voice Output (.wav)
```

## Stack

| Component | Model | Details |
|---|---|---|
| Speaker Encoder | ECAPA-TDNN | SpeechBrain, pretrained on VoxCeleb |
| TTS | XTTS-v2 | Coqui, 17 languages, zero-shot cloning |
| Vocoder | HiFi-GAN | Integrated inside XTTS-v2 |
| UI | Gradio | HuggingFace Spaces compatible |

## Project Structure

```
voice-cloning-system/
├── src/
│   ├── audio/
│   │   └── preprocessor.py     # Audio cleaning & normalization
│   ├── encoder/
│   │   └── speaker_encoder.py  # ECAPA-TDNN speaker embedding
│   └── synthesizer/
│       └── tts_engine.py       # XTTS-v2 inference
├── src/pipeline.py             # VoiceCloner — end-to-end glue layer
├── app.py                      # Gradio web UI
├── config.py                   # Centralized settings
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/RudraKsh0091/voice-cloning-system
cd voice-cloning-system
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` in your browser.

## Key Engineering Details

- **Inference only** — no training. Uses pretrained ECAPA-TDNN and XTTS-v2
- **Modular pipeline** — each component (preprocessor, encoder, TTS) is independently testable
- **Windows compatible** — patched torchaudio.load and torch.amp for Windows + PyTorch 2.1
- **Auto device detection** — runs on CPU locally, GPU on cloud automatically

## Performance

| Environment | Inference Time | Quality |
|---|---|---|
| CPU (local) | ~60-80s | Good |
| T4 GPU (Kaggle/Colab) | ~5-8s | Excellent |

## Author

**Rudraksh** — B.Tech CSE, IIIT Una  
[GitHub](https://github.com/RudraKsh0091) · [HuggingFace](https://huggingface.co/RudraKsh-00)