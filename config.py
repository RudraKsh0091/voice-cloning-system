import os
from pathlib import Path

# ── Project Root ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
SRC_DIR  = ROOT_DIR / "src"

# ── Directories ─────────────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT_DIR / "outputs"
SAMPLES_DIR = ROOT_DIR / "samples"

OUTPUTS_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

# ── Audio Preprocessing ──────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Hz — ECAPA-TDNN expects 16kHz
TARGET_DURATION   = 6.0     # seconds — ideal reference clip length
MIN_DURATION      = 3.0     # seconds — reject clips shorter than this
MAX_DURATION      = 30.0    # seconds — trim clips longer than this
TOP_DB = 30

# ── Speaker Encoder ──────────────────────────────────────────────────────────────
SPEAKER_ENCODER_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIM          = 192   # ECAPA-TDNN output dimension

# ── TTS Model (XTTS-v2) ─────────────────────────────────────────────────────────
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE    = "en"

# ── Output Audio ────────────────────────────────────────────────────────────────
OUTPUT_SAMPLE_RATE = 24000  # Hz — XTTS outputs at 24kHz

# ── Device ──────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"