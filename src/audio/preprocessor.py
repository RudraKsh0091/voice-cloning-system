import librosa
import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import logging
from pathlib import Path

from config import SAMPLE_RATE, MIN_DURATION, MAX_DURATION, TARGET_DURATION, TOP_DB

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self):
        self.target_sr = SAMPLE_RATE
        self.min_duration = MIN_DURATION
        self.max_duration = MAX_DURATION
        self.target_dur = TARGET_DURATION
        
    # ── Public API ────────────────────────────────────────────────────────────
     
    def process(self, audio_path: str | Path) -> tuple[torch.Tensor, int]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Processing audio: {audio_path.name}")
        
        waveform, sr = self._load(audio_path)
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sr)
        waveform = self._trim_silence(waveform)
        waveform = self._validate_duration(waveform)
        waveform = self._normalize(waveform)
        
        duration = waveform.shape[1] / self.target_sr
        logger.info(f"Preprocessed audio: {duration:.2f}s @ {self.target_sr}Hz mono")
        
        return waveform, self.target_sr
    
    # ── Private Steps ─────────────────────────────────────────────────────────
     
    def _load(self, path: Path) -> tuple[torch.Tensor, int]:
        """Load audio using soundfile (reliable on Windows, no torchcodec needed)."""
        try:
            data, sr = sf.read(str(path), always_2d=True)
            waveform = torch.from_numpy(data.T).float()
            logger.info(f"Loaded: {waveform.shape}, sr={sr}")
            return waveform, sr
        except Exception as e:
            raise ValueError(f"Could not load audio file '{path.name}': {e}")
        
    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > 1:
            logger.info(f"Converting {waveform.shape[0]}-channel audio to mono")
            waveform = waveform.mean(dim = 0, keepdim = True)
        return waveform
    
    def _resample(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        if original_sr != self.target_sr:
            logger.info(f"Resampling {original_sr}Hz -> {self.target_sr}Hz")
            resampler = T.Resample(orig_freq = original_sr, new_freq = self.target_sr)
            waveform = resampler(waveform)
        return waveform
    
    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        signal = waveform.squeeze(0).numpy()

        trimmed, _ = librosa.effects.trim(
            signal,
            top_db=TOP_DB
        )

        trimmed = torch.from_numpy(trimmed).unsqueeze(0)

        original = waveform.shape[1] / self.target_sr
        after = trimmed.shape[1] / self.target_sr

        logger.info(
            f"Silence trimmed: {original:.2f}s -> {after:.2f}s"
        )

        return trimmed
    
    def _validate_duration(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[1]
        duration = num_samples / self.target_sr
        if duration < self.min_duration:
            raise ValueError(
                f"Audio too short: {duration:.2f}s"
                f"(minimum required: {self.min_duration}s)"
                f"Please provide a longer clip"
            )
            
        if duration > self.max_duration:
            logger.info(f"Audio too long ({duration:.2f}s), trimming to {self.max_duration}s")
            max_samples = int(self.max_duration * self.target_sr)
            waveform    = waveform[:, :max_samples]
            
        return waveform
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.95
        return waveform
    
    