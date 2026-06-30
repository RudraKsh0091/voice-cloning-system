import torch
import soundfile as sf
import logging
from pathlib import Path

from config import XTTS_MODEL_NAME, TTS_LANGUAGE, OUTPUT_SAMPLE_RATE, DEVICE

logger = logging.getLogger(__name__)

class TTSEngine:
    """
    Wraps Coqui XTTS-v2 for zero-shot voice cloning.
    Takes text + reference audio path, outputs a cloned waveform.
    """
    
    def __init__(self):
        self.device = DEVICE
        self.model_name = XTTS_MODEL_NAME
        self.language = TTS_LANGUAGE
        self.model = None
        self._load_model()
        
    # ── Public API ────────────────────────────────────────────────────────────
    
    def synthesize(self, text: str, reference_audio_path: str | Path, output_path: str | Path = None) -> tuple[torch.Tensor, int]:
        self._validate_inputs(text, reference_audio_path)

        reference_audio_path = str(Path(reference_audio_path).resolve())
        logger.info(f"Synthesizing: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        logger.info(f"Reference   : {Path(reference_audio_path).name}")

        import torchaudio

        def patched_torchaudio_load(path, *args, **kwargs):
            data, sr = sf.read(str(path), always_2d=True)
            waveform = torch.from_numpy(data.T).float()
            return waveform, sr

        original_torchaudio_load = torchaudio.load
        torchaudio.load = patched_torchaudio_load

        wav_list = self.model.tts(
            text=text,
            speaker_wav=reference_audio_path,
            language=self.language,
        )

        torchaudio.load = original_torchaudio_load

        waveform = torch.tensor(wav_list).unsqueeze(0)
        logger.info(f"Synthesized {waveform.shape[1] / OUTPUT_SAMPLE_RATE:.2f}s of audio")

        if output_path:
            self._save(waveform, output_path)

        return waveform, OUTPUT_SAMPLE_RATE
    
    # ── Private Methods ───────────────────────────────────────────────────────
    
    def _load_model(self):
        logger.info(f"Loading XTTS-v2 — this may take a moment...")
        logger.info("First run will download ~1.8GB of model weights...")

        try:
            import os
            os.environ["COQUI_TOS_AGREED"] = "1"
            
            import torch
            original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, "weights_only": False})

            import torchaudio
            import soundfile as sf
            import numpy as np

            def patched_torchaudio_load(path, *args, **kwargs):
                data, sr = sf.read(str(path), always_2d=True)
                waveform = torch.from_numpy(data.T).float()
                return waveform, sr

            original_torchaudio_load = torchaudio.load
            torchaudio.load = patched_torchaudio_load

            from TTS.api import TTS

            self.model = TTS(
                model_name=self.model_name,
                progress_bar=True,
            ).to(self.device)

            torch.load = original_torch_load
            torchaudio.load = original_torchaudio_load

            self.model.synthesizer.tts_model.eval()
            logger.info(f"XTTS-v2 loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS-v2: {e}")
        
    def _save(self, waveform: torch.Tensor, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_np = waveform.squeeze(0).numpy()
        sf.write(str(output_path), audio_np, OUTPUT_SAMPLE_RATE)
        logger.info(f"Saved output audio → {output_path}")
    
    def _validate_inputs(self, text: str, reference_audio_path: str | Path):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > 500:
            raise ValueError(f"Text too long: {len(text)} chars (max 500). Split into smaller chunks.")

        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            raise ValueError(f"Reference audio not found: {ref_path}")

        suffix = ref_path.suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".m4a"}:
            raise ValueError(f"Unsupported audio format: {suffix}")