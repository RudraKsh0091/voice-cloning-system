import torch
import logging
import soundfile as sf
from pathlib import Path

from config import DEVICE, OUTPUTS_DIR, ECAPA_SAMPLE_RATE, XTTS_SAMPLE_RATE
from src.audio.preprocessor import AudioPreprocessor
from src.encoder.speaker_encoder import SpeakerEncoder
from src.synthesizer.tts_engine import TTSEngine
from src.audio.quality_analyzer import AudioQualityAnalyzer

logger = logging.getLogger(__name__)

class VoiceCloner:
    def __init__(self):
        logger.info("Initializing VoiceCloner Pipeline...")
        self.ecapa_preprocessor = AudioPreprocessor(ECAPA_SAMPLE_RATE)
        self.xtts_preprocessor = AudioPreprocessor(XTTS_SAMPLE_RATE)
        self.tts = TTSEngine()
        self.encoder = SpeakerEncoder()
        self.analyzer = AudioQualityAnalyzer()
        logger.info("VoiceCloner ready!")
        
    # ── Public API ────────────────────────────────────────────────────────────
    
    def clone(self, reference_audio: str | Path, text: str, output_path: str | Path = None) -> Path:
        reference_audio = Path(reference_audio)
        output_path = Path(output_path) if output_path else self._auto_output_path(reference_audio)
        
        logger.info("=" * 50)
        logger.info("Voice Cloning Started")
        logger.info(f"  Reference : {reference_audio.name}")
        logger.info(f"  Text      : {text[:60]}{'...' if len(text) > 60 else ''}")
        logger.info(f"  Output    : {output_path}")
        logger.info("=" * 50)
        
        logger.info("[1/3] Preprocessing reference audio...")
        
        # For ECAPA
        ecapa_waveform, _ = self.ecapa_preprocessor.process(reference_audio)
        
        # For XTTS
        xtts_waveform, xtts_sr = self.xtts_preprocessor.process(reference_audio)
        processed_reference = OUTPUTS_DIR / "processed_reference.wav"
        sf.write(
            processed_reference,
            xtts_waveform.squeeze(0).numpy(),
            xtts_sr,
        )
        
        logger.info("[2/3] Extracting speaker embedding...")
        embedding = self.encoder.encode(ecapa_waveform)
        logger.info(f"  Embedding norm : {embedding.norm(p=2, dim=1).item():.4f}")
        
        logger.info("[3/3] Synthesizing cloned speech...")
        logger.info(f"XTTS reference audio: {xtts_sr} Hz")
        _, output_sr = self.tts.synthesize(
            text = text,
            reference_audio_path = processed_reference,
            output_path = output_path,
        )

        logger.info("=" * 50)
        logger.info(f"Voice Cloning Complete → {output_path}")
        logger.info("=" * 50)
        
        return output_path
    
    
    def compare_voices(self, audio_path_1: str | Path, audio_path_2: str | Path) -> float:
        """
        Compare two audio clips and return their speaker similarity score.
        Useful for verifying how close the cloned voice is to the original.

        Returns:
            float: cosine similarity between -1 and 1
                   > 0.85 = very likely same speaker
                   > 0.70 = probably same speaker
                   < 0.50 = likely different speakers
        """
        
        wav1, _ = self.ecapa_preprocessor.process(audio_path_1)
        wav2, _ = self.ecapa_preprocessor.process(audio_path_2)

        emb1 = self.encoder.encode(wav1)
        emb2 = self.encoder.encode(wav2)

        similarity = self.encoder.similarity(emb1, emb2)
        logger.info(f"Voice similarity: {similarity:.4f}")

        return similarity
    
    def analyze_reference(self, audio_path):
        return self.analyzer.analyze(audio_path)
    
    # ── Private Methods ───────────────────────────────────────────────────────

    def _auto_output_path(self, reference_audio: Path) -> Path:
        stem = reference_audio.stem
        path = OUTPUTS_DIR / f"{stem}_cloned.wav"

        # Avoid overwriting existing files
        counter = 1
        while path.exists():
            path = OUTPUTS_DIR / f"{stem}_cloned_{counter}.wav"
            counter += 1

        return path