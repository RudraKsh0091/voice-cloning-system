import torch
import functools

if not hasattr(torch.amp, 'custom_fwd'):
    def custom_fwd(fwd=None, *, device_type='cuda', **kwargs):
        if fwd is None:
            return functools.partial(custom_fwd, device_type=device_type)
        return torch.cuda.amp.custom_fwd(fwd)
    torch.amp.custom_fwd = custom_fwd

if not hasattr(torch.amp, 'custom_bwd'):
    def custom_bwd(bwd):
        return torch.cuda.amp.custom_bwd(bwd)
    torch.amp.custom_bwd = custom_bwd
    
import logging
from pathlib import Path

from config import SPEAKER_ENCODER_SOURCE, EMBEDDING_DIM, DEVICE, MIN_DURATION

logger = logging.getLogger(__name__)

class SpeakerEncoder:
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self._load_model()
        
    # ── Public API ────────────────────────────────────────────────────────────
    
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self._validate_input(waveform)
        
        audio = waveform.squeeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_batch(audio)
            embedding = embedding.squeeze(1)
            embedding = torch.nn.functional.normalize(embedding, p = 2, dim = 1)
            
        logger.info(F"Speaker embedding extracted: {embedding.shape}")
        
        return embedding.cpu()
    
    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two speaker embeddings.
        Returns a value between -1 and 1. Above 0.75 = likely same speaker.
        """
        sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim = 1)
        return sim.item()
    
    # ── Private Methods ───────────────────────────────────────────────────────
     
    def _load_model(self):
        logger.info(f"Loading ECAPA-TDNN from: {SPEAKER_ENCODER_SOURCE}")
        logger.info("First run will download ~80MB of model weights...")

        try:
            import os
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            self.model = EncoderClassifier.from_hparams(source=SPEAKER_ENCODER_SOURCE, savedir="pretrained_models/ecapa_tdnn", run_opts={"device": self.device}, local_strategy=LocalStrategy.COPY)
            self.model.eval()
            logger.info(f"ECAPA-TDNN loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load speaker encoder: {e}")
        
    def _validate_input(self, waveform: torch.Tensor):
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(f"Expected waveform shape (1, num_samples), got {waveform.shape}")
        
        min_samples = 16000 * MIN_DURATION
        if waveform.shape[1] < min_samples:
            raise ValueError(
                f"Waveform too short: {waveform.shape[1]} samples "
                f"(minimum {min_samples})"
            )