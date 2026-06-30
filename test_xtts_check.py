from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hello, this is a test of the voice cloning system and check for errors.",
    speaker_wav="samples/hf_rudra_1.wav",
    language="en",
    file_path="official.wav"
)

from src.audio import preprocessor
from src.encoder import speaker_encoder
ecapa_preprocessor = preprocessor.AudioPreprocessor(16000)

wav1, _ = ecapa_preprocessor.process("samples/hf_rudra_1.wav")
wav2, _ = ecapa_preprocessor.process("outputs/official.wav")

encoder = speaker_encoder.SpeakerEncoder()
emb1 = encoder.encode(wav1)
emb2 = encoder.encode(wav2)

similarity = encoder.similarity(emb1, emb2)
print(f"Voice similarity: {similarity:.4f}")
