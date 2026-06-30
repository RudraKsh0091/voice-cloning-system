import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

from config import MIN_DURATION, MAX_DURATION, TOP_DB

class AudioQualityAnalyzer:
    """
    Analyze a reference audio file and estimate how suitable it is for voice cloning.
    """

    # ── Public API ────────────────────────────────────────────────────────────
    
    def analyze(self, audio_path: str | Path) -> dict:
        audio_path = Path(audio_path)

        audio, sr = sf.read(str(audio_path))

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sr

        peak = float(np.max(np.abs(audio)))

        clipped = peak >= 0.99

        rms = float(np.sqrt(np.mean(audio ** 2)))

        trimmed_audio, index = librosa.effects.trim(
            audio,
            top_db=TOP_DB
        )
        

        trimmed_length = len(trimmed_audio)
        original_length = len(audio)

        silence_percent = ((original_length - trimmed_length) / original_length) * 100

        noise_region = audio[np.abs(audio) < 0.01]

        if len(noise_region) > 0:
            noise_floor = np.mean(np.abs(noise_region))
        else:
            noise_floor = 0
            
        if noise_floor < 0.002:
            noise = "Low"

        elif noise_floor < 0.01:
            noise = "Moderate"

        else:
            noise = "High"
        
        score = 100

        # Duration
        if duration < MIN_DURATION:
            score -= 30
        elif duration < 6:
            score -= 10
        elif duration > MAX_DURATION:
            score -= 10

        # Clipping
        if clipped:
            score -= 20

        # Loudness
        if rms < 0.02:
            score -= 15

        # Silence
        if silence_percent > 30:
            score -= 20
        elif silence_percent > 15:
            score -= 10

        score = max(0, min(score, 100))

        if score >= 90:
            rating = "🟢 Excellent"
        elif score >= 75:
            rating = "🟡 Good"
        elif score >= 60:
            rating = "🟠 Fair"
        else:
            rating = "🔴 Poor"
            
            
        recommendations = []
        
        if clipped:
            recommendations.append(
                "Avoid microphone clipping."
            )

        if silence_percent > 20:
            recommendations.append(
                "Trim long silent sections."
            )

        if duration < 6:
            recommendations.append(
                "Use an 8–10 second recording."
            )

        if noise == "High":
            recommendations.append(
                "Reduce background noise."
            )

        if rms < 0.02:
            recommendations.append(
                "Speak slightly louder."
            )
            
        if len(recommendations) == 0:
            recommendations.append(
                "Excellent reference audio."
            )

        return {
            "score": score,
            "rating": rating,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "peak": round(peak, 3),
            "rms": round(rms, 4),
            "silence": round(silence_percent, 1),
            "clipping": clipped,
            "noise" : noise,
            "recommendations" : recommendations
        }