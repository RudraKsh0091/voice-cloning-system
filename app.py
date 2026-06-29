# ── Compatibility patches (must be before all other imports) ──────────────────
import librosa  # force librosa to initialize before speechbrain

import sys
import unittest.mock
import functools
import torch

# Patch torch.amp for SpeechBrain + PyTorch 2.1 compatibility
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

# ── Imports ───────────────────────────────────────────────────────────────────
import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
import logging
from pathlib import Path

from src.pipeline import VoiceCloner
from config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Load pipeline once at startup ─────────────────────────────────────────────
print("Initializing Voice Cloning System...")
cloner = VoiceCloner()
print("System ready!")

# ── Core inference function ───────────────────────────────────────────────────

def clone_voice(reference_audio, text):
    """
    Gradio inference function.

    Args:
        reference_audio : path to uploaded audio file (provided by Gradio)
        text            : text to synthesize

    Returns:
        output_audio  : path to generated wav file
        similarity_md : markdown string showing similarity score
        status_md     : markdown string showing status
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if reference_audio is None:
        return None, "", "⚠️ Please upload a reference audio file."

    if not text or not text.strip():
        return None, "", "⚠️ Please enter some text to synthesize."

    if len(text.strip()) < 3:
        return None, "", "⚠️ Text is too short. Please enter at least a few words."

    if len(text) > 500:
        return None, "", f"⚠️ Text too long ({len(text)} chars). Please keep it under 500 characters."

    try:
        logger.info(f"Cloning voice | text='{text[:50]}...'")

        # Save output to a temp file Gradio can serve
        output_path = OUTPUTS_DIR / "gradio_output.wav"
        
        quality = cloner.analyze_reference(reference_audio)

        quality_md = f"""
        ### 🎤 Reference Audio Analysis

        **Quality Score:** {quality['score']}/100

        **Rating:** {quality['rating']}

        | Metric | Value |
        |--------|------|
        | Duration | {quality['duration']} s |
        | Sample Rate | {quality['sample_rate']} Hz |
        | RMS Loudness | {quality['rms']} |
        | Silence | {quality['silence']} % |
        | Clipping | {"Yes" if quality["clipping"] else "No"} |
        """

        output_path = cloner.clone(
            reference_audio = reference_audio,
            text            = text.strip(),
            output_path     = output_path,
        )

        # Compute similarity between original and cloned
        similarity = cloner.compare_voices(reference_audio, output_path)
        sim_pct    = similarity * 100

        # Similarity feedback
        if similarity > 0.75:
            sim_label = "🟢 Excellent match"
        elif similarity > 0.55:
            sim_label = "🟡 Good match"
        elif similarity > 0.35:
            sim_label = "🟠 Moderate match"
        else:
            sim_label = "🔴 Low match (try a cleaner recording)"

        similarity_md = f"**Speaker Similarity:** {sim_pct:.1f}% — {sim_label}"
        status_md     = "✅ Voice cloning complete!"

        return str(output_path), quality_md, similarity_md, status_md

    except ValueError as e:
        return None, "", "", f"⚠️ Input error: {str(e)}"
    except Exception as e:
        logger.error(f"Cloning failed: {e}")
        return None, "", "", f"❌ Error: {str(e)}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Voice Cloning System") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("""
        # 🎙️ Voice Cloning System
        Clone any voice from a short audio sample using ECAPA-TDNN + XTTS-v2.
        
        **How to use:**
        1. Upload a **4-10 second** audio clip of the target voice (wav, mp3, flac)
        2. Enter the **text** you want spoken in that voice
        3. Click **Clone Voice** and wait for generation
        """)

        # ── Main layout ───────────────────────────────────────────────────────
        with gr.Row():

            # Left column — inputs
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Reference Audio")
                reference_audio = gr.Audio(
                    label     = "Upload voice sample (4-10 seconds)",
                    type      = "filepath",
                    sources   = ["upload", "microphone"],
                )

                gr.Markdown("### 📝 Text to Synthesize")
                text_input = gr.Textbox(
                    label       = "Enter text (max 500 characters)",
                    placeholder = "Hello! This is a test of the voice cloning system.",
                    lines       = 4,
                    max_lines   = 8,
                )

                char_count = gr.Markdown("0 / 500 characters")
                text_input.change(
                    fn     = lambda t: f"{len(t)} / 500 characters",
                    inputs = text_input,
                    outputs= char_count,
                )

                clone_btn = gr.Button(
                    "🔊 Clone Voice",
                    variant = "primary",
                    size    = "lg",
                )

            # Right column — outputs
            with gr.Column(scale=1):
                gr.Markdown("### 🔈 Cloned Voice Output")
                quality_display = gr.Markdown("")
                output_audio = gr.Audio(
                    label    = "Generated audio",
                    type     = "filepath",
                    autoplay = True,
                )

                similarity_display = gr.Markdown("")
                status_display     = gr.Markdown("")

        # ── Examples ──────────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 💡 Tips for best results")
        gr.Markdown("""
        - Use a **clean recording** with no background noise or music
        - **WAV format** works better than MP3
        - Speak **naturally** in the reference clip — not too fast, not too slow
        - Longer reference clips (8-10s) generally produce better results
        - Run on **GPU** for faster inference and higher quality output
        """)

        # ── Wire up the button ────────────────────────────────────────────────
        clone_btn.click(
            fn      = clone_voice,
            inputs  = [reference_audio, text_input],
            outputs = [output_audio, quality_display, similarity_display, status_display],
        )

    return demo


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
    server_name = "0.0.0.0",
    server_port = 7860,
    share       = False,
    theme       = gr.themes.Soft(),
)