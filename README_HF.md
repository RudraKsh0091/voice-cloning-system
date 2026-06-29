---
title: Voice Cloning System
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🎙️ Voice Cloning System

Clone any voice from a 4-10 second audio sample and synthesize speech in that voice.

Built with **ECAPA-TDNN** (speaker encoder) + **XTTS-v2** (zero-shot TTS) + **HiFi-GAN** (vocoder).

## How to Use

1. Upload a short audio clip (4-10 seconds) of the target voice
2. Enter any text you want spoken in that voice  
3. Click **Clone Voice**
4. Listen and download the result

> ⚠️ Running on CPU — inference takes ~60-80 seconds. GPU recommended for real-time performance.