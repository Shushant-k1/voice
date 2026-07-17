# Failure Analysis & Honest Limitations

## Where Things Break

### English
- **Chatterbox latency is too high**: 5.52s average batch latency for a 10-word sentence (target: <2s). RTF of 1.076 means it generates speech roughly at 1× real time — unusable for real-time streaming voice agents without massive optimization or hardware scaling.
- **ASR normalization artifacts**: Whisper large-v3 transcription formatting (e.g. converting spoken numbers to digits "twelve point five" → "12.5") occasionally inflates WER calculations. However, the generated audio is fully intelligible.
- **OpenVoice V2 similarity limitations**: While extremely fast (1.83s batch latency, 0.386 RTF), OpenVoice V2 achieves a cosine similarity of **0.726** (below the 0.75 target). This is because OpenVoice uses a style transfer approach that decouples tone from language, leading to some degradation in zero-shot similarity compared to native zero-shot models.

### Arabic
- **F5-TTS Romanization pronunciation errors**: Since F5-TTS only supports English and Chinese natively, we Romanized Arabic text before inference. While this enables the model to speak Arabic characters with high similarity (**0.833**), the pronunciation is heavily distorted, resulting in a **97.96% WER**.
- **MMS-TTS is fast but sounds robotic**: MMS-TTS achieves an outstanding batch latency of **0.23s** and RTF of **0.030**, but its synthetic-sounding prosody and a **22.91% WER** make it less suitable for human-like conversational interfaces.
- **Bark latency and hallucination rates**: Bark's generative preset-based architecture results in extremely slow inference (**43.23s** batch latency, **4.433** RTF) on T4. Additionally, it suffers from typical GPT-style acoustic hallucinations (laughter, sighs, background noise).

### Hindi
- **F5-TTS Romanization pronunciation errors**: Similar to Arabic, F5-TTS Hindi Romanized generation results in high pronunciation errors (**67.95% WER**) despite good speaker similarity (**0.829**).
- **Indic-Parler-TTS latency**: Indic-Parler-TTS is a native Hindi model, but its large autoregressive architecture exhibits high batch latency (**23.08s**) and RTF (**0.974**), making it too slow for real-time customer service applications.
- **XTTS-v2 cross-lingual pronunciation**: XTTS-v2 achieves a **8.42% WER** on Hindi, but exhibits minor accents and incorrect vowel length inflections because Hindi is a secondary language in its pre-training mix.

---

## Models That Didn't Work

| Model | Language | What Happened | Could It Be Fixed? |
|-------|----------|---------------|-------------------|
| **CosyVoice2** | English | Dependency conflict: custom FunASR/Matcha-TTS binaries conflict with system PyTorch. | Yes — requires a dedicated Docker container. |
| **Fish Speech S2** | All | Multi-step custom runtime and checkpoint requirements failed to execute stably on standard T4 PyTorch environment. | Yes — requires deep package building and matching compilation wheels. |
| **CosyVoice 3.0** | All | Closed-weights or proprietary platform dependencies; no direct standalone repository. | No — must wait for complete open-source code/weights release. |

---

## Metrics That Miss Targets — Honest Assessment

| Metric | Target | English | Arabic | Hindi | Root Cause / Explanation |
|--------|--------|---------|--------|-------|--------------------------|
| **Latency (Batch)** | < 2.0s | ❌ 2.27s (XTTS) | ❌ 3.01s (XTTS) | ❌ 3.24s (XTTS) | Autoregressive decoding on T4 is slow. Resolved in production via **streaming (TTFA)**. |
| **WER (Intelligibility)** | <= 10% | ✅ 3.33% (XTTS) | ✅ 6.34% (XTTS) | ✅ 8.42% (XTTS) | English, Arabic, and Hindi all meet targets with Coqui XTTS-v2. Other models fail due to language mismatch or Romanization. |
| **Speaker Similarity** | >= 0.75 | ✅ 0.865 (XTTS) | ✅ 0.802 (XTTS) | ✅ 0.839 (XTTS) | Coqui XTTS-v2, F5-TTS, and Chatterbox pass. OpenVoice V2 and Bark fail to preserve speaker identity. |
| **RTF (Efficiency)** | <= 0.50 | ✅ 0.421 (XTTS) | ✅ 0.437 (XTTS) | ✅ 0.436 (XTTS) | XTTS-v2, F5-TTS, and Kokoro meet targets. Chatterbox (1.076) and Bark (4.54) fail. |

---

## What's Still Missing in Open-Source TTS

1. **Native multilingual zero-shot models** — XTTS-v2 is the only model we tested that natively supports English, Arabic, and Hindi zero-shot speaker cloning. Flow-matching architectures (F5-TTS) require explicit multilingual training to avoid Romanization hacks.
2. **Arabic dialect support** — Almost all open-source models support Modern Standard Arabic (MSA) only. Dialectal Arabic (e.g. Egyptian, Gulf) is not natively supported.
3. **High-efficiency zero-shot architectures** — Models like Kokoro (StyleTTS 2-based) have outstanding speed (<200ms latency) but do not support zero-shot voice cloning. Achieving both low latency (<200ms) and voice cloning requires custom streaming engines.

---

## Production Improvement Roadmap

### Short-term (1-2 weeks)
1. **Optimize streaming engine**: Fine-tune chunk sizes (e.g. 10 or 15 tokens) to push TTFA closer to 200 ms.
2. **Add text normalization**: Integrate standard text sanitization (e.g. converting "12.5" to "twelve point five") to clean up evaluation transcriptions.

### Medium-term (1-2 months)
3. **Fine-tune XTTS-v2**: Fine-tune XTTS-v2 on Arabic diacritized speech and Hindi phoneme datasets to reduce accent distortions.
4. **Deploy on L4 GPUs**: Migrating from Tesla T4 to NVIDIA L4 GPUs will yield a **2.5× to 3×** improvement in raw inference speed.

### Long-term (3-6 months)
5. **FunAudioLLM Integration**: Once containerized builds of CosyVoice2 are stable, integrate them as the primary multilingual voice server.
