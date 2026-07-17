# Multilingual TTS Benchmarks: Comparative Evaluation Report

This report documents our technical evaluation of open-source Text-to-Speech (TTS) models to build a voice agent platform for English, Arabic, and Hindi on an NVIDIA Tesla T4 GPU.

---

## 1. Model Landscape & Shortlist

We surveyed 21 open-source TTS models/families on Hugging Face, GitHub, and community benchmarks.

| Model | Architecture | Params | Languages | Voice Cloning | License | Shortlist? | Reason for Exclude / Shortlist |
|---|---|---|---|---|---|---|---|
| **XTTS-v2** | AR GPT + HiFi-GAN | ~450M | 17 (incl. AR, HI) | Zero-shot (3s ref) | MPL-2.0 | Yes | Native multilingual support & zero-shot cloning. |
| **F5-TTS** | Flow-Matching Transformer | ~330M | EN/ZH primary | Zero-shot | CC-BY-NC-4.0 | Yes | Fast inference. Tested with Romanization workaround for AR/HI. |
| **Chatterbox** | Llama-based AR | ~500M | EN primary | Zero-shot | MIT | Yes | High quality in English. Used as AR baseline. |
| **MMS-TTS** | VITS | ~250M | 1,100+ | None | CC-BY-NC-4.0 | Yes | Fast, lightweight VITS baseline for Arabic/Hindi. |
| **Indic-Parler** | Transformer AR | 938M | 22 Indic langs | Description-based | Gated | Yes | Native Indic model baseline. |
| **Kokoro** | StyleTTS 2 based | 82M | 8 (incl. EN, HI) | Presets only | Apache 2.0 | Yes | Ultra-fast feed-forward preset baseline. |
| **OpenVoice V2** | VITS + Tone Converter | ~300M | EN/ZH/FR/ES | Tone transfer | MIT | Yes | Style transfer baseline for English. |
| **Bark** | GPT Semantic/Acoustic | ~1B | Multilingual | None (generative) | MIT | Yes | Generative baseline, prone to hallucination. |
| **CosyVoice2** | Causal Flow-Matching | 500M | ZH/EN/JA/KO | Zero-shot | Apache 2.0 | No | Custom FunASR/Matcha dependency conflict with PyTorch 2.6. |
| **Fish Speech S2**| Dual-AR Codec | ~500M | 13+ (incl. AR) | Zero-shot | Apache 2.0 | No | Multi-step pipeline failed to run stably on T4 VRAM. |
| **IndexTTS-2** | GPT Latent + Duration | — | ZH/EN | Zero-shot | Apache 2.0 | No | No Arabic/Hindi support. |
| **OmniVoice** | Diffusion LM | — | 600+ | Zero-shot | Apache 2.0 | No | Requires manual Kaldi/k2 compilation on standard T4. |
| **SILMA TTS** | F5-TTS fine-tune | 150M | AR/EN | Zero-shot | Apache 2.0 | No | Not directly installable via pip. |
| **IndicF5** | F5-TTS fine-tune | ~330M | 11 Indic | Reference-based | Research | No | Released after our benchmark window. |
| **Orpheus-TTS** | Llama AR | — | EN/AR/HI | Zero-shot | Apache 2.0 | No | Quantized checkpoints for T4 are not public. |
| **Piper** | VITS | ~15M | 30+ | None | MIT | No | Designed for CPU edge; no cloning. |

---

## 2. Three Architectural Approaches Evaluated

For each target language, we evaluated three distinct architectural approaches to assess the trade-off between cloning capability, latency, and quality:

1. **Approach 1: Autoregressive Zero-Shot Voice Cloning (GPT + Vocoder)**
   * *Models*: Coqui XTTS-v2, Suno Bark, Chatterbox (English), Indic-Parler-TTS (Hindi).
   * *Trade-off*: Excellent speaker similarity and naturalness, but slow due to sequential token generation.
2. **Approach 2: Non-Autoregressive Flow-Matching / Diffusion**
   * *Model*: F5-TTS.
   * *Trade-off*: Fast, stable alignments, but restricted vocabulary requires Romanization scripts for Arabic/Hindi.
3. **Approach 3: Feed-Forward / VITS or Preset Models**
   * *Models*: Kokoro (English/Hindi), Meta MMS-TTS (Arabic/Hindi), OpenVoice V2 (English).
   * *Trade-off*: Blazing-fast inference (RTF < 0.05) and low VRAM footprint, but no zero-shot cloning (fixed preset voice models).

---

## 3. Benchmark Results

Controlled benchmarks were run using 10 customer-service sentences per language. 
* **Speaker Similarity**: Cosine distance of Resemblyzer d-vector embeddings against the reference English WAV.
* **Intelligibility (WER)**: Evaluated using Whisper large-v3 back-transcription.
* **Naturalness**: Rated via UTMOS (English bias) and a Human Panel (self-rated, N=1).

### 3.1 English Benchmarks

| Approach | Model | Streaming TTFA | Batch Latency | RTF | Cosine Similarity | Normalized WER | Human MOS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | **-** | **< 500 ms** | **< 2.0 s** | **<= 0.50** | **>= 0.75** | **<= 10.00%** | **>= 4.00** |
| **App 1 (AR Clone)** | **XTTS-v2 (Winner)** | **460.1 ms** | 2.27s | **0.421** | **0.865** | **3.33%** | **4.70** |
| **App 1 (AR Clone)** | Chatterbox | 5.52s (Batch) | 5.52s | 1.076 | **0.864** | **4.05%** | N/A |
| **App 1 (AR Presets)**| Bark | 8.50s (Batch) | 8.50s | 1.540 | 0.557 | **4.17%** | 3.70 |
| **App 2 (Flow-Match)**| F5-TTS | 3.72s | 1.95s | **0.381** | **0.842** | **4.05%** | 4.50 |
| **App 3 (Feedforward)**| Kokoro | 680.9 ms | **0.30s** | **0.068** | 0.490 | **6.67%** | **4.00** |
| **App 3 (Tone Trans)**| OpenVoice V2 | 1.83s (Batch) | **1.83s** | **0.386** | 0.726 | **4.17%** | **4.00** |

* **Winner**: **XTTS-v2** is the best choice. It meets the streaming latency target (460.1ms TTFA), speaker similarity (0.865), and intelligibility (3.33% WER).
* **Alternative**: **Kokoro** is an excellent alternative if voice cloning is not required, generating audio in under 300ms.

### 3.2 Arabic Benchmarks

| Approach | Model | Streaming TTFA | Batch Latency | RTF | Cosine Similarity | Normalized WER | Human MOS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | **-** | **< 500 ms** | **< 2.0 s** | **<= 0.50** | **>= 0.75** | **<= 10.00%** | **>= 4.00** |
| **App 1 (AR Clone)** | **XTTS-v2 (Winner)** | **359.6 ms** | 3.01s | **0.437** | **0.802** | **6.34%** | **4.40** |
| **App 1 (AR Presets)**| Bark | 43.23s (Batch)| 43.23s | 4.433 | 0.557 | 117.42% | 3.50 |
| **App 2 (Flow-Match)**| F5-TTS | 5.21s | 3.05s | **0.401** | **0.826** | 97.96% (Phonetic) | N/A |
| **App 3 (VITS)** | MMS-TTS | 852.1 ms | **0.23s** | **0.030** | N/A | 22.91% | N/A |

* **Winner**: **XTTS-v2** is the only candidate that handles Arabic cloning (0.802 similarity) while outputting correct pronunciation (6.34% WER).
* **Failure Mode**: F5-TTS achieves high similarity (0.826) but fails pronunciation (97.96% WER) because its vocabulary lacks Arabic script tokens, requiring phonetic Romanization that results in heavy accents and incorrect phrasing.

### 3.3 Hindi Benchmarks

| Approach | Model | Streaming TTFA | Batch Latency | RTF | Cosine Similarity | Normalized WER | Human MOS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | **-** | **< 500 ms** | **< 2.0 s** | **<= 0.50** | **>= 0.75** | **<= 10.00%** | **>= 4.00** |
| **App 1 (AR Clone)** | **XTTS-v2 (Winner)** | **445.6 ms** | 3.24s | **0.436** | **0.839** | **8.42%** | **4.40** |
| **App 1 (AR Presets)**| Bark | 30.22s (Batch)| 30.22s | 4.541 | 0.500 | 22.44% | 3.40 |
| **App 1 (AR Presets)**| Indic-Parler-TTS| 23.08s (Batch)| 23.08s | 0.974 | N/A | 115.41% | N/A |
| **App 2 (Flow-Match)**| F5-TTS | 3.36s | 2.77s | **0.415** | **0.834** | 67.95% (Phonetic) | N/A |
| **App 3 (Feedforward)**| Kokoro | 749.5 ms | **0.19s** | **0.031** | 0.551 | **7.78%** | **4.20** |
| **App 3 (VITS)** | MMS-TTS | 905.1 ms | **0.17s** | **0.031** | N/A | 21.95% | N/A |

* **Winner**: **XTTS-v2** meets the cloning (0.839 similarity) and WER targets (8.42%).
* **Alternative**: **Kokoro** provides excellent naturalness (4.20 Human MOS) and speed (0.19s batch) if preset voices are acceptable.

---

## 4. Technical Failure Analysis & Mitigations

During development, we solved several implementation blockers:

### 1. PyTorch weights_only Loading Crash (XTTS-v2)
* *Problem*: PyTorch 2.6.0 defaults to `weights_only=True` for security. This causes XTTS-v2 configuration loaders to crash when loading checkpoints because their custom classes are not in PyTorch's safe allowlist.
* *Fix*: Implemented a runtime monkeypatch in `src/utils.py` that intercepts `torch.load` calls during evaluation to force `weights_only=False` for Coqui model files.

### 2. Grapheme-to-Phoneme Mismatches (F5-TTS)
* *Problem*: F5-TTS is pre-trained primarily on English and Chinese. Directly feeding raw Arabic script or Hindi Devanagari text causes the alignment model to fail, resulting in babbling or silence.
* *Fix*: Applied automated Romanization (Latin transliteration) as a pre-processing step. While this generates phonetic approximations, it causes a high pronunciation error rate (97.96% WER for Arabic), proving that F5-TTS requires native language fine-tuning to be usable in production.

### 3. High Autoregressive Latency
* *Problem*: Autoregressive generation takes 2.2s to 3.2s to write a full sentence WAV file, violating the 2.0s real-time target.
* *Fix*: Implemented a streaming engine (`src/benchmark_streaming.py`) that yields chunked audio frames every **20 tokens**, achieving a **359 ms - 460 ms** Time-to-First-Audio (TTFA).

---

## 5. Fine-Tuning Guide

To train or fine-tune these architectures on proprietary dataset voice clips:

### 1. Coqui XTTS-v2
Format your voice data in the LJSpeech format (mono 22,050Hz WAV files + `metadata.csv` mapping paths to text transcripts).
```bash
pip install TTS
# Train speaker adapter layers while freezing GPT weights
python TTS/bin/train_tts.py --config_path recipes/ljspeech/xtts_v2/config.json
```

### 2. F5-TTS
Prepare a CSV manifest with columns mapping audio file paths to transcripts:
```bash
git clone https://github.com/lucidrains/f5-tts.git && cd f5-tts
# Launch flow-matching training
python train.py --dataset_name "custom_dataset" --exp_name "f5_run" --learning_rate 1e-4
```

---

## 6. Reproduction Checklist

Run the following commands on a Tesla T4 GPU node to verify all benchmarks:

```bash
# 1. Run inference pipelines (generates audio clips in audio/)
python src/pipeline_english.py
python src/pipeline_arabic.py
python src/pipeline_hindi.py

# 2. Run streaming TTFA test
python src/benchmark_streaming.py

# 3. Calculate metrics
python src/evaluate.py
python src/mos_evaluation.py

# 4. Generate final results summary
python src/compile_results.py
```
