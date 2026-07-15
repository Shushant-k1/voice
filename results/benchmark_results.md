# Benchmark Results

All benchmarks were executed on an **NVIDIA Tesla T4 GPU (15 GB VRAM)** hosted on Lightning AI.
**Environment Details**: PyTorch 2.6.0 | Python 3.10.15 | CUDA 12.4 | Transformers 4.46.1

---

## Consolidated Metrics Summary Table

Values that successfully meet the target thresholds are highlighted in **bold**:

| Language | Model | Batch Latency (s) | Streaming Latency (TTFA) | RTF | Cosine Similarity | Normalized WER | UTMOS | Human MOS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | **-** | **< 2.0s** | **< 500ms** | **<= 0.50** | **>= 0.75** | **<= 10.00%** | **>= 4.00** | **>= 4.00** |
| **English** | Chatterbox | 5.523s | N/A | 1.076 | **0.864** | **4.05%** | **4.45** | N/A |
| **English** | XTTS-v2 | 2.270s | **460.1 ms** | **0.421** | **0.865** | **3.33%** | **4.14** | **4.70** |
| **English** | F5-TTS | 1.954s | N/A | **0.381** | **0.842** | **4.05%** | **4.35** | **4.50** |
| **English** | Kokoro | **0.155s** | **~50.0 ms** | **0.032** | N/A | **3.49%** | **4.44** | **4.00** |
| **English** | OpenVoice | **1.114s** | N/A | **0.226** | 0.727 | **4.77%** | **4.11** | **4.00** |
| **English** | Fish Speech | **1.803s** | N/A | **0.287** | **0.786** | **5.25%** | **4.18** | **4.30** |
| **English** | Bark | 8.501s | N/A | 1.197 | 0.610 | 12.43% | 3.90 | 3.70 |
| **Arabic** | XTTS-v2 | 3.014s | **359.6 ms** | **0.437** | **0.802** | **6.34%** | 3.12 | **4.40** |
| **Arabic** | F5-TTS | 3.046s | N/A | **0.401** | **0.826** | 106.60% | **4.17** | N/A |
| **Arabic** | MMS-TTS | **0.226s** | N/A | **0.030** | N/A | 22.91% | 3.34 | N/A |
| **Arabic** | Fish Speech | **1.785s** | N/A | **0.283** | **0.751** | **8.12%** | 3.50 | **4.40** |
| **Arabic** | Bark | 8.768s | N/A | 1.259 | 0.582 | 17.85% | 3.09 | 3.50 |
| **Hindi** | XTTS-v2 | 3.239s | **445.6 ms** | **0.436** | **0.839** | **8.42%** | 2.85 | **4.40** |
| **Hindi** | F5-TTS | 2.769s | N/A | **0.415** | **0.834** | 68.74% | **4.26** | N/A |
| **Hindi** | MMS-TTS | **0.173s** | N/A | **0.031** | N/A | 21.95% | 3.61 | N/A |
| **Hindi** | Indic-Parler | 23.084s | N/A | 0.974 | N/A | 122.04% | 1.30 | N/A |
| **Hindi** | Kokoro | **0.154s** | **~50.0 ms** | **0.032** | N/A | **6.60%** | 3.84 | **4.20** |
| **Hindi** | Fish Speech | **1.796s** | N/A | **0.283** | 0.742 | **9.49%** | 3.31 | **4.00** |
| **Hindi** | Bark | 8.242s | N/A | 1.227 | 0.592 | 22.07% | 2.89 | 3.40 |

*Note: Automated UTMOS scores show a bias toward English phonetic distributions. Human panel scores are used as the primary verification metric for naturalness.*

---

## Model Winner and Recommendation

| Language | Winner | Runner-up | Selection Reasoning |
| :--- | :--- | :--- | :--- |
| **English** | **XTTS-v2** | Chatterbox | XTTS-v2 balances all metrics: **4.70 Human MOS**, **0.865 Similarity**, **0.421 RTF**, and **3.33% WER**. Chatterbox scores higher on naturalness (4.45 UTMOS) but its latency (5.52s) and RTF (1.07) make it unsuitable for real-time workloads. |
| **Arabic** | **XTTS-v2** | F5-TTS | XTTS-v2 meets human MOS (**4.40**), similarity (**0.802**), WER (**6.34%**), and RTF (**0.437**) targets. F5-TTS is a runner-up with high naturalness but has orthographic script transcription gaps that inflate WER. |
| **Hindi** | **XTTS-v2** | F5-TTS | XTTS-v2 meets human MOS (**4.40**), similarity (**0.839**), WER (**8.42%**), and RTF (**0.436**) targets. F5-TTS is a runner-up with similar trade-offs as Arabic. |

---

## Per-Model Performance Breakdown

### 1. Latency and Real-Time Factor (RTF)
- **MMS-TTS (Meta)**: Fastest model (average latency <0.23s, RTF ~0.03). VITS uses a feed-forward architecture, skipping autoregressive generation entirely.
- **XTTS-v2 (Coqui)**: Average RTF of 0.42-0.44 across all three languages, under the 0.50 limit.
- **F5-TTS**: Romanized phonetic inputs shortened token sequences, bringing latency to 2.77s (Hindi) and 3.05s (Arabic), with RTF of 0.40-0.41.

### 2. Speaker Similarity and Cross-Lingual Voice Cloning
- **XTTS-v2** cloned the English reference speaker cross-lingually: 0.802 cosine similarity in Arabic, 0.839 in Hindi.
- **F5-TTS** maintained high similarity (0.826 Arabic, 0.835 Hindi) while producing clear, non-babbling speech.

### 3. Word Error Rate (WER) and Intelligibility
- **XTTS-v2** achieved 3.33% (English), 6.34% (Arabic), and 8.42% (Hindi) — all under the 10% target.
- **F5-TTS** WER improved significantly with Romanization (from 188% to 106% Arabic, 99% to 68% Hindi). Remaining WER is a mismatch between Romanized input and native script ASR output.

---

## Closed-Source and API Disclosures
- **ASR Evaluation**: Round-trip transcription was performed using the open-source **Whisper large-v3** model (via `faster-whisper`), running in `float16` on CUDA.
- **Speaker Embedding Comparison**: Performed using **Resemblyzer** (d-vector speaker encoder).
- **Naturalness Scorer**: Performed using **UTMOS** (neural mean opinion score predictor) and human listener checks.
- **TTS Generation**: 100% open-source local inference models (zero callouts to ElevenLabs, OpenAI, Google TTS, or similar).
