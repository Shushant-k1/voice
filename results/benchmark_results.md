# Benchmark Results

## Consolidated Summary Table

All benchmarks run on **NVIDIA Tesla T4 (15 GB VRAM)** via Lightning AI Studio.
PyTorch 2.6.0+cu124 | Python 3.10.15 | CUDA 12.4

| Language | Model | Batch Latency (s) | Streaming Latency (TTFA) | RTF | Cosine Similarity | Normalized WER | UTMOS | Human MOS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | **-** | **< 2.0s** | **< 500ms** | **<= 0.50** | **>= 0.75** | **<= 10.00%** | **>= 4.00** | **>= 4.00** |
| **Arabic** | Bark | 43.23 | 43.23s (Batch) | 4.433 | 0.557 | 117.42% | 2.96 | 3.50 |
| **Arabic** | F5-TTS | 3.17 | 5209 ms | **0.418** | **0.833** | 97.96% | **4.17** | N/A |
| **Arabic** | MMS-TTS | **0.23** | 852.1 ms | **0.030** | N/A | 22.91% | 3.34 | N/A |
| **Arabic** | **XTTS-v2** (Winner) | 3.01 | **359.6 ms** | **0.437** | **0.802** | **6.34%** | 3.12 | **4.40** |
| | | | | | | | | |
| **English** | Bark | 27.34 | 27.34s (Batch) | 4.625 | 0.557 | **4.17%** | 3.72 | 3.70 |
| **English** | Chatterbox | 5.52 | 5.52s (Batch) | 1.076 | **0.864** | **3.33%** | **4.43** | N/A |
| **English** | F5-TTS | 2.95 | 3721 ms | **0.450** | **0.848** | **3.33%** | **4.30** | N/A |
| **English** | Kokoro | **0.30** | 680.9 ms | **0.068** | 0.490 | **6.67%** | **4.51** | **4.00** |
| **English** | OpenVoice | **1.83** | 1.83s (Batch) | **0.386** | 0.726 | **4.17%** | **4.02** | **4.00** |
| **English** | **XTTS-v2** (Winner) | 2.27 | **460.1 ms** | **0.421** | **0.865** | **3.33%** | **4.14** | **4.70** |
| | | | | | | | | |
| **Hindi** | Bark | 30.22 | 30.22s (Batch) | 4.541 | 0.500 | 22.44% | 2.80 | 3.40 |
| **Hindi** | F5-TTS | 2.86 | 3362 ms | **0.428** | **0.829** | 67.95% | **4.25** | N/A |
| **Hindi** | Indic-Parler-TTS | 23.08 | 23.08s (Batch) | 0.974 | N/A | 115.41% | 1.30 | N/A |
| **Hindi** | Kokoro | **0.19** | 749.5 ms | **0.031** | 0.551 | **7.78%** | **4.27** | **4.20** |
| **Hindi** | MMS-TTS | **0.17** | 905.1 ms | **0.031** | N/A | 21.95% | 3.61 | N/A |
| **Hindi** | **XTTS-v2** (Winner) | 3.24 | **445.6 ms** | **0.436** | **0.839** | **8.42%** | 2.85 | **4.40** |

## Real-Time Streaming Latency (TTFA) Results

| Language | Model | Streaming TTFA (ms) | Target | Status |
| :--- | :--- | :---: | :---: | :---: |
| **English** | **XTTS-v2** (Streaming) | **460.1 ms** | < 500 ms | **Pass** |
| **Arabic** | **XTTS-v2** (Streaming) | **359.6 ms** | < 500 ms | **Pass** |
| **Hindi** | **XTTS-v2** (Streaming) | **445.6 ms** | < 500 ms | **Pass** |

## Winner Selection

| Language | Winner | Runner-up | Reasoning |
|----------|--------|-----------|-----------|
| **English** | **XTTS-v2** | F5-TTS | Best combination of zero-shot voice cloning similarity (**0.865**), low WER (**3.33%**), and low streaming latency (**460.1 ms**). |
| **Arabic** | **XTTS-v2** | F5-TTS | Highest cloning similarity (**0.833**) among secondary candidates, though it requires Romanization for pronunciation. |
| **Hindi** | **XTTS-v2** | F5-TTS | Good cloning similarity (**0.829**) but exhibits zero-shot pronunciation issues without fine-tuning. |

---

## Hardware & Environment

- **GPU**: NVIDIA Tesla T4 (15 GB VRAM) — Lightning AI
- **CUDA**: 12.4
- **PyTorch**: 2.6.0
- **Python**: 3.10.15
- **Models loaded sequentially** to fit within 15 GB VRAM.
- **GPU memory cleared** between model loads via `torch.cuda.empty_cache()`.
