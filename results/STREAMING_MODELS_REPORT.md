# Comparative Evaluation: Real-Time Streaming TTS Models

This report presents a targeted comparative evaluation of open-source Text-to-Speech (TTS) models that natively support **real-time streaming (chunk-by-chunk generation)**. 

To meet the requirements of a low-latency conversational voice agent (<500ms TTFA), we isolated and compared the two primary streaming-native architectures evaluated in this pipeline: **Coqui XTTS-v2** and **Kokoro-82M**.

---

## 1. Consolidated Streaming Metrics Table

All benchmarks were run on an **NVIDIA Tesla T4 GPU (15 GB VRAM)** via Lightning AI Studio. Streaming Latency (TTFA) represents the actual empirical time measured from sending a text prompt to receiving the first playable audio chunk.

| Language | Model | Streaming Latency (TTFA) | Batch Latency (s) | Real-Time Factor (RTF) | Speaker Similarity | Normalized WER | UTMOS (Auto) | Human MOS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | **-** | **< 500 ms** | **< 2.0 s** | **<= 0.50** | **>= 0.75** | **<= 10.00%** | **>= 4.00** | **>= 4.00** |
| **English** | **XTTS-v2** | **460.1 ms** | 2.27s | **0.421** | **0.865** | **3.33%** | **4.14** | **4.70** |
| **English** | **Kokoro** | 680.9 ms | **0.30s** | **0.068** | 0.490 | **6.67%** | **4.51** | **4.00** |
| | | | | | | | | |
| **Hindi** | **XTTS-v2** | **445.6 ms** | 3.24s | **0.436** | **0.839** | **8.42%** | 2.85 | **4.40** |
| **Hindi** | **Kokoro** | 749.5 ms | **0.19s** | **0.031** | 0.551 | **7.78%** | **4.27** | **4.20** |
| | | | | | | | | |
| **Arabic** | **XTTS-v2** | **359.6 ms** | 3.01s | **0.437** | **0.802** | **6.34%** | 3.12 | **4.40** |
| **Arabic** | **Kokoro** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |

*Note: Kokoro-82M does not natively support Arabic in the evaluated v0.19 release.*

---

## 2. Key Insights & Architecture Comparison

### Coqui XTTS-v2 (Autoregressive GPT + HiFi-GAN)
* **Voice Cloning & Similarity**: Outstanding zero-shot voice cloning capability. It successfully clones reference voices cross-lingually (English reference cloned to Arabic and Hindi) with speaker similarity scores exceeding **0.80**.
* **Streaming Performance**: Meets the **< 500ms TTFA** target across all three languages due to its autoregressive token chunking mechanism.
* **Verdict**: **Recommended primary production model** where voice identity matching (cloning) is required.

### Kokoro-82M (Modified StyleTTS 2)
* **Speed & Latency**: Extremely lightweight (82M parameters). It yields the lowest Batch Latency (**0.19s - 0.30s**) and Real-Time Factor (**0.03 - 0.06**), making it highly CPU-friendly.
* **Streaming Behavior**: Because it is so fast, the entire generation finishes quickly, but its PyTorch first-chunk overhead on the GPU resulted in an empirical TTFA of **680ms - 750ms** under benchmark conditions.
* **Voice Cloning**: Does not support zero-shot voice cloning (uses fixed preset voices only), resulting in a low similarity score against the reference speaker.
* **Verdict**: **Recommended fallback/routing option** for English and Hindi if zero-shot voice cloning is not required and raw throughput/resource cost is the primary constraint.

---

## 3. Excluded Non-Streaming Architectures
For clarity, models like **Bark, Chatterbox, OpenVoice, F5-TTS, and MMS-TTS** have been excluded from this streaming-specific report. Since they operate entirely in batch mode, they must generate the entire sentence's waveform before outputting any audio, making their TTFA equal to their full batch latency (often > 2.0s, failing the real-time interaction target).
