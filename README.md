# Multilingual TTS Benchmarking Harness

This repository contains the benchmarking harness, evaluation scripts, and empirical results for English, Arabic, and Hindi text-to-speech pipelines using open-source engines (XTTS-v2, F5-TTS, MMS-TTS, Indic-Parler-TTS, Kokoro, Chatterbox, OpenVoice).

A complete landscape survey and deep-dive analysis are available in the **[Detailed Technical Report](results/DETAILED_REPORT.md)**.

---

## Results Summary

All benchmarks were run on an **NVIDIA Tesla T4 GPU (15 GB VRAM)**.
*Runtime Environment*: PyTorch 2.6.0 | CUDA 12.4 | Python 3.10.15

### Comparative Model Evaluation Results

All benchmarks were run on an **NVIDIA Tesla T4 GPU (15 GB VRAM)**. 
Values meeting target thresholds are highlighted in **bold**.

| Language | Model | Architecture Type | Batch Latency (s) | Streaming TTFA | RTF | Cosine Similarity | Normalized WER | UTMOS | Human MOS |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | **-** | **-** | **< 2.0s** | **< 500 ms** | **<= 0.50** | **>= 0.75** | **<= 10.00%** | **>= 4.00** | **>= 4.00** |
| **English** | **XTTS-v2 (Winner)**| AR Zero-Shot Clone | 2.27 | **460.1 ms** | **0.421** | **0.865** | **3.33%** | **4.14** | **4.70** |
| | **F5-TTS** | Non-AR Flow-Matching | 2.95 | 3.72s (Batch) | **0.381** | **0.842** | **4.05%** | **4.30** | **4.50** |
| | **Chatterbox** | Conversational AR | 5.52 | 5.52s (Batch) | 1.076 | **0.864** | **3.33%** | **4.43** | N/A |
| | **Kokoro** | StyleTTS 2 / Presets | **0.30** | 680.9 ms | **0.068** | 0.490 | **6.67%** | **4.51** | **4.00** |
| | **OpenVoice V2** | VITS + Tone Converter | **1.83** | 1.83s (Batch) | **0.386** | 0.726 | **4.17%** | **4.02** | **4.00** |
| | **Bark** | GPT Presets | 27.34 | 27.34s (Batch)| 4.625 | 0.557 | **4.17%** | 3.72 | 3.70 |
| | | | | | | | | | |
| **Arabic** | **XTTS-v2 (Winner)**| AR Zero-Shot Clone | 3.01 | **359.6 ms** | **0.437** | **0.802** | **6.34%** | 3.12 | **4.40** |
| | **F5-TTS** | Non-AR Flow-Matching | 3.17 | 5.21s (Batch) | **0.418** | **0.833** | 97.96% | **4.17** | N/A |
| | **MMS-TTS** | VITS Baseline | **0.23** | 852.1 ms | **0.030** | N/A | 22.91% | 3.34 | N/A |
| | **Bark** | GPT Presets | 43.23 | 43.23s (Batch)| 4.433 | 0.557 | 117.42% | 2.96 | 3.50 |
| | | | | | | | | | |
| **Hindi** | **XTTS-v2 (Winner)**| AR Zero-Shot Clone | 3.24 | **445.6 ms** | **0.436** | **0.839** | **8.42%** | 2.85 | **4.40** |
| | **F5-TTS** | Non-AR Flow-Matching | 2.86 | 3.36s (Batch) | **0.428** | **0.829** | 67.95% | **4.25** | N/A |
| | **Kokoro** | StyleTTS 2 / Presets | **0.19** | 749.5 ms | **0.031** | 0.551 | **7.78%** | **4.27** | **4.20** |
| | **MMS-TTS** | VITS Baseline | **0.17** | 905.1 ms | **0.031** | N/A | 21.95% | 3.61 | N/A |
| | **Indic-Parler** | Indic AR | 23.08 | 23.08s (Batch)| 0.974 | N/A | 115.41% | 1.30 | N/A |
| | **Bark** | GPT Presets | 30.22 | 30.22s (Batch)| 4.541 | 0.500 | 22.44% | 2.80 | 3.40 |

*Note on MOS: UTMOS is an automated neural MOS model trained on English. While UTMOS scores for Arabic and Hindi drop below 4.00 due to English-only training bias, human listening checks confirm that XTTS-v2 sounds natural (>= 4.40) in both languages.*

---

## Technical Approach & Winner Rationale

We evaluated three distinct architectural approaches for each language:
1. **Autoregressive Voice Cloning (GPT + Vocoder)**: XTTS-v2, Bark, Chatterbox.
2. **Non-Autoregressive Flow-Matching / Diffusion**: F5-TTS.
3. **Feed-Forward / VITS or Preset Models**: Kokoro, MMS-TTS, OpenVoice.

### Winner: Coqui XTTS-v2
* **Why it won**: It is the only open-source model that successfully performs cross-lingual voice cloning (cloning the English speaker prompt into Arabic and Hindi) while keeping pronunciation accurate (WER < 10%). 
* **Latency Mitigation**: While batch generation is slow (often > 2.5s), its streaming implementation reduces Time-to-First-Audio (TTFA) to **359 ms - 460 ms**, satisfying real-time application constraints.
* **Fallback Options**: If zero-shot cloning is not required, **Kokoro** (English and Hindi) and **MMS-TTS** (Arabic) are much faster alternatives (RTF < 0.05).

---

## Setup & Execution

### Installation
```bash
git clone https://github.com/Shushant-k1/voice.git
cd voice
pip install -r requirements.txt
```

### Running the Pipelines
Generates speech waveforms and outputs basic timing metrics:
```bash
python src/pipeline_english.py
python src/pipeline_arabic.py
python src/pipeline_hindi.py
```

### Running Streaming Benchmark
Measures the Time-to-First-Audio (TTFA) for XTTS-v2 streaming:
```bash
python src/benchmark_streaming.py
```

### Running Evaluation Scripts
Calculates WER (via Whisper large-v3) and speaker similarity (via Resemblyzer d-vector distance):
```bash
python src/evaluate.py
python src/mos_evaluation.py
```

### Compiling Results
```bash
python src/compile_results.py
```

---

## Repository Structure

* `src/`: Core Python pipeline and evaluation scripts.
  * `pipeline_english.py`, `pipeline_arabic.py`, `pipeline_hindi.py`: Main batch pipeline entries (each runs 3+ distinct models).
  * `benchmark_streaming.py`: XTTS-v2 streaming benchmark.
  * `evaluate.py`: ASR/WER and Similarity evaluation.
  * `utils.py`: PyTorch 2.6 patching and GPU cleanup helpers.
* `results/`: Consolidated CSV and Markdown tables.
* `audio/`: Reference voice files and all generated samples.
