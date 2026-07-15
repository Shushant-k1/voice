"""
Bark TTS Pipeline — Runs Suno Bark on English, Arabic, and Hindi sentences.
Designed to run on T4 GPU (15 GB VRAM).
"""
import os
import sys
import json
import torch
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_bark(sentences, language, lang_code, output_dir, collector):
    """Run Bark TTS on sentences for a given language."""
    print(f"\n{'=' * 60}")
    print(f"[AUDIO] {language.upper()} — Bark (Suno)")
    print("=" * 60)
    print_gpu_status()

    from bark import SAMPLE_RATE, generate_audio, preload_models

    # Preload models to GPU
    preload_models()

    # Bark speaker prompts per language
    speaker_map = {
        "english": "v2/en_speaker_6",
        "arabic": "v2/ar_speaker_0",
        "hindi": "v2/hi_speaker_0",
    }
    speaker = speaker_map.get(language, "v2/en_speaker_6")

    for i, text in enumerate(sentences):
        print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
        output_path = os.path.join(output_dir, f"bark_{i:02d}.wav")

        with Timer("generate") as t:
            audio_array = generate_audio(text, history_prompt=speaker)

        save_audio(audio_array, output_path, sample_rate=SAMPLE_RATE)
        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')

        collector.add_result(
            language=language, model_name="BARK", sentence_idx=i,
            text=text, latency_s=t.elapsed, rtf=rtf,
            audio_duration_s=duration, audio_path=output_path
        )
        print(f"    Latency: {t.elapsed:.3f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

    clear_gpu()


def main():
    with open("test_sentences.json") as f:
        sentences = json.load(f)

    collector = BenchmarkCollector(output_dir="results")

    for lang in ["english", "arabic", "hindi"]:
        lang_codes = {"english": "en", "arabic": "ar", "hindi": "hi"}
        output_dir = f"audio/{lang}"
        os.makedirs(output_dir, exist_ok=True)
        run_bark(sentences[lang], lang, lang_codes[lang], output_dir, collector)

    collector.save_csv("bark_benchmark.csv")
    collector.save_json("bark_benchmark.json")
    print("\n[DONE] Bark pipeline complete.")


if __name__ == "__main__":
    main()
