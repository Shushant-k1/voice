import os
import sys
import json
import torch
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector

def run_f5_english(test_sentences_path="test_sentences.json",
                   reference_wav="audio/reference/reference_voice.wav",
                   output_base="audio/english",
                   results_dir="results"):
    print("\n" + "=" * 60)
    print("[AUDIO] ENGLISH — F5-TTS (Zero-Shot Flow Matching)")
    print("=" * 60)
    print_gpu_status()

    # Load sentences
    with open(test_sentences_path, 'r', encoding='utf-8') as f:
        all_sentences = json.load(f)
    sentences = all_sentences["english"]

    os.makedirs(output_base, exist_ok=True)
    collector = BenchmarkCollector(results_dir)

    try:
        from f5_tts.api import F5TTS
        model = F5TTS()

        for i, text in enumerate(sentences):
            print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
            output_path = os.path.join(output_base, f"f5_tts_{i:02d}.wav")

            with Timer("generate") as t:
                audio, sr, _ = model.infer(
                    ref_file=reference_wav,
                    ref_text="Some call me nature. Others call me Mother Nature.",
                    gen_text=text,
                )

            save_audio(audio, output_path, sample_rate=sr)
            duration = get_audio_duration(output_path)
            rtf = t.elapsed / duration if duration > 0 else float('inf')

            collector.add_result(
                language="english", model_name="F5-TTS",
                sentence_idx=i, text=text,
                latency_s=t.elapsed, rtf=rtf,
                audio_duration_s=duration, audio_path=output_path
            )
            print(f"  [SUCCESS] Latency: {t.elapsed:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

        del model
        clear_gpu()

        collector.save_csv("english_f5_benchmark.csv")
        collector.print_summary()

    except Exception as e:
        print(f"  [ERROR] F5-TTS failed: {e}")

if __name__ == "__main__":
    run_f5_english()
