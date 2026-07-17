"""
Kokoro TTS Pipeline — Runs Kokoro v1.0 (82M) on English and Hindi sentences.
Designed to run on T4 GPU or even CPU.
Note: Kokoro supports EN/HI/FR/ZH/JA/IT/ES/PT-BR. No Arabic support.
      Kokoro does NOT support voice cloning — uses preset voice styles only.
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_kokoro(sentences, language, voice_id, output_dir, collector):
    """Run Kokoro TTS on sentences for a given language."""
    print(f"\n{'=' * 60}")
    print(f"[AUDIO] {language.upper()} — Kokoro v1.0 (82M)")
    print("=" * 60)
    print_gpu_status()

    import kokoro

    # Initialize Kokoro pipeline
    pipeline = kokoro.KPipeline(lang_code=voice_id[0])  # 'a' for American English, 'h' for Hindi

    for i, text in enumerate(sentences):
        print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
        output_path = os.path.join(output_dir, f"kokoro_{i:02d}.wav")

        with Timer("generate") as t:
            # Generate audio using Kokoro
            generator = pipeline(text, voice=voice_id, speed=1.0)
            # Kokoro returns a generator of (graphemes, phonemes, audio) tuples
            audio_chunks = []
            for gs, ps, audio in generator:
                audio_chunks.append(audio)

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
            else:
                print(f"    [WARNING] No audio generated for sentence {i}")
                continue

        save_audio(full_audio, output_path, sample_rate=24000)
        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')

        collector.add_result(
            language=language, model_name="KOKORO", sentence_idx=i,
            text=text, latency_s=t.elapsed, rtf=rtf,
            audio_duration_s=duration, audio_path=output_path
        )
        print(f"    Latency: {t.elapsed:.3f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

    clear_gpu()


def main():
    with open("test_sentences.json") as f:
        sentences = json.load(f)

    collector = BenchmarkCollector(output_dir="results")

    # English — American English voice
    output_dir_en = "audio/english"
    os.makedirs(output_dir_en, exist_ok=True)
    run_kokoro(sentences["english"], "english", "af_heart", output_dir_en, collector)

    # Hindi — Hindi voice
    output_dir_hi = "audio/hindi"
    os.makedirs(output_dir_hi, exist_ok=True)
    run_kokoro(sentences["hindi"], "hindi", "hf_alpha", output_dir_hi, collector)

    collector.save_csv("kokoro_benchmark.csv")
    collector.save_json("kokoro_benchmark.json")
    print("\n[DONE] Kokoro v1.0 pipeline complete.")


if __name__ == "__main__":
    main()
