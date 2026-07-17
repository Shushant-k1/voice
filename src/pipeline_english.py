"""
English TTS Pipeline — Compares Chatterbox, XTTS-v2, and CosyVoice2.
Designed to run on Google Colab with T4 GPU.
"""
import os
import sys
import json
import torch
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_chatterbox(sentences, reference_wav, output_dir, collector):
    """Run Chatterbox TTS on English sentences."""
    print("\n" + "=" * 60)
    print("[AUDIO] ENGLISH — Model 1: Chatterbox (Resemble AI)")
    print("=" * 60)
    print_gpu_status()

    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device="cuda")

    for i, text in enumerate(sentences):
        print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
        output_path = os.path.join(output_dir, f"chatterbox_{i:02d}.wav")

        with Timer("generate") as t:
            wav = model.generate(text, audio_prompt_path=reference_wav)

        # Save audio
        if hasattr(wav, 'squeeze'):
            wav_np = wav.squeeze().cpu().numpy() if hasattr(wav, 'cpu') else wav.squeeze().numpy()
        else:
            wav_np = np.array(wav)
        save_audio(wav_np, output_path, sample_rate=24000)

        # Compute metrics
        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')

        collector.add_result(
            language="english", model_name="Chatterbox",
            sentence_idx=i, text=text,
            latency_s=t.elapsed, rtf=rtf,
            audio_duration_s=duration, audio_path=output_path
        )
        print(f"  [SUCCESS] Latency: {t.elapsed:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

    # Cleanup
    del model
    clear_gpu()


def run_xtts_english(sentences, reference_wav, output_dir, collector):
    """Run XTTS-v2 on English sentences."""
    print("\n" + "=" * 60)
    print("[AUDIO] ENGLISH — Model 2: XTTS-v2 (Coqui)")
    print("=" * 60)
    print_gpu_status()

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    for i, text in enumerate(sentences):
        print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
        output_path = os.path.join(output_dir, f"xtts_v2_{i:02d}.wav")

        with Timer("generate") as t:
            tts.tts_to_file(
                text=text,
                speaker_wav=reference_wav,
                language="en",
                file_path=output_path
            )

        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')

        collector.add_result(
            language="english", model_name="XTTS-v2",
            sentence_idx=i, text=text,
            latency_s=t.elapsed, rtf=rtf,
            audio_duration_s=duration, audio_path=output_path
        )
        print(f"  [SUCCESS] Latency: {t.elapsed:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

    del tts
    clear_gpu()


def run_f5_english(sentences, reference_wav, output_dir, collector):
    """Run F5-TTS on English sentences."""
    print("\n" + "=" * 60)
    print("[AUDIO] ENGLISH — Model 3: F5-TTS (Zero-Shot Flow Matching)")
    print("=" * 60)
    print_gpu_status()

    try:
        from f5_tts.api import F5TTS
        model = F5TTS()

        for i, text in enumerate(sentences):
            print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
            output_path = os.path.join(output_dir, f"f5_tts_{i:02d}.wav")

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

    except ImportError:
        print("  [WARNING] F5-TTS not installed. Skipping.")
    except Exception as e:
        print(f"  [WARNING] F5-TTS failed: {e}")


def run_english_pipeline(test_sentences_path="test_sentences.json",
                         reference_wav="audio/reference/reference_voice.wav",
                         output_base="audio/english",
                         results_dir="results"):
    """Run all English TTS models and collect benchmarks."""
    print("\n" + "* " * 20)
    print("  ENGLISH PIPELINE — Comparing 3 Models")
    print("* " * 20)

    # Load sentences
    with open(test_sentences_path, 'r', encoding='utf-8') as f:
        all_sentences = json.load(f)
    sentences = all_sentences["english"]

    os.makedirs(output_base, exist_ok=True)
    collector = BenchmarkCollector(results_dir)

    # Run each model
    run_chatterbox(sentences, reference_wav, output_base, collector)
    run_xtts_english(sentences, reference_wav, output_base, collector)
    run_f5_english(sentences, reference_wav, output_base, collector)

    # Save results
    collector.save_csv("english_benchmark.csv")
    collector.save_json("english_benchmark.json")
    collector.print_summary()

    return collector


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="English TTS Pipeline")
    parser.add_argument("--text", type=str, help="Single text to synthesize")
    parser.add_argument("--reference", type=str, default="audio/reference/reference_voice.wav")
    parser.add_argument("--output", type=str, default="audio/english")
    args = parser.parse_args()

    if args.text:
        # Single inference mode
        print(f"Generating: {args.text}")
        collector = BenchmarkCollector("results")
        run_chatterbox([args.text], args.reference, args.output, collector)
    else:
        # Full benchmark mode
        run_english_pipeline(reference_wav=args.reference, output_base=args.output)
