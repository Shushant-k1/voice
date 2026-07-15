"""
Fish Speech Pipeline — Runs Fish Speech on English, Arabic, and Hindi.
Uses the fish-speech package for zero-shot voice cloning.
Designed to run on T4 GPU (15 GB VRAM).
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_fish_speech(sentences, language, reference_wav, output_dir, collector):
    """Run Fish Speech TTS on sentences for a given language."""
    print(f"\n{'=' * 60}")
    print(f"[AUDIO] {language.upper()} — Fish Speech")
    print("=" * 60)
    print_gpu_status()

    try:
        from fish_speech.inference import TTSInference

        # Initialize Fish Speech
        tts = TTSInference(
            model_name="fishaudio/fish-speech-1.5",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        for i, text in enumerate(sentences):
            print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
            output_path = os.path.join(output_dir, f"fish_{i:02d}.wav")

            with Timer("generate") as t:
                audio = tts.synthesize(
                    text=text,
                    reference_audio=reference_wav,
                )

            if audio is not None:
                save_audio(audio, output_path, sample_rate=44100)
                duration = get_audio_duration(output_path)
                rtf = t.elapsed / duration if duration > 0 else float('inf')

                collector.add_result(
                    language=language, model_name="FISH_SPEECH", sentence_idx=i,
                    text=text, latency_s=t.elapsed, rtf=rtf,
                    audio_duration_s=duration, audio_path=output_path
                )
                print(f"    Latency: {t.elapsed:.3f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")
            else:
                print(f"    [WARNING] No audio generated for sentence {i}")

    except ImportError:
        print("[ERROR] fish-speech not installed. Install with: pip install fish-speech")
        print("        Attempting fallback via HuggingFace transformers API...")

        # Fallback: use transformers pipeline if available
        try:
            from transformers import pipeline as hf_pipeline

            pipe = hf_pipeline(
                "text-to-speech",
                model="fishaudio/fish-speech-1.5",
                device=0 if torch.cuda.is_available() else -1,
            )

            for i, text in enumerate(sentences):
                print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
                output_path = os.path.join(output_dir, f"fish_{i:02d}.wav")

                with Timer("generate") as t:
                    result = pipe(text)

                if result and "audio" in result:
                    audio = np.array(result["audio"])
                    sr = result.get("sampling_rate", 44100)
                    save_audio(audio, output_path, sample_rate=sr)
                    duration = get_audio_duration(output_path)
                    rtf = t.elapsed / duration if duration > 0 else float('inf')

                    collector.add_result(
                        language=language, model_name="FISH_SPEECH", sentence_idx=i,
                        text=text, latency_s=t.elapsed, rtf=rtf,
                        audio_duration_s=duration, audio_path=output_path
                    )
                    print(f"    Latency: {t.elapsed:.3f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

        except Exception as e:
            print(f"[ERROR] Fish Speech fallback also failed: {e}")

    clear_gpu()


def main():
    with open("test_sentences.json") as f:
        sentences = json.load(f)

    reference_wav = "audio/reference/reference_voice.wav"
    collector = BenchmarkCollector(output_dir="results")

    for lang in ["english", "arabic", "hindi"]:
        output_dir = f"audio/{lang}"
        os.makedirs(output_dir, exist_ok=True)
        run_fish_speech(sentences[lang], lang, reference_wav, output_dir, collector)

    collector.save_csv("fish_speech_benchmark.csv")
    collector.save_json("fish_speech_benchmark.json")
    print("\n[DONE] Fish Speech pipeline complete.")


if __name__ == "__main__":
    main()
