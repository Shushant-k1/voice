"""
OpenVoice V2 Pipeline — Runs MyShell OpenVoice on English sentences + tone color transfer.
Designed to run on T4 GPU (15 GB VRAM).
Note: OpenVoice V2 natively supports EN/ES/FR/ZH/JA/KO. No native Arabic/Hindi.
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_openvoice(sentences, reference_wav, output_dir, collector):
    """Run OpenVoice V2 TTS on English sentences with voice cloning."""
    print(f"\n{'=' * 60}")
    print("[AUDIO] ENGLISH — OpenVoice V2 (MyShell AI)")
    print("=" * 60)
    print_gpu_status()

    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS

    # Initialize base TTS (MeloTTS) and tone color converter
    ckpt_converter = "checkpoints_v2/converter"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
    tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

    # Extract reference speaker embedding
    target_se, _ = se_extractor.get_se(reference_wav, tone_color_converter, vad=False)

    # Initialize base TTS
    base_tts = MeloTTS(language="EN", device=device)
    speaker_ids = base_tts.hps.data.spk2id

    # Get source speaker embedding
    src_path = os.path.join(output_dir, "openvoice_tmp_src.wav")

    for i, text in enumerate(sentences):
        print(f"\n  [{i+1}/{len(sentences)}] \"{text[:60]}...\"")
        output_path = os.path.join(output_dir, f"openvoice_{i:02d}.wav")

        with Timer("generate") as t:
            # Step 1: Generate base speech with MeloTTS
            base_tts.tts_to_file(text, speaker_ids["EN-US"], src_path, speed=1.0)
            # Step 2: Extract source speaker embedding
            source_se, _ = se_extractor.get_se(src_path, tone_color_converter, vad=False)
            # Step 3: Apply tone color conversion (voice cloning)
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
            )

        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')

        collector.add_result(
            language="english", model_name="OPENVOICE_V2", sentence_idx=i,
            text=text, latency_s=t.elapsed, rtf=rtf,
            audio_duration_s=duration, audio_path=output_path
        )
        print(f"    Latency: {t.elapsed:.3f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

    # Cleanup temp
    if os.path.exists(src_path):
        os.remove(src_path)
    clear_gpu()


def main():
    with open("test_sentences.json") as f:
        sentences = json.load(f)

    reference_wav = "audio/reference/reference_voice.wav"
    collector = BenchmarkCollector(output_dir="results")

    output_dir = "audio/english"
    os.makedirs(output_dir, exist_ok=True)
    run_openvoice(sentences["english"], reference_wav, output_dir, collector)

    collector.save_csv("openvoice_benchmark.csv")
    collector.save_json("openvoice_benchmark.json")
    print("\n[DONE] OpenVoice V2 pipeline complete.")


if __name__ == "__main__":
    main()
