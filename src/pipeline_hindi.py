"""
Hindi TTS Pipeline — Compares Indic Parler-TTS, XTTS-v2, and MMS-TTS.
"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_indic_parler(sentences, output_dir, collector):
    """Indic Parler-TTS uses text prompts for style, not reference audio."""
    print("\n" + "=" * 60)
    print("[AUDIO] HINDI — Model 1: Indic Parler-TTS (AI4Bharat)")
    print("=" * 60)
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    except Exception as e:
        print(f"  [WARNING] Failed to load Indic Parler-TTS: {e}")
        print("  This model is gated. Please ensure you have accepted the terms on HF and set HF_TOKEN.")
        return

    description = "A female speaker with a calm, clear voice speaks Hindi at a moderate pace in a quiet environment."

    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] \"{text[:50]}\"")
        output_path = os.path.join(output_dir, f"indic_parler_{i:02d}.wav")
        with Timer("generate") as t:
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to("cuda")
            prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
            audio = generation.cpu().numpy().squeeze()
        save_audio(audio, output_path, sample_rate=22050)
        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')
        collector.add_result("hindi", "Indic-Parler-TTS", i, text, t.elapsed, rtf, duration, output_path)
        print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
    del model, tokenizer; clear_gpu()


def run_xtts_hindi(sentences, reference_wav, output_dir, collector):
    print("\n" + "=" * 60)
    print("[AUDIO] HINDI — Model 2: XTTS-v2 (Coqui)")
    print("=" * 60)
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] \"{text[:50]}\"")
        output_path = os.path.join(output_dir, f"xtts_v2_{i:02d}.wav")
        with Timer("generate") as t:
            tts.tts_to_file(text=text, speaker_wav=reference_wav, language="hi", file_path=output_path)
        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')
        collector.add_result("hindi", "XTTS-v2", i, text, t.elapsed, rtf, duration, output_path)
        print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
    del tts; clear_gpu()


def run_mms_tts_hindi(sentences, output_dir, collector):
    print("\n" + "=" * 60)
    print("[AUDIO] HINDI — Model 3: MMS-TTS (Meta) — no cloning")
    print("=" * 60)
    from transformers import VitsModel, AutoTokenizer
    model = VitsModel.from_pretrained("facebook/mms-tts-hin").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] \"{text[:50]}\"")
        output_path = os.path.join(output_dir, f"mms_tts_{i:02d}.wav")
        with Timer("generate") as t:
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model(**inputs)
            waveform = output.waveform.squeeze().cpu().numpy()
        save_audio(waveform, output_path, sample_rate=model.config.sampling_rate)
        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')
        collector.add_result("hindi", "MMS-TTS", i, text, t.elapsed, rtf, duration, output_path)
        print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
    del model, tokenizer; clear_gpu()


def run_f5_hindi(sentences, reference_wav, output_dir, collector):
    print("\n" + "=" * 60)
    print("[AUDIO] HINDI — Model 4: F5-TTS (Zero-Shot Flow Matching)")
    print("=" * 60)
    
    romanized_hindi_sentences = [
        "Namaste, hamari grahak seva mein aapka swagat hai. Main aapki kaise madad kar sakta hoon?",
        "Aaj ka mausam saaf rahega aur taapmaan paintis degree tak pahunch sakta hai.",
        "Kripya apna order number batayein taaki hum aapki shikayat darj kar sakein.",
        "Aapka package kal subah nau baje se dopahar baarah baje ke beech deliver kiya jayega.",
        "Dhanyavaad aapke dhairya ke liye. Aapki call hamare liye bahut mahatvapurna hai.",
        "Kya aap kripya dobara bata sakte hain? Maine theek se nahi suna.",
        "Hamari nayi yojna mein aseemit calling aur pachaas GB data shaamil hai.",
        "Aapka khaata safaltapoorvak update kar diya gaya hai.",
        "Agli baithak somvaar ko subah das baje nirdharit ki gayi hai.",
        "Shubh raatri, aur humse sampark karne ke liye dhanyavaad."
    ]

    try:
        from f5_tts.api import F5TTS
        model = F5TTS()
        for i, text in enumerate(sentences):
            gen_text = text
            if i < len(romanized_hindi_sentences):
                gen_text = romanized_hindi_sentences[i]
                print(f"  [{i+1}/{len(sentences)}] Using Romanized phonetic text: \"{gen_text}\"")
            else:
                print(f"  [{i+1}/{len(sentences)}] \"{text[:50]}\"")
                
            output_path = os.path.join(output_dir, f"f5_tts_{i:02d}.wav")
            with Timer("generate") as t:
                audio, sr, _ = model.infer(
                    ref_file=reference_wav,
                    ref_text="Some call me nature. Others call me Mother Nature.",
                    gen_text=gen_text,
                )
            save_audio(audio, output_path, sample_rate=sr)
            duration = get_audio_duration(output_path)
            rtf = t.elapsed / duration if duration > 0 else float('inf')
            collector.add_result("hindi", "F5-TTS", i, text, t.elapsed, rtf, duration, output_path)
            print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
        del model; clear_gpu()
    except ImportError:
        print("  [WARNING] F5-TTS not installed. Skipping.")
    except Exception as e:
        print(f"  [WARNING] F5-TTS failed: {e}")


def run_hindi_pipeline(test_sentences_path="test_sentences.json",
                       reference_wav="audio/reference/reference_voice.wav",
                       output_base="audio/hindi", results_dir="results"):
    print("\n  HINDI PIPELINE — Comparing 4 Models")
    with open(test_sentences_path, 'r', encoding='utf-8') as f:
        sentences = json.load(f)["hindi"]
    os.makedirs(output_base, exist_ok=True)
    collector = BenchmarkCollector(results_dir)
    run_indic_parler(sentences, output_base, collector)
    run_xtts_hindi(sentences, reference_wav, output_base, collector)
    run_mms_tts_hindi(sentences, output_base, collector)
    run_f5_hindi(sentences, reference_wav, output_base, collector)
    collector.save_csv("hindi_benchmark.csv")
    collector.print_summary()
    return collector

if __name__ == "__main__":
    run_hindi_pipeline()
