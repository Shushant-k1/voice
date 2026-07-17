"""
Arabic TTS Pipeline — Compares XTTS-v2, Fish-Speech, and MMS-TTS.
"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_xtts_arabic(sentences, reference_wav, output_dir, collector):
    print("\n" + "=" * 60)
    print("[AUDIO] ARABIC — Model 1: XTTS-v2 (Coqui)")
    print("=" * 60)
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] \"{text[:50]}\"")
        output_path = os.path.join(output_dir, f"xtts_v2_{i:02d}.wav")
        with Timer("generate") as t:
            tts.tts_to_file(text=text, speaker_wav=reference_wav, language="ar", file_path=output_path)
        duration = get_audio_duration(output_path)
        rtf = t.elapsed / duration if duration > 0 else float('inf')
        collector.add_result("arabic", "XTTS-v2", i, text, t.elapsed, rtf, duration, output_path)
        print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
    del tts; clear_gpu()


def run_mms_tts_arabic(sentences, output_dir, collector):
    print("\n" + "=" * 60)
    print("[AUDIO] ARABIC — Model 2: MMS-TTS (Meta) — no cloning")
    print("=" * 60)
    from transformers import VitsModel, AutoTokenizer
    model = VitsModel.from_pretrained("facebook/mms-tts-ara").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
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
        collector.add_result("arabic", "MMS-TTS", i, text, t.elapsed, rtf, duration, output_path)
        print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
    del model, tokenizer; clear_gpu()


def run_f5_arabic(sentences, reference_wav, output_dir, collector):
    print("\n" + "=" * 60)
    print("[AUDIO] ARABIC — Model 3: F5-TTS (Zero-Shot Flow Matching)")
    print("=" * 60)
    
    romanized_arabic_sentences = [
        "Marhaban bikum fee khidmat al-umala. Kayfa yumkinuni musa'adatukum al-yawm?",
        "Al-taqsu al-yawma mushmisun ma'a darajati hararatin tasilu ila khamsatin wa-thalatheena darajatan mi'awiyyah.",
        "Yurja al-ittisalu bina 'ala al-raqm sifr wahid ithnan thalathah arba'ah khamsah.",
        "Shukran li-sabrikum. Sayatimm taheel mukalamatukum ila al-qism al-mukhtass.",
        "Maw'id al-ijtima' al-qadim yawm al-ahad al-sa'ah al-'ashirah sabahan.",
        "Na'tadiru 'an ayyi iz'ajin qad sabbabnahu lakum wa-na'idukum bi-tahseeni khadamatina.",
        "Yumkinukum ziyaratu mawqi'ina al-iliktroni lil-ittila' 'ala ahdath al-urood wal-khusoomat.",
        "Tamma irsal al-talab bi-najah wa-sayatimm al-tawseel khilal thalathat ayyam 'amal.",
        "Hal tureed al-istimrar fee al-lughah al-arabiyyah am al-tahweel ila al-lughah al-injleeziyyah?",
        "Ma'a al-salamah wa-natamanna lakum yawman sa'eedan."
    ]

    try:
        from f5_tts.api import F5TTS
        model = F5TTS()
        for i, text in enumerate(sentences):
            gen_text = text
            if i < len(romanized_arabic_sentences):
                gen_text = romanized_arabic_sentences[i]
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
            collector.add_result("arabic", "F5-TTS", i, text, t.elapsed, rtf, duration, output_path)
            print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
        del model; clear_gpu()
    except ImportError:
        print("  [WARNING] F5-TTS not installed. Skipping.")
    except Exception as e:
        print(f"  [WARNING] F5-TTS failed: {e}")


def run_arabic_pipeline(test_sentences_path="test_sentences.json",
                        reference_wav="audio/reference/reference_voice.wav",
                        output_base="audio/arabic", results_dir="results"):
    print("\n  ARABIC PIPELINE — Comparing 3 Models")
    with open(test_sentences_path, 'r', encoding='utf-8') as f:
        sentences = json.load(f)["arabic"]
    os.makedirs(output_base, exist_ok=True)
    collector = BenchmarkCollector(results_dir)
    run_xtts_arabic(sentences, reference_wav, output_base, collector)
    run_mms_tts_arabic(sentences, output_base, collector)
    run_f5_arabic(sentences, reference_wav, output_base, collector)
    collector.save_csv("arabic_benchmark.csv")
    collector.print_summary()
    return collector

if __name__ == "__main__":
    run_arabic_pipeline()
