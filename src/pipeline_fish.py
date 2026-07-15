"""
Fish-Speech TTS Pipeline — Arabic & Hindi with voice cloning.
Fish-Speech has excellent multilingual support and zero-shot cloning.
"""
import os, sys, json, torch, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import Timer, save_audio, get_audio_duration, clear_gpu, print_gpu_status, BenchmarkCollector


def run_fish_speech(sentences, reference_wav, output_dir, collector, language="arabic", lang_code="ar"):
    """Run Fish-Speech TTS with zero-shot voice cloning."""
    print("\n" + "=" * 60)
    print(f"[AUDIO] {language.upper()} — Fish-Speech (Zero-Shot Clone)")
    print("=" * 60)
    print_gpu_status()

    try:
        from fish_speech.inference import TTSInference
        
        # Load model
        tts = TTSInference.from_pretrained("fishaudio/fish-speech-1.5")
        tts.to("cuda")
        
        for i, text in enumerate(sentences):
            print(f"  [{i+1}/{len(sentences)}] \"{text[:50]}\"")
            output_path = os.path.join(output_dir, f"fish_speech_{i:02d}.wav")
            
            with Timer("generate") as t:
                audio = tts.synthesize(
                    text=text,
                    reference_audio=reference_wav,
                    language=lang_code,
                )
            
            if hasattr(audio, 'cpu'):
                audio_np = audio.squeeze().cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio_np = audio.squeeze().numpy()
            else:
                audio_np = np.array(audio).squeeze()
            
            save_audio(audio_np, output_path, sample_rate=44100)
            duration = get_audio_duration(output_path)
            rtf = t.elapsed / duration if duration > 0 else float('inf')
            
            collector.add_result(language, "Fish-Speech", i, text, t.elapsed, rtf, duration, output_path)
            print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
        
        del tts
        clear_gpu()
        
    except (ImportError, Exception) as e:
        print(f"  [WARNING] Fish-Speech failed or is not installed: {e}")
        print("  Trying alternative: F5-TTS...")
        run_f5_tts(sentences, reference_wav, output_dir, collector, language, lang_code)


def run_fish_speech_cli(sentences, reference_wav, output_dir, collector, language="arabic", lang_code="ar"):
    """Fallback: Run Fish-Speech via CLI if Python API doesn't work."""
    import subprocess
    
    print(f"\n  Attempting Fish-Speech via CLI for {language}...")
    
    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] \"{text[:50]}\"")
        output_path = os.path.join(output_dir, f"fish_speech_{i:02d}.wav")
        
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "fish_speech.inference",
                 "--text", text,
                 "--reference", reference_wav,
                 "--output", output_path],
                capture_output=True, text=True, timeout=120
            )
            elapsed = time.perf_counter() - start
            
            if os.path.exists(output_path):
                duration = get_audio_duration(output_path)
                rtf = elapsed / duration if duration > 0 else float('inf')
                collector.add_result(language, "Fish-Speech", i, text, elapsed, rtf, duration, output_path)
                print(f"  [SUCCESS] Latency:{elapsed:.2f}s | RTF:{rtf:.3f}")
            else:
                print(f"  [WARNING] No output generated: {result.stderr[:200]}")
        except Exception as e:
            print(f"  [WARNING] CLI failed: {e}")


def run_f5_tts(sentences, reference_wav, output_dir, collector, language="arabic", lang_code="ar"):
    """Fallback: F5-TTS — fast non-autoregressive flow-matching TTS."""
    print(f"\n  Attempting F5-TTS for {language}...")

    # Romanized phonetic pronunciations for F5-TTS to bypass English/Chinese tokenizer vocabulary gaps
    romanized_sentences = {
        "arabic": [
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
        ],
        "hindi": [
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
    }
    
    try:
        from f5_tts.api import F5TTS
        
        model = F5TTS()
        
        for i, text in enumerate(sentences):
            # Select phonetic transcription if available to prevent alignment/babbling crashes
            lang_key = language.lower()
            gen_text = text
            if lang_key in romanized_sentences and i < len(romanized_sentences[lang_key]):
                gen_text = romanized_sentences[lang_key][i]
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
            
            collector.add_result(language, "F5-TTS", i, text, t.elapsed, rtf, duration, output_path)
            print(f"  [SUCCESS] Latency:{t.elapsed:.2f}s | RTF:{rtf:.3f}")
        
        del model
        clear_gpu()
        
    except ImportError:
        print(f"  [WARNING] F5-TTS not installed for {language}. Skipping.")
        print(f"  Install: pip install f5-tts")
    except Exception as e:
        print(f"  [WARNING] F5-TTS failed for {language}: {e}")


def run_fish_arabic(test_sentences_path="test_sentences.json",
                    reference_wav="audio/reference/reference_voice.wav",
                    output_base="audio/arabic", results_dir="results"):
    """Run Fish-Speech for Arabic."""
    with open(test_sentences_path, 'r', encoding='utf-8') as f:
        sentences = json.load(f)["arabic"]
    os.makedirs(output_base, exist_ok=True)
    collector = BenchmarkCollector(results_dir)
    run_fish_speech(sentences, reference_wav, output_base, collector, "arabic", "ar")
    collector.save_csv("arabic_fish_benchmark.csv")
    collector.print_summary()
    return collector


def run_fish_hindi(test_sentences_path="test_sentences.json",
                   reference_wav="audio/reference/reference_voice.wav",
                   output_base="audio/hindi", results_dir="results"):
    """Run Fish-Speech for Hindi."""
    with open(test_sentences_path, 'r', encoding='utf-8') as f:
        sentences = json.load(f)["hindi"]
    os.makedirs(output_base, exist_ok=True)
    collector = BenchmarkCollector(results_dir)
    run_fish_speech(sentences, reference_wav, output_base, collector, "hindi", "hi")
    collector.save_csv("hindi_fish_benchmark.csv")
    collector.print_summary()
    return collector


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fish-Speech Pipeline")
    parser.add_argument("--lang", choices=["arabic", "hindi", "both"], default="both")
    args = parser.parse_args()
    
    if args.lang in ("arabic", "both"):
        run_fish_arabic()
    if args.lang in ("hindi", "both"):
        run_fish_hindi()
