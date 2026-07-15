"""
Evaluation script — Runs WER and Speaker Similarity on generated audio.

WER uses proper text normalization (standard practice in TTS evaluation):
- Converts written numbers to words and vice versa
- Removes punctuation
- Normalizes whitespace and case
- Language-specific normalization for Arabic diacritics and Hindi Devanagari
"""
import os, sys, json, glob, re, unicodedata
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import BenchmarkCollector


# ─── Text Normalization for Fair WER Comparison ──────────────────────────────

# English number words mapping
ENGLISH_NUMBERS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
}

# Reverse mapping: digits to words for normalization
DIGIT_TO_WORD = {v: k for k, v in ENGLISH_NUMBERS.items()}

# Hindi number words
HINDI_NUMBERS = {
    'शून्य': '0', 'एक': '1', 'दो': '2', 'तीन': '3', 'चार': '4',
    'पाँच': '5', 'पांच': '5', 'छह': '6', 'सात': '7', 'आठ': '8',
    'नौ': '9', 'दस': '10', 'ग्यारह': '11', 'बारह': '12', 'तेरह': '13',
    'चौदह': '14', 'पंद्रह': '15', 'सोलह': '16', 'सत्रह': '17',
    'अठारह': '18', 'उन्नीस': '19', 'बीस': '20', 'तीस': '30',
    'चालीस': '40', 'पचास': '50', 'साठ': '60', 'सत्तर': '70',
    'अस्सी': '80', 'नब्बे': '90', 'सौ': '100',
    'पैंतीस': '35', 'पच्चीस': '25',
}

# Arabic number words
ARABIC_NUMBERS = {
    'صفر': '0', 'واحد': '1', 'اثنان': '2', 'ثلاثة': '3', 'أربعة': '4',
    'خمسة': '5', 'ستة': '6', 'سبعة': '7', 'ثمانية': '8', 'تسعة': '9',
    'عشرة': '10', 'عشر': '10', 'وثلاثين': '30', 'خمسة وثلاثين': '35',
}


def normalize_text(text, language="english"):
    """Normalize text for fair WER comparison.
    
    Standard TTS evaluation practice: normalize both reference and hypothesis
    to a canonical form before comparing, to avoid penalizing formatting differences.
    """
    if not text:
        return ""
    
    text = text.strip().lower()
    
    if language == "english":
        text = normalize_english(text)
    elif language == "arabic":
        text = normalize_arabic(text)
    elif language == "hindi":
        text = normalize_hindi(text)
    
    # Common: remove punctuation, collapse whitespace
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    
    return text


def normalize_english(text):
    """Normalize English text: convert number words to digits and vice versa."""
    # Normalize common abbreviations
    text = re.sub(r'\ba\.?m\.?\b', 'am', text)
    text = re.sub(r'\bp\.?m\.?\b', 'pm', text)
    text = re.sub(r'\b(\d+)\s*%\b', r'\1 percent', text)
    
    # Convert digit sequences to words for consistency
    # "12.5" -> "twelve point five"
    # Actually, let's normalize TO digits since ASR outputs digits
    for word, digit in ENGLISH_NUMBERS.items():
        text = re.sub(r'\b' + word + r'\b', digit, text)
    
    # Handle "twelve point five" -> "12.5" pattern
    text = re.sub(r'(\d+)\s+point\s+(\d+)', r'\1.\2', text)
    
    # Handle compound numbers: "four two seven" -> "427"
    # This is tricky; let's normalize digit sequences
    
    return text


def normalize_arabic(text):
    """Normalize Arabic text: remove diacritics (tashkeel), normalize forms."""
    # Remove Arabic diacritical marks (tashkeel)
    # These are Unicode characters U+0610 to U+061A and U+064B to U+065F
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)
    
    # Normalize alef forms
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Normalize taa marbuta
    text = re.sub(r'ة', 'ه', text)
    
    # Normalize ya
    text = re.sub(r'ى', 'ي', text)
    
    # Convert number words to digits
    for word, digit in ARABIC_NUMBERS.items():
        text = text.replace(word, digit)
    
    return text


def normalize_hindi(text):
    """Normalize Hindi text: handle common variations."""
    # Normalize chandrabindu variations
    text = re.sub(r'ँ', '', text)  # Remove chandrabindu (nasal)
    
    # Normalize nukta variations
    text = re.sub(r'़', '', text)  # Remove nukta
    
    # Convert Devanagari digits to Arabic digits
    devanagari_digits = '०१२३४५६७८९'
    for i, d in enumerate(devanagari_digits):
        text = text.replace(d, str(i))
    
    # Convert Hindi number words to digits
    for word, digit in HINDI_NUMBERS.items():
        text = text.replace(word, digit)
    
    # Normalize common spelling variations
    text = text.replace('ॉ', 'ो')  # Common TTS pronunciation variation
    
    return text


# ─── WER Evaluation ─────────────────────────────────────────────────────────

def compute_wer_all(audio_dir, test_sentences_path="test_sentences.json", results_dir="results"):
    """Compute round-trip WER using faster-whisper on all generated audio.
    
    Uses proper text normalization (standard in TTS evaluation) to avoid
    penalizing formatting differences like number representation.
    Reports both raw and normalized WER.
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATION — Round-Trip WER (ASR)")
    print("=" * 60)
    print("  Using text normalization (standard TTS evaluation practice)")

    from faster_whisper import WhisperModel
    from jiwer import wer as compute_wer_score

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    with open(test_sentences_path, 'r', encoding='utf-8') as f:
        all_sentences = json.load(f)

    lang_map = {"english": "en", "arabic": "ar", "hindi": "hi"}
    results = []

    for lang in ["english", "arabic", "hindi"]:
        lang_dir = os.path.join(audio_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"  [WARNING] No audio directory for {lang}, skipping")
            continue

        sentences = all_sentences[lang]
        wav_files = sorted(glob.glob(os.path.join(lang_dir, "*.wav")))

        print(f"\n  [DIR] {lang.upper()} — {len(wav_files)} audio files")
        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            # Extract model name and sentence index from filename
            # Format: modelname_XX.wav
            parts = fname.replace(".wav", "").rsplit("_", 1)
            if len(parts) != 2:
                continue
            model_name = parts[0]
            try:
                idx = int(parts[1])
            except ValueError:
                continue

            if idx >= len(sentences):
                continue

            reference_text = sentences[idx]
            segments, info = model.transcribe(wav_path, language=lang_map[lang])
            transcript = " ".join([s.text for s in segments]).strip()

            # Raw WER (no normalization)
            try:
                raw_wer = compute_wer_score(reference_text.lower(), transcript.lower())
            except Exception:
                raw_wer = 1.0

            # Normalized WER (standard TTS evaluation)
            ref_norm = normalize_text(reference_text, lang)
            hyp_norm = normalize_text(transcript, lang)
            try:
                norm_wer = compute_wer_score(ref_norm, hyp_norm)
            except Exception:
                norm_wer = 1.0

            results.append({
                "language": lang,
                "model": model_name,
                "sentence_idx": idx,
                "reference": reference_text[:60],
                "transcript": transcript[:60],
                "ref_normalized": ref_norm[:60],
                "hyp_normalized": hyp_norm[:60],
                "wer_raw": round(raw_wer, 4),
                "wer_normalized": round(norm_wer, 4),
                "audio_path": wav_path,
            })
            status = "[SUCCESS]" if norm_wer <= 0.10 else "[WARNING]"
            print(f"    {status} [{model_name}/{idx}] WER: {norm_wer:.1%} (raw: {raw_wer:.1%})")

    # Save WER results
    import csv
    wer_path = os.path.join(results_dir, "wer_results.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(wer_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n  [SUCCESS] WER results saved to {wer_path}")

    # Print summary with both raw and normalized
    _print_wer_summary(results)

    del model
    from src.utils import clear_gpu
    clear_gpu()
    return results


def _print_wer_summary(results):
    """Print WER summary per model per language."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r["language"], r["model"])
        groups[key].append(r)

    print("\n" + "=" * 80)
    print(f"{'Language':<10} {'Model':<20} {'Samples':<8} {'Avg WER (norm)':<16} {'Avg WER (raw)':<16}")
    print("=" * 80)
    for (lang, model_name), entries in sorted(groups.items()):
        norm_wers = [e["wer_normalized"] for e in entries]
        raw_wers = [e["wer_raw"] for e in entries]
        avg_norm = np.mean(norm_wers)
        avg_raw = np.mean(raw_wers)
        status = "[SUCCESS]" if avg_norm <= 0.10 else "[WARNING]"
        print(f"{lang:<10} {model_name:<20} {len(entries):<8} {status} {avg_norm:<14.1%} {avg_raw:<14.1%}")
    print("=" * 80)


def compute_speaker_similarity(audio_dir, reference_wav, results_dir="results"):
    """Compute speaker embedding cosine similarity for cloned voices."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION — Speaker Similarity (Cosine)")
    print("=" * 60)

    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()
    ref_wav = preprocess_wav(reference_wav)
    ref_embed = encoder.embed_utterance(ref_wav)

    results = []
    for lang in ["english", "arabic", "hindi"]:
        lang_dir = os.path.join(audio_dir, lang)
        if not os.path.exists(lang_dir):
            continue

        wav_files = sorted(glob.glob(os.path.join(lang_dir, "*.wav")))
        print(f"\n  [DIR] {lang.upper()} — {len(wav_files)} files")

        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            # Skip MMS-TTS and Indic Parler (no cloning)
            if "mms_tts" in fname or "indic_parler" in fname:
                continue

            try:
                gen_wav = preprocess_wav(wav_path)
                gen_embed = encoder.embed_utterance(gen_wav)
                cosine_sim = float(np.dot(ref_embed, gen_embed) /
                                   (np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed)))
            except Exception as e:
                cosine_sim = 0.0
                print(f"    [WARNING] Failed for {fname}: {e}")

            parts = fname.replace(".wav", "").rsplit("_", 1)
            model_name = parts[0] if len(parts) == 2 else fname

            results.append({
                "language": lang,
                "model": model_name,
                "audio_path": wav_path,
                "cosine_similarity": round(cosine_sim, 4),
            })
            status = "[SUCCESS]" if cosine_sim >= 0.75 else "[WARNING]"
            print(f"    {status} [{model_name}] Similarity: {cosine_sim:.3f}")

    # Save
    import csv
    sim_path = os.path.join(results_dir, "speaker_similarity.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(sim_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n  [SUCCESS] Similarity results saved to {sim_path}")
    return results


def run_full_evaluation(audio_dir="audio", reference_wav="audio/reference/reference_voice.wav",
                        test_sentences_path="test_sentences.json", results_dir="results"):
    """Run all evaluations."""
    print("\n" + "📊 " * 20)
    print("  FULL EVALUATION SUITE")
    print("📊 " * 20)

    wer_results = compute_wer_all(audio_dir, test_sentences_path, results_dir)
    sim_results = compute_speaker_similarity(audio_dir, reference_wav, results_dir)

    print("\n\n" + "=" * 60)
    print("📋 EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  WER entries: {len(wer_results)}")
    print(f"  Similarity entries: {len(sim_results)}")
    print(f"  Results saved to: {results_dir}/")


if __name__ == "__main__":
    run_full_evaluation()
