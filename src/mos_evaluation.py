"""
Automated MOS scoring using UTMOS — a neural MOS predictor.
Also runs self-MOS collection for human scores.
"""
import os
import sys
import json
import glob
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_utmos(audio_dir="audio", results_dir="results"):
    """Compute UTMOS (neural MOS predictor) on all generated audio."""
    print("\n" + "=" * 60)
    print("📊 AUTOMATED MOS — UTMOS Neural Predictor")
    print("=" * 60)

    try:
        import torch
        import torchaudio
    except ImportError:
        print("  [WARNING] torch/torchaudio not available")
        return []

    # Try to load UTMOS predictor
    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        )
        predictor = predictor.cuda() if torch.cuda.is_available() else predictor
        predictor.eval()
        print("  [SUCCESS] UTMOS model loaded successfully")
    except Exception as e:
        print(f"  [WARNING] Could not load UTMOS: {e}")
        print("  Falling back to basic signal-level analysis")
        return compute_signal_mos_proxy(audio_dir, results_dir)

    results = []
    for lang in ["english", "arabic", "hindi"]:
        lang_dir = os.path.join(audio_dir, lang)
        if not os.path.exists(lang_dir):
            continue

        wav_files = sorted(glob.glob(os.path.join(lang_dir, "*.wav")))
        print(f"\n  [DIR] {lang.upper()} — {len(wav_files)} files")

        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            parts = fname.replace(".wav", "").rsplit("_", 1)
            if len(parts) != 2:
                continue
            model_name = parts[0]

            try:
                waveform, sr = torchaudio.load(wav_path)
                # Resample to 16kHz if needed (UTMOS expects 16kHz)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)

                # UTMOS expects mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                if torch.cuda.is_available():
                    waveform = waveform.cuda()

                with torch.no_grad():
                    score = predictor(waveform, 16000)

                mos_score = float(score.item()) if hasattr(score, 'item') else float(score)
                mos_score = max(1.0, min(5.0, mos_score))  # Clamp to [1, 5]

            except Exception as e:
                print(f"    [WARNING] Failed for {fname}: {e}")
                mos_score = None

            results.append({
                "language": lang,
                "model": model_name,
                "audio_path": wav_path,
                "utmos_score": round(mos_score, 3) if mos_score else None,
            })

            if mos_score:
                status = "[SUCCESS]" if mos_score >= 4.0 else "[WARNING]"
                print(f"    {status} [{model_name}/{fname}] UTMOS: {mos_score:.2f}")

    # Save results
    mos_path = os.path.join(results_dir, "utmos_results.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(mos_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n  [SUCCESS] UTMOS results saved to {mos_path}")

    # Print summary
    print_mos_summary(results)

    return results


def compute_signal_mos_proxy(audio_dir="audio", results_dir="results"):
    """Fallback: compute signal-level MOS proxy metrics."""
    print("\n  📊 Computing signal-level MOS proxy (SNR, spectral quality)...")

    import soundfile as sf
    from scipy import signal as sig

    results = []
    for lang in ["english", "arabic", "hindi"]:
        lang_dir = os.path.join(audio_dir, lang)
        if not os.path.exists(lang_dir):
            continue

        wav_files = sorted(glob.glob(os.path.join(lang_dir, "*.wav")))
        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            parts = fname.replace(".wav", "").rsplit("_", 1)
            if len(parts) != 2:
                continue
            model_name = parts[0]

            try:
                audio, sr = sf.read(wav_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Compute basic quality metrics
                rms = np.sqrt(np.mean(audio ** 2))
                peak = np.max(np.abs(audio))
                dynamic_range = 20 * np.log10(peak / (rms + 1e-10))

                # Spectral centroid as proxy for brightness/naturalness
                freqs = np.fft.rfftfreq(len(audio), d=1.0/sr)
                spectrum = np.abs(np.fft.rfft(audio))
                spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

                results.append({
                    "language": lang,
                    "model": model_name,
                    "audio_path": wav_path,
                    "rms_energy": round(float(rms), 4),
                    "dynamic_range_db": round(float(dynamic_range), 1),
                    "spectral_centroid_hz": round(float(spectral_centroid), 1),
                })
            except Exception as e:
                print(f"    [WARNING] Failed for {fname}: {e}")

    # Save
    proxy_path = os.path.join(results_dir, "signal_quality.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(proxy_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"  [SUCCESS] Signal quality results saved to {proxy_path}")
    return results


def print_mos_summary(results):
    """Print aggregated MOS summary per model per language."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r.get("utmos_score") is not None:
            key = (r["language"], r["model"])
            groups[key].append(r["utmos_score"])

    print("\n" + "=" * 70)
    print(f"{'Language':<10} {'Model':<20} {'Samples':<8} {'Avg UTMOS':<10} {'Min':<8} {'Max':<8}")
    print("=" * 70)
    for (lang, model), scores in sorted(groups.items()):
        avg = np.mean(scores)
        status = "[SUCCESS]" if avg >= 4.0 else "[WARNING]"
        print(f"{lang:<10} {model:<20} {len(scores):<8} {status} {avg:<8.2f} {min(scores):<8.2f} {max(scores):<8.2f}")
    print("=" * 70)


def collect_human_mos(audio_dir="audio", results_dir="results"):
    """Interactive human MOS collection — run locally, rate clips 1-5."""
    print("\n" + "=" * 60)
    print("[MOS] HUMAN MOS COLLECTION")
    print("=" * 60)
    print("Listen to each clip and rate naturalness 1-5:")
    print("  1 = Bad (robotic, unintelligible)")
    print("  2 = Poor (clearly synthetic)")
    print("  3 = Fair (synthetic but understandable)")
    print("  4 = Good (mostly natural, minor artifacts)")
    print("  5 = Excellent (indistinguishable from human)")
    print()

    results = []
    for lang in ["english", "arabic", "hindi"]:
        lang_dir = os.path.join(audio_dir, lang)
        if not os.path.exists(lang_dir):
            continue

        wav_files = sorted(glob.glob(os.path.join(lang_dir, "*.wav")))
        print(f"\n[DIR] {lang.upper()} ({len(wav_files)} clips)")

        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            parts = fname.replace(".wav", "").rsplit("_", 1)
            model_name = parts[0] if len(parts) == 2 else fname

            print(f"\n  [AUDIO] Playing: {fname} (model: {model_name})")
            # Try to play audio
            try:
                os.system(f"play '{wav_path}' 2>/dev/null || aplay '{wav_path}' 2>/dev/null")
            except:
                print(f"    (Play manually: {wav_path})")

            while True:
                try:
                    score = input("  Rate (1-5, or 's' to skip): ").strip()
                    if score.lower() == 's':
                        break
                    score = int(score)
                    if 1 <= score <= 5:
                        results.append({
                            "language": lang,
                            "model": model_name,
                            "audio_path": wav_path,
                            "human_mos": score,
                            "rater": "self",
                        })
                        break
                    else:
                        print("    Please enter 1-5")
                except (ValueError, EOFError):
                    print("    Please enter 1-5 or 's'")

    # Save
    mos_path = os.path.join(results_dir, "human_mos.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(mos_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n[SUCCESS] Human MOS saved to {mos_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MOS Evaluation")
    parser.add_argument("--mode", choices=["utmos", "human", "signal", "all"], default="utmos",
                        help="Evaluation mode: utmos (neural), human (interactive), signal (proxy), all")
    parser.add_argument("--audio-dir", default="audio", help="Audio directory")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    if args.mode in ("utmos", "all"):
        compute_utmos(args.audio_dir, args.results_dir)
    if args.mode in ("signal", "all"):
        compute_signal_mos_proxy(args.audio_dir, args.results_dir)
    if args.mode in ("human", "all"):
        collect_human_mos(args.audio_dir, args.results_dir)
