"""
Master Benchmark Runner — Runs ALL model pipelines sequentially.
Installs dependencies, runs each model, collects results.

Usage (on GPU machine like Lightning AI Studio, Kaggle, or Colab):
    pip install -r requirements.txt
    python src/run_all_benchmarks.py

This will run:
  1. XTTS-v2     (English, Arabic, Hindi)  — Already done
  2. F5-TTS      (English, Arabic, Hindi)  — Already done
  3. Chatterbox   (English)                — Already done
  4. MMS-TTS     (Arabic, Hindi)           — Already done
  5. Indic-Parler (Hindi)                  — Already done
  6. Bark        (English, Arabic, Hindi)  — NEW
  7. Kokoro      (English, Hindi)          — NEW
  8. OpenVoice   (English)                 — NEW
  9. Fish Speech (English, Arabic, Hindi)  — NEW
"""
import subprocess
import sys
import os
import traceback


def install_package(package_name, pip_name=None):
    """Install a package if not already installed."""
    pip_name = pip_name or package_name
    try:
        __import__(package_name)
        print(f"  [OK] {package_name} already installed")
        return True
    except ImportError:
        print(f"  [INSTALLING] {pip_name}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  [OK] {pip_name} installed successfully")
            return True
        else:
            print(f"  [FAIL] Could not install {pip_name}: {result.stderr[:200]}")
            return False


def run_pipeline(script_name, description):
    """Run a pipeline script and handle errors."""
    print(f"\n{'#' * 70}")
    print(f"# RUNNING: {description}")
    print(f"# Script:  src/{script_name}")
    print(f"{'#' * 70}")

    try:
        result = subprocess.run(
            [sys.executable, f"src/{script_name}"],
            capture_output=False, text=True, timeout=1800  # 30 min timeout
        )
        if result.returncode == 0:
            print(f"\n[SUCCESS] {description} completed.")
            return True
        else:
            print(f"\n[FAILED] {description} exited with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {description} exceeded 30-minute limit. Skipping.")
        return False
    except Exception as e:
        print(f"\n[ERROR] {description} failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("MASTER BENCHMARK RUNNER — All Open-Source TTS Models")
    print("=" * 70)

    results = {}

    # ─── Phase 1: Install dependencies for new models ──────────────────
    print("\n--- Phase 1: Installing dependencies ---")

    # Bark
    install_package("bark", "suno-bark")

    # Kokoro
    install_package("kokoro", "kokoro>=0.9")

    # OpenVoice
    install_package("openvoice", "openvoice")
    install_package("melo", "melo-tts")

    # Fish Speech (may not have a simple pip install)
    install_package("fish_speech", "fish-speech")

    # ─── Phase 2: Run existing pipelines (skip if results already exist) ─
    print("\n--- Phase 2: Running existing pipelines ---")

    existing_pipelines = [
        ("pipeline_english.py", "English (XTTS-v2 + Chatterbox)"),
        ("pipeline_arabic.py", "Arabic (XTTS-v2 + MMS-TTS)"),
        ("pipeline_hindi.py", "Hindi (XTTS-v2 + MMS-TTS + Indic-Parler)"),
        ("pipeline_fish.py", "F5-TTS (English + Arabic + Hindi)"),
        ("run_f5_english.py", "F5-TTS English"),
    ]

    for script, desc in existing_pipelines:
        if os.path.exists(f"src/{script}"):
            results[desc] = run_pipeline(script, desc)
        else:
            print(f"  [SKIP] src/{script} not found")
            results[desc] = False

    # ─── Phase 3: Run NEW model pipelines ──────────────────────────────
    print("\n--- Phase 3: Running NEW model pipelines ---")

    new_pipelines = [
        ("pipeline_bark.py", "Bark (English + Arabic + Hindi)"),
        ("pipeline_kokoro.py", "Kokoro v1.0 (English + Hindi)"),
        ("pipeline_openvoice.py", "OpenVoice V2 (English)"),
        ("pipeline_fish_speech.py", "Fish Speech (English + Arabic + Hindi)"),
    ]

    for script, desc in new_pipelines:
        if os.path.exists(f"src/{script}"):
            results[desc] = run_pipeline(script, desc)
        else:
            print(f"  [SKIP] src/{script} not found")
            results[desc] = False

    # ─── Phase 4: Run evaluations ─────────────────────────────────────
    print("\n--- Phase 4: Running evaluations ---")
    results["Evaluation (WER + Similarity)"] = run_pipeline("evaluate.py", "WER + Speaker Similarity")
    results["UTMOS Scoring"] = run_pipeline("mos_evaluation.py", "UTMOS Scoring")
    results["Streaming Benchmark"] = run_pipeline("benchmark_streaming.py", "Streaming TTFA")

    # ─── Phase 5: Compile results ─────────────────────────────────────
    print("\n--- Phase 5: Compiling results ---")
    results["Compile Results"] = run_pipeline("compile_results.py", "Compile Final Summary")

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL STATUS")
    print("=" * 70)
    for desc, status in results.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {desc}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Total: {passed}/{total} pipelines completed successfully.")


if __name__ == "__main__":
    main()
