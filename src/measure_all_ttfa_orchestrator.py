import subprocess
import csv
import os
import sys

def run_orchestrator():
    # Model list to run
    tasks = [
        # English
        ("en", "xtts_v2"),
        ("en", "chatterbox"),
        ("en", "f5_tts"),
        ("en", "kokoro"),
        ("en", "openvoice"),
        ("en", "bark"),
        # Arabic
        ("ar", "xtts_v2"),
        ("ar", "f5_tts"),
        ("ar", "mms"),
        ("ar", "bark"),
        # Hindi
        ("hi", "xtts_v2"),
        ("hi", "f5_tts"),
        ("hi", "indic_parler"),
        ("hi", "kokoro"),
        ("hi", "mms"),
        ("hi", "bark"),
    ]

    results = []

    print("Orchestrator: Starting TTFA measurements for all models...")

    for lang, model in tasks:
        print(f"\n--- Running Task: Language={lang}, Model={model} ---")
        try:
            # We must use the remote python environment
            # Usually 'python' is fine if run in the active venv
            cmd = [sys.executable, "src/measure_ttfa.py", "--model", model, "--lang", lang]
            print(f"Running command: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate(timeout=180) # 3 min timeout per model
            
            print("STDOUT:")
            print(stdout)
            print("STDERR:")
            print(stderr)
            
            ttfa = None
            for line in stdout.split("\n"):
                if line.startswith("RESULT:"):
                    ttfa = float(line.split(":")[1])
            
            if ttfa is not None:
                results.append({"language": lang, "model": model, "ttfa_ms": ttfa})
                print(f"Success! TTFA: {ttfa:.2f} ms")
            else:
                results.append({"language": lang, "model": model, "ttfa_ms": "N/A"})
                print("Failed to parse RESULT line.")
        except Exception as e:
            results.append({"language": lang, "model": model, "ttfa_ms": "N/A"})
            print(f"Error executing task: {e}")

    # Write to CSV
    os.makedirs("results", exist_ok=True)
    out_path = "results/all_streaming_latency.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "model", "ttfa_ms"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\n[DONE] Saved all TTFA results to {out_path}")

if __name__ == "__main__":
    run_orchestrator()
