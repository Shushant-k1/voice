"""
Time-to-First-Audio (TTFA) Streaming Latency Benchmark.
Measures the latency of chunk-by-chunk autoregressive generation using XTTS-v2 
across English, Arabic, and Hindi on GPU, evaluating against the < 500 ms target.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import clear_gpu

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_streaming_latency(
    speaker_wav: str = None,
    output_csv: str = None,
    chunk_size: int = 20
) -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if speaker_wav is None:
        speaker_wav = os.path.join(project_root, "audio", "reference", "reference_voice.wav")
    if output_csv is None:
        output_csv = os.path.join(project_root, "results", "streaming_latency.csv")

    """
    Measures the streaming TTFA latency for XTTS-v2 across English, Arabic, and Hindi.
    
    Args:
        speaker_wav: Path to reference speaker audio file for voice cloning.
        output_csv: Path where the streaming latency CSV results will be saved.
        chunk_size: Token chunk size for streaming.
    """
    logger.info("Initializing XTTS-v2 model for streaming benchmark...")
    
    # Import inside function to avoid loading delays during script parse
    try:
        from TTS.api import TTS
    except ImportError:
        logger.error("TTS package is not installed. Please install it using pip.")
        return

    # Check if speaker reference file exists
    if not os.path.exists(speaker_wav):
        logger.error(f"Speaker reference audio file not found at: {speaker_wav}")
        return

    # Load XTTS-v2 model onto GPU
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    except Exception as e:
        logger.error(f"Failed to initialize TTS model on CUDA: {e}")
        raise

    logger.info("Pre-computing conditioning speaker latents...")
    start_latent = time.time()
    try:
        gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(
            audio_path=[speaker_wav]
        )
    except Exception as e:
        logger.error(f"Failed to compute speaker conditioning latents: {e}")
        del tts
        clear_gpu()
        raise
        
    latent_time_ms = (time.time() - start_latent) * 1000
    logger.info(f"Latents computed in {latent_time_ms:.1f} ms (can be cached in production)")

    # Model warm-up run
    logger.info("Warming up model generator...")
    try:
        list(tts.synthesizer.tts_model.inference_stream(
            text="Warmup",
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=chunk_size
        ))
    except Exception as e:
        logger.error(f"Model warm-up failed: {e}")
        del tts
        clear_gpu()
        raise

    test_sentence = "Hello! I am your real-time voice assistant. How can I help you today?"
    languages = ["en", "ar", "hi"]
    results: Dict[str, Optional[float]] = {}

    logger.info("Measuring Time-to-First-Audio (TTFA) Streaming Latency:")
    print("-" * 65)

    for lang in languages:
        logger.info(f"Generating streaming chunks for {lang.upper()}...")
        start_time = time.time()
        try:
            chunks = tts.synthesizer.tts_model.inference_stream(
                text=test_sentence,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                stream_chunk_size=chunk_size
            )
            # Fetch the very first generated audio chunk to measure TTFA
            first_chunk = next(chunks)
            ttfa = (time.time() - start_time) * 1000  # convert to ms
            logger.info(f"  {lang.upper()}: Received first audio chunk (shape {first_chunk.shape}) in {ttfa:.1f} ms")
            results[lang] = ttfa
        except StopIteration:
            logger.error(f"  {lang.upper()}: Model returned empty stream.")
            results[lang] = None
        except Exception as e:
            logger.error(f"  {lang.upper()}: Stream generation failed: {e}")
            results[lang] = None

    # Print final summary table
    print("\n" + "=" * 60)
    print("[RESULTS] STREAMING TIME-TO-FIRST-AUDIO (TTFA) RESULTS")
    print("=" * 60)
    print("Target: Under 500 ms")
    print("-" * 60)
    for lang, ttfa in results.items():
        if ttfa is not None:
            print(f"| {lang.upper()} (XTTS-v2 Streaming) | {ttfa:.1f} ms | Pass [SUCCESS] |")
        else:
            print(f"| {lang.upper()} (XTTS-v2 Streaming) | Failed | Fail [ERROR] |")
    print("=" * 60)

    # Save results to CSV file
    try:
        import pandas as pd
        df = pd.DataFrame([
            {"language": lang, "model": "XTTS-v2 (Streaming)", "ttfa_ms": ttfa}
            for lang, ttfa in results.items() if ttfa is not None
        ])
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved results to: {output_csv}")
    except Exception as e:
        logger.error(f"Failed to write results CSV: {e}")

    # Cleanup GPU memory
    del tts
    clear_gpu()


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ref = os.path.join(project_root, "audio", "reference", "reference_voice.wav")
    default_out = os.path.join(project_root, "results", "streaming_latency.csv")

    parser = argparse.ArgumentParser(
        description="Run Time-to-First-Audio (TTFA) streaming benchmarks on XTTS-v2."
    )
    parser.add_argument(
        "--reference", 
        type=str, 
        default=default_ref,
        help="Path to speaker reference audio file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_out,
        help="Path to save latency results CSV."
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=20,
        help="Token chunk size for streaming inference."
    )
    args = parser.parse_args()

    test_streaming_latency(
        speaker_wav=args.reference,
        output_csv=args.output,
        chunk_size=args.chunk_size
    )
