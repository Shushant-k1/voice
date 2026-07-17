"""
Shared utilities for timing, audio I/O, and metrics collection.
"""
import os
import time
import json
import csv
import numpy as np
import soundfile as sf
from functools import wraps

# Patch torch.load for PyTorch 2.6+ compatibility with Coqui TTS
try:
    import torch
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
except ImportError:
    pass

# Patch transformers.pytorch_utils.isin_mps_friendly for Coqui TTS streaming compatibility
try:
    import torch
    import transformers.pytorch_utils
    original_isin_mps_friendly = transformers.pytorch_utils.isin_mps_friendly
    def patched_isin_mps_friendly(elements, test_elements):
        if not isinstance(elements, torch.Tensor):
            elements = torch.tensor(elements)
        if not isinstance(test_elements, torch.Tensor):
            test_elements = torch.tensor(test_elements)
        if elements.device != test_elements.device:
            test_elements = test_elements.to(elements.device)
        return original_isin_mps_friendly(elements, test_elements)
    transformers.pytorch_utils.isin_mps_friendly = patched_isin_mps_friendly
except (ImportError, AttributeError):
    pass

# Patch StreamGenerationConfig for newer transformers compatibility in XTTS-v2 streaming
try:
    from TTS.tts.layers.xtts.stream_generator import StreamGenerationConfig
    StreamGenerationConfig._eos_token_tensor = None
    StreamGenerationConfig._bos_token_tensor = None
    StreamGenerationConfig._pad_token_tensor = None
except ImportError:
    pass

# Patch GPT2InferenceModel for newer transformers compatibility in XTTS-v2 streaming
try:
    from TTS.tts.layers.xtts.gpt import GPT2InferenceModel
    from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    
    def patched_get_logits_warper(self, generation_config):
        warpers = LogitsProcessorList()
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))
        return warpers
        
    GPT2InferenceModel._get_logits_warper = patched_get_logits_warper
except ImportError:
    pass

# Patch GenerationMixin._update_model_kwargs_for_generation to handle missing cache_position
try:
    from transformers.generation.utils import GenerationMixin
    original_update_kwargs = GenerationMixin._update_model_kwargs_for_generation
    
    def patched_update_kwargs(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1, **kwargs):
        if "cache_position" not in model_kwargs:
            import torch
            device = "cuda"
            seq_len = 0
            if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
                device = model_kwargs["attention_mask"].device
                seq_len = model_kwargs["attention_mask"].shape[-1]
            elif "input_ids" in model_kwargs and model_kwargs["input_ids"] is not None:
                device = model_kwargs["input_ids"].device
                seq_len = model_kwargs["input_ids"].shape[-1]
            elif hasattr(outputs, "device"):
                device = outputs.device
            model_kwargs["cache_position"] = torch.arange(seq_len, device=device)
        return original_update_kwargs(self, outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens, **kwargs)
        
    GenerationMixin._update_model_kwargs_for_generation = patched_update_kwargs
except ImportError:
    pass







# Audio helper functions

def save_audio(audio_data, path, sample_rate=24000):
    """Save audio tensor/numpy array to wav file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(audio_data, 'numpy'):
        audio_data = audio_data.numpy()
    if hasattr(audio_data, 'cpu'):
        audio_data = audio_data.cpu().numpy()
    # Flatten if needed
    if audio_data.ndim > 1:
        audio_data = audio_data.squeeze()
    sf.write(path, audio_data, sample_rate)
    return path


def get_audio_duration(path):
    """Get duration of audio file in seconds."""
    info = sf.info(path)
    return info.duration


# Timer class

class Timer:
    """Context manager for timing operations."""
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"  [{self.label}] took {self.elapsed:.3f}s")


# Metrics collector for evaluations

class BenchmarkCollector:
    """Collects and saves benchmark results across models and languages."""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def add_result(self, language, model_name, sentence_idx, text,
                   latency_s, rtf, audio_duration_s, audio_path,
                   wer=None, speaker_similarity=None, mos=None):
        self.results.append({
            "language": language,
            "model": model_name,
            "sentence_idx": sentence_idx,
            "text": text[:80],
            "latency_s": round(latency_s, 4),
            "rtf": round(rtf, 4),
            "audio_duration_s": round(audio_duration_s, 4),
            "audio_path": audio_path,
            "wer": round(wer, 4) if wer is not None else None,
            "speaker_similarity": round(speaker_similarity, 4) if speaker_similarity is not None else None,
            "mos": mos,
        })

    def save_csv(self, filename="benchmark_raw.csv"):
        path = os.path.join(self.output_dir, filename)
        if not self.results:
            print("No results to save.")
            return
        keys = self.results[0].keys()
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"[SUCCESS] Saved {len(self.results)} results to {path}")
        return path

    def save_json(self, filename="benchmark_raw.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Saved {len(self.results)} results to {path}")
        return path

    def get_summary(self):
        """Get aggregated summary per model per language."""
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            key = (r["language"], r["model"])
            groups[key].append(r)

        summary = []
        for (lang, model), entries in groups.items():
            latencies = [e["latency_s"] for e in entries]
            rtfs = [e["rtf"] for e in entries]
            wers = [e["wer"] for e in entries if e["wer"] is not None]
            sims = [e["speaker_similarity"] for e in entries if e["speaker_similarity"] is not None]
            mos_scores = [e["mos"] for e in entries if e["mos"] is not None]

            summary.append({
                "language": lang,
                "model": model,
                "n_samples": len(entries),
                "avg_latency_s": round(np.mean(latencies), 3),
                "avg_rtf": round(np.mean(rtfs), 3),
                "avg_wer": round(np.mean(wers), 3) if wers else None,
                "avg_speaker_similarity": round(np.mean(sims), 3) if sims else None,
                "avg_mos": round(np.mean(mos_scores), 2) if mos_scores else None,
            })
        return summary

    def print_summary(self):
        """Print a formatted summary table."""
        summary = self.get_summary()
        print("\n" + "=" * 90)
        print(f"{'Language':<10} {'Model':<20} {'Samples':<8} {'Latency':<10} {'RTF':<8} {'WER':<8} {'SimCos':<8} {'MOS':<6}")
        print("=" * 90)
        for s in summary:
            wer_str = f"{s['avg_wer']:.1%}" if s['avg_wer'] is not None else "—"
            sim_str = f"{s['avg_speaker_similarity']:.3f}" if s['avg_speaker_similarity'] is not None else "—"
            mos_str = f"{s['avg_mos']:.1f}" if s['avg_mos'] is not None else "—"
            print(f"{s['language']:<10} {s['model']:<20} {s['n_samples']:<8} "
                  f"{s['avg_latency_s']:<10.3f} {s['avg_rtf']:<8.3f} "
                  f"{wer_str:<8} {sim_str:<8} {mos_str:<6}")
        print("=" * 90)


# Loading input test sentences

def load_test_sentences(path="test_sentences.json"):
    """Load test sentences from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# GPU memory utilities

def clear_gpu():
    """Free GPU memory between model loads."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"  [CLEANUP] GPU memory cleared. Free: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB / {torch.cuda.mem_get_info()[1]/1e9:.1f}GB")
    except Exception as e:
        print(f"  [WARNING] GPU clear failed: {e}")


def print_gpu_status():
    """Print current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            print(f"  [GPU] {used/1e9:.1f}GB used / {total/1e9:.1f}GB total ({free/1e9:.1f}GB free)")
        else:
            print("  [WARNING] No CUDA GPU available")
    except:
        print("  [WARNING] Could not query GPU status")
