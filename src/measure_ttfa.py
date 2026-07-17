import os
import sys
import argparse
import time
import torch
import numpy as np

def measure_ttfa():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model.lower()
    lang = args.lang.lower()

    text_map = {
        "en": "Welcome to our customer support line. How may I help you today?",
        "ar": "مرحبا بكم في خدمة العملاء. كيف يمكنني مساعدتكم اليوم؟",
        "hi": "नमस्ते, आप कैसे हैं?"
    }
    text = text_map.get(lang, "Hello.")
    ref_wav = "audio/reference/reference_voice.wav"

    torch.cuda.empty_cache()
    
    ttfa_ms = None

    if model_name == "xtts_v2":
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=[ref_wav])
        
        start = time.time()
        chunks = tts.synthesizer.tts_model.inference_stream(
            text=text,
            language="en" if lang=="en" else ("ar" if lang=="ar" else "hi"),
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding
        )
        first_chunk = next(chunks)
        ttfa_ms = (time.time() - start) * 1000

    elif model_name == "kokoro":
        import kokoro
        voice = "af_heart" if lang == "en" else "hf_alpha"
        pipeline = kokoro.KPipeline(lang_code=voice[0])
        
        start = time.time()
        generator = pipeline(text, voice=voice, speed=1.0)
        gs, ps, audio = next(generator)
        ttfa_ms = (time.time() - start) * 1000

    elif model_name == "f5_tts":
        from f5_tts.api import F5TTS
        f5 = F5TTS()
        # F5-TTS fallback phonetic representations if needed, but for TTFA we just pass the text
        start = time.time()
        wav, sr, spect = f5.infer(
            ref_file=ref_wav,
            ref_text="Welcome to our customer support line.",
            gen_text=text,
        )
        ttfa_ms = (time.time() - start) * 1000

    elif model_name == "openvoice":
        if lang != "en":
            print("N/A (OpenVoice only supports English)")
            sys.exit(0)
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS as MeloTTS
        
        ckpt_converter = "checkpoints_v2/converter"
        tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device="cuda")
        tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
        
        target_se, _ = se_extractor.get_se(ref_wav, tone_color_converter, vad=False)
        base_tts = MeloTTS(language="EN", device="cuda")
        speaker_ids = base_tts.hps.data.spk2id
        src_path = "results/openvoice_tmp_measure.wav"
        
        start = time.time()
        base_tts.tts_to_file(text, speaker_ids["EN-US"], src_path, speed=1.0)
        source_se, _ = se_extractor.get_se(src_path, tone_color_converter, vad=False)
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path="results/openvoice_out_measure.wav",
        )
        ttfa_ms = (time.time() - start) * 1000
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists("results/openvoice_out_measure.wav"):
            os.remove("results/openvoice_out_measure.wav")

    elif model_name == "bark":
        from bark import generate_audio, preload_models
        preload_models()
        speaker = "v2/en_speaker_6" if lang in ["en", "ar"] else "v2/hi_speaker_0"
        
        start = time.time()
        audio = generate_audio(text, history_prompt=speaker)
        ttfa_ms = (time.time() - start) * 1000

    elif model_name == "mms":
        from transformers import VitsModel, AutoTokenizer
        mms_id = "facebook/mms-tts-ara" if lang == "ar" else "facebook/mms-tts-hin"
        model = VitsModel.from_pretrained(mms_id).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(mms_id)
        
        start = time.time()
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs["input_ids"].to("cuda"))
        ttfa_ms = (time.time() - start) * 1000

    elif model_name == "chatterbox":
        if lang != "en":
            print("N/A (Chatterbox only supports English)")
            sys.exit(0)
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device="cuda")
        
        start = time.time()
        audio = model.synthesize(text, voice_path=ref_wav)
        ttfa_ms = (time.time() - start) * 1000

    elif model_name == "indic_parler":
        if lang != "hi":
            print("N/A (Indic-Parler only supports Hindi)")
            sys.exit(0)
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        
        prompt = "A female speaker with a clear voice."
        
        start = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        ttfa_ms = (time.time() - start) * 1000

    else:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    if ttfa_ms is not None:
        print(f"RESULT:{ttfa_ms:.3f}")

if __name__ == "__main__":
    measure_ttfa()
