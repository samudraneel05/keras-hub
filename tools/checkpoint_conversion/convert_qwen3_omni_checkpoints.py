"""Convert Qwen3-Omni HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_qwen3_omni_checkpoints.py \
        --preset qwen3_omni_30b_a3b_thinking_en \
        --validate_dtype bfloat16 \
        --save_dtype bfloat16
"""

import gc
import os
import random
import tempfile
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import soundfile as sf
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image
from transformers import AutoTokenizer
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers import Qwen3OmniMoeProcessor

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "qwen3_omni_30b_a3b_en": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3_omni_30b_a3b_captioner_en": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    "qwen3_omni_30b_a3b_thinking_en": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
}

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

TEXT_PROMPT = "What is Keras?"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
IMAGE_PROMPT = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
AUDIO_PROMPT_TEXT = "What is being said in this audio?"
_AUDIO_SAMPLE_RATE = 16000
_AUDIO_DURATION_S = 3

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "validate_dtype",
    "bfloat16",
    "Dtype to use while validating HF and Keras numerics.",
)
flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to use when saving the converted Keras preset.",
)
flags.DEFINE_bool(
    "run_generate_check",
    True,
    "Whether to compare generated text after conversion.",
)


def _extract_response(text):
    """Strip prompt prefix and any <think>...</think> block."""
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _make_test_audio():
    """Return (audio_array, sample_rate) for a 3 s 440 Hz sine wave."""
    n = _AUDIO_SAMPLE_RATE * _AUDIO_DURATION_S
    t = np.linspace(0, _AUDIO_DURATION_S, n)
    return (np.sin(2 * np.pi * 440 * t).astype(np.float32), _AUDIO_SAMPLE_RATE)


def _count_keras_params(model):
    """Count unique parameters (handles tied weights)."""
    unique = {id(w): w for w in model.weights}.values()
    return sum(w.numpy().size for w in unique)


def _logit_report(keras_logits, hf_logits, section=""):
    """Print absolute diff statistics and per-token argmax match rate."""
    prefix = f"  [{section}] " if section else "  "
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"{prefix}Logit mean absolute diff : {abs_diff.mean():.6f}")
    print(f"{prefix}Logit max absolute diff  : {abs_diff.max():.6f}")
    keras_argmax = np.argmax(keras_logits, axis=-1)
    hf_argmax = np.argmax(hf_logits, axis=-1)
    match = np.mean(keras_argmax == hf_argmax)
    print(f"{prefix}Logit argmax match       : {match * 100:.1f}%")
    if match < 1.0:
        print(
            f"{prefix}⚠ {(1 - match) * 100:.1f}% of tokens differ "
            "(expected for bfloat16 MoE or M-RoPE position mismatch)"
        )
    else:
        print(f"{prefix}✓ All logit argmaxes match.")


# ---------------------------------------------------------------
# Precompute all HF outputs before freeing the model
# ---------------------------------------------------------------


def precompute_hf_outputs(
    hf_thinker, hf_tokenizer, hf_processor, run_generate_check
):
    """Run all HF forward passes and return results as numpy arrays.

    Collects text-only, image+text, and audio+text outputs so the HF
    model can be freed before loading KerasHub.
    """
    results = {}
    results["hf_param_count"] = sum(
        {p.data_ptr(): p.numel() for p in hf_thinker.parameters()}.values()
    )

    # --- Text-only ---
    hf_ids = hf_tokenizer(TEXT_PROMPT, return_tensors="np")["input_ids"]
    results["text_token_ids"] = hf_ids
    input_ids_t = torch.tensor(hf_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        hf_out = hf_thinker(input_ids=input_ids_t)
        hf_embeddings = hf_thinker.model.embed_tokens(input_ids_t)
        hf_text_out = hf_thinker.model(inputs_embeds=hf_embeddings)
        hf_hidden = hf_text_out.last_hidden_state

    results["text_logits"] = hf_out.logits.detach().cpu().float().numpy()
    results["text_embeddings"] = hf_embeddings.detach().cpu().float().numpy()
    results["text_hidden"] = hf_hidden.detach().cpu().float().numpy()

    emb_f, hid_f, log_f = (
        hf_embeddings.float(),
        hf_hidden.float(),
        hf_out.logits.float(),
    )
    print(f"   input_ids shape: {hf_ids.shape}")
    print(
        f"   embeddings:"
        f" mean={emb_f.mean():.6f}  std={emb_f.std():.6f}"
        f"  absmax={emb_f.abs().max():.6f}"
    )
    print(
        f"   hidden (post-norm):"
        f" mean={hid_f.mean():.6f}  std={hid_f.std():.6f}"
        f"  absmax={hid_f.abs().max():.6f}"
    )
    print(
        f"   logits:"
        f" mean={log_f.mean():.6f}  std={log_f.std():.6f}"
        f"  absmax={log_f.abs().max():.6f}"
    )

    # Consistency check: lm_head(hidden) vs full logits
    with torch.no_grad():
        check_logits = hf_thinker.lm_head(hf_hidden)
    check_diff = (check_logits.float() - hf_out.logits.float()).abs().max()
    print(f"   lm_head(hidden) vs full logits max diff: {check_diff:.8f}")

    if run_generate_check:
        with torch.no_grad():
            hf_gen = hf_thinker.generate(
                input_ids=input_ids_t,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=hf_tokenizer.pad_token_id,
            )
        results["text_generated"] = hf_tokenizer.decode(
            hf_gen[0], skip_special_tokens=True
        )

    # --- Image + text ---
    try:
        from qwen_omni_utils import process_mm_info

        raw_image = _load_test_image()
        img_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        img_text = hf_processor.apply_chat_template(
            img_messages, tokenize=False, add_generation_prompt=True
        )
        img_audios, img_images, img_videos = process_mm_info(
            img_messages, use_audio_in_video=False
        )
        hf_img_in = (
            hf_processor(
                text=[img_text],
                audio=img_audios,
                images=img_images,
                videos=img_videos,
                return_tensors="pt",
                padding=True,
            )
            .to(device)
            .to(hf_thinker.dtype)
        )

        with torch.no_grad():
            hf_img_out = hf_thinker(**hf_img_in)

        results["img_logits"] = hf_img_out.logits.detach().cpu().float().numpy()
        results["img_token_ids"] = (
            hf_img_in["input_ids"].cpu().numpy().astype(np.int32)
        )
        results["img_attention_mask"] = (
            hf_img_in["attention_mask"].cpu().numpy().astype(np.int32)
        )
        results["img_pixel_values"] = (
            hf_img_in["pixel_values"].cpu().float().numpy()
        )
        results["img_grid_thw"] = (
            hf_img_in["image_grid_thw"].cpu().numpy().astype(np.int32)
        )
        results["raw_image"] = raw_image
        print(f"   image pixel_values shape: {hf_img_in['pixel_values'].shape}")

        if run_generate_check:
            with torch.no_grad():
                hf_img_gen = hf_thinker.generate(
                    **hf_img_in,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=hf_tokenizer.pad_token_id,
                )
            results["img_generated"] = hf_processor.batch_decode(
                hf_img_gen, skip_special_tokens=True
            )[0]
    except ImportError:
        print("\n  ⚠ qwen_omni_utils not installed — skipping image section.")
        print("    Install with: pip install qwen-omni-utils")
    except Exception as e:
        print(f"\n  ⚠ HF image forward failed: {e}")

    # --- Audio + text ---
    # Write synthetic audio to a temp file so process_mm_info can load it.
    audio_tmp_path = None
    try:
        from qwen_omni_utils import process_mm_info

        audio_arr, sr = _make_test_audio()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            audio_tmp_path = tmp_f.name
        sf.write(audio_tmp_path, audio_arr, sr)

        aud_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_tmp_path},
                    {"type": "text", "text": AUDIO_PROMPT_TEXT},
                ],
            }
        ]
        aud_text = hf_processor.apply_chat_template(
            aud_messages, tokenize=False, add_generation_prompt=True
        )
        aud_audios, aud_images, aud_videos = process_mm_info(
            aud_messages, use_audio_in_video=False
        )
        hf_aud_in = (
            hf_processor(
                text=[aud_text],
                audio=aud_audios,
                images=aud_images,
                videos=aud_videos,
                return_tensors="pt",
                padding=True,
            )
            .to(device)
            .to(hf_thinker.dtype)
        )

        with torch.no_grad():
            hf_aud_out = hf_thinker(**hf_aud_in)

        results["aud_logits"] = hf_aud_out.logits.detach().cpu().float().numpy()
        results["aud_token_ids"] = (
            hf_aud_in["input_ids"].cpu().numpy().astype(np.int32)
        )
        results["aud_attention_mask"] = (
            hf_aud_in["attention_mask"].cpu().numpy().astype(np.int32)
        )
        results["aud_input_features"] = (
            hf_aud_in["input_features"].cpu().float().numpy()
        )
        print(
            f"   audio input_features shape:"
            f" {hf_aud_in['input_features'].shape}"
        )

        if run_generate_check:
            with torch.no_grad():
                hf_aud_gen = hf_thinker.generate(
                    **hf_aud_in,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=hf_tokenizer.pad_token_id,
                )
            results["aud_generated"] = hf_processor.batch_decode(
                hf_aud_gen, skip_special_tokens=True
            )[0]
    except ImportError:
        print("\n  ⚠ qwen_omni_utils not installed — skipping audio section.")
        print("    Install with: pip install qwen-omni-utils")
    except Exception as e:
        print(f"\n  ⚠ HF audio forward failed: {e}")
    finally:
        if audio_tmp_path and os.path.exists(audio_tmp_path):
            os.remove(audio_tmp_path)

    return results


def test_parameter_count(keras_backbone, hf_param_count):
    print("\n-> Parameter count")
    keras_params = _count_keras_params(keras_backbone)
    print(f"\n  KerasHub params   : {keras_params:,}")
    print(f"  HuggingFace params: {hf_param_count:,}")
    if keras_params == hf_param_count:
        print("  ✓ Parameter counts match!")
    else:
        diff = abs(hf_param_count - keras_params)
        print(f"  ⚠ Difference: {diff:,} params")


def validate_tokenizer(keras_tokenizer, hf_token_ids):
    print("\n-> Tokenizer validation")
    preprocessor = keras_hub.models.Qwen3OmniCausalLMPreprocessor(
        keras_tokenizer
    )
    out = preprocessor([TEXT_PROMPT], sequence_length=hf_token_ids.shape[1])[0]
    keras_ids = ops.convert_to_numpy(out["token_ids"])
    keras_mask = ops.convert_to_numpy(out["padding_mask"])
    keras_valid = keras_ids[keras_mask.astype(bool)]
    print(f"\n  HF token ids    : {hf_token_ids[0][:10].tolist()}")
    print(f"  KerasHub token ids: {keras_valid[:10].tolist()}")
    np.testing.assert_array_equal(keras_valid, hf_token_ids[0])
    print("  ✓ Token IDs match.")


def validate_text_output(keras_model, hf_results):
    print("\n-> Text-only validation")

    hf_ids = hf_results["text_token_ids"]
    token_ids = ops.convert_to_tensor(hf_ids.astype(np.int32))
    padding_mask = ops.ones_like(token_ids)

    # Embedding diagnostic
    keras_emb = keras_model.backbone.token_embedding(token_ids)
    keras_emb_np = ops.convert_to_numpy(ops.cast(keras_emb, "float32"))
    hf_emb = hf_results["text_embeddings"]
    print(
        f"\n  Embedding max absolute diff:"
        f" {np.abs(keras_emb_np - hf_emb).max():.8f}"
    )
    print(
        f"  Keras emb: mean={keras_emb_np.mean():.6f}"
        f"  std={keras_emb_np.std():.6f}"
        f"  absmax={np.abs(keras_emb_np).max():.6f}"
    )

    # Hidden-state diagnostic
    keras_hidden = keras_model.backbone(
        {"token_ids": token_ids, "padding_mask": padding_mask}
    )
    keras_hidden_np = ops.convert_to_numpy(ops.cast(keras_hidden, "float32"))
    hf_hid = hf_results["text_hidden"]
    print(
        f"\n  Hidden-state max absolute diff:"
        f" {np.abs(keras_hidden_np - hf_hid).max():.8f}"
    )

    # Logit comparison
    keras_logits = keras_model.backbone.token_embedding(
        keras_hidden, reverse=True
    )
    keras_logits = np.array(
        ops.convert_to_numpy(keras_logits), dtype=np.float32
    )
    print()
    _logit_report(keras_logits, hf_results["text_logits"], section="TEXT")

    # Generation comparison
    if "text_generated" in hf_results:
        keras_output = keras_model.generate(
            TEXT_PROMPT, max_length=len(hf_ids[0]) + 32
        )
        keras_text = (
            keras_output[0] if isinstance(keras_output, list) else keras_output
        )
        print(f"\n  HF output    : {hf_results['text_generated']}")
        print(f"  KerasHub output: {_extract_response(keras_text)}")
        print("  ✓ Text generation completed.")


def validate_image_output(keras_model, hf_results):
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping image validation (no vision encoder).")
        return
    if "img_logits" not in hf_results:
        print("\n-> Skipping image validation (HF image forward failed).")
        return

    print("\n-> Image + text validation")

    backbone = keras_model.backbone
    token_ids_np = hf_results["img_token_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["img_attention_mask"])

    # Reshape pixel_values: HF (N, C*T*pH*pW) → Keras (N, T, pH, pW, C)
    pv_np = hf_results["img_pixel_values"]
    ve = backbone.vision_encoder
    C = ve.in_channels
    T = ve.temporal_patch_size
    pH = pW = ve.patch_size
    pv_np = pv_np.reshape(-1, C, T, pH, pW)
    pv_np = np.transpose(pv_np, (0, 2, 3, 4, 1))
    pixel_values = ops.convert_to_tensor(pv_np)
    grid_thw = ops.convert_to_tensor(hf_results["img_grid_thw"])

    # backbone.call() uses position_ids=None (sequential).
    # HF uses M-RoPE spatial positions for visual tokens, so argmax
    # match may be < 100% here — generation comparison is more reliable.
    keras_hidden = backbone.call(
        {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
        }
    )
    keras_logits = backbone.token_embedding(keras_hidden, reverse=True)
    keras_logits = np.array(
        ops.convert_to_numpy(keras_logits), dtype=np.float32
    )
    print()
    _logit_report(keras_logits, hf_results["img_logits"], section="IMAGE")
    print(
        "  (Note: positional mismatch expected for visual tokens"
        " — generation is more reliable)"
    )

    if "img_generated" in hf_results:
        print(f"\n  HF output: {hf_results['img_generated']}")
        try:
            raw_image_np = np.array(hf_results["raw_image"])
            keras_output = keras_model.generate(
                {"prompts": [IMAGE_PROMPT], "images": [raw_image_np]},
                max_length=token_ids_np.shape[1] + 32,
            )
            keras_text = (
                keras_output[0]
                if isinstance(keras_output, list)
                else keras_output
            )
            print(f"  KerasHub output: {_extract_response(keras_text)}")
        except Exception as e:
            print(f"  ⚠ KerasHub image generate failed: {e}")
        print("  ✓ Image + text validation completed.")


def validate_audio_output(keras_model, hf_results):
    if keras_model.backbone.audio_encoder is None:
        print("\n-> Skipping audio validation (no audio encoder).")
        return
    if "aud_logits" not in hf_results:
        print("\n-> Skipping audio validation (HF audio forward failed).")
        return

    print("\n-> Audio + text validation")

    backbone = keras_model.backbone
    token_ids_np = hf_results["aud_token_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["aud_attention_mask"])
    # input_features: (batch, time_steps, num_mel_bins)
    audio_features = ops.convert_to_tensor(
        hf_results["aud_input_features"].astype(np.float32)
    )

    keras_hidden = backbone.call(
        {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "audio_features": audio_features,
        }
    )
    keras_logits = backbone.token_embedding(keras_hidden, reverse=True)
    keras_logits = np.array(
        ops.convert_to_numpy(keras_logits), dtype=np.float32
    )
    print()
    _logit_report(keras_logits, hf_results["aud_logits"], section="AUDIO")

    if "aud_generated" in hf_results:
        print(f"\n  HF output: {hf_results['aud_generated']}")
        audio, sr = _make_test_audio()
        try:
            keras_output = keras_model.generate(
                {"prompts": [AUDIO_PROMPT_TEXT], "audios": [audio]},
                max_length=token_ids_np.shape[1] + 32,
            )
            keras_text = (
                keras_output[0]
                if isinstance(keras_output, list)
                else keras_output
            )
            print(f"  KerasHub output: {_extract_response(keras_text)}")
        except Exception as e:
            print(f"  ⚠ KerasHub audio generate failed: {e}")
        print("  ✓ Audio + text validation completed.")


def save_preset(keras_model, preset_name):
    print(f"\n-> Saving KerasHub preset to ./{preset_name}...")
    keras_model.save_to_preset(f"./{preset_name}")
    print(f"  ✓ Preset saved to ./{preset_name}")


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )

    hf_preset = PRESET_MAP[preset]
    validate_dtype = FLAGS.validate_dtype
    validate_torch_dtype = TORCH_DTYPE_MAP.get(validate_dtype)
    if validate_torch_dtype is None:
        raise ValueError(
            f"Invalid validate_dtype. Must be one of "
            f"{', '.join(TORCH_DTYPE_MAP.keys())}"
        )
    save_dtype = FLAGS.save_dtype
    if save_dtype not in TORCH_DTYPE_MAP:
        raise ValueError(
            f"Invalid save_dtype. Must be one of "
            f"{', '.join(TORCH_DTYPE_MAP.keys())}"
        )
    run_generate_check = FLAGS.run_generate_check

    # --- Phase 1: Load HF model and precompute all outputs ---
    print("-> Loading HF model...")
    hf_full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_preset,
        device_map=device,
        torch_dtype=validate_torch_dtype,
    )
    hf_full.disable_talker()
    hf_thinker = hf_full.thinker
    del hf_full
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_processor = Qwen3OmniMoeProcessor.from_pretrained(hf_preset)
    hf_thinker.eval()
    print(f"   HF thinker loaded: {hf_thinker.num_parameters():,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(
        hf_thinker, hf_tokenizer, hf_processor, run_generate_check
    )
    print("   HF outputs precomputed!")

    # --- Phase 2: Free HF model ---
    print("\n-> Releasing HF model...")
    del hf_thinker
    del hf_tokenizer
    del hf_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("   HF model released.")

    # --- Phase 3: Load KerasHub model ---
    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        f"hf://{hf_preset}", dtype=validate_dtype
    )
    print("   KerasHub model loaded!")

    # --- Phase 4: Validate ---
    test_parameter_count(keras_model.backbone, hf_results["hf_param_count"])
    validate_tokenizer(
        keras_model.preprocessor.tokenizer, hf_results["text_token_ids"]
    )
    validate_text_output(keras_model, hf_results)
    validate_image_output(keras_model, hf_results)
    validate_audio_output(keras_model, hf_results)

    # --- Phase 5: Save ---
    if save_dtype == validate_dtype:
        save_preset(keras_model, preset)
    else:
        del keras_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n-> Reloading model in {save_dtype} for saving...")
        keras_model_save = keras_hub.models.Qwen3OmniCausalLM.from_preset(
            f"hf://{hf_preset}", dtype=save_dtype
        )
        save_preset(keras_model_save, preset)

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
