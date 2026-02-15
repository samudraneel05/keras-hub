"""Qwen-Image model preset configurations."""

# TODO(presets): These presets are placeholders only.
# Cannot be loaded until Qwen2.5-VL and VAE are implemented.

presets = {
    "qwen_image_base": {
        "metadata": {
            "description": (
                "Qwen-Image base model for high-quality text-to-image "
                "generation. Excels at rendering complex text and supports "
                "various artistic styles."
            ),
            "params": 4_000_000_000,
            "official_name": "Qwen-Image",
            "path": "qwen_image",
            "model_card": "https://huggingface.co/Qwen/Qwen-Image",
        },
        "kaggle_handle": "kaggle://qwen/qwen-image/keras/qwen_image_base",
    },
}
