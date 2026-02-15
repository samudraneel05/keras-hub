"""Qwen-Image-Edit model preset configurations."""

# TODO(presets): These presets are placeholders only.
# Cannot be loaded until Qwen2.5-VL and VAE are implemented.

presets = {
    "qwen_image_edit_2511": {
        "metadata": {
            "description": (
                "Qwen-Image-Edit-2511 model for advanced image editing. "
                "Features improved character consistency, multi-image "
                "composition, integrated LoRA capabilities, and enhanced "
                "geometric reasoning."
            ),
            "params": 4_000_000_000,
            "official_name": "Qwen-Image-Edit-2511",
            "path": "qwen_image_edit",
            "model_card": "https://huggingface.co/Qwen/Qwen-Image-Edit-2511",
        },
        "kaggle_handle": "kaggle://qwen/qwen-image-edit/keras/qwen_image_edit_2511",
    },
    "qwen_image_edit_2509": {
        "metadata": {
            "description": (
                "Qwen-Image-Edit-2509 model with multi-image editing "
                "support. Capable of editing multiple images simultaneously "
                "with various combinations like person+person, "
                "person+product, person+scene."
            ),
            "params": 4_000_000_000,
            "official_name": "Qwen-Image-Edit-2509",
            "path": "qwen_image_edit",
            "model_card": "https://huggingface.co/Qwen/Qwen-Image-Edit-2509",
        },
        "kaggle_handle": "kaggle://qwen/qwen-image-edit/keras/qwen_image_edit_2509",
    },
}
