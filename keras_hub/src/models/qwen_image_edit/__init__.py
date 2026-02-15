# TODO(INCOMPLETE): This module is not production-ready.
# Missing critical dependencies:
# - Qwen2.5-VL vision-language model (text encoder)
# - Qwen2VLProcessor for multi-image composition
# - AutoencoderKLQwenImage (VAE for image encoding/decoding)
# - FlowMatchEulerDiscreteScheduler
# Current implementation is architectural skeleton only.
# Note: Qwen-Image-Edit uses the SAME architecture as Qwen-Image with condition injection.

from keras_hub.src.models.qwen_image_edit.qwen_image_edit_backbone import (
    QwenImageEditBackbone,
)
from keras_hub.src.models.qwen_image_edit.qwen_image_edit_image_to_image import (
    QwenImageEditImageToImage,
)
from keras_hub.src.models.qwen_image_edit.qwen_image_edit_image_to_image_preprocessor import (
    QwenImageEditImageToImagePreprocessor,
)
from keras_hub.src.models.qwen_image_edit.qwen_image_edit_presets import presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(presets, QwenImageEditBackbone)
