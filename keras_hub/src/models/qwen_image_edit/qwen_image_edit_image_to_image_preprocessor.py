# TODO(qwen2.5-vl): This preprocessor requires Qwen2.5-VL integration
# Currently a stub implementation - needs:
# 1. Qwen2Tokenizer for tokenization
# 2. Qwen2VLProcessor for multi-image handling
# 3. Qwen2.5-VL text encoder for embedding generation

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor


@keras_hub_export("keras_hub.models.QwenImageEditImageToImagePreprocessor")
class QwenImageEditImageToImagePreprocessor(Preprocessor):
    """Preprocessor for Qwen-Image-Edit image-to-image editing.

    This preprocessor handles both image encoding and text tokenization for
    the Qwen-Image-Edit model, supporting multi-image inputs and advanced
    editing operations.

    Args:
        tokenizer: A tokenizer instance for processing text prompts.
        text_encoder: Optional text encoder model.
        image_encoder: Optional image encoder for processing reference images.

    Example:
    ```python
    preprocessor = (
        keras_hub.models.QwenImageEditImageToImagePreprocessor
        .from_preset("qwen_image_edit_2511")
    )

    # Preprocess for generation
    inputs = preprocessor.generate_preprocess("Add a red hat to the person")
    ```
    """

    def __init__(
        self,
        tokenizer=None,
        text_encoder=None,
        image_encoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def generate_preprocess(self, x):
        """Preprocess text prompts for generation.

        Args:
            x: Text prompt or list of text prompts.

        Returns:
            Tokenized and encoded text ready for the model.
        """
        if self.tokenizer is not None:
            return self.tokenizer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": self.tokenizer,
            }
        )
        return config
