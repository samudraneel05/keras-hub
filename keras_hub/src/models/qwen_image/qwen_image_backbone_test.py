import pytest

from keras_hub.src.models.qwen_image.qwen_image_backbone import (
    QwenImageBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class QwenImageBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "image_size": 256,
            "patch_size": 2,
            "latent_channels": 4,
            "text_encoder_dim": 256,
        }
        self.input_data = {
            "latents": self.random_uniform(shape=(2, 32, 32, 4)),
            "text_embeddings": self.random_uniform(shape=(2, 10, 256)),
            "timesteps": self.random_integer(shape=(2,), minval=0, maxval=1000),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=QwenImageBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 32, 32, 4),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=QwenImageBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
