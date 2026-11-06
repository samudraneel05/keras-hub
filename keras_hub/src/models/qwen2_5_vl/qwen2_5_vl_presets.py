"""Qwen2.5-VL preset configurations."""

backbone_presets = {
    "qwen2_5_vl_3b_instruct": {
        "metadata": {
            "description": (
                "3 billion parameter Qwen2.5-VL vision-language model. "
                "Instruction-tuned for image and video understanding with "
                "window-attention vision encoder and multimodal rotary "
                "position embeddings."
            ),
            "params": 3754625024,
            "path": "qwen2_5_vl",
        },
        "kaggle_handle": "kaggle://keras/qwen2_5_vl/keras/qwen2_5_vl_3b_instruct/1",
    },
}

# backbone_presets["qwen2_5_vl_7b_instruct"] = {
#     "metadata": {
#         "description": (
#             "7 billion parameter Qwen2.5-VL vision-language model. "
#             "Instruction-tuned for image and video understanding."
#         ),
#         "params": 8291361792,
#         "path": "qwen2_5_vl",
#     },
#     "kaggle_handle": "kaggle://keras/qwen2_5_vl/keras/qwen2_5_vl_7b_instruct/1",
# }
# backbone_presets["qwen2_5_vl_32b_instruct"] = {
#     "metadata": {
#         "description": (
#             "32 billion parameter Qwen2.5-VL vision-language model. "
#             "Instruction-tuned for image and video understanding."
#         ),
#         "params": 33107603456,
#         "path": "qwen2_5_vl",
#     },
#     "kaggle_handle": "kaggle://keras/qwen2_5_vl/keras/qwen2_5_vl_32b_instruct/1",
# }
# backbone_presets["qwen2_5_vl_72b_instruct"] = {
#     "metadata": {
#         "description": (
#             "72 billion parameter Qwen2.5-VL vision-language model. "
#             "Instruction-tuned for image and video understanding."
#         ),
#         "params": 73405560320,
#         "path": "qwen2_5_vl",
#     },
#     "kaggle_handle": "kaggle://keras/qwen2_5_vl/keras/qwen2_5_vl_72b_instruct/1",
# }
