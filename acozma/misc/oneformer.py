import numpy as np
import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

model_id = "shi-labs/oneformer_ade20k_swin_large"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = OneFormerProcessor.from_pretrained(model_id)
model = OneFormerForUniversalSegmentation.from_pretrained(model_id)


def get_oneformer_masks(image: Image.Image):
    instance_inputs = processor(
        images=image, task_inputs=["panoptic"], return_tensors="pt"
    )
    instance_outputs = model(**instance_inputs)

    predicted_instance_map = processor.post_process_instance_segmentation(
        instance_outputs,
        target_sizes=[image.size[::-1]],
    )[0]["segmentation"]

    predicted_instance_map = predicted_instance_map.cpu().numpy()

    masks_list = []
    for instance_id in np.unique(predicted_instance_map):
        masks_list.append((predicted_instance_map == instance_id).astype(np.uint8))

    return masks_list
