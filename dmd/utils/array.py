from typing import Union, List

import numpy as np
import PIL.Image
import torch


def numpy_to_pillow(x: np.ndarray) -> PIL.Image.Image:
    x = np.array((x * 127.5 + 128).clip(0, 255), dtype=np.uint8)
    return PIL.Image.fromarray(np.transpose(x, (1, 2, 0)), mode="RGB")


def torch_to_pillow(x: torch.Tensor, idx: int = None) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
    if x.ndim == 3:
        image_np = (x * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return PIL.Image.fromarray(image_np, mode="RGB")

    image_np = (x * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    if idx is not None:
        return PIL.Image.fromarray(image_np[idx], mode="RGB")
    return [PIL.Image.fromarray(image_np[i], mode="RGB") for i in range(image_np.shape[0])]
