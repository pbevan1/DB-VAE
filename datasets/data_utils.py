import torch
import torchvision.transforms as transforms
import os
import uuid
from typing import Optional, NamedTuple

class DatasetOutput(NamedTuple):
    image: torch.FloatTensor
    label: int
    idx: int

# Default transform
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def visualize_tensor(img_tensor: torch.Tensor):
    pil_transformer = transforms.ToPILImage()
    pil_transformer(img_tensor).show()

def save_images(torch_tensors: torch.Tensor, path_to_folder: str):
    rand_filenames = str(uuid.uuid4())[:8]
    pil_transformer = transforms.ToPILImage()
    image_folder = f"results/{path_to_folder}/debug/images/{rand_filenames}/"
    os.makedirs(image_folder, exist_ok=True)

    for i, img in enumerate(torch_tensors):
        pil_img = pil_transformer(img)
        pil_img.save(f"{image_folder}/{rand_filenames}_{i}.jpg")

    return torch_tensors
