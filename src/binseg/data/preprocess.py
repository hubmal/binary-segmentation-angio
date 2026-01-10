import torch
import numpy as np
import cv2
from torchvision import tv_tensors
from torchvision.transforms import v2
import torchvision.transforms.functional as F


class ImproveContrast(v2.Transform):
    def forward(self, img, mask):
        img_np = img.permute(1, 2, 0).numpy() if isinstance(img, torch.Tensor) else np.array(img)
        
        if img_np.ndim == 3 and img_np.shape[-1] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        img_neg = cv2.bitwise_not(img_np)
        kernel = np.ones((50, 50), np.uint8)
        mask_top = cv2.morphologyEx(img_neg, cv2.MORPH_TOPHAT, kernel)
        img_np = cv2.subtract(img_np, mask_top)
        img_np = np.clip(img_np, 0, 255)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)

        img_tensor = F.to_tensor(img_np)
        mask_tensor = (F.pil_to_tensor(mask) / 255).float()

        return tv_tensors.Image(img_tensor), tv_tensors.Mask(mask_tensor)
    
def get_transforms(img_size=(512, 512), train=True):
    if train:
        return v2.Compose([
            ImproveContrast(),
            v2.Resize(img_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        return v2.Compose([
            ImproveContrast(),
            v2.Resize(img_size),
            v2.ToDtype(torch.float32, scale=True),
        ])