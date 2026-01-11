import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import tv_tensors

from binseg.model.binary_segmentation import BinarySegmentationTrainer
from binseg.data.preprocess import get_transforms
from binseg.config import IMGSZ

def run_inference(image_path: str, checkpoint_path: str, output_path: str = "result.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_module = BinarySegmentationTrainer.load_from_checkpoint(checkpoint_path)
    trainer_module.to(device)
    trainer_module.eval()

    img_pil = Image.open(image_path).convert("RGB")
    img_v2 = tv_tensors.Image(img_pil)
    transform = get_transforms(IMGSZ, train=False)
    
    with torch.no_grad():
        img_t = transform(img_v2)
        img_t = img_t.unsqueeze(0).to(device)

        logits = trainer_module(img_t)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    pred_np = preds.squeeze().cpu().numpy()
    pred_np = (pred_np * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, pred_np)
    print(f"Result saved in: {output_path}")

if __name__ == "__main__":
    IMG_PATH = "/home/shared/angio-binary-dataset/images/1.png"
    CKPT_PATH = "/home/hubert/binary-segmentation-angio/checkpoints/binary_segmentation-epoch-epoch=99-v1.ckpt"
    
    run_inference(IMG_PATH, CKPT_PATH)