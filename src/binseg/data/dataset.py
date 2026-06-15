import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union, List
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2


class ICABinaryDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        _set=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root)
        
        if _set is not None:
            self.input_dir = self.root / _set / 'images'
            self.label_dir = self.root / _set / 'masks'
        else:
            self.input_dir = self.root / 'images'
            self.label_dir = self.root / 'masks'

        if not self.input_dir.is_dir():
            raise RuntimeError(f"Input directory '{self.input_dir}' not found.")
        if not self.label_dir.is_dir():
            raise RuntimeError(f"Label directory '{self.label_dir}' not found.")

        filenames = os.listdir(self.input_dir)
        filenames.sort()
        self.samples = []

        for filename in filenames:
            input_path = self.input_dir / filename
            label_path = self.label_dir / filename

            if label_path.exists():
                self.samples.append((input_path, label_path))
            else:
                print(f"Warning: Corresponding label for {filename} not found at {label_path}. Skipping.")

        print(f"Found {len(self.samples)} image-label pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_path, label_path = self.samples[index]

        input_image = Image.open(input_path).convert("L")
        label_mask = Image.open(label_path).convert("L")

        if self.transforms is not None:
            input_image, label_mask = self.transforms(input_image, label_mask)

        return input_image, label_mask
    
class BinarySegmentationDataset(VisionDataset):
    def __init__(self, root, split, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root)
        self.split_dir = self.root / split

        if not self.split_dir.is_dir():
            raise RuntimeError(f"Directory not found: '{self.split_dir}'")

        self.samples = []
        all_files = os.listdir(self.split_dir)
        mask_files = [f for f in all_files if f.endswith('.mask.png')]
        
        for mask_name in mask_files:
            mask_path = self.split_dir / mask_name
            base_id = mask_name.replace('.mask.png', '')
            image_name = f"{base_id}.png"
            image_path = self.split_dir / image_name

            if image_path.exists():
                self.samples.append((image_path, mask_path))

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, label_path = self.samples[index]

        input_image = Image.open(input_path).convert("L")
        label_mask = Image.open(label_path).convert("L")

        if self.transforms is not None:
            input_image, label_mask = self.transforms(input_image, label_mask)

        return input_image, label_mask