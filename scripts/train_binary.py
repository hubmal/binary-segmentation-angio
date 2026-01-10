from binseg.train import train_with_clearml
from binseg.model.binary_segmentation import BinarySegmentationTrainer
from binseg.data.lightning_binary import ICABinaryDataModule

train_with_clearml(
    "binary_segmentation",
    BinarySegmentationTrainer(lr=1e-4),
    ICABinaryDataModule(root="/home/shared/angio-binary-dataset")
)