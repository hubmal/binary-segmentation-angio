import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation


class DeepLab(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier = segmentation.deeplabv3.DeepLabHead(2048, 1)
        if in_channels != 3:
            self.model.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels, out_channels=64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
    
    def forward(self, x):
        return self.model(x)["out"]
