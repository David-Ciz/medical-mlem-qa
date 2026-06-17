from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    FCN_ResNet101_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    fcn_resnet101,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)


MODEL_NAMES = (
    "unet",
    "deeplabv3",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
    "fcn_resnet50",
    "fcn_resnet101",
    "lraspp_mobilenet_v3_large",
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class TorchvisionSegmentationWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def _replace_classifier_head(head: nn.Sequential, in_channels: int) -> None:
    head[-1] = nn.Conv2d(in_channels, 1, kernel_size=1)


def build_model(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "unet":
        return SimpleUNet()
    if normalized == "deeplabv3":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        _replace_classifier_head(model.classifier, 256)
        return TorchvisionSegmentationWrapper(model)
    if normalized == "deeplabv3_resnet101":
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        _replace_classifier_head(model.classifier, 256)
        return TorchvisionSegmentationWrapper(model)
    if normalized == "deeplabv3_mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        _replace_classifier_head(model.classifier, 256)
        return TorchvisionSegmentationWrapper(model)
    if normalized == "fcn_resnet50":
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        _replace_classifier_head(model.classifier, 512)
        if model.aux_classifier is not None:
            _replace_classifier_head(model.aux_classifier, 256)
        return TorchvisionSegmentationWrapper(model)
    if normalized == "fcn_resnet101":
        model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)
        _replace_classifier_head(model.classifier, 512)
        if model.aux_classifier is not None:
            _replace_classifier_head(model.aux_classifier, 256)
        return TorchvisionSegmentationWrapper(model)
    if normalized == "lraspp_mobilenet_v3_large":
        model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier.low_classifier = nn.Conv2d(40, 1, kernel_size=1)
        model.classifier.high_classifier = nn.Conv2d(128, 1, kernel_size=1)
        return TorchvisionSegmentationWrapper(model)
    raise ValueError(f"Unknown model: {name}")
