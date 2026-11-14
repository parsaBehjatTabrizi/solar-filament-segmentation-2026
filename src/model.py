"""U-Net with a torchvision ResNet encoder (final model: ResNet-50, ImageNet initialisation when training)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ENCODER_SPECS = {
    # constructor, ImageNet weights used for initialisation when training, channels of the 5 encoder stages (/2, /4, /8, /16, /32)
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V1, [64, 256, 512, 1024, 2048]),
    "resnet101": (torchvision.models.resnet101, torchvision.models.ResNet101_Weights.IMAGENET1K_V1, [64, 256, 512, 1024, 2048]),
}


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class ResNetUNet(nn.Module):
    """Input (B, 1, H, W) in [0, 1] -> logits (B, 1, H, W).

    The grayscale image is repeated to 3 channels and ImageNet-normalised inside the model.
    `pretrained=True` downloads the public ImageNet weights (training only); for inference use pretrained=False and `load_weights`.
    """

    def __init__(self, encoder: str = "resnet50", pretrained: bool = False):
        super().__init__()
        ctor, weights, c = ENCODER_SPECS[encoder]
        r = ctor(weights=weights if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.d4 = ConvBlock(c[4] + c[3], 256)
        self.d3 = ConvBlock(256 + c[2], 128)
        self.d2 = ConvBlock(128 + c[1], 64)
        self.d1 = ConvBlock(64 + c[0], 64)
        self.d0 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def encoder_parameters(self):
        for mod in (self.stem, self.l1, self.l2, self.l3, self.l4):
            yield from mod.parameters()

    def decoder_parameters(self):
        for mod in (self.d4, self.d3, self.d2, self.d1, self.d0, self.out):
            yield from mod.parameters()

    def forward(self, x):
        x3 = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        s0 = self.stem(x3)                       # stride 2
        s1 = self.l1(self.pool(s0))              # stride 4
        s2 = self.l2(s1)                         # stride 8
        s3 = self.l3(s2)                         # stride 16
        s4 = self.l4(s3)                         # stride 32
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        d = self.d4(torch.cat([up(s4, s3), s3], 1))
        d = self.d3(torch.cat([up(d, s2), s2], 1))
        d = self.d2(torch.cat([up(d, s1), s1], 1))
        d = self.d1(torch.cat([up(d, s0), s0], 1))
        d = self.d0(F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.out(d)


def load_weights(model: nn.Module, path, device="cpu") -> nn.Module:
    """Load a state dict (half-precision tensors are converted to float32) and switch to eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
