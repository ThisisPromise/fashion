"""Mostly-frozen MobileNetV3-Small backbone + a small trainable FPN decoder.

Frozen backbone params don't count toward the 2M cap. Optionally the last
few backbone layers can be unfrozen (see unfreeze_last_n_backbone_layers)
to let the deepest features adapt to garment boundaries, still well within
the cap. Decoder taps three feature maps (strides 4/8/16), fuses them
top-down, and predicts background/body/sleeve at stride 4 before
upsampling to input size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

NUM_CLASSES = 3  # background, body, sleeve

# Layer indices in MobileNetV3-Small's .features Sequential to tap for the
# decoder (found by running a dummy tensor through and checking shapes).
FEATURE_TAPS = {"low": 3, "mid": 6, "high": 12}


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightDecoder(nn.Module):
    """FPN-style top-down fusion of three backbone feature maps -> logits."""

    def __init__(self, low_ch, mid_ch, high_ch, decoder_ch=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.reduce_low = nn.Conv2d(low_ch, decoder_ch, 1)
        self.reduce_mid = nn.Conv2d(mid_ch, decoder_ch, 1)
        self.reduce_high = nn.Conv2d(high_ch, decoder_ch, 1)

        self.smooth_mid = ConvBNAct(decoder_ch, decoder_ch)
        self.smooth_low = ConvBNAct(decoder_ch, decoder_ch)

        self.head = nn.Sequential(
            ConvBNAct(decoder_ch, decoder_ch),
            nn.Conv2d(decoder_ch, num_classes, 1),
        )

    def forward(self, low, mid, high):
        low = self.reduce_low(low)
        mid = self.reduce_mid(mid)
        high = self.reduce_high(high)

        mid = mid + F.interpolate(high, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        mid = self.smooth_mid(mid)

        low = low + F.interpolate(mid, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.smooth_low(low)

        return self.head(low)


class TinyGarmentSegModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, decoder_ch=32, pretrained_backbone=True,
                 unfreeze_last_n_backbone_layers=0):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features

        for p in self.features.parameters():
            p.requires_grad = False

        n_layers = len(self.features)
        self.unfrozen_layer_indices = set(
            range(n_layers - unfreeze_last_n_backbone_layers, n_layers)
        ) if unfreeze_last_n_backbone_layers else set()
        for i in self.unfrozen_layer_indices:
            for p in self.features[i].parameters():
                p.requires_grad = True

        # Channel counts read off empirically rather than hardcoded, so this
        # stays correct if torchvision's MobileNetV3 implementation changes.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self._extract_taps(dummy)
        low_ch, mid_ch, high_ch = (f.shape[1] for f in feats)

        self.decoder = LightDecoder(low_ch, mid_ch, high_ch, decoder_ch=decoder_ch, num_classes=num_classes)
        self.train(True)

    def _extract_taps(self, x):
        outputs = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == FEATURE_TAPS["low"]:
                outputs["low"] = x
            elif i == FEATURE_TAPS["mid"]:
                outputs["mid"] = x
            elif i == FEATURE_TAPS["high"]:
                outputs["high"] = x
        return outputs["low"], outputs["mid"], outputs["high"]

    def train(self, mode=True):
        super().train(mode)
        # Frozen backbone layers always stay in eval mode (fixed BN stats);
        # any explicitly unfrozen tail layers follow the requested mode.
        for i, layer in enumerate(self.features):
            layer.train(mode and i in self.unfrozen_layer_indices)
        return self

    def forward(self, x):
        input_size = x.shape[-2:]
        low, mid, high = self._extract_taps(x)
        logits = self.decoder(low, mid, high)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def backbone_trainable_parameters(self):
        return [p for i in self.unfrozen_layer_indices for p in self.features[i].parameters()]


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


if __name__ == "__main__":
    model = TinyGarmentSegModel()
    print("Trainable parameters (decoder only):", count_trainable_parameters(model))
    print("Frozen parameters (backbone):", count_frozen_parameters(model))
    x = torch.zeros(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
