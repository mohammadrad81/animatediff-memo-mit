"""Inflated 3D ConvNet (I3D, Carreira & Zisserman 2017), RGB stream, for the weights of DeepMind's Kinetics-400
model converted to PyTorch (https://github.com/piergiaj/pytorch-i3d, models/rgb_imagenet.pt: ImageNet-inflated,
trained on Kinetics-400). The module and parameter names follow that conversion, so its state dict loads as is.
Input: RGB videos [batch, 3, frames, 224, 224] in [-1, 1]."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_URL = "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"


def same_padding(x, kernel, stride):
    """TensorFlow 'SAME' padding, as in the original model (the extra pixel goes at the end)."""
    pad = []
    for size, k, s in zip(x.shape[2:], kernel, stride):
        p = max(k - s, 0) if size % s == 0 else max(k - size % s, 0)
        pad = [p // 2, p - p // 2] + pad  # F.pad wants the last dimension first
    return F.pad(x, pad)


class Unit3D(nn.Module):
    """Conv3d with 'SAME' padding, batch norm and ReLU."""

    def __init__(self, cin, cout, kernel=(1, 1, 1), stride=(1, 1, 1), bn=True, relu=True, bias=False):
        super().__init__()
        self.kernel, self.stride, self.relu = kernel, stride, relu
        self.conv3d = nn.Conv3d(cin, cout, kernel, stride, padding=0, bias=bias)
        self.bn = nn.BatchNorm3d(cout, eps=0.001, momentum=0.01) if bn else None

    def forward(self, x):
        x = self.conv3d(same_padding(x, self.kernel, self.stride))
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x) if self.relu else x


class MaxPool(nn.Module):
    def __init__(self, kernel, stride):
        super().__init__()
        self.kernel, self.stride = kernel, stride

    def forward(self, x):  # zero padding is fine: the inputs come after a ReLU
        return F.max_pool3d(same_padding(x, self.kernel, self.stride), self.kernel, self.stride)


class Inception(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.b0 = Unit3D(cin, c[0])
        self.b1a, self.b1b = Unit3D(cin, c[1]), Unit3D(c[1], c[2], (3, 3, 3))
        self.b2a, self.b2b = Unit3D(cin, c[3]), Unit3D(c[3], c[4], (3, 3, 3))
        self.b3a, self.b3b = MaxPool((3, 3, 3), (1, 1, 1)), Unit3D(cin, c[5])

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1b(self.b1a(x)), self.b2b(self.b2a(x)), self.b3b(self.b3a(x))], dim=1)


class I3D(nn.Module):
    def __init__(self, num_classes=400):
        super().__init__()
        layers = [
            ("Conv3d_1a_7x7", Unit3D(3, 64, (7, 7, 7), (2, 2, 2))),
            ("MaxPool3d_2a_3x3", MaxPool((1, 3, 3), (1, 2, 2))),
            ("Conv3d_2b_1x1", Unit3D(64, 64)),
            ("Conv3d_2c_3x3", Unit3D(64, 192, (3, 3, 3))),
            ("MaxPool3d_3a_3x3", MaxPool((1, 3, 3), (1, 2, 2))),
            ("Mixed_3b", Inception(192, [64, 96, 128, 16, 32, 32])),
            ("Mixed_3c", Inception(256, [128, 128, 192, 32, 96, 64])),
            ("MaxPool3d_4a_3x3", MaxPool((3, 3, 3), (2, 2, 2))),
            ("Mixed_4b", Inception(480, [192, 96, 208, 16, 48, 64])),
            ("Mixed_4c", Inception(512, [160, 112, 224, 24, 64, 64])),
            ("Mixed_4d", Inception(512, [128, 128, 256, 24, 64, 64])),
            ("Mixed_4e", Inception(512, [112, 144, 288, 32, 64, 64])),
            ("Mixed_4f", Inception(528, [256, 160, 320, 32, 128, 128])),
            ("MaxPool3d_5a_2x2", MaxPool((2, 2, 2), (2, 2, 2))),
            ("Mixed_5b", Inception(832, [256, 160, 320, 32, 128, 128])),
            ("Mixed_5c", Inception(832, [384, 192, 384, 48, 128, 128])),
        ]
        self.names = [n for n, _ in layers]
        for n, m in layers:
            self.add_module(n, m)
        self.logits = Unit3D(1024, num_classes, bn=False, relu=False, bias=True)

    def forward(self, x):
        """-> (features [batch, 1024], logits [batch, 400]). features: the last Inception block, averaged over
        space and time (the input of the classifier); logits: the Kinetics-400 class scores, averaged over time."""
        for n in self.names:
            x = getattr(self, n)(x)
        x = F.avg_pool3d(x, (2, 7, 7), stride=1)  # as in the original: [batch, 1024, frames/8 - 1, 1, 1]
        logits = self.logits(x).flatten(2).mean(2)
        return x.flatten(2).mean(2), logits


def load_i3d(path=None, device="cuda"):
    """The Kinetics-400 I3D from a local copy of rgb_imagenet.pt, or downloaded. Only tensors are loaded
    (weights_only), so the file cannot run code."""
    model = I3D()
    if path:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    else:
        sd = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.eval().to(device)
