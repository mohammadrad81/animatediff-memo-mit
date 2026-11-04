"""SSCD copy-detection descriptor (Pizzi et al., "A Self-Supervised Descriptor for Image Copy Detection", CVPR 2022),
for the Classy Vision checkpoint sscd_disc_mixup.classy.pt of https://github.com/facebookresearch/sscd-copy-detection:
ResNet-50 -> GeM pooling (p = 3) -> linear 2048 -> 512 -> L2 normalization. The checkpoint's Classy Vision names are
mapped onto torchvision's ResNet-50, and only tensors are loaded (weights_only), so the file cannot run code.
Input: RGB images [batch, 3, H, W] in [0, 1]; the official evaluation resizes the short side to 288."""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.classy.pt"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def torchvision_names(sd):
    """Classy Vision ResNet-50 names -> torchvision ResNet-50 names."""
    out = {}
    for k, v in sd.items():
        if k.startswith("embeddings."):
            out[k.replace("embeddings.1.", "proj.")] = v
            continue
        k = k.replace("backbone.initial_block._module.0.", "trunk.conv1.")
        k = k.replace("backbone.initial_block._module.1.", "trunk.bn1.")
        k = re.sub(r"backbone\.blocks\.(\d)\.block\d-(\d+)\.",
                   lambda m: f"trunk.layer{int(m.group(1)) + 1}.{m.group(2)}.", k)
        for classy, tv in [("convolutional_block.0.", "conv1."), ("convolutional_block.1.", "bn1."),
                           ("convolutional_block.3.", "conv2."), ("convolutional_block.4.", "bn2."),
                           ("convolutional_block.6.", "conv3."), (".bn.", ".bn3.")]:
            k = k.replace(classy, tv)
        out[k] = v
    return out


class SSCD(nn.Module):
    def __init__(self, gem_p=3.0):
        super().__init__()
        self.trunk = resnet50()
        self.trunk.fc = nn.Identity()
        self.trunk.avgpool = nn.Identity()  # GeM pooling instead
        self.proj = nn.Linear(2048, 512)
        self.gem_p = gem_p

    def forward(self, x):
        """x: [batch, 3, H, W] in [0, 1] -> [batch, 512] unit-length descriptors (cosine = dot product)."""
        t = self.trunk
        x = (x - x.new_tensor(MEAN)[:, None, None]) / x.new_tensor(STD)[:, None, None]
        x = t.maxpool(t.relu(t.bn1(t.conv1(x))))
        x = t.layer4(t.layer3(t.layer2(t.layer1(x))))
        x = x.float().clamp(min=1e-6).pow(self.gem_p).mean((2, 3)).pow(1 / self.gem_p)  # GeM pooling
        return F.normalize(self.proj(x), dim=1)


def load_sscd(path=None, device="cuda"):
    """SSCD from a local copy of sscd_disc_mixup.classy.pt, or downloaded."""
    if path:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    else:
        sd = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", weights_only=True)
    model = SSCD()
    model.load_state_dict(torchvision_names(sd), strict=True)  # every weight must be mapped, and nothing left over
    return model.eval().to(device)
