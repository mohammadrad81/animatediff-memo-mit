"""Enumerate the attention layers of an AnimateDiff UNetMotionModel and select subsets of them.

Three kinds of attention live in the AnimateDiff UNet:

* ``spatial``  - self-attention inside one frame (``attentions.*.attn1``): every spatial
                 location of a frame attends to every other location of the same frame.
* ``cross``    - cross-attention to the text tokens (``attentions.*.attn2``), per frame.
* ``temporal`` - self-attention of the motion modules (``motion_modules.*``): every
                 spatial location attends to the same location in all other frames.
"""
from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import asdict, dataclass

ATTN_TYPES = ("spatial", "cross", "temporal")


@dataclass
class LayerInfo:
    idx: int
    name: str
    attn_type: str  # spatial | cross | temporal
    block: str  # down.0 ... | mid | up.3 ...
    heads: int
    head_dim: int
    latent_div: int  # spatial downsampling factor of this layer w.r.t. the latent

    def resolution(self, latent_h: int, latent_w: int) -> tuple[int, int]:
        return math.ceil(latent_h / self.latent_div), math.ceil(latent_w / self.latent_div)

    def to_dict(self) -> dict:
        return asdict(self)


def block_of(name: str) -> str:
    m = re.match(r"(down|up)_blocks\.(\d+)\.", name)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    if name.startswith("mid_block"):
        return "mid"
    return "other"


def attn_type_of(name: str, module) -> str:
    if "motion_modules" in name:
        return "temporal"
    return "cross" if getattr(module, "is_cross_attention", False) else "spatial"


def enumerate_layers(unet) -> list[LayerInfo]:
    from diffusers.models.attention_processor import Attention

    n_down, n_up = len(unet.down_blocks), len(unet.up_blocks)
    layers = []
    for name, module in unet.named_modules():
        if not isinstance(module, Attention):
            continue
        block = block_of(name)
        if block.startswith("down"):
            div = 2 ** int(block.split(".")[1])
        elif block.startswith("up"):
            div = 2 ** (n_up - 1 - int(block.split(".")[1]))
        else:
            div = 2 ** (n_down - 1)
        heads = module.heads
        layers.append(
            LayerInfo(
                idx=len(layers),
                name=name,
                attn_type=attn_type_of(name, module),
                block=block,
                heads=heads,
                head_dim=module.to_q.out_features // heads,
                latent_div=div,
            )
        )
    return layers


def parse_index_list(spec, n: int) -> list[int]:
    """Parse ``all`` | ``3`` | ``0,5,-1`` | ``0:10:2`` (python slice syntax) into sorted indices < n."""
    if spec is None or str(spec).strip().lower() == "all":
        return list(range(n))
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bounds = [int(x) if x.strip() else None for x in part.split(":")]
            out.extend(range(n)[slice(*bounds)])
        else:
            i = int(part)
            i = i + n if i < 0 else i
            if not 0 <= i < n:
                raise ValueError(f"index {part} out of range for size {n}")
            out.append(i)
    return sorted(set(out))


def parse_csv(spec) -> list[str] | None:
    if spec is None or str(spec).strip().lower() == "all":
        return None
    return [s.strip() for s in str(spec).split(",") if s.strip()]


def select_layers(
    layers: list[LayerInfo],
    attn_types=None,
    blocks=None,
    patterns=None,
    layer_ids=None,
) -> list[LayerInfo]:
    """Intersection of all given filters. ``None`` means "no filtering" for that criterion.

    attn_types: e.g. ``["cross", "temporal"]``
    blocks:     e.g. ``["down", "mid", "up.3"]`` (``down`` matches every down block)
    patterns:   fnmatch globs on the full module name, e.g. ``["*up_blocks.3*attn2"]``
    layer_ids:  indices printed by ``list-layers``
    """
    types = parse_csv(attn_types) if isinstance(attn_types, str) or attn_types is None else attn_types
    blks = parse_csv(blocks) if isinstance(blocks, str) or blocks is None else blocks
    ids = None if layer_ids is None else set(parse_index_list(layer_ids, len(layers)))
    for t in types or []:
        if t not in ATTN_TYPES:
            raise ValueError(f"unknown attention type {t!r}; choose from {ATTN_TYPES}")

    selected = []
    for layer in layers:
        if types and layer.attn_type not in types:
            continue
        if blks and not any(layer.block == b or layer.block.split(".")[0] == b for b in blks):
            continue
        if patterns and not any(fnmatch.fnmatch(layer.name, p) for p in patterns):
            continue
        if ids is not None and layer.idx not in ids:
            continue
        selected.append(layer)
    return selected


def format_layer_table(layers: list[LayerInfo], latent_h: int, latent_w: int) -> str:
    rows = [f"{'id':>3}  {'type':<8} {'block':<6} {'heads':>5} {'hdim':>4} {'res':>7}  name"]
    for l in layers:
        h, w = l.resolution(latent_h, latent_w)
        rows.append(f"{l.idx:>3}  {l.attn_type:<8} {l.block:<6} {l.heads:>5} {l.head_dim:>4} {f'{h}x{w}':>7}  {l.name}")
    return "\n".join(rows)
