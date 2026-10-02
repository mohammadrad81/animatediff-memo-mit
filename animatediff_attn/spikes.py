"""Attention-spike detection following GUARD (Zhao et al., 2026), Sec. 5.2.

For every attended position i (text token for cross-attention, key location for spatial
self-attention, key frame for temporal attention):

    M_i = max_q AttnScore[q, i]
    Z_i = (M_i - mu) / sigma,  mu/sigma over {M_i}
    S   = { i : Z_i > tau }

Matching the official implementation (local_sd_pipeline.py), by default:
  * AttnScore are the *pre-softmax logits* (``--score max_logit``); the paper text speaks of
    attention scores, use ``--score max_prob`` for post-softmax probabilities;
  * the statistics are computed per (batch row, head); in AnimateDiff a cross-attention batch
    row is one frame, so this is per (frame, head);
  * for cross-attention, position 0 (<sot>) is excluded from mu/sigma and never flagged.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import torch

SCORES = ("max_logit", "max_prob", "mass")
AGGS = ("none", "mean", "max")


def zscore(M: torch.Tensor, exclude_first: bool) -> torch.Tensor:
    content = M[..., 1:] if exclude_first else M
    mu = content.mean(-1, keepdim=True)
    sd = content.std(-1, keepdim=True)
    return (M - mu) / (sd + 1e-6)


def detect(M: torch.Tensor, tau: float, exclude_first: bool) -> tuple[torch.Tensor, torch.Tensor]:
    z = zscore(M.float(), exclude_first)
    spikes = z > tau
    if exclude_first:
        spikes[..., 0] = False
    return z, spikes


def _reduce(x: torch.Tensor, dim: int, how: str) -> torch.Tensor:
    if how == "none":
        return x
    return x.mean(dim, keepdim=True) if how == "mean" else x.amax(dim, keepdim=True)


def position_scores(stats: dict, attn_type: str, score: str = "max_logit", heads=None, frames=None,
                    head_agg: str = "none", frame_agg: str = "none") -> torch.Tensor:
    """Returns M with shape [F', H', P] (temporal layers have F' = 1)."""
    M = stats[score].float()
    if attn_type == "temporal":
        M = M[None]  # [1, H, F]
    elif frames is not None:
        M = M[[f for f in frames if f < M.shape[0]]]
    if heads is not None:
        M = M[:, [h for h in heads if h < M.shape[1]]]
    M = _reduce(M, 1, head_agg)
    if attn_type != "temporal":
        M = _reduce(M, 0, frame_agg)
    return M


def exclude_first_for(attn_type: str) -> bool:
    return attn_type == "cross"


@dataclass
class SpikeRow:
    layer_id: int
    layer: str
    attn_type: str
    block: str
    step: int
    timestep: float
    frame: str
    head: str
    position: int
    label: str
    M: float
    z: float


def position_label(attn_type, pos, token_labels, hw):
    if attn_type == "cross":
        return token_labels[pos] if pos < len(token_labels) else str(pos)
    if attn_type == "temporal":
        return f"frame{pos}"
    h, w = hw
    return f"({pos // w},{pos % w})"


def find_spikes(record: dict, layer_names, steps, tau=3.0, score="max_logit", heads=None, frames=None,
                head_agg="none", frame_agg="none") -> list[SpikeRow]:
    rows = []
    meta = record["meta"]
    for name in layer_names:
        info = record["layers"][name]
        hw = tuple(record["resolutions"].get(name, (1, 1)))
        for step in steps:
            stats = record["stats"][name].get(step)
            if stats is None:
                continue
            M = position_scores(stats, info["attn_type"], score, heads, frames, head_agg, frame_agg)
            z, spk = detect(M, tau, exclude_first_for(info["attn_type"]))
            for f, h, p in spk.nonzero().tolist():
                frame = "-" if info["attn_type"] == "temporal" else (
                    frame_agg if frame_agg != "none" else str(frames[f] if frames else f))
                head = head_agg if head_agg != "none" else str(heads[h] if heads else h)
                rows.append(SpikeRow(
                    info["idx"], name, info["attn_type"], info["block"], step, record["timesteps"][step],
                    frame, head, p, position_label(info["attn_type"], p, meta["token_labels"], hw),
                    float(M[f, h, p]), float(z[f, h, p]),
                ))
    return rows


def write_csv(rows: list[SpikeRow], path):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(SpikeRow.__dataclass_fields__))
        for r in rows:
            writer.writerow(list(r.__dict__.values()))


def summarize(rows: list[SpikeRow], record: dict, layer_names, steps, heads=None, frames=None,
              head_agg="none", frame_agg="none") -> dict:
    """How often each position is flagged, per attention type, as a fraction of all tests."""
    summary = {}
    for attn_type in ("cross", "temporal", "spatial"):
        names = [n for n in layer_names if record["layers"][n]["attn_type"] == attn_type]
        tests = 0
        for n in names:
            for s in steps:
                stats = record["stats"][n].get(s)
                if stats is None:
                    continue
                M = position_scores(stats, attn_type, "mass", heads, frames, head_agg, frame_agg)
                tests += M.shape[0] * M.shape[1]
        if not tests:
            continue
        counts = {}
        for r in rows:
            if r.attn_type == attn_type:
                if attn_type == "cross":
                    key = f"{r.label} ({r.position})"
                elif attn_type == "spatial":
                    key = f"L{r.layer_id}:{r.label}"
                else:
                    key = r.label
                counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        summary[attn_type] = {
            "num_tests": tests,
            "num_spikes": sum(counts.values()),
            "top_positions": [{"position": k, "count": c, "fraction_of_tests": c / tests} for k, c in ranked[:30]],
        }
    return summary
