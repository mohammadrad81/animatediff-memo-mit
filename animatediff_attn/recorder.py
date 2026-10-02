"""Record attention statistics and maps from the AnimateDiff UNet during sampling.

The recorder wraps the attention processor of every selected layer. The original processor
still produces the layer output (so generation is untouched); the wrapper recomputes
queries/keys for one classifier-free-guidance branch and stores:

``stats`` (small, recorded at every ``stat_step``) - per attended position ``i``:
    max_logit[i] = max_q  QK^T/sqrt(d)  (pre-softmax; what the GUARD code thresholds)
    max_prob[i]  = max_q  softmax(...)  (post-softmax "attention score" of the paper)
    mass[i]      = mean_q softmax(...)  ("cross-attention mass", Fig. 1 of the paper)

    cross:    tensors of shape [F, H, 77]     (frames, heads, text tokens)
    temporal: tensors of shape [H, F]         (heads, key frames; max/mean over space & query frames)
              + mean_map/max_map/max_logit_map [H, Fq, Fk] and entropy [H]
    spatial:  tensors of shape [F, H, h*w]    (key locations; only at map steps, they are large)

``maps`` (large, recorded only at ``map_steps``, for the selected heads/frames):
    cross:    probs        [F', H', h, w, T]  (T = selected token positions)
    spatial:  query_maps   [F', H', P, h, w]  (P = query points)
    temporal: probs        [H', h, w, Fq, Fk]
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from .selection import LayerInfo


@dataclass
class RecordConfig:
    num_frames: int
    latent_h: int
    latent_w: int
    stat_steps: list[int]
    map_steps: list[int]
    heads: list[int] | None = None  # heads stored in maps (None = all)
    frames: list[int] | None = None  # frames stored in maps (None = all)
    map_tokens: list[int] | None = None  # cross-attention token positions stored in maps
    query_points: list[tuple[float, float]] = field(default_factory=lambda: [(0.5, 0.5)])
    branch: str = "cond"  # which CFG branch to analyse: cond | uncond
    cfg: bool = True
    query_chunk: int = 1024

    def to_dict(self) -> dict:
        return asdict(self)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


class RecordingProcessor:
    """Delegates to the layer's original processor, capturing statistics on the side."""

    def __init__(self, recorder: "AttentionRecorder", layer: LayerInfo, base):
        self.recorder, self.layer, self.base = recorder, layer, base

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        if self.recorder.wants(self.layer):
            self.recorder.capture(self.layer, attn, hidden_states, encoder_hidden_states, temb)
        return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)


class AttentionRecorder:
    def __init__(self, cfg: RecordConfig, layers: list[LayerInfo]):
        self.cfg = cfg
        self.layers = {l.name: l for l in layers}
        self.step = -1
        self.timesteps: list[float] = []
        self.stats: dict[str, dict[int, dict[str, torch.Tensor]]] = {l.name: {} for l in layers}
        self.maps: dict[str, dict[int, dict[str, torch.Tensor]]] = {l.name: {} for l in layers}
        self.resolutions: dict[str, tuple[int, int]] = {}
        self._orig_processors = {}
        self._hook = None

    # ------------------------------------------------------------------ wiring
    def attach(self, unet):
        modules = dict(unet.named_modules())
        for name, layer in self.layers.items():
            module = modules[name]
            self._orig_processors[name] = module.processor
            module.set_processor(RecordingProcessor(self, layer, module.processor))
        self._hook = unet.register_forward_pre_hook(self._on_unet_call, with_kwargs=True)

    def detach(self, unet):
        modules = dict(unet.named_modules())
        for name, proc in self._orig_processors.items():
            modules[name].set_processor(proc)
        self._orig_processors.clear()
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _on_unet_call(self, module, args, kwargs):
        # One UNet call per denoising step (CFG branches are batched together).
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        self.step += 1
        self.timesteps.append(float(t.flatten()[0]) if torch.is_tensor(t) else float(t))

    def wants(self, layer: LayerInfo) -> bool:
        if self.step in self.cfg.map_steps:
            return True
        return self.step in self.cfg.stat_steps and layer.attn_type != "spatial"

    # ---------------------------------------------------------------- capture
    def _project(self, attn, hs, ehs, temb):
        """Replicates the pre-processing of diffusers' AttnProcessor2_0 up to q/k, for one CFG branch."""
        x = hs
        if attn.spatial_norm is not None:
            x = attn.spatial_norm(x, temb)
        if x.ndim == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            x = attn.group_norm(x.transpose(1, 2)).transpose(1, 2)
        if ehs is None:
            ctx = x
        else:
            ctx = attn.norm_encoder_hidden_states(ehs) if attn.norm_cross else ehs

        rows = x.shape[0]
        if self.cfg.cfg:
            per = rows // 2
            sl = slice(per, rows) if self.cfg.branch == "cond" else slice(0, per)
        else:
            per, sl = rows, slice(0, rows)
        x, ctx = x[sl], ctx[sl]

        heads = attn.heads
        q, k = attn.to_q(x), attn.to_k(ctx)
        d = q.shape[-1] // heads
        q = q.view(per, -1, heads, d).transpose(1, 2)  # [rows, H, N, d]
        k = k.view(per, -1, heads, d).transpose(1, 2)
        if getattr(attn, "norm_q", None) is not None:
            q = attn.norm_q(q)
        if getattr(attn, "norm_k", None) is not None:
            k = attn.norm_k(k)
        return q.float(), k.float(), attn.scale

    def _hw(self, layer: LayerInfo, n: int) -> tuple[int, int]:
        h, w = layer.resolution(self.cfg.latent_h, self.cfg.latent_w)
        if h * w != n:  # fall back to inferring a square-ish grid
            div = round(math.sqrt(self.cfg.latent_h * self.cfg.latent_w / n))
            h, w = math.ceil(self.cfg.latent_h / div), math.ceil(self.cfg.latent_w / div)
            if h * w != n:
                h, w = 1, n
        self.resolutions[layer.name] = (h, w)
        return h, w

    def _sel(self, idx, n):
        return list(range(n)) if idx is None else [i for i in idx if i < n]

    @torch.no_grad()
    def capture(self, layer: LayerInfo, attn, hs, ehs, temb):
        q, k, scale = self._project(attn, hs, ehs, temb)
        is_map_step = self.step in self.cfg.map_steps
        if layer.attn_type == "cross":
            stats, maps = self._capture_cross(layer, q, k, scale, is_map_step)
        elif layer.attn_type == "spatial":
            stats, maps = self._capture_spatial(layer, q, k, scale)
        else:
            stats, maps = self._capture_temporal(layer, q, k, scale, is_map_step)
        if stats:
            self.stats[layer.name][self.step] = stats
        if maps:
            self.maps[layer.name][self.step] = maps

    def _capture_cross(self, layer, q, k, scale, is_map_step):
        n_frames, heads, n, _ = q.shape
        h, w = self._hw(layer, n)
        heads_sel = self._sel(self.cfg.heads, heads)
        frames_sel = self._sel(self.cfg.frames, n_frames)
        tokens = self.cfg.map_tokens if self.cfg.map_tokens is not None else list(range(k.shape[2]))
        max_logit, max_prob, mass, ent, maps = [], [], [], [], []
        for f in range(n_frames):
            logits = (q[f] @ k[f].transpose(-1, -2)) * scale  # [H, N, L]
            p = logits.softmax(-1)
            max_logit.append(logits.amax(1))
            max_prob.append(p.amax(1))
            mass.append(p.mean(1))
            ent.append(_entropy(p).mean(-1))
            if is_map_step and f in frames_sel:
                m = p[heads_sel][:, :, tokens].reshape(len(heads_sel), h, w, len(tokens))
                maps.append(m.half().cpu())
        stats = {
            "max_logit": torch.stack(max_logit).cpu(),
            "max_prob": torch.stack(max_prob).cpu(),
            "mass": torch.stack(mass).cpu(),
            "entropy": torch.stack(ent).cpu(),
        }
        return stats, ({"probs": torch.stack(maps)} if maps else None)

    def _capture_spatial(self, layer, q, k, scale):
        n_frames, heads, n, _ = q.shape
        h, w = self._hw(layer, n)
        heads_sel = self._sel(self.cfg.heads, heads)
        frames_sel = self._sel(self.cfg.frames, n_frames)
        q_idx = [
            min(round(y * (h - 1)), h - 1) * w + min(round(x * (w - 1)), w - 1) for y, x in self.cfg.query_points
        ]
        chunk = self.cfg.query_chunk
        max_logit, max_prob, mass, ent, qmaps = [], [], [], [], []
        for f in range(n_frames):
            ml = torch.full((heads, n), -float("inf"), device=q.device)
            mp = torch.zeros(heads, n, device=q.device)
            ms = torch.zeros(heads, n, device=q.device)
            en = torch.zeros(heads, device=q.device)
            for s in range(0, n, chunk):
                logits = (q[f, :, s : s + chunk] @ k[f].transpose(-1, -2)) * scale  # [H, c, N]
                p = logits.softmax(-1)
                ml = torch.maximum(ml, logits.amax(1))
                mp = torch.maximum(mp, p.amax(1))
                ms += p.sum(1)
                en += _entropy(p).sum(-1)
            max_logit.append(ml)
            max_prob.append(mp)
            mass.append(ms / n)
            ent.append(en / n)
            if f in frames_sel:
                logits = (q[f][:, q_idx] @ k[f].transpose(-1, -2)) * scale  # [H, P, N]
                qmaps.append(logits.softmax(-1)[heads_sel].reshape(len(heads_sel), len(q_idx), h, w).half().cpu())
        stats = {
            "max_logit": torch.stack(max_logit).half().cpu(),
            "max_prob": torch.stack(max_prob).half().cpu(),
            "mass": torch.stack(mass).half().cpu(),
            "entropy": torch.stack(ent).cpu(),
        }
        maps = {"query_maps": torch.stack(qmaps), "query_index": torch.tensor(q_idx)} if qmaps else None
        return stats, maps

    def _capture_temporal(self, layer, q, k, scale, is_map_step):
        n_loc, heads, n_frames, _ = q.shape  # rows are spatial locations, sequence is frames
        h, w = self._hw(layer, n_loc)
        logits = (q @ k.transpose(-1, -2)) * scale  # [HW, H, Fq, Fk]
        p = logits.softmax(-1)
        stats = {
            "max_logit": logits.amax(dim=(0, 2)).cpu(),  # [H, Fk]
            "max_prob": p.amax(dim=(0, 2)).cpu(),
            "mass": p.mean(dim=(0, 2)).cpu(),
            "mean_map": p.mean(0).cpu(),  # [H, Fq, Fk]
            "max_map": p.amax(0).cpu(),
            "max_logit_map": logits.amax(0).cpu(),
            "entropy": _entropy(p).mean(dim=(0, 2)).cpu(),  # [H]
        }
        maps = None
        if is_map_step:
            heads_sel = self._sel(self.cfg.heads, heads)
            pm = p[:, heads_sel].permute(1, 0, 2, 3).reshape(len(heads_sel), h, w, n_frames, n_frames)
            maps = {"probs": pm.half().cpu()}
        return stats, maps

    # ------------------------------------------------------------------- save
    def state_dict(self) -> dict:
        return {
            "config": self.cfg.to_dict(),
            "layers": {n: l.to_dict() for n, l in self.layers.items()},
            "resolutions": self.resolutions,
            "timesteps": self.timesteps,
            "stats": self.stats,
            "maps": self.maps,
        }
