"""Custom attention processor for the AnimateDiff UNet.

It replaces diffusers' attention in every layer and can
  1. record attention maps of the prompt-conditioned stream (for plotting), and
  2. apply GUARD's cross-attention attenuation to the "positive" stream.

The UNet batch is made of "streams", each with one row per frame:
  2 streams: [uncond, cond]                 (normal classifier-free guidance)
  3 streams: [uncond, positive, negative]   (GUARD)
Stream 1 (rows n:2n) is the one we record / attenuate.

Inside the UNet there are three kinds of attention layers:
  spatial  (attentions.*.attn1):      pixels of a frame  -> pixels of the same frame
  cross    (attentions.*.attn2):      pixels of a frame  -> the 77 text tokens
  temporal (motion_modules.*.attn*):  one pixel location -> the same location in all frames
For spatial/cross a batch row is one frame; for temporal a batch row is one pixel location.
"""
import torch
import torch.nn.functional as F


def layer_kind(name):
    if "motion_modules" in name:
        return "temporal"
    return "cross" if name.endswith("attn2") else "spatial"


def spike_scores(peak, skip_first=False):
    """Spike score of GUARD's detector (paper Sec. 5.2 / official code).

    peak[..., i] = M_i = max over queries of the attention logit given to key i (e.g. text token i).
    Returns Z_i = (M_i - mean(M)) / std(M), with mean/std over the keys of the same row (last dim).
    For cross-attention, token 0 (<start of text>) gets most of the attention for every prompt, so it is
    left out of mean/std and gets Z = -inf, i.e. it is never a spike (same as the official code).
    """
    peak = peak.float()
    ref = peak[..., 1:] if skip_first else peak
    z = (peak - ref.mean(-1, keepdim=True)) / (ref.std(-1, keepdim=True) + 1e-6)
    if skip_first:
        z[..., 0] = -float("inf")
    return z


def find_spikes(peak, tau, skip_first=False):
    """Key i is a spike if Z_i > tau."""
    return spike_scores(peak, skip_first) > tau


class Controller:
    """State shared by all attention processors."""

    def __init__(self, num_streams, record=False, guard_layers=(), tau=0.1, alpha=0.3):
        self.num_streams = num_streams          # 2 = CFG, 3 = GUARD
        self.record = record                    # store attention maps of stream 1
        self.guard_layers = set(guard_layers)   # cross-attn layers where GUARD attenuates spikes
        self.tau, self.alpha = tau, alpha       # spike threshold and logit scaling factor
        self.maps = {}     # layer -> list over steps of (attn, peak)  (filled when record=True)
        self.spikes = {}   # layer -> list over steps of ([77] count of attenuated (frame, head) rows, number of rows)


class Processor:
    def __init__(self, name, ctrl):
        self.name, self.kind, self.ctrl = name, layer_kind(name), ctrl

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        # q, k, v: [batch rows, heads, sequence, head_dim]
        q = attn.head_to_batch_dim(attn.to_q(hidden_states), out_dim=4)
        k = attn.head_to_batch_dim(attn.to_k(context), out_dim=4)
        v = attn.head_to_batch_dim(attn.to_v(context), out_dim=4)

        # Normal (fast) attention for every row; stream 1 is recomputed below if needed.
        out = F.scaled_dot_product_attention(q, k, v)
        n = q.shape[0] // self.ctrl.num_streams
        s1 = slice(n, 2 * n)
        if self.name in self.ctrl.guard_layers:
            out[s1] = self.attenuated(attn, q[s1], k[s1], v[s1])
        elif self.ctrl.record:
            out[s1] = self.recorded(attn, q[s1], k[s1], v[s1])

        b, h, seq, d = out.shape
        out = out.transpose(1, 2).reshape(b, seq, h * d)
        return attn.to_out[1](attn.to_out[0](out))  # linear projection + dropout

    def attenuated(self, attn, q, k, v):
        """GUARD positive target: scale down the logits of spiky tokens by alpha before the softmax."""
        logits = (q.float() @ k.float().transpose(-1, -2)) * attn.scale         # [frames, heads, pixels, 77]
        spikes = find_spikes(logits.amax(dim=2), self.ctrl.tau, skip_first=True)  # [frames, heads, 77]
        logits = torch.where(spikes[:, :, None, :], logits * self.ctrl.alpha, logits)
        self.ctrl.spikes.setdefault(self.name, []).append((spikes.sum((0, 1)).cpu(), spikes.shape[0] * spikes.shape[1]))
        return (logits.softmax(-1) @ v.float()).to(v.dtype)

    def recorded(self, attn, q, k, v):
        """Plain attention, computed a few rows at a time (spatial maps can be 4096 x 4096), while storing:
        attn = max over queries of the softmax probabilities (the strongest attention each key receives)
        peak = max over queries of the logits (M_i of the spike detector)."""
        rows, heads, nq, nk = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
        chunk = max(1, 2**27 // (heads * nq * nk))
        outs, attn_max, peak = [], [], []
        for i in range(0, rows, chunk):
            logits = (q[i:i + chunk] @ k[i:i + chunk].transpose(-1, -2)) * attn.scale  # [rows, heads, nq, nk]
            probs = logits.softmax(-1)
            outs.append(probs @ v[i:i + chunk])
            if self.kind == "temporal":  # rows = pixel locations: keep the frame x frame map, max over pixels
                attn_max.append(probs.amax(0))
                peak.append(logits.amax((0, 2)))
            else:                        # rows = frames: max over the query pixels
                attn_max.append(probs.amax(2))
                peak.append(logits.amax(2))
        if self.kind == "temporal":   # [heads, F, F], [heads, F]
            attn_max, peak = torch.stack(attn_max).amax(0), torch.stack(peak).amax(0)
        else:                         # [F, heads, keys] each
            attn_max, peak = torch.cat(attn_max), torch.cat(peak)
        self.ctrl.maps.setdefault(self.name, []).append((attn_max.half().cpu(), peak.half().cpu()))
        return torch.cat(outs)


def install(unet, ctrl):
    """Put our processor in every attention layer of the UNet."""
    unet.set_attn_processor({key: Processor(key.removesuffix(".processor"), ctrl) for key in unet.attn_processors})


def cross_layers(unet, blocks):
    """Names of the cross-attention layers in the given UNet parts, e.g. blocks=["down", "mid"]."""
    names = [key.removesuffix(".processor") for key in unet.attn_processors]
    return [n for n in names if layer_kind(n) == "cross" and n.split("_")[0] in blocks]
