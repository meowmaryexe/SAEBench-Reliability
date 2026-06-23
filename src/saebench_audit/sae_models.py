"""
Independent reimplementations of the SAEBench / dictionary_learning SAE forward passes.

These are written from the architecture descriptions in the SAEBench paper and the public
`dictionary_learning/dictionary.py` reference, NOT imported from that library. The point of a
reproducibility study is to re-derive the computation; we only use the released *weights*.

Each class exposes:
    encode(x) -> features f         (post-nonlinearity activations, >= 0 for ReLU-family)
    decode(f) -> x_hat
    forward(x) -> x_hat
and a `.dict_size` / `.activation_dim`.

`load_sae(path, arch)` reads a released `ae.pt` state_dict and returns the right module.
The decoder-normalization step (April-update SAEs store un-normalized decoders) is a no-op for the
reconstruction x_hat — it only rescales the encoder/feature magnitudes — so we skip it for Loss
Recovered (x_hat is bit-identical) and note it explicitly where L0/feature magnitudes would matter.
"""
from __future__ import annotations
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Standard ReLU autoencoder  (dict_class "AutoEncoder", StandardTrainerAprilUpdate)
#   encode(x) = ReLU( W_enc (x - b_dec) + b_enc )
#   decode(f) = W_dec f + b_dec
# state_dict keys: encoder.weight [F,D], encoder.bias [F], decoder.weight [D,F], bias [D]
# --------------------------------------------------------------------------------------
class StandardAE(nn.Module):
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(torch.zeros(activation_dim))          # b_dec
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

    def encode(self, x):
        return torch.relu(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# --------------------------------------------------------------------------------------
# Gated SAE  (dict_class "GatedAutoEncoder")
#   x_enc = W_enc (x - b_dec)
#   gate  = 1[x_enc + b_gate > 0]
#   mag   = ReLU( exp(r_mag) * x_enc + b_mag )
#   f     = gate * mag ;  x_hat = W_dec f + b_dec
# keys: encoder.weight, decoder.weight, decoder_bias, r_mag, gate_bias, mag_bias
# --------------------------------------------------------------------------------------
class GatedAE(nn.Module):
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(torch.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False)
        self.r_mag = nn.Parameter(torch.zeros(dict_size))
        self.gate_bias = nn.Parameter(torch.zeros(dict_size))
        self.mag_bias = nn.Parameter(torch.zeros(dict_size))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

    def encode(self, x):
        x_enc = self.encoder(x - self.decoder_bias)
        f_gate = (x_enc + self.gate_bias > 0).to(x.dtype)
        f_mag = torch.relu(self.r_mag.exp() * x_enc + self.mag_bias)
        return f_gate * f_mag

    def decode(self, f):
        return self.decoder(f) + self.decoder_bias

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# --------------------------------------------------------------------------------------
# JumpReLU SAE  (dict_class "JumpReluAutoEncoder")
#   pre = x W_enc + b_enc ;  f = ReLU(pre) * 1[pre > threshold]
#   x_hat = f W_dec + b_dec
# keys: W_enc [D,F], b_enc [F], W_dec [F,D], b_dec [D], threshold [F]
# --------------------------------------------------------------------------------------
class JumpReluAE(nn.Module):
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(torch.zeros(activation_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.zeros(dict_size, activation_dim))
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        self.threshold = nn.Parameter(torch.ones(dict_size) * 0.001)
        self.apply_b_dec_to_input = False

    def encode(self, x):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        return torch.relu(pre * (pre > self.threshold))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# --------------------------------------------------------------------------------------
# TopK SAE  (dict_class "AutoEncoderTopK")
#   pre = W_enc (x - b_dec) + b_enc ;  keep top-k per row, ReLU ;  x_hat = W_dec f + b_dec
# keys: encoder.weight [F,D], encoder.bias [F], decoder.weight [D,F], b_dec [D], k (buffer)
# --------------------------------------------------------------------------------------
class TopKAE(nn.Module):
    def __init__(self, activation_dim, dict_size, k):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", torch.tensor(int(k)))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x):
        pre = self.encoder(x - self.b_dec)
        k = int(self.k.item())
        vals, idx = pre.topk(k, dim=-1)
        vals = torch.relu(vals)
        f = torch.zeros_like(pre)
        f.scatter_(-1, idx, vals)
        return f

    def decode(self, f):
        return self.decoder(f) + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# --------------------------------------------------------------------------------------
# BatchTopK / MatryoshkaBatchTopK  (dict_class "BatchTopKSAE" / "MatryoshkaBatchTopKSAE")
# At INFERENCE time both use a learned per-feature `threshold` (jump-style), not batch top-k:
#   pre = W_enc (x - b_dec) + b_enc ;  f = pre * 1[pre > threshold] (ReLU) ;  x_hat = W_dec f + b_dec
# keys: encoder.weight, encoder.bias, decoder.weight, b_dec, threshold (scalar or [F]), k (buffer)
# Matryoshka shares the identical inference forward (the nesting only matters during training).
# --------------------------------------------------------------------------------------
class BatchTopKAE(nn.Module):
    def __init__(self, activation_dim, dict_size, k, use_threshold=True):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.use_threshold = use_threshold
        self.register_buffer("k", torch.tensor(int(k)))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x):
        pre = self.encoder(x - self.b_dec)
        post = torch.relu(pre)
        thr = float(self.threshold.item())
        if self.use_threshold and thr >= 0:
            return post * (post > thr)
        # fallback: batch-level top-(k*B) selection
        k = int(self.k.item())
        flat = post.reshape(-1, self.dict_size)
        n = flat.shape[0]
        topk_total = k * n
        vals, idx = flat.flatten().topk(topk_total)
        mask = torch.zeros_like(flat.flatten())
        mask[idx] = 1.0
        return (flat.flatten() * mask).reshape(post.shape)

    def decode(self, f):
        return self.decoder(f) + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# --------------------------------------------------------------------------------------
# MatryoshkaBatchTopK  (dict_class "MatryoshkaBatchTopKSAE")
# Distinct from BatchTopK: stores raw W_enc/W_dec (matmul) + a per-tensor learned threshold,
# plus group_sizes (nesting structure, only used in training — ignored at inference).
#   pre = (x - b_dec) @ W_enc + b_enc ;  f = ReLU(pre) * 1[ReLU(pre) > threshold]
#   x_hat = f @ W_dec + b_dec
# keys: W_enc [D,F], b_enc [F], W_dec [F,D], b_dec [D], threshold (scalar), k (buffer), group_sizes
# --------------------------------------------------------------------------------------
class MatryoshkaAE(nn.Module):
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(torch.zeros(activation_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.zeros(dict_size, activation_dim))
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.register_buffer("k", torch.tensor(0))

    def encode(self, x):
        post = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        thr = float(self.threshold.item())
        if thr >= 0:
            post = post * (post > thr)
        return post

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# --------------------------------------------------------------------------------------
# Baselines: Identity (residual stream) and PCA  (SAEBench custom_saes/identity_sae.py, pca_sae.py)
#   Identity: encode(x)=x, decode(f)=f  -> perfect reconstruction, L0 = d_model
#   PCA:      encode(x)=(x-mean) W_enc,  decode(f)=f W_dec + mean ;  full-rank orthonormal basis ->
#             perfect reconstruction, L0 ~ d_model.
# --------------------------------------------------------------------------------------
class IdentitySAE(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim
        self.W_enc = nn.Parameter(torch.eye(activation_dim))
        self.W_dec = nn.Parameter(torch.eye(activation_dim))

    def encode(self, x): return x @ self.W_enc
    def decode(self, f): return f @ self.W_dec
    def forward(self, x, output_features=False):
        f = self.encode(x); xh = self.decode(f)
        return (xh, f) if output_features else xh


class PCASAE(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim
        self.mean = nn.Parameter(torch.zeros(activation_dim))
        self.W_enc = nn.Parameter(torch.eye(activation_dim))   # [D, D] eigenvectors (columns)
        self.W_dec = nn.Parameter(torch.eye(activation_dim))   # [D, D] = W_enc.T

    def encode(self, x): return (x - self.mean) @ self.W_enc
    def decode(self, f): return f @ self.W_dec + self.mean
    def forward(self, x, output_features=False):
        f = self.encode(x); xh = self.decode(f)
        return (xh, f) if output_features else xh

    @torch.no_grad()
    def fit(self, activations_ND):
        """Full-rank PCA via eigendecomposition of the covariance (matches PCASAE: full basis)."""
        x = activations_ND.to(torch.float32)
        self.mean.data = x.mean(dim=0)
        xc = x - self.mean
        cov = (xc.T @ xc) / (xc.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)          # ascending; columns are orthonormal eigenvectors
        evecs = evecs.flip(1)                            # descending variance (PCA component order)
        self.W_enc.data = evecs                          # project onto components
        self.W_dec.data = evecs.T                        # inverse (orthonormal) -> perfect reconstruction
        return self


# --------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------
ARCH_ALIASES = {
    "standard": "standard", "autoencoder": "standard", "panneal": "standard",
    "p_anneal": "standard", "standardtraineraprilupdate": "standard",
    "gated": "gated", "gatedsae": "gated", "gatedautoencoder": "gated",
    "jumprelu": "jumprelu", "jumpreluautoencoder": "jumprelu",
    "topk": "topk", "autoencodertopk": "topk",
    "batchtopk": "batchtopk", "batchtopksae": "batchtopk",
    "matryoshka": "matryoshka", "matryoshkabatchtopk": "matryoshka",
    "matryoshkabatchtopksae": "matryoshka",
}


def _norm_arch(arch: str) -> str:
    key = arch.lower().replace("-", "").replace("_", "").replace(" ", "")
    for k, v in ARCH_ALIASES.items():
        if k.replace("_", "") == key:
            return v
    if "matryoshka" in key:
        return "matryoshka"
    if "batchtopk" in key:
        return "batchtopk"
    if "topk" in key:
        return "topk"
    if "jump" in key:
        return "jumprelu"
    if "gated" in key:
        return "gated"
    return "standard"


@torch.no_grad()
def _normalize_decoder_(m):
    """Fold decoder column norms into the encoder so decoder columns are unit-norm — matches
    SAEBench's loader (custom_saes/relu_sae.py: check_decoder_norms -> normalize_decoder), applied
    for April-update SAEs (Standard/PAnneal) whose decoders are not trained to unit norm. This leaves
    x_hat and L0 unchanged but rescales feature magnitudes (so L1 / feature norms match SAEBench).
    No-op when decoder columns are already unit-norm (TopK/BatchTopK/JumpReLU/Matryoshka)."""
    if hasattr(m, "encoder"):                       # StandardAE / GatedAE (Linear decoder [D,F])
        W_dec = m.decoder.weight.data              # [D, F]
        norms = W_dec.norm(dim=0)                   # per-feature (column) norm
        if torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
            return
        m.decoder.weight.data = W_dec / norms[None, :]
        if hasattr(m, "encoder") and m.encoder.weight.shape[0] == norms.shape[0]:
            m.encoder.weight.data = m.encoder.weight.data * norms[:, None]
            if m.encoder.bias is not None:
                m.encoder.bias.data = m.encoder.bias.data * norms
    elif hasattr(m, "W_dec"):                        # JumpReLU / Matryoshka (W_dec [F,D])
        norms = m.W_dec.data.norm(dim=1)            # per-feature norm
        if torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
            return
        m.W_dec.data = m.W_dec.data / norms[:, None]
        m.W_enc.data = m.W_enc.data * norms[None, :]
        m.b_enc.data = m.b_enc.data * norms


def load_sae(state_dict_path: str, arch: str, device="cpu", dtype=torch.float32,
             normalize_decoder: bool = True):
    """Load a released ae.pt into the matching independent module.

    normalize_decoder: fold decoder norms into the encoder (matches SAEBench's loader). Affects only
    feature-magnitude metrics (L1); leaves reconstruction, L0 and Loss Recovered unchanged.
    """
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and "encoder.weight" not in sd:
        sd = sd["state_dict"]
    a = _norm_arch(arch)

    if a == "standard":
        F, D = sd["encoder.weight"].shape
        m = StandardAE(D, F)
        m.load_state_dict(sd, strict=True)
    elif a == "gated":
        F, D = sd["encoder.weight"].shape
        m = GatedAE(D, F)
        m.load_state_dict(sd, strict=True)
    elif a == "jumprelu":
        D, F = sd["W_enc"].shape
        m = JumpReluAE(D, F)
        m.load_state_dict(sd, strict=True)
    elif a == "topk":
        F, D = sd["encoder.weight"].shape
        k = int(sd["k"].item()) if "k" in sd else 0
        m = TopKAE(D, F, k)
        m.load_state_dict(sd, strict=False)
    elif a == "batchtopk":
        F, D = sd["encoder.weight"].shape
        k = int(sd["k"].item()) if "k" in sd else 0
        m = BatchTopKAE(D, F, k)
        m.load_state_dict(sd, strict=False)
    elif a == "matryoshka":
        D, F = sd["W_enc"].shape
        m = MatryoshkaAE(D, F)
        m.load_state_dict(sd, strict=False)   # ignores group_sizes (training-only)
    else:
        raise ValueError(f"Unknown arch {arch}")

    # April-update SAEs (Standard/PAnneal) ship un-normalized decoders; SAEBench's loader normalizes
    # them. TopK/BatchTopK/JumpReLU/Gated/Matryoshka are trained unit-norm, so this is a no-op there.
    if normalize_decoder and a == "standard":
        _normalize_decoder_(m)

    m = m.to(device=device, dtype=dtype)
    m.eval()
    return m
