"""
train_transformer.py
====================

A fairly complete, from-scratch training pipeline for a decoder-only
Transformer language model.

Features
--------
* Character-level tokenizer (trivially swappable for BPE)
* Pre-norm Transformer blocks with RMSNorm
* Rotary position embeddings (RoPE)
* SwiGLU feed-forward network
* Fused attention via torch.nn.functional.scaled_dot_product_attention
* Weight tying between embedding and LM head
* KV-cache for fast autoregressive sampling
* AdamW with decoupled weight decay + parameter grouping
* Cosine LR schedule with linear warmup
* Mixed precision (bf16 or fp16 + GradScaler)
* Gradient accumulation and gradient clipping
* Checkpointing / resuming
* Deterministic seeding
* Sampling with temperature / top-k / top-p
* Optional torch.compile

Usage
-----
    python train_transformer.py                        # synthetic demo corpus
    python train_transformer.py --data_path my.txt     # train on your own text
    python train_transformer.py --compile --max_steps 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # --- data ---
    data_path: Optional[str] = None
    out_dir: str = "out"
    block_size: int = 256

    # --- model ---
    vocab_size: int = 0  # filled in at runtime
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    ffn_mult: float = 4.0
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5

    # --- optimization ---
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    label_smoothing: float = 0.0

    # --- bookkeeping ---
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    ckpt_interval: int = 500
    seed: int = 1337
    compile: bool = False
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Minimal character-level tokenizer with JSON (de)serialization."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BlockDataset(Dataset):
    """Fixed-length windows over a token stream."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        assert tokens.numel() > block_size + 1, "corpus too small for block_size"
        self.tokens = tokens
        self.block_size = block_size
        self.n_windows = tokens.numel() - block_size

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int):
        chunk = self.tokens[i : i + self.block_size + 1]
        return chunk[:-1].long(), chunk[1:].long()


def cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for rotary position embeddings."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._build(0)

    def _build(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        # Duplicate so that rotate_half (which splits in halves) matches up.
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self._seq_len_cached = seq_len

    def forward(self, seq_len: int):
        if seq_len > self._seq_len_cached:
            self._build(max(seq_len, 2 * self._seq_len_cached, 16))
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_base)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(start_pos + T)
        cos = cos[start_pos : start_pos + T]
        sin = sin[start_pos : start_pos + T]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_cache = (k, v)

        # is_causal only makes sense when q and k have the same length.
        is_causal = T > 1
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache


class SwiGLU(nn.Module):
    """Gated feed-forward:  W2( SiLU(W1 x) * W3 x )."""

    def __init__(self, cfg: Config):
        super().__init__()
        # 2/3 factor keeps parameter count comparable to a 4x MLP.
        hidden = int(cfg.d_model * cfg.ffn_mult * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64  # round to a nice multiple
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, kv_cache=None, start_pos=0):
        h, new_cache = self.attn(self.norm1(x), kv_cache, start_pos)
        x = x + self.drop(h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, new_cache


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scaled init for residual output projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model] {n_params/1e6:.2f}M parameters "
              f"({n_params - self.tok_emb.weight.numel()/1e6:.2f}M non-embedding)")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        caches: Optional[list] = None,
        start_pos: int = 0,
    ):
        x = self.drop(self.tok_emb(idx))
        new_caches: Optional[list] = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, nc = block(x, cache, start_pos)
            if use_cache:
                new_caches.append(nc)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                label_smoothing=self.cfg.label_smoothing,
            )
        return logits, loss, new_caches


# ---------------------------------------------------------------------------
# LR schedule & optimizer
# ---------------------------------------------------------------------------


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def build_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    seen = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: TransformerLM,
    idx: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    model.eval()
    caches = None

    for _ in range(max_new_tokens):
        if use_cache:
            if caches is None:
                inp, pos = idx, 0
            else:
                inp, pos = idx[:, -1:], idx.size(1) - 1
        else:
            inp = idx[:, -model.cfg.block_size :]
            pos = 0

        logits, _, caches = model(
            inp, use_cache=use_cache, caches=caches, start_pos=pos
        )
        logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits
                )
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    return idx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(model, val_iter, cfg, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = next(val_iter)
        with ctx:
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Sample corpus (used when --data_path is not given)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
The transformer is a neural network architecture built entirely from attention
mechanisms. Unlike recurrent networks, which process a sequence one element at
a time, a transformer looks at every position simultaneously. This parallelism
makes it possible to train on enormous corpora, and it is the reason modern
language models can be trained at all.

At the heart of the transformer lies self attention. Each token produces a
query, a key, and a value. The query of one token is compared against the keys
of every other token, producing a set of scores. Those scores become weights
after a softmax, and the values are mixed accordingly. In the causal variant,
a token may only attend to itself and to tokens that came before it, which is
what allows the model to be used for next token prediction.

Positional information is injected separately, because attention itself is
permutation invariant. Early models used fixed sinusoidal encodings or learned
embeddings. More recent ones rotate the query and key vectors by an angle that
depends on position, so that the dot product between two tokens depends only on
their relative distance. This scheme, called rotary position embedding, is now
the default in most open models.

Training such a model is an exercise in patience. One streams batches of token
sequences, computes the cross entropy between predictions and the next token,
and takes a small step downhill. Warmup prevents early instability, and a
cosine decay gradually reduces the learning rate as the run approaches its end.
Gradient clipping keeps the occasional sharp loss spike from wrecking the
weights. Mixed precision halves memory use and doubles throughput.

The result, after enough steps, is a model that can continue a passage in a
style it has never explicitly been told to imitate. It has learned the shape of
the language, its rhythms, its habits, and something of its content.
""".strip()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: Config) -> None:
    # ---- seeding ----
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ---- data ----
    if cfg.data_path:
        with open(cfg.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[data] loaded {len(text):,} characters from {cfg.data_path}")
    else:
        text = (SAMPLE_TEXT + "\n\n") * 40
        print(f"[data] using built-in sample corpus ({len(text):,} characters)")

    tokenizer = CharTokenizer.from_text(text)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"[data] vocabulary size: {cfg.vocab_size}")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(0.9 * ids.numel())
    train_ids, val_ids = ids[:n_train], ids[n_train:]

    train_ds = BlockDataset(train_ids, cfg.block_size)
    val_ds = BlockDataset(val_ids, cfg.block_size)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, num_workers=cfg.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        drop_last=True, num_workers=cfg.num_workers, pin_memory=pin,
    )
    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)
    print(f"[data] {len(train_ds):,} train windows | {len(val_ds):,} val windows")

    # ---- model ----
    model = TransformerLM(cfg).to(device)
    if cfg.compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("[model] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[model] torch.compile unavailable: {e}")

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=1e-8,
    )

    # ---- mixed precision ----
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (
        use_amp and torch.cuda.is_bf16_supported()
    ) else torch.float16
    scaler = None
    if use_amp and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):  # older torch
            scaler = torch.cuda.amp.GradScaler()
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )
    print(f"[amp] autocast={'on (' + str(amp_dtype).split('.')[-1] + ')' if use_amp else 'off'}")

    # ---- checkpointing ----
    os.makedirs(cfg.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(cfg.out_dir, "tokenizer.json"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    start_step = 0
    best_val = float("inf")
    if os.path.exists(ckpt_path):
        print(f"[ckpt] resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"]
        # Strip torch.compile prefix if present.
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", float("inf"))

    # ---- training loop ----
    model.train()
    t0 = time.time()
    tokens_seen = 0
    running_loss = 0.0
    print(f"\n[train] starting at step {start_step} / {cfg.max_steps}\n")

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tokens_seen += x.numel()

            with ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / cfg.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        running_loss += accum_loss

        # ---- logging ----
        if (step + 1) % cfg.log_interval == 0:
            avg = running_loss / cfg.log_interval
            running_loss = 0.0
            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            tok_per_s = tokens_seen / max(elapsed, 1e-9)
            eta = elapsed / max(steps_done, 1) * (cfg.max_steps - step - 1)
            print(
                f"step {step+1:>6}/{cfg.max_steps} | "
                f"loss {avg:7.4f} | lr {lr:.2e} | "
                f"{tok_per_s/1e3:6.1f}k tok/s | "
                f"elapsed {elapsed/60:5.1f}m | eta {eta/60:5.1f}m"
            )

        # ---- evaluation ----
        if (step + 1) % cfg.eval_interval == 0 or step + 1 == cfg.max_steps:
            val_loss = estimate_loss(model, val_iter, cfg, ctx)
            print(f"\n[eval] step {step+1} | val loss {val_loss:.4f} | "
                  f"ppl {math.exp(val_loss):.2f}")

            prompt = "The transformer is"
            ids_prompt = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            out = generate(
                model, ids_prompt, max_new_tokens=120,
                temperature=0.8, top_k=40, top_p=0.95,
            )
            print("[sample] " + tokenizer.decode(out[0].tolist()).replace("\n", " "))
            print()

            if val_loss < best_val:
                best_val = val_loss

        # ---- checkpointing ----
        if (step + 1) % cfg.ckpt_interval == 0 or step + 1 == cfg.max_steps:
            raw_model = getattr(model, "_orig_mod", model)
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val": best_val,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved {ckpt_path}")

    print(f"\n[train] done in {(time.time() - t0)/60:.1f} minutes. "
          f"best val loss {best_val:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train a small transformer LM.")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    a = p.parse_args()
    return Config(**vars(a))


if __name__ == "__main__":
    train(parse_args())