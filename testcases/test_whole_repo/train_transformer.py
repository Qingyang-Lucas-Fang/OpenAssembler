#!/usr/bin/env python3
"""Train a small causal Transformer language model on a built-in corpus."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_TEXT = (
    "transformers learn relationships between tokens.\n"
    "small examples make training loops easier to understand.\n"
    "attention lets each token use earlier context.\n"
) * 200


@dataclass(frozen=True)
class Vocabulary:
    """Bidirectional character vocabulary used by the language model."""

    stoi: Dict[str, int]
    itos: Tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        symbols = tuple(sorted(set(text)))
        if not symbols:
            raise ValueError("Training text must contain at least one character.")
        return cls(stoi={symbol: index for index, symbol in enumerate(symbols)}, itos=symbols)

    def encode(self, text: str) -> Tensor:
        try:
            return torch.tensor([self.stoi[character] for character in text], dtype=torch.long)
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is not in the vocabulary.") from error

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)


class CausalTransformer(nn.Module):
    """A compact decoder-only language model built from causal attention blocks."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(DecoderBlock(d_model, nhead, dropout) for _ in range(num_layers))
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return next-token logits with shape (batch, sequence, vocabulary)."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch_size, sequence_length).")
        _, sequence_length = tokens.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block size {self.block_size}."
            )

        positions = torch.arange(sequence_length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms.to(dtype=x.dtype)) * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by nhead")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_width)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_width ** -0.5)
        future = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(attended)


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.attention = CausalSelfAttention(width, heads, dropout)
        self.mlp_norm = RMSNorm(width)
        self.mlp_in = nn.Linear(width, 4 * width)
        self.mlp_out = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        return x + self.dropout(self.mlp_out(F.gelu(self.mlp_in(self.mlp_norm(x)))))


def make_batch(data: Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Sample random fixed-length input and next-token target sequences."""
    if data.ndim != 1:
        raise ValueError("data must be a one-dimensional tensor of token ids.")
    max_start = data.numel() - block_size - 1
    if max_start < 0:
        raise ValueError("Training data must be at least block_size + 1 tokens long.")

    starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
    offsets = torch.arange(block_size, device=device)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
    return data[indices], data[indices + 1]


def choose_device(requested: str) -> torch.device:
    """Use the requested accelerator when available, otherwise select automatically."""
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if requested == "auto":
        for name in ("cuda", "mps", "cpu"):
            if available[name]:
                return torch.device(name)
    elif available[requested]:
        return torch.device(requested)
    raise RuntimeError(f"Requested device '{requested}' is not available.")


def train(
    model: CausalTransformer,
    data: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    timing_warmup: int = 0,
    profile_steps: int = 0,
) -> float:
    """Train next-token prediction and optionally save CUDA-event stage timings."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_value = float("nan")
    stages = ("batch", "forward_loss", "backward", "optimizer", "end_to_end")
    samples: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {name: [] for name in stages}
    if timing_warmup < 0 or profile_steps < 0:
        raise ValueError("timing_warmup and profile_steps must be nonnegative")

    for step in range(timing_warmup + steps):
        measured = (device.type == "cuda" and step >= timing_warmup
                    and step - timing_warmup < profile_steps)
        events = ({name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                   for name in stages} if measured else {})
        if measured:
            events["end_to_end"][0].record()
            events["batch"][0].record()
        inputs, targets = make_batch(data, batch_size, model.block_size, device)
        if measured:
            events["batch"][1].record()
            events["forward_loss"][0].record()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        if measured:
            events["forward_loss"][1].record()
            events["backward"][0].record()
        loss.backward()
        if measured:
            events["backward"][1].record()
            events["optimizer"][0].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if measured:
            events["optimizer"][1].record()
            events["end_to_end"][1].record()
            for name in stages:
                samples[name].append(events[name])
        loss_value = loss.item()

        display_step = step - timing_warmup + 1
        if step == 0 or step + 1 == timing_warmup + steps or display_step % 50 == 0:
            print(f"step {max(display_step, 0):4d}/{steps}: loss={loss_value:.4f}")

    if samples["end_to_end"]:
        torch.cuda.synchronize(device)
        timings = {}
        for name, pairs in samples.items():
            values = [start.elapsed_time(end) for start, end in pairs]
            timings[name] = {"samples_ms": values, "mean_ms": statistics.fmean(values),
                             "median_ms": statistics.median(values),
                             "p90_ms": sorted(values)[min(len(values) - 1, int(0.9 * len(values)))]}
        step_ms = timings["end_to_end"]["median_ms"]
        timings["tokens_per_second"] = (batch_size * model.block_size * 1000 / step_ms) if step_ms else None
        report_path = os.environ.get("OPENASSEMBLER_STAGE_TIMING_PATH")
        if report_path:
            Path(report_path).write_text(json.dumps({"schema": 1, "device": torch.cuda.get_device_name(device),
                "profile_steps": len(samples["end_to_end"]), "timing_warmup_steps": timing_warmup,
                "batch_size": batch_size, "sequence_length": model.block_size, "stages": timings}, indent=2) + "\n")

    return loss_value


@torch.no_grad()
def generate(
    model: CausalTransformer,
    prompt: Tensor,
    *,
    new_tokens: int,
    temperature: float = 0.8,
) -> Tensor:
    """Autoregressively append sampled tokens to a single-token-batch prompt."""
    if prompt.ndim != 2 or prompt.size(0) != 1:
        raise ValueError("prompt must have shape (1, sequence_length).")
    if prompt.size(1) == 0:
        raise ValueError("prompt must contain at least one token.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    was_training = model.training
    model.eval()
    generated = prompt
    for _ in range(new_tokens):
        context = generated[:, -model.block_size :]
        logits = model(context)[:, -1, :] / temperature
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    model.train(was_training)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of optimizer steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Sequences per optimizer step.")
    parser.add_argument("--block-size", type=int, default=48, help="Maximum context length in characters.")
    parser.add_argument("--d-model", type=int, default=96, help="Embedding width.")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads per layer.")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="AdamW learning rate.")
    parser.add_argument("--new-tokens", type=int, default=160, help="Characters to generate after training.")
    parser.add_argument("--timing-warmup", type=int, default=2, help="Optimizer steps excluded before CUDA-event timing.")
    parser.add_argument("--profile-steps", type=int, default=0, help="Measured CUDA steps; 0 disables stage timing output.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("steps must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if args.block_size < 2:
        raise ValueError("block_size must be at least 2.")
    if args.d_model < 1 or args.nhead < 1 or args.num_layers < 1:
        raise ValueError("d_model, nhead, and num_layers must each be at least 1.")
    if args.d_model % args.nhead != 0:
        raise ValueError("d_model must be divisible by nhead.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1).")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if args.new_tokens < 0:
        raise ValueError("new_tokens cannot be negative.")
    if args.timing_warmup < 0 or args.profile_steps < 0:
        raise ValueError("timing_warmup and profile_steps must be nonnegative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    vocabulary = Vocabulary.from_text(DEFAULT_TEXT)
    data = vocabulary.encode(DEFAULT_TEXT).to(device)
    model = CausalTransformer(
        vocab_size=len(vocabulary.itos),
        block_size=args.block_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"device={device.type}, vocabulary={len(vocabulary.itos)}, parameters={sum(p.numel() for p in model.parameters()):,}")
    train(
        model,
        data,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        timing_warmup=args.timing_warmup if args.profile_steps else 0,
        profile_steps=args.profile_steps,
    )

    prompt = vocabulary.encode("transformer ").unsqueeze(0).to(device)
    sample = generate(model, prompt, new_tokens=args.new_tokens).squeeze(0).tolist()
    print("\nGenerated sample:\n" + vocabulary.decode(sample))


if __name__ == "__main__":
    main()
