#!/usr/bin/env python3
"""Train a small causal Transformer language model on a built-in corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    """A compact Transformer encoder configured as a causal language model."""

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
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
        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=tokens.device,
                dtype=hidden.dtype,
            ),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask, is_causal=True)
        return self.output(self.final_norm(hidden))


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
) -> float:
    """Run next-token training and return the final scalar loss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_value = float("nan")

    for step in range(1, steps + 1):
        inputs, targets = make_batch(data, batch_size, model.block_size, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_value = loss.item()

        if step == 1 or step == steps or step % 50 == 0:
            print(f"step {step:4d}/{steps}: loss={loss_value:.4f}")

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
    )

    prompt = vocabulary.encode("transformer ").unsqueeze(0).to(device)
    sample = generate(model, prompt, new_tokens=args.new_tokens).squeeze(0).tolist()
    print("\nGenerated sample:\n" + vocabulary.decode(sample))


if __name__ == "__main__":
    main()
