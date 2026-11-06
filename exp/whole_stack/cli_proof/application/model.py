"""Deterministic transformer workload used by the A100 whole-stack study."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)).to(dtype) * self.weight


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm_attn = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = RMSNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        batch, seq, width = x.shape
        h = self.norm_attn(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (batch, seq, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).contiguous().view(batch, seq, width)
        x = x + self.proj(h)
        h = self.norm_mlp(x)
        return x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, width: int, heads: int, layers: int, max_seq: int):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(max_seq, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, vocab)

    def forward(self, tokens, targets=None):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(positions).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits[:, -1]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def summarize(samples):
    values = [float(value) for value in samples]
    return {"samples_ms": values, "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "p90_ms": sorted(values)[min(len(values) - 1, int(0.9 * len(values)))]}


def elapsed(start, end):
    end.synchronize()
    return start.elapsed_time(end)


def profiler_summary(prof):
    """Keep CUDA events separate from overlapping CPU operator attribution."""
    kernels = {}
    for event in prof.events():
        if str(event.device_type).split(".")[-1] != "CUDA":
            continue
        row = kernels.setdefault(event.name, {"name": event.name, "count": 0, "cuda_us": 0.0})
        row["count"] += 1
        row["cuda_us"] += float(event.device_time_total)
    operators = []
    for event in prof.key_averages():
        if str(event.device_type).split(".")[-1] != "CPU":
            continue
        operators.append({"name": event.key, "count": event.count,
                          "self_cpu_us": event.self_cpu_time_total,
                          "self_cuda_us": event.self_device_time_total,
                          "cuda_us": event.device_time_total})
    return {"cuda_kernels": sorted(kernels.values(), key=lambda row: row["cuda_us"], reverse=True),
            "cuda_launch_count": sum(row["count"] for row in kernels.values()),
            "cuda_kernel_time_us": sum(row["cuda_us"] for row in kernels.values()),
            "cpu_operator_attribution": sorted(operators, key=lambda row: row["self_cuda_us"], reverse=True),
            "scope": "separate post-timing diagnostic step; CUDA kernel totals exclude CPU operator attribution; fused kernels are indivisible"}


def run_training(args, model, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    tokens = torch.randint(args.vocab, (args.batch, args.seq + 1), device=device)
    names = ("batch", "forward_loss", "backward", "optimizer", "end_to_end")
    event_pairs = {name: [] for name in names}
    loss_value = None
    for step in range(args.warmup + args.steps):
        optimizer.zero_grad(set_to_none=True)
        measured = step >= args.warmup
        events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for name in names} if measured else {}
        if measured:
            events["end_to_end"][0].record()
            events["batch"][0].record()
        starts = torch.randint(0, tokens.shape[0], (args.batch,), device=device)
        batch = tokens[starts]
        x, y = batch[:, :args.seq], batch[:, 1:]
        if measured:
            events["batch"][1].record()
            events["forward_loss"][0].record()
        with torch.profiler.record_function("stage/forward_loss"):
            loss = model(x, y)
        if measured:
            events["forward_loss"][1].record()
            events["backward"][0].record()
        with torch.profiler.record_function("stage/backward"):
            loss.backward()
        if measured:
            events["backward"][1].record()
            events["optimizer"][0].record()
        with torch.profiler.record_function("stage/optimizer"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        loss_value = loss.detach()
        if measured:
            events["optimizer"][1].record()
            events["end_to_end"][1].record()
            for name in names:
                event_pairs[name].append(events[name])
    torch.cuda.synchronize()
    samples = {name: [start.elapsed_time(end) for start, end in pairs]
               for name, pairs in event_pairs.items()}
    output = {name: summarize(values) for name, values in samples.items()}
    step_ms = output["end_to_end"]["median_ms"]
    output["tokens_per_second"] = args.batch * args.seq * 1000.0 / step_ms
    output["last_loss"] = float(loss_value.float().item())
    if args.profile:
        optimizer.zero_grad(set_to_none=True)
        x, y = tokens[:args.batch, :args.seq], tokens[:args.batch, 1:args.seq + 1]
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            with torch.profiler.record_function("stage/forward_loss"):
                profile_loss = model(x, y)
            with torch.profiler.record_function("stage/backward"):
                profile_loss.backward()
            with torch.profiler.record_function("stage/optimizer"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        torch.cuda.synchronize()
        output["profiler"] = profiler_summary(prof)
    return output


def run_inference(args, model, device):
    model.eval()
    prompt = torch.randint(args.vocab, (args.batch, args.seq), device=device)
    with torch.no_grad():
        for _ in range(3):
            model(prompt)
        torch.cuda.synchronize()
        prefill = []
        for _ in range(args.reps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); model(prompt); end.record()
            prefill.append(elapsed(start, end))

        # Fixed-window recompute decode keeps the graph shape stable. It does
        # not claim KV-cached serving behavior.
        for _ in range(2):
            context = prompt[:1]
            for _ in range(args.decode_tokens):
                next_token = model(context).argmax(dim=-1, keepdim=True)
                context = torch.cat((context, next_token), dim=1)[:, -args.seq:]
        torch.cuda.synchronize()
        decode = []
        for _ in range(args.decode_reps):
            context = prompt[:1]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.decode_tokens):
                next_token = model(context).argmax(dim=-1, keepdim=True)
                context = torch.cat((context, next_token), dim=1)[:, -args.seq:]
            end.record()
            decode.append(elapsed(start, end))
    prefill_ms = statistics.median(prefill)
    decode_ms = statistics.median(decode)
    result = {"prefill": summarize(prefill),
            "prefill_tokens_per_second": args.batch * args.seq * 1000.0 / prefill_ms,
            "fixed_window_recompute_decode": summarize(decode),
            "decode_tokens_per_second": args.decode_tokens * 1000.0 / decode_ms,
            "decode_tokens": args.decode_tokens,
            "decode_final_token_ids": context.detach().cpu().tolist()}
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.no_grad(), torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            model(prompt)
        torch.cuda.synchronize()
        result["prefill_profiler"] = profiler_summary(prof)
        context = prompt[:1].clone()
        with torch.no_grad(), torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            for _ in range(args.decode_tokens):
                next_token = model(context).argmax(dim=-1, keepdim=True)
                context = torch.cat((context, next_token), dim=1)[:, -args.seq:]
        torch.cuda.synchronize()
        result["decode_profiler"] = profiler_summary(prof)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "infer"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument("--decode-reps", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.width % args.heads or args.seq <= 0:
        raise ValueError("width must divide evenly by heads and sequence must be positive")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = TinyTransformer(args.vocab, args.width, args.heads, args.layers, args.seq).to(device)
    if args.mode == "train":
        result = {"mode": "train", "stages": run_training(args, model, device)}
    else:
        result = {"mode": "infer", **run_inference(args, model, device)}
    result.update({"device": torch.cuda.get_device_name(), "shape": vars(args),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "tf32": torch.backends.cuda.matmul.allow_tf32,
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "final_gradients_available": sum(p.grad is not None for p in model.parameters())})
    path = os.environ.get("OPENASSEMBLER_STAGE_TIMING_PATH")
    if path:
        Path(path).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mode": result["mode"], "parameters": result["parameters"],
                      "final_gradients_available": result["final_gradients_available"]}))


if __name__ == "__main__":
    main()
