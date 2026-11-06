"""Bounded BF16 numerical sensitivity study; never changes benchmark gates."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = ROOT / "exp/whole_stack" if (ROOT / "exp/whole_stack").is_dir() else _SCRIPT.parent
APP = modal.App("openassembler-bf16-numerical-diagnostic")
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/openassembler", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(HERE), "/root/whole_stack", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc", "runs/**", "native_builds/**"]))


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=240, max_containers=1, retries=0)
def execute():
    import copy
    import hashlib
    import sys
    import time
    import torch
    import torch._inductor.config as inductor_config
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    sys.path.insert(0, "/root/whole_stack")
    from model import TinyTransformer

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    inductor_config.fallback_random = True
    inductor_config.compile_threads = 1

    width, batch, seq = 512, 8, 128
    initial = TinyTransformer(512, width, 8, 2, seq).cuda()
    token_pool = torch.randint(512, (batch, seq + 1), device="cuda")
    indices = [torch.randint(0, batch, (batch,), device="cuda") for _ in range(28)]
    batches = [token_pool[index] for index in indices] + [token_pool]
    initial_weights = {name: parameter.detach().cpu().clone() for name, parameter in initial.named_parameters()}
    graphs = []

    def compile_graph(graph, inputs):
        graphs.append({"nodes": len(list(graph.graph.nodes)),
                       "operations": [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]})
        return make_boxed_func(torch._inductor.compile(graph, inputs))

    backend = aot_autograd(fw_compiler=compile_graph, bw_compiler=compile_graph)
    compiled_forward = torch.compile(TinyTransformer.forward, backend=backend, dynamic=False, fullgraph=True)

    def full_logits(model, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = model.token(tokens) + model.position(positions).unsqueeze(0)
        for block in model.blocks:
            x = block(x)
        return model.lm_head(model.norm(x))

    compiled_full_logits = torch.compile(full_logits, backend=backend, dynamic=False, fullgraph=True)

    def tensor_stats(a, b):
        af, bf = a.float(), b.float()
        delta = (af - bf).abs()
        return {"max_abs_difference": float(delta.max()),
                "relative_l2_difference": float(torch.linalg.vector_norm(af - bf) /
                                                torch.linalg.vector_norm(af).clamp_min(1e-30)),
                "eager_l2": float(torch.linalg.vector_norm(af)),
                "compiled_l2": float(torch.linalg.vector_norm(bf)),
                "eager_max_abs": float(af.abs().max()),
                "compiled_max_abs": float(bf.abs().max()),
                "elements_outside_benchmark_tolerance": int((delta > 0.005 + 0.05 * af.abs()).sum()),
                "elements": a.numel()}

    def parameter_comparison(a, b):
        rows = []
        for name in a:
            stats = tensor_stats(a[name], b[name])
            if stats["elements_outside_benchmark_tolerance"]:
                rows.append({"name": name, **stats})
        return {"failed_parameter_tensors": len(rows), "failed_parameters": rows,
                "all_parameter_elements_checked": sum(value.numel() for value in a.values())}

    def run_method(compiled, epsilon):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, eps=epsilon)
        checkpoints = {}
        for step, tokens in enumerate(batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            if step == 1:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = compiled_full_logits(model, tokens[:, :seq]) if compiled else full_logits(model, tokens[:, :seq])
                initial_logits = logits.detach().cpu().clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = compiled_forward(model, tokens[:, :seq], tokens[:, 1:]) if compiled else model(tokens[:, :seq], tokens[:, 1:])
            loss.backward()
            if step in (1, 29):
                raw_bias_grads = {name: parameter.grad.detach().cpu().clone()
                                  for name, parameter in model.named_parameters()
                                  if name.endswith("qkv.bias")}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step in (1, 29):
                clipped_bias_grads = {name: parameter.grad.detach().cpu().clone()
                                      for name, parameter in model.named_parameters()
                                      if name.endswith("qkv.bias")}
            optimizer.step()
            if step in (1, 29):
                checkpoints[step] = {
                    "loss": float(loss.detach()), "preclip_total_gradient_norm": float(norm),
                    "raw_qkv_bias_gradients": raw_bias_grads,
                    "clipped_qkv_bias_gradients": clipped_bias_grads,
                    "parameters": {name: parameter.detach().cpu().clone()
                                   for name, parameter in model.named_parameters()},
                }
                if step == 1:
                    checkpoints[step]["initial_full_logits"] = initial_logits
        torch.cuda.synchronize()
        return checkpoints

    results = []
    for epsilon in (1e-8, 1e-4):
        eager = run_method(False, epsilon)
        compiled = run_method(True, epsilon)
        checkpoints = []
        for step in (1, 29):
            a, b = eager[step], compiled[step]
            biases = []
            for name in a["raw_qkv_bias_gradients"]:
                for index, component in enumerate(("query", "key", "value")):
                    section = slice(index * width, (index + 1) * width)
                    ga = a["raw_qkv_bias_gradients"][name][section]
                    gb = b["raw_qkv_bias_gradients"][name][section]
                    pa = a["parameters"][name][section]
                    pb = b["parameters"][name][section]
                    initial_bias = initial_weights[name][section]
                    biases.append({"name": name, "component": component,
                                   "raw_gradient": tensor_stats(ga, gb),
                                   "clipped_gradient": tensor_stats(a["clipped_qkv_bias_gradients"][name][section],
                                                                     b["clipped_qkv_bias_gradients"][name][section]),
                                   "parameter": tensor_stats(pa, pb),
                                   "eager_parameter_delta_l2": float(torch.linalg.vector_norm(pa - initial_bias)),
                                   "compiled_parameter_delta_l2": float(torch.linalg.vector_norm(pb - initial_bias)),
                                   "raw_gradient_sign_disagreement_count": int(((ga.sign() != gb.sign()) & ((ga != 0) | (gb != 0))).sum()),
                                   "raw_gradient_abs_below_1e_6_eager": int((ga.abs() < 1e-6).sum()),
                                   "raw_gradient_abs_below_1e_6_compiled": int((gb.abs() < 1e-6).sum())})
            checkpoints.append({"step": step, "eager_loss": a["loss"], "compiled_loss": b["loss"],
                                "eager_preclip_norm": a["preclip_total_gradient_norm"],
                                "compiled_preclip_norm": b["preclip_total_gradient_norm"],
                                "parameter_check": parameter_comparison(a["parameters"], b["parameters"]),
                                "qkv_bias_components": biases})
            if step == 1:
                checkpoints[-1]["initial_full_logits"] = {
                    **tensor_stats(a["initial_full_logits"], b["initial_full_logits"]),
                    "shape": list(a["initial_full_logits"].shape),
                    "eager_sha256": hashlib.sha256(a["initial_full_logits"].view(torch.uint8).numpy().tobytes()).hexdigest(),
                    "compiled_sha256": hashlib.sha256(b["initial_full_logits"].view(torch.uint8).numpy().tobytes()).hexdigest(),
                    "all_elements_compared": True,
                }
        results.append({"adam_epsilon": epsilon, "checkpoints": checkpoints})
    return {"schema": 1, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "elapsed_s": time.monotonic() - started,
            "model_source_sha256": hashlib.sha256(Path("/root/whole_stack/model.py").read_bytes()).hexdigest(),
            "precision": "FP32 master parameters with BF16 autocast; TF32 disabled",
            "shape": {"batch": batch, "seq": seq, "width": width, "heads": 8, "layers": 2, "vocab": 512},
            "comparison": "same initial model/data/batches; exact worker-style AOTAutograd+Inductor forward/backward lowering; 29 AdamW updates; steps1 and29; eps fixed independently per matched pair",
            "benchmark_gate": {"rtol": 0.05, "atol": 0.005},
            "scope": "numerical sensitivity diagnostic; epsilon variants are separate training policies, not performance improvements or changes to existing verification gates",
            "graphs": graphs, "results": results}


@APP.local_entrypoint()
def main(out: str = ""):
    result = execute.remote()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(out) if out else Path("exp/whole_stack/runs") / f"bf16-diagnostic-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
