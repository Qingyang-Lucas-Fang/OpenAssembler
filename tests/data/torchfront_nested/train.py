import torch
from model.layers import Model
models = [Model().cuda(), Model().cuda()]
optimizer = torch.optim.AdamW([p for model in models for p in model.parameters()], lr=0.001, eps=0.0001)
for step in range(2):
    x = torch.randn(16, 8, device="cuda")
    loss = sum(model(x).square().mean() for model in models)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"step={step + 1} loss={loss.item():.6f}")
