"""Optional small nonlinear classifier; exported inference uses NumPy only."""

from __future__ import annotations

import numpy as np

from .training import _device, _torch


def fit_mlp_detectors(x, y, *, checkpoints=(25, 50, 100, 200, 400), seed=48101, device="auto"):
    """Fit on training rows only and return fixed epoch candidates for validation."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=bool)
    if (
        x.ndim != 2
        or x.shape[1] < 1
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or len(np.unique(y)) != 2
        or not checkpoints
        or any(not isinstance(e, int) or e < 1 for e in checkpoints)
    ):
        raise ValueError(
            "Need finite training features, both labels and positive epoch checkpoints"
        )
    center = np.median(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-4)
    z = np.clip((x - center) / scale, -10, 10)
    torch = _torch()
    torch.manual_seed(seed)
    selected_device = _device(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    ).to(selected_device)
    values = torch.as_tensor(z, dtype=torch.float32, device=selected_device)
    labels = torch.as_tensor(y.astype(np.float32), device=selected_device)
    balanced = np.where(y, 0.5 / y.sum(), 0.5 / (~y).sum())
    weights = torch.as_tensor(balanced, dtype=torch.float32, device=selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    results = []
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        logits = model(values).squeeze(1)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            * weights
        ).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite nonlinear detector loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if epoch in checkpoints:
            layers = [
                {
                    "weight": layer.weight.detach().cpu().numpy().tolist(),
                    "bias": layer.bias.detach().cpu().numpy().tolist(),
                }
                for layer in model
                if isinstance(layer, torch.nn.Linear)
            ]
            results.append(
                {
                    "schema_version": 1,
                    "classifier": "tanh_mlp_v1",
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                    "layers": layers,
                    "regularization": 0.01,
                    "epochs": epoch,
                    "seed": seed,
                    "training_loss": float(loss.detach().cpu()),
                    "score_note": "Logit score, not a calibrated probability",
                }
            )
            print(
                f"Nonlinear head epoch {epoch}: training loss {float(loss.detach().cpu()):.4f}",
                flush=True,
            )
    return results
