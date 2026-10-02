from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .continuous import ContinuousData, ContinuousParameters


@dataclass(frozen=True)
class TorchContinuousData:
    times: torch.Tensor
    types: torch.Tensor
    basis: torch.Tensor
    end: float

    @classmethod
    def from_numpy(cls, data: ContinuousData, device: torch.device,
                   dtype: torch.dtype = torch.float64) -> "TorchContinuousData":
        return cls(
            torch.as_tensor(data.times, device=device, dtype=dtype),
            torch.as_tensor(data.types, device=device, dtype=torch.long),
            torch.as_tensor(data.basis_values, device=device, dtype=dtype),
            data.end,
        )


def require_cuda(device: str = "cuda") -> torch.device:
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to PyTorch. Run on a host/session with the NVIDIA "
            "driver mounted; verify with `nvidia-smi` and "
            "`python -c \"import torch; print(torch.cuda.is_available())\"`."
        )
    return selected


def device_report(device: torch.device) -> dict[str, object]:
    report: dict[str, object] = {
        "requested_device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        report.update({
            "active_device": f"cuda:{index}",
            "gpu_name": props.name,
            "gpu_memory_gib": round(props.total_memory / 2**30, 2),
            "current_allocated_mib": round(torch.cuda.memory_allocated(index) / 2**20, 2),
            "current_reserved_mib": round(torch.cuda.memory_reserved(index) / 2**20, 2),
        })
    else:
        report["active_device"] = "cpu"
    return report


def torch_continuous_log_likelihood(
    data: TorchContinuousData,
    log_nu: torch.Tensor,
    alpha_coef: torch.Tensor,
    beta_coef: torch.Tensor,
    max_history_seconds: float | None = None,
    smoothness: float = 0.0,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Differentiable M2 likelihood, chunked across target events on GPU.

    Both alpha(z) and beta(z) are evaluated at every source event. A finite
    history cutoff is a numerical acceleration only; validate it against the
    CPU/direct reference before freezing it.
    """
    nu = torch.exp(log_nu)
    # [past event, source type, target type]
    alpha = torch.exp(torch.einsum("nr,str->nst", data.basis, alpha_coef))
    beta = torch.exp(torch.einsum("nr,str->nst", data.basis, beta_coef))
    row = torch.arange(len(data.times), device=data.times.device)
    source = data.types
    observed = torch.nonzero(data.times >= 0, as_tuple=False).flatten()
    event_term = torch.zeros((), device=data.times.device, dtype=data.times.dtype)

    for begin in range(0, len(observed), chunk_size):
        current = observed[begin:begin + chunk_size]
        lags = data.times[current, None] - data.times[None, :]
        mask = lags > 0
        if max_history_seconds is not None:
            mask = mask & (lags <= max_history_seconds)
        intensities = nu[None, :].expand(len(current), 2).clone()
        for target in range(2):
            a = alpha[row, source, target]
            b = beta[row, source, target]
            contribution = a[None, :] * torch.exp(-b[None, :] * torch.clamp(lags, min=0))
            intensities[:, target] += torch.sum(torch.where(mask, contribution, 0.0), dim=1)
        event_term = event_term + torch.log(
            intensities.gather(1, data.types[current, None]).squeeze(1)
        ).sum()

    lower = torch.clamp(-data.times, min=0)
    horizon = data.end - data.times
    compensator = data.end * nu.sum()
    for target in range(2):
        a = alpha[row, source, target]
        b = beta[row, source, target]
        valid = horizon > lower
        integrals = a / b * (torch.exp(-b * lower) - torch.exp(-b * horizon))
        compensator = compensator + torch.where(valid, integrals, 0.0).sum()

    penalty = smoothness * (
        torch.diff(alpha_coef, n=2, dim=-1).square().sum()
        + torch.diff(beta_coef, n=2, dim=-1).square().sum()
    )
    return event_term - compensator - penalty


def parameters_from_torch(log_nu, alpha_coef, beta_coef) -> ContinuousParameters:
    return ContinuousParameters(
        torch.exp(log_nu).detach().cpu().numpy(),
        alpha_coef.detach().cpu().numpy(), beta_coef.detach().cpu().numpy(),
    )
