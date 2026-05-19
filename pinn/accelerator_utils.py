"""Torch accelerator helpers for the PINN sandbox."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def macos_version() -> str | None:
    """Return the macOS product version when `sw_vers` is available."""
    try:
        out = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    except Exception:  # pragma: no cover - best-effort environment probe
        return None
    return out or None


def probe_torch_accelerators(torch_module: Any) -> dict[str, Any]:
    """Collect conservative Torch accelerator diagnostics."""
    torch = torch_module
    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    mps_tensor_ok = False
    mps_error = None
    if mps_built:
        try:
            x = torch.ones(4, device="mps")
            y = (x * x).sum()
            mps_tensor_ok = str(y.device) == "mps"
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime
            mps_error = f"{type(exc).__name__}: {exc}"

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "macos_version": macos_version(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_tensor_ok": mps_tensor_ok,
        "mps_error": mps_error,
    }


def select_torch_device(torch_module: Any, requested: str = "auto"):
    """Return `(torch.device, diagnostics)` for a requested device.

    `auto` is intentionally conservative: CUDA wins when available, then MPS
    only if a real tensor operation succeeds, otherwise CPU.  Explicit `mps` or
    `cuda` requests fail if the backend cannot execute a trivial tensor op.
    """
    torch = torch_module
    requested = requested.lower()
    diagnostics = probe_torch_accelerators(torch)
    diagnostics["requested_device"] = requested

    if requested == "cpu":
        diagnostics["selected_device"] = "cpu"
        diagnostics["fallback_reason"] = None
        return torch.device("cpu"), diagnostics

    if requested == "cuda":
        if diagnostics["cuda_available"]:
            diagnostics["selected_device"] = "cuda"
            diagnostics["fallback_reason"] = None
            return torch.device("cuda"), diagnostics
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    if requested == "mps":
        if diagnostics["mps_tensor_ok"]:
            diagnostics["selected_device"] = "mps"
            diagnostics["fallback_reason"] = None
            return torch.device("mps"), diagnostics
        reason = diagnostics.get("mps_error") or "torch.backends.mps.is_available() is false"
        raise RuntimeError(f"MPS was requested, but it is not usable: {reason}")

    if requested != "auto":
        raise ValueError(f"Unknown device request: {requested}")

    if diagnostics["cuda_available"]:
        diagnostics["selected_device"] = "cuda"
        diagnostics["fallback_reason"] = None
        return torch.device("cuda"), diagnostics
    if diagnostics["mps_tensor_ok"]:
        diagnostics["selected_device"] = "mps"
        diagnostics["fallback_reason"] = None
        return torch.device("mps"), diagnostics

    reason = diagnostics.get("mps_error")
    if reason:
        diagnostics["fallback_reason"] = f"MPS probe failed: {reason}"
    elif diagnostics["mps_built"]:
        diagnostics["fallback_reason"] = "MPS is built but not available in this runtime."
    else:
        diagnostics["fallback_reason"] = "No CUDA or MPS backend is available."
    diagnostics["selected_device"] = "cpu"
    return torch.device("cpu"), diagnostics
