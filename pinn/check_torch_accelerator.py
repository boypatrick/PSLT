#!/usr/bin/env python3
"""Check Torch accelerator availability for the PINN sandbox."""

from __future__ import annotations

import argparse
import json
import time

from accelerator_utils import probe_torch_accelerators, select_torch_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--benchmark", action="store_true", help="Run a small matmul benchmark on the selected device.")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--repeat", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed in this Python environment.") from exc

    try:
        device, diagnostics = select_torch_device(torch, args.device)
    except Exception as exc:
        diagnostics = probe_torch_accelerators(torch)
        diagnostics["requested_device"] = args.device
        diagnostics["selected_device"] = None
        diagnostics["selection_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(diagnostics, indent=2))
        raise SystemExit(2) from exc

    if args.benchmark:
        torch.manual_seed(17)
        a = torch.randn(args.n, args.n, device=device)
        b = torch.randn(args.n, args.n, device=device)
        # Warm up once.  MPS/CUDA are asynchronous, so synchronize when possible.
        c = a @ b
        if str(device) == "mps":
            torch.mps.synchronize()
        elif str(device) == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            c = a @ b
        if str(device) == "mps":
            torch.mps.synchronize()
        elif str(device) == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        diagnostics["benchmark"] = {
            "op": "matmul",
            "n": args.n,
            "repeat": args.repeat,
            "elapsed_s": elapsed,
            "mean_s": elapsed / max(args.repeat, 1),
            "checksum": float(c.detach().cpu()[0, 0]),
        }

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
