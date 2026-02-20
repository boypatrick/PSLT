#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess, sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=False)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    run([sys.executable, "run_rho_window_grid_refine.py"])
    run([sys.executable, "transition_band_scan.py", "transition-band", "--preset", "band",
         "--tcoh-mode", "geom", "--tcoh-extrap", "linear-log"])
    print("\n[OK] Full v3p7 pipeline finished.")

if __name__ == "__main__":
    main()
