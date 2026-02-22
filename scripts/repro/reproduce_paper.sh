#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_DIR="$ROOT_DIR/repro/runs"
MAP_FILE="$ROOT_DIR/repro/artifact_map.csv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_ID=""
WITH_PAPER=0
PACKAGE_ONLY=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/repro/reproduce_paper.sh [options]

Options:
  --run-id <id>      Use explicit run id (default: UTC timestamp + git short sha).
  --with-paper       Compile paper/main.tex via latexmk after generation steps.
  --package-only     Skip generation; package current outputs only.
  --dry-run          Print steps without executing commands.
  -h, --help         Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --with-paper)
      WITH_PAPER=1
      shift
      ;;
    --package-only)
      PACKAGE_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  short_sha="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo no-git)"
  RUN_ID="${stamp}_${short_sha}"
fi

RUN_DIR="$RUNS_DIR/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
STEPS_TSV="$RUN_DIR/steps.tsv"
RUN_META="$RUN_DIR/run_meta.json"

mkdir -p "$LOG_DIR"
printf "step\tname\tstatus\tstart_utc\tend_utc\tduration_s\tcommand\n" > "$STEPS_TSV"

run_step() {
  local step="$1"
  local name="$2"
  local cmd="$3"
  local log_file="$LOG_DIR/${step}_${name}.log"
  local start_utc end_utc status duration rc t0 t1

  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[step $step] $name"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $cmd" | tee "$log_file"
    end_utc="$start_utc"
    duration="0"
    status="dry-run"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$step" "$name" "$status" "$start_utc" "$end_utc" "$duration" "$cmd" >> "$STEPS_TSV"
    return 0
  fi

  t0="$(date +%s)"
  set +e
  (
    cd "$ROOT_DIR"
    eval "$cmd"
  ) > "$log_file" 2>&1
  rc=$?
  set -e
  t1="$(date +%s)"
  duration="$((t1 - t0))"
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "$rc" -eq 0 ]]; then
    status="ok"
  else
    status="failed(${rc})"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$step" "$name" "$status" "$start_utc" "$end_utc" "$duration" "$cmd" >> "$STEPS_TSV"

  if [[ "$rc" -ne 0 ]]; then
    echo "[error] step $step failed: $name" >&2
    echo "[error] see log: $log_file" >&2
    tail -n 30 "$log_file" >&2 || true
    exit "$rc"
  fi
}

sync_one() {
  local src="$1"
  local dst="$2"
  local required="$3"
  if [[ -f "$ROOT_DIR/$src" ]]; then
    cp -f "$ROOT_DIR/$src" "$ROOT_DIR/$dst"
    echo "[sync] $src -> $dst"
    return 0
  fi
  if [[ "$required" -eq 1 ]]; then
    echo "[sync-error] missing required source: $src" >&2
    return 1
  fi
  echo "[sync-warn] missing optional source: $src"
  return 0
}

sync_paper_assets() {
  sync_one "output/three_generation_phase_diagram.png" "paper/three_generation_phase_diagram.png" 1
  sync_one "output/three_generation_bars.png" "paper/three_generation_bars.png" 1
  sync_one "output/hmumu_exclusion.png" "paper/hmumu_exclusion.png" 1
  sync_one "output/hmumu_signal_strength.png" "paper/hmumu_signal_strength.png" 1
  sync_one "output/hll_signal_strength/hll_signal_strength_maps.png" "paper/hll_signal_strength_maps.png" 1
  sync_one "output/hll_signal_strength/hll_signal_strength_summary.csv" "paper/hll_signal_strength_summary.csv" 1
  sync_one "output/true_single_track/Veff_true_D12.png" "paper/Veff_true_D12.png" 1
  sync_one "output/true_single_track/true_single_track.png" "paper/true_single_track.png" 1
  sync_one "output/chi_fp_2d/chi_wavefunction_contours_D6_12_18.png" "paper/chi_wavefunction_contours_D6_12_18.png" 1
  sync_one "output/chi_fp_2d/chi_scale_stress_test.png" "paper/chi_scale_stress_test.png" 1
  sync_one "output/chi_fp_2d/surrogate_vs_action_points.csv" "paper/surrogate_vs_action_points.csv" 1
  sync_one "output/chi_fp_2d/surrogate_vs_action_points_summary.csv" "paper/surrogate_vs_action_points_summary.csv" 1
  sync_one "output/robustness/core_param_robustness_table.csv" "paper/core_param_robustness.csv" 1
  sync_one "output/gn_fp_impact/gn_profile_impact.csv" "paper/gn_profile_impact.csv" 1
  sync_one "output/gn_fp_impact/gn_nmax_convergence.csv" "paper/gn_nmax_convergence.csv" 1
  sync_one "output/gn_fp_impact/gn_cardy_vs_phase_space.csv" "paper/gn_cardy_vs_phase_space.csv" 1
  sync_one "output/gn_fp_impact/gn_cardy_vs_phase_space.png" "paper/gn_cardy_vs_phase_space.png" 1
  sync_one "output/chi_open_system/chi_open_system_sensitivity.csv" "paper/chi_open_system_sensitivity.csv" 1
  sync_one "output/first_principles_migration/first_principles_migration_summary.csv" "paper/first_principles_migration_summary.csv" 1
  sync_one "output/first_principles_migration/first_principles_migration_summary.png" "paper/first_principles_migration_summary.png" 1
  sync_one "output/tcoh_fp_1d/tcoh_profile_impact.csv" "paper/tcoh_profile_impact.csv" 1
  sync_one "output/eta_fp_1d/eta_profile_impact.csv" "paper/eta_profile_impact.csv" 1
  sync_one "output/superrad_fp_1d/superrad_profile_impact.csv" "paper/superrad_profile_impact.csv" 1
  sync_one "output/chi_fp_2d/chi_profile_robustness.csv" "paper/chi_profile_robustness.csv" 1
}

cat > "$RUN_META" <<EOF
{
  "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "run_id": "$RUN_ID",
  "repo_root": "$ROOT_DIR",
  "package_only": $PACKAGE_ONLY,
  "with_paper": $WITH_PAPER,
  "dry_run": $DRY_RUN
}
EOF

if [[ "$PACKAGE_ONLY" -eq 0 ]]; then
  run_step "01" "extract_chi_localized_knots" \
    "$PYTHON_BIN code/extract_chi_localized_2d.py --Ds 6,12,18"
  run_step "02" "extract_chi_localized_fullscan" \
    "$PYTHON_BIN code/extract_chi_localized_2d.py --full-scan"
  run_step "03" "scan_chi_profile_robustness" \
    "$PYTHON_BIN code/scan_chi_profile_robustness.py"
  run_step "03b" "scan_surrogate_vs_action_points" \
    "$PYTHON_BIN code/scan_surrogate_vs_action_points.py"
  run_step "03c" "scan_core_param_robustness" \
    "$PYTHON_BIN code/scan_core_param_robustness.py"
  run_step "04" "plot_chi_wavefunction_contours" \
    "$PYTHON_BIN code/plot_chi_wavefunction_contours.py"
  run_step "05" "plot_chi_scale_stress_test" \
    "$PYTHON_BIN code/plot_chi_scale_stress_test.py"
  run_step "06" "extract_omega_exact_convergence" \
    "$PYTHON_BIN code/extract_omega_exact_convergence.py --Ds 6,12,18"
  run_step "07" "extract_tcoh_knots" \
    "$PYTHON_BIN code/extract_tcoh_dephasing_1d.py --Ds 6,12,18"
  run_step "08" "extract_tcoh_fullscan" \
    "$PYTHON_BIN code/extract_tcoh_dephasing_1d.py --full-scan"
  run_step "09" "scan_tcoh_profile_impact" \
    "$PYTHON_BIN code/scan_tcoh_profile_impact.py"
  run_step "10" "extract_eta_knots" \
    "$PYTHON_BIN code/extract_eta_prefactor_1d.py --Ds 6,12,18"
  run_step "11" "extract_eta_fullscan" \
    "$PYTHON_BIN code/extract_eta_prefactor_1d.py --full-scan"
  run_step "12" "scan_eta_profile_impact" \
    "$PYTHON_BIN code/scan_eta_profile_impact.py"
  run_step "13" "extract_superrad_knots" \
    "$PYTHON_BIN code/extract_superrad_prefactor_1d.py --Ds 6,12,18"
  run_step "14" "extract_superrad_fullscan" \
    "$PYTHON_BIN code/extract_superrad_prefactor_1d.py --full-scan"
  run_step "15" "scan_superrad_profile_impact" \
    "$PYTHON_BIN code/scan_superrad_profile_impact.py"
  run_step "16" "verify_omega_geometric_origin" \
    "$PYTHON_BIN code/verify_omega_geometric_origin.py"
  run_step "17" "extract_gn_phase_space_1d" \
    "$PYTHON_BIN code/extract_gn_phase_space_candidate.py --D 12"
  run_step "18" "extract_gn_phase_space_2d" \
    "$PYTHON_BIN code/extract_gn_phase_space_2d.py --Ds 6,12,18"
  run_step "19" "scan_gn_profile_impact" \
    "$PYTHON_BIN code/scan_gn_profile_impact.py"
  run_step "19b" "scan_gn_nmax_convergence" \
    "$PYTHON_BIN code/scan_gn_nmax_convergence.py --nmax-list 20,30,40"
  run_step "19c" "plot_gn_cardy_vs_phase_space" \
    "$PYTHON_BIN code/plot_gn_cardy_vs_phase_space.py"
  run_step "20" "lindblad_chi_minimal" \
    "$PYTHON_BIN code/lindblad_chi_minimal.py"
  run_step "21" "extract_chi_open_knots" \
    "$PYTHON_BIN code/extract_chi_open_system_geometry.py --Ds 6,12,18"
  run_step "22" "extract_chi_open_fullscan" \
    "$PYTHON_BIN code/extract_chi_open_system_geometry.py --full-scan"
  run_step "23" "scan_chi_open_sensitivity" \
    "$PYTHON_BIN code/scan_chi_open_system_sensitivity.py"
  run_step "23b" "scan_first_principles_migration_summary" \
    "$PYTHON_BIN code/scan_first_principles_migration_summary.py"
  run_step "24" "generate_plots" \
    "$PYTHON_BIN code/generate_plots.py"
  run_step "25" "scan_hll_signal_strengths" \
    "$PYTHON_BIN code/scan_hll_signal_strengths.py"
  run_step "26" "verify_robustness" \
    "$PYTHON_BIN code/verify_robustness.py"
  run_step "27" "validate_2d_laplacian" \
    "$PYTHON_BIN code/validate_2d_laplacian.py"
  run_step "28" "splitting_action_analysis" \
    "$PYTHON_BIN code/splitting_action_analysis.py"
  run_step "29" "true_single_track_solver" \
    "$PYTHON_BIN code/true_single_track_solver.py"
  run_step "30" "generate_physical_intuition_figs" \
    "$PYTHON_BIN code/generate_physical_intuition_figs.py"
  run_step "31" "sync_paper_assets" "sync_paper_assets"
fi

if [[ "$WITH_PAPER" -eq 1 ]]; then
  run_step "32" "compile_paper" \
    "cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex"
fi

run_step "40" "package_outputs" \
  "$PYTHON_BIN scripts/repro/package_repro_outputs.py --root '$ROOT_DIR' --run-dir '$RUN_DIR' --map-file '$MAP_FILE'"

if [[ "$DRY_RUN" -eq 0 ]]; then
  ln -sfn "runs/$RUN_ID" "$ROOT_DIR/repro/latest"
fi

echo "[done] run_id=$RUN_ID"
echo "[done] run_dir=$RUN_DIR"
