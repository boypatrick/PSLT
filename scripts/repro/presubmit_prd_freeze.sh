#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_DIR="$ROOT_DIR/repro/runs"
MAP_FILE="$ROOT_DIR/repro/artifact_map.csv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_ID=""
SKIP_RECOMPUTE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/repro/presubmit_prd_freeze.sh [options]

Options:
  --run-id <id>      Use explicit run id (default: UTC timestamp + git short sha + _prd_freeze).
  --skip-recompute   Skip scan recomputation; run compile/package/report only.
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
    --skip-recompute)
      SKIP_RECOMPUTE=1
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
  RUN_ID="${stamp}_${short_sha}_prd_freeze"
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
    tail -n 50 "$log_file" >&2 || true
    exit "$rc"
  fi
}

cat > "$RUN_META" <<EOF
{
  "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "run_id": "$RUN_ID",
  "repo_root": "$ROOT_DIR",
  "skip_recompute": $SKIP_RECOMPUTE,
  "dry_run": $DRY_RUN
}
EOF

if [[ "$SKIP_RECOMPUTE" -eq 0 ]]; then
  run_step "01" "py_compile_core" \
    "PYTHONPYCACHEPREFIX=/tmp/pycache $PYTHON_BIN -m py_compile code/eft_rge.py code/pslt_lib.py code/hll_observable.py code/scan_hll_signal_strengths.py code/scan_hll_uv_to_eft_matching.py code/scan_hll_rge_sensitivity.py code/scan_hll_uv_envelope.py code/publish_full_direct_map.py code/build_artifact_status_registry.py code/generate_plots.py code/extract_chi_open_system_micro.py code/calibrate_kappa_env_micro_anchor.py code/scan_chi_open_system_micro_sensitivity.py code/assess_open_system_micro_baseline_candidate.py"
  run_step "02" "scan_hll_uv_to_eft_matching" \
    "MPLCONFIGDIR=/tmp/mpl PYTHONPYCACHEPREFIX=/tmp/pycache $PYTHON_BIN code/scan_hll_uv_to_eft_matching.py"
  run_step "03" "scan_hll_signal_strengths" \
    "MPLCONFIGDIR=/tmp/mpl PYTHONPYCACHEPREFIX=/tmp/pycache $PYTHON_BIN code/scan_hll_signal_strengths.py"
  run_step "04" "scan_hll_rge_sensitivity" \
    "PYTHONPYCACHEPREFIX=/tmp/pycache $PYTHON_BIN code/scan_hll_rge_sensitivity.py"
  run_step "04b" "publish_full_direct_map" \
    "MPLCONFIGDIR=/tmp/mpl PYTHONPYCACHEPREFIX=/tmp/pycache $PYTHON_BIN code/publish_full_direct_map.py"
  run_step "04c" "scan_hll_uv_envelope" \
    "MPLCONFIGDIR=/tmp/mpl PYTHONPYCACHEPREFIX=/tmp/pycache $PYTHON_BIN code/scan_hll_uv_envelope.py --chain-mode full_direct --d-num 21 --eta-num 41"
  run_step "04d" "build_artifact_status_registry" \
    "$PYTHON_BIN code/build_artifact_status_registry.py"
fi

run_step "05" "compile_paper" \
  "cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex"

run_step "06" "package_outputs" \
  "$PYTHON_BIN scripts/repro/package_repro_outputs.py --root '$ROOT_DIR' --run-dir '$RUN_DIR' --map-file '$MAP_FILE'"

run_step "07" "build_presubmit_report" \
  "$PYTHON_BIN scripts/repro/build_presubmit_prd_report.py --root '$ROOT_DIR' --run-id '$RUN_ID' --out-json '$RUN_DIR/presubmit_prd_report.json' --out-md '$RUN_DIR/presubmit_prd_report.md'"

run_step "08" "sync_report_to_paper" \
  "cp -f '$RUN_DIR/presubmit_prd_report.json' '$ROOT_DIR/paper/presubmit_prd_report_latest.json' && cp -f '$RUN_DIR/presubmit_prd_report.md' '$ROOT_DIR/paper/presubmit_prd_report_latest.md'"

if [[ "$DRY_RUN" -eq 0 ]]; then
  ln -sfn "runs/$RUN_ID" "$ROOT_DIR/repro/latest"
fi

echo "[done] run_id=$RUN_ID"
echo "[done] run_dir=$RUN_DIR"
