# Runtime-Direct Detlin Method Decision Table

This note fixes the current method landscape for the strict all-direct visibility closure line.

## Current Reading

- `cell_direct_runtime_release_tailm2_detlin` remains the trusted deterministic de-anchored stress line.
- The residual structure is still split:
  - `D=4.8`: width overshoot
  - `D=6.4`: UV/g-dominated
  - `D=7.2`: small width tail
- The first method that behaves like a genuinely localized `D=6.4` correction is the new `band-pass gnorm` gate.
- Strict all-direct remains open because `D=4.8` and `D=4.0` are still not closed.

## Method Table

| Method | Status | What it does | Why it is not enough yet | Best evidence |
| --- | --- | --- | --- | --- |
| `cell_direct_runtime_release_tailm2_detlin` | Open baseline | Closes the small-surface `D21xE41` mismatch gate and exposes the remaining drift structure | Still fails badly on drift: `max|Δμ_μμ|=68.39`, `p95=9.99` | `/Users/boypatrick/codex/PSLT_quantam/output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_tailm2_detlin_D21E41.csv` |
| Component decomposition | Established diagnosis | Separates `D=4.8` width overshoot from `D=6.4` UV/g dominance and `D=7.2` width tail | Diagnostic only; not a fix | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_detlin_component_audit_summary.csv` |
| Width-family extension | Partial | Strongly reduces the `D=4.8` width problem and helps `D=7.2` | Reopens `D=6.4` and worsens `D=4.0` mismatch | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_detlin_width_family_extension_summary.csv` |
| Coarse `gnorm` family (`uniform/side`) | Not enough | Applies broad UV/g compression to low-mid `D` | Too blunt; does not isolate `D=6.4` and leaves `D=4.0` untouched | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_detlin_gnorm_family_audit_summary.csv` |
| `gmixgate` | Not enough | Tests a wider norm-plus-side direct-only UV/g gate | No jointly admissible candidate; `D=4.8` stays large and `D=6.4` remains open | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_detlin_gmixgate_audit_summary.csv` |
| Low-metric one-sided `gnorm` | Partial but broad | Proves that low-metric UV/g correction can help | Behaves like general low/mid-`D` compression, not a localized `D=6.4` fix | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_detlin_joint_tailgnorm_lowmetric_audit_summary.csv` |
| Band-pass `gnorm` | Promising local fix | Selectively targets the `D=6.4` strip and improves `D=7.2/8.0` without reopening them | Does not close `D=4.8` width overshoot or `D=4.0` mismatch | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_detlin_joint_tailgnorm_bandpass_focus_compare_summary.csv` |
| `bandcombo` / `bandm2` component probes | Partial component hints | Show low-`D` improvements relative to release baseline | Not demonstrated as a strict all-direct closure route | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_lowd_component_audit_summary.csv` |

## Most Defensible Current Conclusion

The methods that still look genuinely possible are:

1. A localized `band-pass gnorm` route for the `D=6.4` UV/g strip.
2. A separate observable-side mechanism for the `D=4.8` width overshoot.
3. A distinct fix for the `D=4.0` mismatch, which remains untouched by the current UV/g gates.

So the current blocker is no longer “we do not know how to touch `D=6.4`.” The blocker is that the strict all-direct line still needs a second mechanism for `D=4.8` and a third mechanism or reframing for `D=4.0`.
