# S4.5 — Diagnose and Stabilize Stacking Uniform BC

The frozen dataset, expert, task distribution, physics, success detector, and original S4 checkpoint were not changed. The selected S4.5 policy is `relative_phase_balanced_v1`; it reached 9/100 on the frozen development benchmark, compared with 1/100 for S4.

## Diagnosis

The global offline score concealed phase-local errors. Validation results below are mean motion L2 error in millimetres (baseline → selected), with baseline transition counts:

| Phase | N | Error (mm) |
|---|---:|---:|
| RAISE | 480 | 0.195 → 0.153 |
| APPROACH_SOURCE | 1605 | 0.354 → 0.214 |
| DESCEND_SOURCE | 1980 | 0.269 → 0.215 |
| GRASP | 600 | 0.996 → 0.826 |
| LIFT | 1980 | 0.255 → 0.136 |
| TRANSPORT_ABOVE_SUPPORT | 2418 | 0.386 → 0.421 |
| ALIGN | 223 | 2.771 → 1.238 |
| LOWER_TO_PRECONTACT | 1765 | 0.521 → 0.388 |
| LOWER_TO_CONTACT | 546 | 0.574 → 0.473 |
| RELEASE | 720 | 1.218 → 1.283 |
| RETREAT | 1620 | 0.137 → 0.190 |
| VERIFY_STACK | 240 | 0.256 → 0.137 |

ALIGN was both scarce and the least accurately imitated moving phase. Its direction cosine improved from 0.885 to 0.960. Baseline traces also showed one-step gripper spikes and, in the worst sampled episode, 111 command changes. First releases often occurred before contact and at large horizontal error. High offline gripper accuracy therefore did not imply reliable closed-loop switching.

Raw-channel ordering and units are consistent across collection, training, and inference. The 1e-4 epsilon avoids division by zero but does not prevent closed-loop covariate shift: sampled BC rollouts reached normalized source angular velocity near ±4,405 and normalized source orientation near ±3,817. A conservative ±10 normalized-input clipping test reduced the selected policy from 4/20 to 2/20 and was rejected.

## Controlled experiments

All prefix experiments used frozen development tasks 0–19:

| Variant | Change | Success |
|---|---|---:|
| Original S4 | none | 0/20 on prefix; 1/100 frozen result |
| `relative_v1` | observable relative positions | 0/20 |
| `phase_balanced_v1` | 5× effective weight for four precision phases | 0/20 |
| Original + debounce3 | three consistent predictions per gripper switch | 0/20 |
| `relative_phase_balanced_v1` | both supervised changes | 4/20 |
| Selected + clip10 | normalized input clipping | 2/20 |

The selected representation remains 29D. It retains absolute EE position, replaces source position by source-minus-EE, and replaces support position by source-minus-support. All remaining channels are unchanged. Training adds four extra copies of ALIGN, LOWER_TO_PRECONTACT, LOWER_TO_CONTACT, and RELEASE transitions. Phase is never a policy input. This protocol must be applied identically to all future stacking curriculum methods.

The selected checkpoint uses the original 29→128→128 two-head MLP and all original S4 optimizer settings. Its best checkpoint was epoch 93 with validation normalized motion MSE 0.00343171 and gripper accuracy 1.0.

## Development result and limitations

Full frozen development result: 9/100. Among 91 failures: placement/contact 55, post-release instability 15, grasp/lift 12, transport/alignment 8, and release 1. The largest improvement is ALIGN imitation, but compounding state-distribution error and release/contact precision remain unresolved. The policy is a more meaningful baseline, not a reliable stacking controller.

Expert-to-BC handoff experiments were not implemented; the trace comparison already localized the first major divergence to grasp switching/retention in some tasks and alignment/contact in most lifted tasks.

The GUI runner makes no control or physics steps beyond the shared rollout loop; its only sleep occurs after rollout. It uses the same saved task configuration. The earlier DIRECT success that failed in GUI therefore was not caused by visualization-loop stepping or real-time timing in repository code. A new GUI visual run was not performed in S4.5, so backend contact nondeterminism remains the strongest hypothesis and DIRECT remains the benchmark.

## Reproduction

```bash
conda run -n cs5478 python scripts/diagnose_stacking_bc.py --task-count 10
conda run -n cs5478 python scripts/train_stacking_s4_5_variant.py --variant relative_phase_balanced_v1 --device cpu
conda run -n cs5478 python scripts/evaluate_stacking_s4_5_variant.py \
  --checkpoint outputs/stacking_v1/checkpoints/s4_5/relative_phase_balanced_v1/best.pt \
  --task-count 100 \
  --output outputs/stacking_v1/results/s4_5/relative_phase_balanced_v1/development100.json
conda run -n cs5478 python scripts/inspect_stacking_dataset.py
conda run -n cs5478 python -m pytest -q
conda run -n cs5478 python scripts/run_stacking_expert.py --episodes 20
conda run -n cs5478 python scripts/run_expert.py --episodes 20
```

S4.5 stops here. Difficulty scoring, curriculum training, and held-out final evaluation were not started.
