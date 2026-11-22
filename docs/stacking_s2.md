# S2: Deterministic Scripted Stacking Expert

`ScriptedStackingExpert` is a deterministic finite-state controller for
`PandaStackingEnv`. It moves the real Panda through the environment's 4-D
Cartesian/gripper action interface. It never teleports a cube, creates a grasp
constraint, disables collisions, or fixes the support cube.

## Controller

The FSM is:

```text
RAISE -> APPROACH_SOURCE -> DESCEND_SOURCE -> GRASP -> LIFT
-> TRANSPORT_ABOVE_SUPPORT -> ALIGN -> LOWER_TO_PRECONTACT
-> LOWER_TO_CONTACT -> RELEASE -> RETREAT -> VERIFY_STACK -> DONE
```

Movement phases use closed-loop Cartesian displacement capped at 0.008 m per
control action. `LOWER_TO_CONTACT` uses 0.002 m steps. Every phase has a
configured timeout and terminal failures are classified explicitly.

Waypoints are derived from current EE, source, and support positions. The grasp
target is 0.010 m above the source center. For two 0.050 m cubes, the nominal
contact EE height is support center z + 0.050 m + the 0.010 m grasp offset.
Pre-contact is 0.015 m above that height. The final contact target extends
0.008 m below nominal, but descent stops immediately on physical cube contact.

Transport targets use the current support pose. ALIGN, pre-contact descent, and
contact descent close the loop on observed source-to-support XY error rather
than merely centering the EE. This compensates for the carried cube lagging the
gripper during motion.

The fingers close for ten control actions before lift. Grasp/lift verification
requires at least 0.050 m source height gain and no more than 0.080 m
source-to-EE distance. The gripper opens for twelve actions after contact. The
robot then retreats vertically to z=0.30 m and holds open until the unchanged
S1 detector records 20 consecutive physically stable actions.

## Reliability results

The default fixed task succeeded 20/20 times. Every run used 240 expert actions
and reproduced the same phase history and final trajectory within 1e-6.

The fixed-seed randomized S2 distribution uses:

```text
source x:  [0.46, 0.64] m    source y:  [-0.24, -0.08] m
support x: [0.46, 0.64] m    support y: [ 0.08,  0.24] m
seed: 25478
```

The disjoint y bands provide safe initial separation; no task is filtered or
retried based on outcome. The expert succeeded on 100/100 tasks.

```text
Actions:        mean 236.35, std 12.74, min 203, max 264
EE path length: mean 1.6137 m, std 0.1011, min 1.3598, max 1.8365
Action length:  mean 1.6142 m, std 0.1011, min 1.3600, max 1.8377
```

No S1 success threshold was changed. The expert configuration remains subject
to reassessment if S3 broadens the task distribution beyond these conservative
ranges.

## Reproduction

```bash
conda run -n cs5478 python scripts/run_stacking_expert.py
conda run -n cs5478 python scripts/evaluate_randomized_stacking_expert.py
conda run -n cs5478 python scripts/visualize_stacking_expert.py
conda run -n cs5478 python scripts/visualize_stacking_expert.py --randomized --seed 25478 --task-index 0
conda run -n cs5478 python -m pytest tests/test_stacking_expert.py -q
```

S3 dataset collection, serialization, and train/validation splitting are not
implemented.
