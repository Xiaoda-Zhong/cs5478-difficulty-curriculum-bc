#!/usr/bin/env python3
"""Run deterministic fixed-task stacking expert regression episodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_stacking import PandaStackingEnv  # noqa: E402
from expert.stacking_expert import ScriptedStackingExpert, rollout_stacking_expert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    records = []
    with PandaStackingEnv(mode="direct") as env:
        for index in range(args.episodes):
            env.reset()
            trace = rollout_stacking_expert(env, ScriptedStackingExpert())
            result = trace["result"]
            record = {
                "episode": index + 1, "success": result.success,
                "failure_reason": result.failure_reason, "steps": result.steps,
                "phase_history": list(result.phase_history),
                "final_source_position": result.final_source_position.tolist(),
                "final_support_position": result.final_support_position.tolist(),
                "final_alignment_error": result.final_diagnostics["horizontal_alignment_error"],
                "maximum_support_displacement": result.maximum_support_displacement,
                "final_physical_contact": result.final_diagnostics["source_support_contact"],
                "stable_steps": result.final_diagnostics["stable_steps"],
            }
            records.append(record)
            status = "SUCCESS" if result.success else f"FAILURE ({result.failure_reason})"
            print(f"Episode {index + 1:02d}: {status} | steps={result.steps}")
    successful = sum(row["success"] for row in records)
    deterministic = all(
        row["steps"] == records[0]["steps"]
        and row["phase_history"] == records[0]["phase_history"]
        and np.allclose(row["final_source_position"], records[0]["final_source_position"], atol=1e-6)
        for row in records[1:]
    ) if records else True
    summary = {
        "task": "object_stacking", "distribution": "fixed default task",
        "episodes": args.episodes, "successes": successful,
        "success_rate": successful / args.episodes,
        "deterministic_reproduction": deterministic,
        "failure_reasons": dict(Counter(row["failure_reason"] for row in records if not row["success"])),
        "records": records,
    }
    output = ROOT / "outputs/results/stacking_s2/fixed_task_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSuccess rate: {successful}/{args.episodes}; deterministic={deterministic}")
    print(f"Saved: {output}")
    if successful != args.episodes or not deterministic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
