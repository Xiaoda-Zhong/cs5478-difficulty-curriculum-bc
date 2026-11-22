#!/usr/bin/env python3
"""Evaluate ranking-corruption checkpoints on the M7 development benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ranking_robustness import (  # noqa: E402
    CORRUPTION_LEVELS, DATASET_FINGERPRINT, RANKING_SEED, TRAINING_SEEDS,
    sha256_file,
)
from scripts.evaluate_curricula import evaluate_checkpoint_job  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, choices=CORRUPTION_LEVELS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    levels = tuple(args.levels or CORRUPTION_LEVELS); seeds = tuple(args.seeds or TRAINING_SEEDS)
    if args.smoke:
        levels, seeds = (levels[0],), (seeds[0],)
    root_name = "ranking_robustness_v1_smoke" if args.smoke else "ranking_robustness_v1"
    checkpoint_root = ROOT / "outputs/checkpoints" / root_name
    result_root = ROOT / "outputs/results" / root_name / "evaluations"
    task_limit = 3 if args.smoke else 100
    expected_steps = (20, 40, 60) if args.smoke else (2060, 5150, 10300, 15450, 20600)
    jobs: list[dict[str, Any]] = []
    checkpoint_records = []
    for level in levels:
        ranking_path = ROOT / f"data/metadata/ranking_robustness_v1/learner_ranking_corruption_{level:03d}.json"
        ranking_hash = sha256_file(ranking_path)
        for seed in seeds:
            run_dir = checkpoint_root / f"corruption_{level:03d}" / f"seed_{seed}"
            for step in expected_steps:
                checkpoint_path = run_dir / f"step_{step:06d}.pt"
                if not checkpoint_path.exists():
                    raise SystemExit(f"Missing ranking-robustness checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if checkpoint["dataset_fingerprint"] != DATASET_FINGERPRINT:
                    raise ValueError("Checkpoint dataset fingerprint mismatch")
                if checkpoint["method_name"] != "reverse_learner" or int(checkpoint["training_seed"]) != seed:
                    raise ValueError("Checkpoint method or training seed mismatch")
                if int(checkpoint["optimizer_step"]) != step or checkpoint["ranking_fingerprint"] != ranking_hash:
                    raise ValueError("Checkpoint update or corrupted-ranking fingerprint mismatch")
                output = result_root / f"corruption_{level:03d}" / f"seed_{seed}" / f"step_{step:06d}.json"
                jobs.append({
                    "checkpoint_path": str(checkpoint_path), "output_path": str(output),
                    "method": f"corruption_{level:03d}", "seed": seed,
                    "optimizer_step": step, "task_limit": task_limit,
                    "overwrite": args.overwrite, "horizon": 400,
                    "task_path": str(ROOT / "data/metadata/id_eval_tasks_v1.json"),
                    "evaluation_split": "m7_development_id", "retain_failure_traces": 3,
                })
                checkpoint_records.append({
                    "corruption_percentage": level, "training_seed": seed,
                    "optimizer_step": step, "path": str(checkpoint_path.relative_to(ROOT)),
                    "sha256": sha256_file(checkpoint_path),
                })
    pending = [job for job in jobs if job["overwrite"] or not Path(job["output_path"]).exists()]
    workers = max(1, int(args.workers or config["curriculum"]["evaluation_workers"]))
    print(f"Ranking evaluation jobs: total={len(jobs)}, pending={len(pending)}, workers={workers}")
    if workers == 1:
        for index, job in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {evaluate_checkpoint_job(job)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(evaluate_checkpoint_job, job) for job in pending]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                print(f"[{index}/{len(pending)}] {future.result()}", flush=True)
    manifest = {
        "version": 1, "date_completed_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory_follow_up": True, "dataset_fingerprint": DATASET_FINGERPRINT,
        "ranking_seed": RANKING_SEED, "corruption_levels": list(levels),
        "training_seeds": list(seeds), "evaluation_tasks": "data/metadata/id_eval_tasks_v1.json",
        "evaluation_seed": 15478, "task_count": task_limit, "rollout_horizon": 400,
        "checkpoint_steps": list(expected_steps), "checkpoint_records": checkpoint_records,
        "zero_corruption_reference": "existing M7 Reverse Learner checkpoints and evaluations; not retrained",
        "smoke": args.smoke,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
