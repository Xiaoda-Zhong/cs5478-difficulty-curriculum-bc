#!/usr/bin/env python3
"""Evaluate all fixed M7 final checkpoints on frozen M8 benchmarks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.final_benchmark import (  # noqa: E402
    DATASET_FINGERPRINT, METHODS, SEEDS, final_checkpoint_path, sha256_file,
    validate_final_checkpoint,
)
from scripts.evaluate_curricula import evaluate_checkpoint_job  # noqa: E402

SPLITS = {
    "final_id": ("data/metadata/final_id_eval_tasks_v1.json", 200),
    "precision_ood": ("data/metadata/final_precision_ood_v1.json", 100),
    "spatial_ood": ("data/metadata/final_spatial_ood_v1.json", 100),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = REPOSITORY_ROOT / "data/metadata/final_benchmark_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise SystemExit("Final benchmark manifest fingerprint mismatch")
    split_names = args.splits or (["final_id", "precision_ood", "spatial_ood"] if manifest["spatial_ood_included"] else ["final_id", "precision_ood"])
    for relative, expected_hash in manifest["configuration_sha256"].items():
        if sha256_file(REPOSITORY_ROOT / relative) != expected_hash:
            raise SystemExit(f"Frozen final benchmark configuration changed: {relative}")
    checkpoint_records = []
    for method in METHODS:
        for seed in SEEDS:
            path = final_checkpoint_path(REPOSITORY_ROOT, method, seed)
            validate_final_checkpoint(path, method, seed)
            checkpoint_records.append({"method": method, "seed": seed, "path": str(path.relative_to(REPOSITORY_ROOT)), "sha256": sha256_file(path)})
    jobs = []
    root = REPOSITORY_ROOT / "outputs/results/final_evaluation_v1"
    methods = ("uniform",) if args.smoke else METHODS
    seeds = (5478,) if args.smoke else SEEDS
    for split in split_names:
        relative_path, count = SPLITS[split]
        task_path = REPOSITORY_ROOT / relative_path
        if not task_path.exists():
            raise SystemExit(f"Missing frozen evaluation split: {task_path}")
        payload = json.loads(task_path.read_text())
        if payload["count"] != count:
            raise ValueError(f"Unexpected task count for {split}")
        for method in methods:
            for seed in seeds:
                checkpoint = final_checkpoint_path(REPOSITORY_ROOT, method, seed)
                output = root / ("smoke" if args.smoke else "evaluations") / split / method / f"seed_{seed}" / "final.json"
                jobs.append({
                    "checkpoint_path": str(checkpoint), "output_path": str(output),
                    "method": method, "seed": seed, "optimizer_step": 20600,
                    "task_limit": 3 if args.smoke else count, "overwrite": args.overwrite,
                    "horizon": 400, "task_path": str(task_path),
                    "evaluation_split": split, "retain_failure_traces": 3,
                })
    pending = [job for job in jobs if job["overwrite"] or not Path(job["output_path"]).exists()]
    print(f"Final evaluation jobs: total={len(jobs)}, pending={len(pending)}, workers={args.workers}")
    if args.workers == 1:
        for index, job in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {evaluate_checkpoint_job(job)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(evaluate_checkpoint_job, job) for job in pending]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                print(f"[{index}/{len(pending)}] {future.result()}", flush=True)
    run_manifest = {
        "version": 1, "date_completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": DATASET_FINGERPRINT, "benchmark_manifest": str(manifest_path.relative_to(REPOSITORY_ROOT)),
        "splits": split_names, "methods": list(methods), "seeds": list(seeds),
        "checkpoint_selection": "final.pt at optimizer update 20,600",
        "checkpoint_records": checkpoint_records, "rollout_horizon": 400,
        "workers": args.workers, "smoke": args.smoke,
    }
    destination = root / ("smoke_manifest.json" if args.smoke else "evaluation_manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
