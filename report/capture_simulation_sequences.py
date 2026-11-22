"""Capture publication panels from successful deterministic PyBullet experts.

This script is intentionally report-only: it reads the existing environments and
scripted experts, advances them exclusively through ``env.step()``, and writes two
composite PNG figures.  No experiment artifact or simulator setting is changed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from envs.panda_stacking import PandaStackingEnv  # noqa: E402
from expert.scripted_expert import ExpertPhase, ScriptedExpert  # noqa: E402
from expert.stacking_expert import (  # noqa: E402
    ScriptedStackingExpert,
    StackingPhase,
)


FRAME_WIDTH = 960
FRAME_HEIGHT = 600


@dataclass
class Capture:
    image: Image.Image
    step: int
    phase: str
    note: str


def render_camera(client_id: int) -> Image.Image:
    """Render the shared oblique table view with PyBullet's camera API."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.48, 0.0, 0.12],
        distance=1.18,
        yaw=48.0,
        pitch=-31.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=48.0,
        aspect=FRAME_WIDTH / FRAME_HEIGHT,
        nearVal=0.05,
        farVal=3.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        shadow=1,
        lightDirection=[-1.0, -1.0, 2.0],
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    pixels = np.asarray(rgba, dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
    return Image.fromarray(pixels[:, :, :3], mode="RGB")


def compose(captures: list[Capture], output: Path) -> None:
    """Compose four camera frames without redundant in-image stage titles."""
    if len(captures) != 4:
        raise ValueError("Exactly four captures are required")
    border = 4
    canvas = Image.new(
        "RGB",
        (4 * FRAME_WIDTH + 5 * border, FRAME_HEIGHT + 2 * border),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, capture in enumerate(captures):
        x = border + index * (FRAME_WIDTH + border)
        canvas.paste(capture.image, (x, border))
        draw.rectangle(
            [x, border, x + FRAME_WIDTH - 1, border + FRAME_HEIGHT - 1],
            outline=(85, 85, 85),
            width=2,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True, dpi=(300, 300))


def capture_pick_and_place() -> tuple[list[Capture], dict[str, object]]:
    """Run the fixed deterministic expert and select four physical states."""
    env = PandaPickPlaceEnv(mode="direct")
    expert = ScriptedExpert()
    try:
        observation = env.reset(seed=5478)
        initial_distance = float(
            np.linalg.norm(env.get_object_position()[:2] - env.get_goal_position()[:2])
        )
        expert.reset(env)

        approach: Capture | None = None
        lift: Capture | None = None
        transport: Capture | None = None
        release: Capture | None = None
        transport_score = float("inf")

        while not expert.is_done:
            action = expert.act(observation)
            phase = expert.current_phase
            observation, _ = env.step(action)
            step = expert.steps
            object_position = env.get_object_position()

            if phase == ExpertPhase.DESCEND:
                approach = Capture(
                    render_camera(env.client_id), step, phase.value,
                    f"EE-object={np.linalg.norm(env.get_ee_position() - object_position):.3f} m",
                )
            elif phase == ExpertPhase.LIFT:
                lift = Capture(
                    render_camera(env.client_id), step, phase.value,
                    f"object_z={object_position[2]:.3f} m",
                )
            elif phase == ExpertPhase.TRANSPORT:
                remaining = float(
                    np.linalg.norm(object_position[:2] - env.get_goal_position()[:2])
                )
                progress = 1.0 - remaining / initial_distance
                score = abs(progress - 0.5)
                if score < transport_score:
                    transport_score = score
                    transport = Capture(
                        render_camera(env.client_id), step, phase.value,
                        f"transport_progress={progress:.3f}",
                    )
            elif phase == ExpertPhase.RELEASE:
                release = Capture(
                    render_camera(env.client_id), step, phase.value,
                    f"goal_xy_error={np.linalg.norm(object_position[:2] - env.get_goal_position()[:2]):.3f} m",
                )

        if not expert.success or not env.is_success():
            raise RuntimeError(
                f"Pick-and-place expert did not succeed: {expert.failure_reason}"
            )
        selected = [approach, lift, transport, release]
        if any(item is None for item in selected):
            raise RuntimeError("A pick-and-place publication state was not captured")
        captures = [item for item in selected if item is not None]
        metadata = {
            "trajectory": "fixed default expert rollout, reset seed 5478",
            "steps": expert.steps,
            "success": bool(env.is_success()),
            "captures": [capture.__dict__ | {"image": None} for capture in captures],
        }
        return captures, metadata
    finally:
        env.close()


def capture_stacking() -> tuple[list[Capture], dict[str, object]]:
    """Run the fixed stacking expert and select four physical states."""
    env = PandaStackingEnv(mode="direct")
    expert = ScriptedStackingExpert()
    try:
        observation = env.reset(seed=5478)
        expert.reset(env)

        grasp: Capture | None = None
        align: Capture | None = None
        release: Capture | None = None
        alignment_score = float("inf")

        while not expert.is_done:
            action = expert.act(observation)
            phase = expert.current_phase
            observation, info = env.step(action)
            step = expert.steps
            source = env.get_source_position()
            support = env.get_support_position()

            if phase == StackingPhase.GRASP:
                grasp = Capture(
                    render_camera(env.client_id), step, phase.value,
                    f"gripper={env.get_gripper_opening():.3f} m",
                )
            elif phase in {
                StackingPhase.TRANSPORT_ABOVE_SUPPORT,
                StackingPhase.ALIGN,
            }:
                xy_error = float(np.linalg.norm(source[:2] - support[:2]))
                if xy_error < alignment_score:
                    alignment_score = xy_error
                    align = Capture(
                        render_camera(env.client_id), step, phase.value,
                        f"source-support_xy={xy_error:.3f} m",
                    )
            elif phase == StackingPhase.RELEASE:
                diagnostics = info["stack_diagnostics"]
                release = Capture(
                    render_camera(env.client_id), step, phase.value,
                    "contact=" + str(bool(diagnostics["source_support_contact"])),
                )

        diagnostics = env.get_stack_diagnostics()
        if not expert.success or not env.is_success():
            raise RuntimeError(
                f"Stacking expert did not reach persistent success: {expert.failure_reason}"
            )
        verify = Capture(
            render_camera(env.client_id),
            expert.steps,
            StackingPhase.VERIFY_STACK.value,
            f"is_success=True, stable_steps={env.stable_steps}",
        )
        selected = [grasp, align, release, verify]
        if any(item is None for item in selected):
            raise RuntimeError("A stacking publication state was not captured")
        captures = [item for item in selected if item is not None]
        metadata = {
            "trajectory": "fixed default stacking expert rollout, reset seed 5478",
            "steps": expert.steps,
            "success": bool(env.is_success()),
            "stable_steps": env.stable_steps,
            "final_contact": bool(diagnostics["source_support_contact"]),
            "captures": [capture.__dict__ | {"image": None} for capture in captures],
        }
        return captures, metadata
    finally:
        env.close()


def _print_metadata(name: str, metadata: dict[str, object]) -> None:
    print(f"{name}: {metadata['trajectory']}")
    print(f"  rollout steps: {metadata['steps']}, success: {metadata['success']}")
    for index, capture in enumerate(metadata["captures"], start=1):
        print(
            f"  panel {index}: step {capture['step']}, phase {capture['phase']}, "
            f"{capture['note']}"
        )
    if "stable_steps" in metadata:
        print(
            f"  final stable steps: {metadata['stable_steps']}, "
            f"source-support contact: {metadata['final_contact']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()

    jobs: list[tuple[str, Callable[[], tuple[list[Capture], dict[str, object]]], str]] = [
        (
            "Pick-and-place",
            capture_pick_and_place,
            "pnp_simulation_sequence.png",
        ),
        (
            "Stacking",
            capture_stacking,
            "stacking_simulation_sequence.png",
        ),
    ]

    for name, capture_function, filename in jobs:
        captures, metadata = capture_function()
        output = args.output_dir / filename
        compose(captures, output)
        _print_metadata(name, metadata)
        print(f"  wrote: {output}")


if __name__ == "__main__":
    main()
