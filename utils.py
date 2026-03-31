import numpy as np
import random
from typing import Tuple

import matplotlib.pyplot as plt

from data_generator import (
    GRID_ROWS,
    GRID_COLS,
    MAX_SPEED,
    VEHICLE_EMPTY,
    VEHICLE_CAR,
    VEHICLE_TRUCK,
    VEHICLE_EGO,
    STEER_LEFT,
    STEER_STRAIGHT,
    STEER_RIGHT,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def steering_to_str(idx: int) -> str:
    if idx == STEER_LEFT:
        return "left"
    if idx == STEER_RIGHT:
        return "right"
    return "straight"


def compute_collision(
    grid_types: np.ndarray,
    ego_row: int,
    ego_col: int,
    steering_idx: int,
    collision_distance: int = 2,
) -> bool:
    """Approximate collision measure based on lane and distance.

    - If steering straight: any obstacle in same lane within collision_distance rows ahead.
    - If changing lanes: check target lane within same distance.
    """
    rows, cols = grid_types.shape

    # Determine target lane for this timestep
    lane = ego_col
    if steering_idx == STEER_LEFT:
        lane -= 1
    elif steering_idx == STEER_RIGHT:
        lane += 1

    if lane < 0 or lane >= cols:
        return True  # invalid lane -> treat as unsafe

    r_start = max(0, ego_row - collision_distance)
    for r in range(ego_row - 1, r_start - 1, -1):
        if grid_types[r, lane] in (VEHICLE_CAR, VEHICLE_TRUCK):
            return True
    return False


def _draw_grid_on_axis(
    ax,
    grid_types: np.ndarray,
    grid_speeds: np.ndarray,
    predicted_speed: float,
    predicted_steer_idx: int,
    final_speed: float,
    final_steer_idx: int,
    safe_speed: float,
    title: str | None = None,
    compact: bool = False,
):
    """Core drawing logic used by both single and multi-plot visualizations.

    If compact=True, use a shorter/smaller title to avoid overlapping
    when many subplots are shown in one figure.
    """
    rows, cols = grid_types.shape

    # Color map: empty=white, car=blue, truck=red, ego=green
    color_grid = np.zeros((rows, cols, 3), dtype=float)
    for r in range(rows):
        for c in range(cols):
            t = grid_types[r, c]
            if t == VEHICLE_EMPTY:
                color_grid[r, c] = (1, 1, 1)
            elif t == VEHICLE_CAR:
                color_grid[r, c] = (0.2, 0.4, 0.9)
            elif t == VEHICLE_TRUCK:
                color_grid[r, c] = (0.9, 0.3, 0.2)
            elif t == VEHICLE_EGO:
                color_grid[r, c] = (0.2, 0.8, 0.2)

    ax.imshow(color_grid, origin="upper")

    # Draw grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="gray", linewidth=0.5)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="gray", linewidth=0.5)

    # Annotate speeds
    for r in range(rows):
        for c in range(cols):
            if grid_types[r, c] != VEHICLE_EMPTY:
                ax.text(
                    c,
                    r,
                    f"{grid_speeds[r, c]:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=7,
                )

    ego_pos = np.argwhere(grid_types == VEHICLE_EGO)[0]
    ego_row, ego_col = int(ego_pos[0]), int(ego_pos[1])

    steer_str_pred = steering_to_str(predicted_steer_idx)
    steer_str_final = steering_to_str(final_steer_idx)

    ax.set_xticks([])
    ax.set_yticks([])

    if title is None:
        title = "Human-like Driving Decision"

    if compact:
        # Short, single-line title for multi-case plots
        ax.set_title(
            f"{title}: NN {predicted_speed:.0f} | Safe {safe_speed:.0f} | "
            f"Final {final_speed:.0f} ({steer_str_final})",
            fontsize=8,
        )
    else:
        ax.set_title(
            f"{title}\n"
            f"NN: {predicted_speed:.1f} km/h, {steer_str_pred}  |  "
            f"Safe: {safe_speed:.1f} km/h  |  Final: {final_speed:.1f} km/h, {steer_str_final}",
            fontsize=10,
        )

    # Mark ego position with a thicker border
    ax.add_patch(
        plt.Rectangle(
            (ego_col - 0.5, ego_row - 0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )
    )


def visualize_grid(
    grid_types: np.ndarray,
    grid_speeds: np.ndarray,
    predicted_speed: float,
    predicted_steer_idx: int,
    final_speed: float,
    final_steer_idx: int,
    safe_speed: float,
    title: str | None = None,
):
    """Visualize a single grid in its own window."""
    rows, cols = grid_types.shape
    fig, ax = plt.subplots(figsize=(cols, rows))
    _draw_grid_on_axis(
        ax,
        grid_types,
        grid_speeds,
        predicted_speed,
        predicted_steer_idx,
        final_speed,
        final_steer_idx,
        safe_speed,
        title,
        compact=False,
    )
    plt.tight_layout()
    plt.show()


def visualize_grid_on_axis(
    ax,
    grid_types: np.ndarray,
    grid_speeds: np.ndarray,
    predicted_speed: float,
    predicted_steer_idx: int,
    final_speed: float,
    final_steer_idx: int,
    safe_speed: float,
    title: str | None = None,
):
    """Draw a grid on an existing matplotlib axis (for multi-case figures)."""
    _draw_grid_on_axis(
        ax,
        grid_types,
        grid_speeds,
        predicted_speed,
        predicted_steer_idx,
        final_speed,
        final_steer_idx,
        safe_speed,
        title,
        compact=True,
    )


def flatten_scene(types: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    """Flatten grid into a 1D feature vector [types_norm, speeds_norm]."""
    types_norm = types.astype(np.float32) / 3.0  # max type index
    speeds_norm = speeds.astype(np.float32) / MAX_SPEED
    return np.concatenate([types_norm.flatten(), speeds_norm.flatten()], axis=0)


def find_ego(grid_types: np.ndarray) -> Tuple[int, int]:
    idx = np.argwhere(grid_types == VEHICLE_EGO)
    if idx.size == 0:
        raise ValueError("Ego vehicle not found in grid")
    return int(idx[0, 0]), int(idx[0, 1])
