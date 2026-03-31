import numpy as np

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


def _find_ego(grid_types: np.ndarray):
    idx = np.argwhere(grid_types == VEHICLE_EGO)
    if idx.size == 0:
        raise ValueError("Ego vehicle not found in grid")
    return tuple(idx[0])


def _lane_blocked_hard(
    grid_types: np.ndarray,
    ego_row: int,
    ego_col: int,
    target_lane: int,
    max_rows_ahead: int = 3,
) -> bool:
    """Hard safety check: is there any vehicle too close in target lane?

    - If target lane is outside road -> blocked.
    - If any vehicle is found in the same or next few rows ahead in
      the target lane -> treat as blocked (unsafe for lane change).
    """
    rows, cols = grid_types.shape
    if target_lane < 0 or target_lane >= cols:
        return True

    r_start = max(0, ego_row - max_rows_ahead)
    for r in range(ego_row, r_start - 1, -1):
        if grid_types[r, target_lane] in (VEHICLE_CAR, VEHICLE_TRUCK):
            return True
    return False


def compute_safety_suggestions(
    grid_types: np.ndarray,
    grid_speeds: np.ndarray,
    nn_speed: float,
    nn_steering_logits: np.ndarray,
    look_ahead: int = 4,
):
    """Compute safety-aware speed and steering suggestions.

    This is a simplified repulsive potential field (RPF) styled logic.

    - If obstacles are close ahead -> suggest lower safe speed.
    - If obstacle directly in front -> discourage straight steering.

    Returns
    -------
    safe_speed : float
        Safety-suggested speed (0-100).
    safety_steering_logits : np.ndarray, shape [3]
        Logit-style preferences for [left, straight, right].
    """
    ego_row, ego_col = _find_ego(grid_types)
    ego_speed = float(grid_speeds[ego_row, ego_col])

    # Base suggestions from NN output
    nn_speed_kmh = float(np.clip(nn_speed * MAX_SPEED, 0.0, MAX_SPEED))

    # Repulsive speed effect from obstacles ahead in all lanes
    safe_speed = nn_speed_kmh

    rows, cols = grid_types.shape
    safety_steering_logits = np.zeros(3, dtype=np.float32)

    # Start with mild preference to keep lane
    safety_steering_logits[STEER_STRAIGHT] = 0.5

    # Scan nearby vehicles
    for r in range(max(0, ego_row - look_ahead), ego_row):
        for c in range(cols):
            if grid_types[r, c] in (VEHICLE_CAR, VEHICLE_TRUCK):
                dist_rows = ego_row - r
                if dist_rows <= 0:
                    continue

                rel_speed = grid_speeds[r, c]
                # Repulsive speed influence: closer means stronger braking
                influence = 1.0 / dist_rows
                target_speed = max(0.0, float(rel_speed) - 20.0 * influence)
                safe_speed = min(safe_speed, target_speed)

                # If directly ahead, discourage straight
                if c == ego_col:
                    safety_steering_logits[STEER_STRAIGHT] -= 2.0 * influence

                # If obstacle in left/right lane near ego, discourage steering into it
                if c == ego_col - 1:
                    safety_steering_logits[STEER_LEFT] -= 2.0 * influence
                if c == ego_col + 1:
                    safety_steering_logits[STEER_RIGHT] -= 2.0 * influence

    # Encourage moving to emptier directions if straight is unsafe
    straight_score = safety_steering_logits[STEER_STRAIGHT]
    if straight_score < 0.0:
        # Check lane occupancy ahead for left/right and reward safer side
        for steer, lane_offset in ((STEER_LEFT, -1), (STEER_RIGHT, 1)):
            lane = ego_col + lane_offset
            if lane < 0 or lane >= cols:
                safety_steering_logits[steer] -= 5.0  # impossible direction
                continue
            free_cells = 0
            for r in range(max(0, ego_row - look_ahead), ego_row):
                if grid_types[r, lane] == VEHICLE_EMPTY:
                    free_cells += 1
            safety_steering_logits[steer] += 0.5 * free_cells

    safe_speed = float(np.clip(safe_speed, 0.0, MAX_SPEED))
    return safe_speed, safety_steering_logits


def combine_with_safety(
    nn_speed: float,
    nn_steering_logits: np.ndarray,
    grid_types: np.ndarray,
    grid_speeds: np.ndarray,
    w_nn: float = 0.7,
    w_safe: float = 0.3,
):
    """Combine neural network output with safety layer suggestions.

    final_output = w_nn * NN_output + w_safe * safety_output

    Parameters
    ----------
    nn_speed : float
        Normalized speed from NN in [0,1].
    nn_steering_logits : np.ndarray, shape [3]
    grid_types, grid_speeds : np.ndarray
    w_nn, w_safe : float

    Returns
    -------
    final_speed : float
        Final speed in km/h.
    final_steering_idx : int
        0=left, 1=straight, 2=right
    safe_speed : float
        Safety-only speed for comparison.
    """
    safe_speed, safety_logits = compute_safety_suggestions(
        grid_types, grid_speeds, nn_speed, nn_steering_logits
    )

    # Apply hard safety: directions with a nearby vehicle in the
    # corresponding lane are forbidden by setting their logits very low.
    ego_row, ego_col = _find_ego(grid_types)
    very_negative = -1e6
    # Straight lane
    if _lane_blocked_hard(grid_types, ego_row, ego_col, ego_col):
        safety_logits[STEER_STRAIGHT] = very_negative
    # Left lane
    if _lane_blocked_hard(grid_types, ego_row, ego_col, ego_col - 1):
        safety_logits[STEER_LEFT] = very_negative
    # Right lane
    if _lane_blocked_hard(grid_types, ego_row, ego_col, ego_col + 1):
        safety_logits[STEER_RIGHT] = very_negative

    # Combine speeds in km/h domain
    nn_speed_kmh = float(np.clip(nn_speed * MAX_SPEED, 0.0, MAX_SPEED))
    final_speed = w_nn * nn_speed_kmh + w_safe * safe_speed

    # Combine steering logits
    final_logits = w_nn * nn_steering_logits + w_safe * safety_logits
    final_steering_idx = int(np.argmax(final_logits))

    final_speed = float(np.clip(final_speed, 0.0, MAX_SPEED))
    return final_speed, final_steering_idx, safe_speed
