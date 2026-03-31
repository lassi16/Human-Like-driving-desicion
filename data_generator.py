import numpy as np

GRID_ROWS = 10
GRID_COLS = 5
MAX_SPEED = 100.0

VEHICLE_EMPTY = 0
VEHICLE_CAR = 1
VEHICLE_TRUCK = 2
VEHICLE_EGO = 3

STEER_LEFT = 0
STEER_STRAIGHT = 1
STEER_RIGHT = 2


def generate_random_scene(grid_rows: int = GRID_ROWS, grid_cols: int = GRID_COLS):
    """Generate a single random traffic scene (types + speeds).

    Ego vehicle is placed at the bottom row in a (usually) center lane.
    Other vehicles are placed randomly with random types and speeds.
    """
    grid_types = np.zeros((grid_rows, grid_cols), dtype=np.int64)
    grid_speeds = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    # Place ego vehicle
    ego_row = grid_rows - 1
    ego_col = grid_cols // 2
    grid_types[ego_row, ego_col] = VEHICLE_EGO
    ego_speed = np.random.uniform(40, 80)
    grid_speeds[ego_row, ego_col] = ego_speed

    # Place other vehicles randomly
    num_other = np.random.randint(4, grid_rows * grid_cols // 2 + 1)
    for _ in range(num_other):
        r = np.random.randint(0, grid_rows)
        c = np.random.randint(0, grid_cols)
        if (r == ego_row and c == ego_col) or grid_types[r, c] != VEHICLE_EMPTY:
            continue
        vtype = np.random.choice([VEHICLE_CAR, VEHICLE_TRUCK])
        vspeed = np.random.uniform(20, 100)
        grid_types[r, c] = vtype
        grid_speeds[r, c] = vspeed

    # Ensure that fairly often there is an obstacle ahead of ego in the
    # current lane to encourage lane-change behavior in the labels.
    if _first_obstacle_ahead(grid_types, ego_row, ego_col, max_dist=4) is None:
        if np.random.rand() < 0.7:  # 70% of such scenes
            r = np.random.randint(max(0, ego_row - 4), ego_row)
            c = ego_col
            if grid_types[r, c] == VEHICLE_EMPTY:
                vtype = np.random.choice([VEHICLE_CAR, VEHICLE_TRUCK])
                vspeed = np.random.uniform(20, 80)
                grid_types[r, c] = vtype
                grid_speeds[r, c] = vspeed

    return grid_types, grid_speeds


def _find_ego(grid_types: np.ndarray):
    idx = np.argwhere(grid_types == VEHICLE_EGO)
    if idx.size == 0:
        raise ValueError("Ego vehicle not found in grid")
    return tuple(idx[0])  # (row, col)


def _lane_is_free_ahead(grid_types: np.ndarray, ego_row: int, lane: int, look_ahead: int = 4) -> bool:
    rows, cols = grid_types.shape
    if lane < 0 or lane >= cols:
        return False
    r_start = max(0, ego_row - look_ahead)
    for r in range(ego_row - 1, r_start - 1, -1):
        if grid_types[r, lane] != VEHICLE_EMPTY:
            return False
    return True


def _first_obstacle_ahead(grid_types: np.ndarray, ego_row: int, ego_col: int, max_dist: int = GRID_ROWS) -> tuple | None:
    r_start = max(0, ego_row - max_dist)
    for r in range(ego_row - 1, r_start - 1, -1):
        if grid_types[r, ego_col] in (VEHICLE_CAR, VEHICLE_TRUCK):
            return r, ego_col
    return None


def label_scene(grid_types: np.ndarray, grid_speeds: np.ndarray):
    """Generate human-like labels: target speed (0-100) and steering.

    Heuristic behavior:
    - If obstacle close ahead in the same lane -> reduce speed and prefer lane change.
    - Prefer straight if road ahead is reasonably free.
    - Choose left/right based on which side is free when needed.
    """
    ego_row, ego_col = _find_ego(grid_types)
    ego_speed = grid_speeds[ego_row, ego_col]

    obstacle = _first_obstacle_ahead(grid_types, ego_row, ego_col, max_dist=GRID_ROWS)
    speed_label = ego_speed
    steering_label = STEER_STRAIGHT

    if obstacle is not None:
        obs_row, obs_col = obstacle
        dist = ego_row - obs_row
        obs_speed = grid_speeds[obs_row, obs_col]

        # Speed logic
        if dist <= 2:
            # Very close: brake hard below obstacle speed
            desired = max(0.0, obs_speed - 20.0)
            speed_label = min(speed_label, desired)
        elif dist <= 4:
            # Moderate distance: gently slow down
            factor = (4 - (dist - 1)) / 4.0
            desired = obs_speed - 10.0 * factor
            speed_label = min(speed_label, max(0.0, desired))
        else:
            # Far ahead: keep or slightly reduce
            speed_label = min(speed_label, ego_speed)

        # Steering logic: if obstacle is ahead, strongly encourage lane change
        if dist <= 4:
            left_free = _lane_is_free_ahead(grid_types, ego_row, ego_col - 1)
            right_free = _lane_is_free_ahead(grid_types, ego_row, ego_col + 1)

            if left_free and not right_free:
                steering_label = STEER_LEFT
            elif right_free and not left_free:
                steering_label = STEER_RIGHT
            elif left_free and right_free:
                # Bias toward changing lane when both are free
                steering_label = np.random.choice([STEER_LEFT, STEER_RIGHT])
            else:
                steering_label = STEER_STRAIGHT
    else:
        # No obstacle: accelerate slightly but cap at MAX_SPEED
        speed_label = min(MAX_SPEED, ego_speed + 10.0)
        steering_label = STEER_STRAIGHT

    speed_label = float(np.clip(speed_label, 0.0, MAX_SPEED))
    return speed_label, steering_label


def generate_dataset(num_samples: int):
    """Generate dataset of random scenes and heuristic labels.

    Returns
    -------
    inputs_types: np.ndarray [N, GRID_ROWS, GRID_COLS]
    inputs_speeds: np.ndarray [N, GRID_ROWS, GRID_COLS]
    speed_labels: np.ndarray [N]
    steering_labels: np.ndarray [N]
    """
    inputs_types = np.zeros((num_samples, GRID_ROWS, GRID_COLS), dtype=np.int64)
    inputs_speeds = np.zeros((num_samples, GRID_ROWS, GRID_COLS), dtype=np.float32)
    speed_labels = np.zeros((num_samples,), dtype=np.float32)
    steering_labels = np.zeros((num_samples,), dtype=np.int64)

    for i in range(num_samples):
        g_types, g_speeds = generate_random_scene()
        spd, steer = label_scene(g_types, g_speeds)
        inputs_types[i] = g_types
        inputs_speeds[i] = g_speeds
        speed_labels[i] = spd
        steering_labels[i] = steer

    return inputs_types, inputs_speeds, speed_labels, steering_labels


if __name__ == "__main__":
    # Quick manual test
    g_types, g_speeds = generate_random_scene()
    spd, steer = label_scene(g_types, g_speeds)
    print("Types:\n", g_types)
    print("Speeds:\n", np.round(g_speeds, 1))
    print("Label speed:", spd, "steer:", steer)
