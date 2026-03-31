import numpy as np
import torch
import matplotlib.pyplot as plt

from data_generator import (
    generate_random_scene,
    GRID_ROWS,
    GRID_COLS,
    MAX_SPEED,
)
from model import build_model
from safety_layer import combine_with_safety
from utils import (
    set_seed,
    flatten_scene,
    steering_to_str,
    compute_collision,
    visualize_grid,
    visualize_grid_on_axis,
    find_ego,
)


def load_trained_model(model_path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = build_model(dev)
    ckpt = torch.load(model_path, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, dev


def run_single_scenario(model, device, show_plot: bool = True):
    g_types, g_speeds = generate_random_scene()

    x = flatten_scene(g_types, g_speeds)
    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)

    with torch.no_grad():
        speed_norm, steering_logits_t = model(x_tensor)

    speed_norm = float(speed_norm.cpu().numpy()[0])
    steering_logits = steering_logits_t.cpu().numpy()[0]

    # NN outputs
    nn_speed_kmh = float(np.clip(speed_norm * MAX_SPEED, 0.0, MAX_SPEED))
    nn_steer_idx = int(np.argmax(steering_logits))

    # Safety combination
    final_speed, final_steer_idx, safe_speed = combine_with_safety(
        speed_norm, steering_logits, g_types, g_speeds
    )

    ego_row, ego_col = find_ego(g_types)

    collision_nn = compute_collision(g_types, ego_row, ego_col, nn_steer_idx)
    collision_final = compute_collision(g_types, ego_row, ego_col, final_steer_idx)

    print("Scene types:\n", g_types)
    print("Scene speeds:\n", np.round(g_speeds, 1))
    print(
        f"NN decision: speed={nn_speed_kmh:.1f} km/h, steering={steering_to_str(nn_steer_idx)}, "
        f"collision_risk={collision_nn}"
    )
    print(
        f"Safety-only speed: {safe_speed:.1f} km/h | Final decision: speed={final_speed:.1f} km/h, "
        f"steering={steering_to_str(final_steer_idx)}, collision_risk={collision_final}"
    )

    if show_plot:
        visualize_grid(
            g_types,
            g_speeds,
            predicted_speed=nn_speed_kmh,
            predicted_steer_idx=nn_steer_idx,
            final_speed=final_speed,
            final_steer_idx=final_steer_idx,
            safe_speed=safe_speed,
        )


def evaluate_collision_rate(
    model,
    device,
    num_scenarios: int = 200,
):
    collisions_nn = 0
    collisions_final = 0

    for _ in range(num_scenarios):
        g_types, g_speeds = generate_random_scene()

        x = flatten_scene(g_types, g_speeds)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)

        with torch.no_grad():
            speed_norm, steering_logits_t = model(x_tensor)

        speed_norm = float(speed_norm.cpu().numpy()[0])
        steering_logits = steering_logits_t.cpu().numpy()[0]

        nn_speed_kmh = float(np.clip(speed_norm * MAX_SPEED, 0.0, MAX_SPEED))
        nn_steer_idx = int(np.argmax(steering_logits))

        final_speed, final_steer_idx, safe_speed = combine_with_safety(
            speed_norm, steering_logits, g_types, g_speeds
        )

        ego_row, ego_col = find_ego(g_types)
        if compute_collision(g_types, ego_row, ego_col, nn_steer_idx):
            collisions_nn += 1
        if compute_collision(g_types, ego_row, ego_col, final_steer_idx):
            collisions_final += 1

    print(
        f"Collision rate over {num_scenarios} scenarios - "
        f"NN only: {collisions_nn / num_scenarios:.3f}, "
        f"With safety layer: {collisions_final / num_scenarios:.3f}"
    )


def run_multiple_scenarios_grid(
    model,
    device,
    num_scenarios: int = 6,
    cols: int = 3,
):
    """Generate and visualize several random scenarios at once.

    Shows a grid of subplots, each with its own random traffic scene,
    NN decision, safety suggestion, and final combined decision.
    """
    if num_scenarios <= 0:
        return

    rows = int(np.ceil(num_scenarios / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 5))
    # Ensure axes is always 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif cols == 1:
        axes = np.expand_dims(axes, axis=1)

    scenario_idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if scenario_idx >= num_scenarios:
                ax.axis("off")
                continue

            g_types, g_speeds = generate_random_scene()

            x = flatten_scene(g_types, g_speeds)
            x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)

            with torch.no_grad():
                speed_norm, steering_logits_t = model(x_tensor)

            speed_norm_val = float(speed_norm.cpu().numpy()[0])
            steering_logits = steering_logits_t.cpu().numpy()[0]

            nn_speed_kmh = float(np.clip(speed_norm_val * MAX_SPEED, 0.0, MAX_SPEED))
            nn_steer_idx = int(np.argmax(steering_logits))

            final_speed, final_steer_idx, safe_speed = combine_with_safety(
                speed_norm_val,
                steering_logits,
                g_types,
                g_speeds,
            )

            title = f"Case {scenario_idx + 1}"
            visualize_grid_on_axis(
                ax,
                g_types,
                g_speeds,
                predicted_speed=nn_speed_kmh,
                predicted_steer_idx=nn_steer_idx,
                final_speed=final_speed,
                final_steer_idx=final_steer_idx,
                safe_speed=safe_speed,
                title=title,
            )

            scenario_idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    model_path = "dmn_model.pth"
    model, dev = load_trained_model(model_path)
    print("Running a single scenario with visualization...")
    run_single_scenario(model, dev, show_plot=True)
    print("\nEvaluating collision rate over multiple scenarios...")
    evaluate_collision_rate(model, dev, num_scenarios=200)
    print("\nVisualizing multiple random scenarios in a single figure...")
    run_multiple_scenarios_grid(model, dev, num_scenarios=6, cols=3)
