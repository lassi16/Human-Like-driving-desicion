import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from data_generator import (
    generate_dataset,
    GRID_ROWS,
    GRID_COLS,
    MAX_SPEED,
    STEER_LEFT,
    STEER_STRAIGHT,
    STEER_RIGHT,
)
from model import build_model
from utils import set_seed, flatten_scene


class HumanLikeDrivingDataset(Dataset):
    def __init__(self, num_samples: int):
        inputs_types, inputs_speeds, speed_labels, steering_labels = generate_dataset(num_samples)
        self.types = inputs_types
        self.speeds = inputs_speeds
        self.speed_labels = speed_labels.astype(np.float32) / MAX_SPEED  # normalize to [0,1]
        self.steering_labels = steering_labels.astype(np.int64)

    def __len__(self) -> int:
        return self.types.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.types[idx]
        s = self.speeds[idx]
        x = flatten_scene(t, s)  # np.array
        x = torch.from_numpy(x).float()
        y_speed = torch.tensor(self.speed_labels[idx], dtype=torch.float32)
        y_steer = torch.tensor(self.steering_labels[idx], dtype=torch.long)
        return x, y_speed, y_steer


def train_model(
    num_samples: int = 8000,
    batch_size: int = 32,
    num_epochs: int = 30,
    lr: float = 1e-3,
    lambda_speed: float = 1.0,
    lambda_steer: float = 1.0,
    device: str | None = None,
    model_path: str = "dmn_model.pth",
):
    set_seed(42)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    dataset = HumanLikeDrivingDataset(num_samples=num_samples)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(dev)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_speed_loss = 0.0
        total_steer_loss = 0.0

        for x, y_speed, y_steer in train_loader:
            x = x.to(dev)
            y_speed = y_speed.to(dev)
            y_steer = y_steer.to(dev)

            optimizer.zero_grad()
            pred_speed_norm, pred_steer_logits = model(x)

            loss_speed = mse_loss(pred_speed_norm, y_speed)
            loss_steer = ce_loss(pred_steer_logits, y_steer)
            loss = lambda_speed * loss_speed + lambda_steer * loss_steer

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_speed_loss += loss_speed.item() * x.size(0)
            total_steer_loss += loss_steer.item() * x.size(0)

        avg_loss = total_loss / train_size
        avg_speed_loss = total_speed_loss / train_size
        avg_steer_loss = total_steer_loss / train_size

        # Validation
        model.eval()
        val_loss = 0.0
        val_speed_loss = 0.0
        val_steer_loss = 0.0
        correct_steer = 0
        total_steer = 0
        with torch.no_grad():
            for x, y_speed, y_steer in val_loader:
                x = x.to(dev)
                y_speed = y_speed.to(dev)
                y_steer = y_steer.to(dev)
                pred_speed_norm, pred_steer_logits = model(x)

                loss_speed = mse_loss(pred_speed_norm, y_speed)
                loss_steer = ce_loss(pred_steer_logits, y_steer)
                loss = lambda_speed * loss_speed + lambda_steer * loss_steer

                val_loss += loss.item() * x.size(0)
                val_speed_loss += loss_speed.item() * x.size(0)
                val_steer_loss += loss_steer.item() * x.size(0)

                pred_steer = pred_steer_logits.argmax(dim=1)
                correct_steer += (pred_steer == y_steer).sum().item()
                total_steer += y_steer.size(0)

        avg_val_loss = val_loss / val_size
        avg_val_speed_loss = val_speed_loss / val_size
        avg_val_steer_loss = val_steer_loss / val_size
        steer_acc = correct_steer / total_steer if total_steer > 0 else 0.0

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"TrainLoss={avg_loss:.4f} (speed={avg_speed_loss:.4f}, steer={avg_steer_loss:.4f}) "
            f"ValLoss={avg_val_loss:.4f} (speed={avg_val_speed_loss:.4f}, steer={avg_val_steer_loss:.4f}) "
            f"SteerAcc={steer_acc:.3f}"
        )

    torch.save({"model_state_dict": model.state_dict()}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_model()
