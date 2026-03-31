import torch
import torch.nn as nn

from data_generator import GRID_ROWS, GRID_COLS, MAX_SPEED


class DecisionMakingNetwork(nn.Module):
    """Fully-connected decision-making network.

    Input: flattened grid (types + speeds)
      - types normalized to [0, 1]
      - speeds normalized to [0, 1]
    Output:
      - normalized speed in [0, 1]
      - steering logits for 3 classes (left, straight, right)
    """

    def __init__(self, grid_rows: int = GRID_ROWS, grid_cols: int = GRID_COLS):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        input_dim = grid_rows * grid_cols * 2

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1028),
            nn.ReLU(),
            nn.Linear(1028, 1028),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
        )

        # Speed branch (regression of normalized speed)
        self.speed_head = nn.Linear(512, 1)

        # Steering branch (3-way classification)
        self.steering_head = nn.Linear(512, 3)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape [B, grid_rows * grid_cols * 2]

        Returns
        -------
        speed_norm : torch.Tensor, shape [B]
            Normalized speed in [0,1]. Multiply by MAX_SPEED for physical speed.
        steering_logits : torch.Tensor, shape [B, 3]
        """
        feat = self.shared(x)
        speed_raw = self.speed_head(feat).squeeze(-1)  # [B]
        steering_logits = self.steering_head(feat)  # [B, 3]

        # Constrain speed to [0,1]
        speed_norm = torch.sigmoid(speed_raw)
        return speed_norm, steering_logits


def build_model(device: torch.device | None = None) -> DecisionMakingNetwork:
    model = DecisionMakingNetwork()
    if device is not None:
        model.to(device)
    return model


if __name__ == "__main__":
    net = DecisionMakingNetwork()
    dummy = torch.randn(4, GRID_ROWS * GRID_COLS * 2)
    s, st = net(dummy)
    print("Dummy speed:", s.shape, "steering logits:", st.shape)
