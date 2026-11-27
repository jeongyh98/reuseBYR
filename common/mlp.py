import torch.nn as nn


class MLP(nn.Module):
    """Image embedding -> z-normalized action chunk (the action key of the retrieval score)."""

    def __init__(self, d_in: int, hidden: int, d_out: int, depth: int = 2):
        super().__init__()
        layers, prev = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_mlp(path: str, device: str = "cuda") -> MLP:
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    m = MLP(c["d_in"], c["hidden"], c["d_out"], depth=c["depth"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m
