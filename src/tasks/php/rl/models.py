import torch
from torch import nn
from rsl_rl.models import CNNModel


class PhpDepthEncoder(nn.Module):
    """Holosoma DepthOnlyFCBackbone58x87Small topology, 32-dimensional output."""

    output_channels = None
    output_dim = 32

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ELU(),
        )

    def forward(self, image):
        return self.layers(image)


class PhpStudentModel(CNNModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        kwargs.pop("cnn_cfg", None)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            cnns={"student_depth": PhpDepthEncoder()},
            **kwargs,
        )


class PhpExpertRouter(nn.Module):
    def __init__(self, experts, num_actions):
        super().__init__()
        if len(experts) != 5:
            raise ValueError("PHP requires all five experts in BUNDLES order")
        self.experts = nn.ModuleList(experts)
        self.num_actions = num_actions
        self.register_buffer("loaded", torch.zeros(5, dtype=torch.bool))
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, obs):
        ids = obs["expert_id"].flatten().long()
        if ((ids < 0) | (ids >= 5)).any():
            raise ValueError("Expert IDs must be in [0, 4]")
        actions = obs["teacher"].new_empty(len(ids), self.num_actions)
        for index, expert in enumerate(self.experts):
            rows = ids == index
            if rows.any():
                if not self.loaded[index]:
                    raise RuntimeError(f"Expert {index} has not been loaded")
                actions[rows] = expert(obs[rows])
        return actions
