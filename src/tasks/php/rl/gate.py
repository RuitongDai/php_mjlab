import torch
from torch import nn


class PhaseResidualGate(nn.Module):
    def __init__(self, bin_clips, alpha=0.001, warmup_samples=500_000, low=0.5):
        super().__init__()
        if not 0 < alpha <= 1 or not 0 <= low <= 1 or warmup_samples < 0:
            raise ValueError("Invalid residual gate parameters")
        self.alpha, self.warmup_samples, self.low = alpha, warmup_samples, low
        self.register_buffer("bin_clips", bin_clips.clone())
        self.register_buffer(
            "residual_sum", torch.zeros_like(bin_clips, dtype=torch.float)
        )
        self.register_buffer("count", torch.zeros_like(self.residual_sum))
        self.register_buffer(
            "seen", torch.zeros((), dtype=torch.long, device=bin_clips.device)
        )

    @torch.no_grad()
    def table(self):
        observed = self.count > 1e-8
        gate = torch.ones_like(self.count)
        if self.seen < self.warmup_samples or observed.sum() < 8:
            return gate
        residual = self.residual_sum / self.count.clamp_min(1e-8)
        quantiles = torch.quantile(residual[observed], residual.new_tensor([0.5, 0.95]))
        gate[observed] = (
            (residual[observed] - quantiles[0])
            / (quantiles[1] - quantiles[0]).clamp_min(1e-8)
        ).clamp(0, 1)
        # Anticipate the next second's difficulty, without leaking across clips.
        gate[:-1] = torch.maximum(
            gate[:-1],
            torch.where(self.bin_clips[:-1] == self.bin_clips[1:], gate[1:], gate[:-1]),
        )
        return self.low + (1 - self.low) * gate

    @torch.no_grad()
    def update(self, bins, residual, valid, distributed=False):
        valid = valid.bool() & torch.isfinite(residual)
        size = len(self.count)
        sums = torch.bincount(bins[valid], weights=residual[valid], minlength=size)
        counts = torch.bincount(bins[valid], minlength=size).float()
        stats = torch.stack((sums, counts))
        if distributed:
            torch.distributed.all_reduce(stats)
        self.residual_sum.lerp_(stats[0], self.alpha)
        self.count.lerp_(stats[1], self.alpha)
        self.seen.add_(stats[1].sum().long())
