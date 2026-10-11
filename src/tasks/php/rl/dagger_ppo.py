from copy import deepcopy
from itertools import chain

import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from ..dataset import BUNDLES
from .gate import PhaseResidualGate
from .models import PhpExpertRouter, PhpStudentModel


def hybrid_loss(
    value,
    surrogate,
    entropy,
    mse,
    valid,
    gate,
    ppo_coef,
    value_coef,
    entropy_coef,
    dagger_coef,
):
    coefficient = gate * ppo_coef
    # Deliberately average over the whole batch, as in PHP's finetune recipe.
    behavior = mse * valid
    return (
        value_coef * value
        + (
            coefficient * (surrogate - entropy_coef * entropy)
            + (1 - coefficient) * dagger_coef * behavior
        ).mean()
    )


class PhpDaggerPPO(PPO):
    def __init__(
        self,
        actor,
        critic,
        teacher,
        storage,
        bin_clips,
        dataset_signature,
        dagger_loss_coef=10.0,
        ppo_start_iteration=0,
        dagger_end_iteration=10_000,
        gate_alpha=0.001,
        gate_warmup_samples=500_000,
        gate_low=0.5,
        depth_weight_decay=0.01,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        if kwargs.get("optimizer", "adamw") != "adamw":
            raise ValueError(
                "PHP uses AdamW with separate student/critic/depth/std parameter groups"
            )
        if dagger_end_iteration <= ppo_start_iteration:
            raise ValueError("dagger_end_iteration must exceed ppo_start_iteration")
        if self.rnd or self.symmetry:
            raise ValueError("PHP DAgger does not support RND or symmetry augmentation")
        self.teacher = teacher
        self.gate = PhaseResidualGate(
            bin_clips, gate_alpha, gate_warmup_samples, gate_low
        )
        self.dataset_signature = dataset_signature
        self.dagger_loss_coef = dagger_loss_coef
        self.ppo_start_iteration, self.dagger_end_iteration = (
            ppo_start_iteration,
            dagger_end_iteration,
        )
        self.num_updates = 0
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": actor.mlp.parameters(),
                    "weight_decay": 0.0,
                    "name": "student",
                },
                {"params": critic.parameters(), "weight_decay": 0.0, "name": "critic"},
                {
                    "params": actor.cnns.parameters(),
                    "weight_decay": depth_weight_decay,
                    "name": "depth",
                },
                {
                    "params": actor.distribution.parameters(),
                    "weight_decay": 0.0,
                    "name": "std",
                },
            ],
            lr=self.learning_rate,
        )

    @property
    def ppo_coef(self):
        return min(
            max(
                (self.num_updates - self.ppo_start_iteration)
                / (self.dagger_end_iteration - self.ppo_start_iteration),
                0.0,
            ),
            0.9,
        )

    def act(self, obs):
        with torch.no_grad():
            labels = self.teacher(obs)
            phase = obs["phase_bin"].flatten().long()
            augmented = obs.clone(recurse=False)
            augmented["teacher_actions"] = labels
            augmented["ppo_gate"] = self.gate.table()[phase, None]
            action = super().act(augmented)
            residual = (self.actor.output_mean - labels).square().mean(-1)
            self.gate.update(
                phase, residual, obs["dagger_valid"].flatten(), self.is_multi_gpu
            )
        return action

    def _set_learning_rates(self):
        for group in self.optimizer.param_groups:
            group["lr"] = (
                0.0
                if group["name"] == "std" and self.ppo_coef <= 0.1
                else self.learning_rate
            )

    def update(self):
        if (
            self.num_mini_batches
            > self.storage.num_envs * self.storage.num_transitions_per_env
        ):
            raise ValueError(
                "num_mini_batches exceeds rollout sample count; reduce --agent.algorithm.num-mini-batches"
            )
        self._set_learning_rates()
        keys = (
            "value",
            "surrogate",
            "entropy",
            "dagger",
            "teacher_valid",
            "ppo_gate",
            "total",
        )
        totals = torch.zeros(len(keys), device=self.device)
        expert_stats = torch.zeros(
            5, 3, device=self.device
        )  # squared error, valid, total
        batches = 0
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            obs = batch.observations
            self.actor(obs, stochastic_output=True)
            mean = self.actor.output_mean
            entropy = self.actor.output_entropy
            values = self.critic(obs)
            gate = obs["ppo_gate"].flatten()
            if (
                self.desired_kl is not None
                and self.schedule == "adaptive"
                and self.ppo_coef > 0.1
            ):
                with torch.no_grad():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params,
                        self.actor.output_distribution_params,
                    )
                    pair = torch.stack(((kl * gate).sum(), gate.sum()))
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(pair)
                    kl_mean = pair[0] / pair[1].clamp_min(1e-6)
                    if self.gpu_global_rank == 0:
                        if kl_mean > 2 * self.desired_kl:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0 < kl_mean < self.desired_kl / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        rate = mean.new_tensor(self.learning_rate)
                        torch.distributed.broadcast(rate, 0)
                        self.learning_rate = rate.item()
                    self._set_learning_rates()
            log_prob = self.actor.get_output_log_prob(batch.actions)
            ratio = (log_prob - batch.old_actions_log_prob.flatten()).exp()
            advantage = batch.advantages.flatten()
            if self.normalize_advantage_per_mini_batch:
                advantage = (advantage - advantage.mean()) / (
                    advantage.std(unbiased=False) + 1e-8
                )
            surrogate = torch.maximum(
                -advantage * ratio,
                -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
            )
            value_error = (values - batch.returns).square()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_error = torch.maximum(
                    value_error, (clipped - batch.returns).square()
                )
            value_loss = value_error.mean()
            valid = obs["dagger_valid"].flatten()
            mse = (mean - obs["teacher_actions"]).square().mean(-1)
            loss = hybrid_loss(
                value_loss,
                surrogate,
                entropy,
                mse,
                valid,
                gate,
                self.ppo_coef,
                self.value_loss_coef,
                self.entropy_coef,
                self.dagger_loss_coef,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(
                list(chain(self.actor.parameters(), self.critic.parameters())),
                self.max_grad_norm,
            )
            self.optimizer.step()
            with torch.no_grad():
                self.actor.distribution.std_param.clamp_(min=1e-4)
                totals += torch.stack(
                    (
                        value_loss,
                        surrogate.mean(),
                        entropy.mean(),
                        (mse * valid).mean(),
                        valid.mean(),
                        gate.mean(),
                        loss,
                    )
                ).detach()
                ids = obs["expert_id"].flatten().long()
                for i in range(5):
                    rows = (ids == i).float()
                    expert_stats[i] += torch.stack(
                        ((mse * valid * rows).sum(), (valid * rows).sum(), rows.sum())
                    )
            batches += 1
        if self.is_multi_gpu:
            torch.distributed.all_reduce(totals)
            totals /= self.gpu_world_size
            torch.distributed.all_reduce(expert_stats)
        result = {key: (totals[i] / batches).item() for i, key in enumerate(keys)}
        result["ppo_coef"] = self.ppo_coef
        result["gate_seen_samples"] = self.gate.seen.item()
        for i, name in enumerate(BUNDLES):
            stats = expert_stats[i]
            result[f"{name}/dagger_mse_valid"] = (
                stats[0] / stats[1].clamp_min(1)
            ).item()
            result[f"{name}/teacher_valid"] = (stats[1] / stats[2].clamp_min(1)).item()
            result[f"{name}/sample_fraction"] = (
                stats[2] / expert_stats[:, 2].sum().clamp_min(1)
            ).item()
        self.storage.clear()
        self.num_updates += 1
        return result

    def train_mode(self):
        super().train_mode()
        self.teacher.eval()

    def eval_mode(self):
        super().eval_mode()
        self.teacher.eval()

    def save(self):
        saved = super().save()
        saved.update(
            php_version=2,
            teacher_state_dict=self.teacher.state_dict(),
            gate_state_dict=self.gate.state_dict(),
            php_num_updates=self.num_updates,
            php_learning_rate=self.learning_rate,
            php_dataset_signature=self.dataset_signature,
            php_expert_order=tuple(BUNDLES),
        )
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        config = (
            {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "teacher": True,
            }
            if load_cfg is None
            else load_cfg
        )
        if config.get("actor") or config.get("iteration"):
            if loaded_dict.get("php_version") != 2:
                raise ValueError(
                    "PHP Student now uses 11 command dimensions (104 proprio inputs). "
                    "The old 15-command Student checkpoint cannot be loaded directly; "
                    "start a new Student run. Teacher checkpoints are unchanged."
                )
        if config.get("iteration"):
            if loaded_dict.get("php_dataset_signature") != self.dataset_signature:
                raise ValueError(
                    "PHP resume dataset differs: phase bins must retain their clip/frame mapping"
                )
            self.num_updates = loaded_dict["php_num_updates"]
            self.learning_rate = loaded_dict["php_learning_rate"]
            self.gate.load_state_dict(loaded_dict["gate_state_dict"], strict=True)
        if config.get("teacher"):
            if tuple(loaded_dict.get("php_expert_order", ())) != tuple(BUNDLES):
                raise ValueError("Checkpoint expert order is incompatible")
            self.teacher.load_state_dict(loaded_dict["teacher_state_dict"], strict=True)
        result = super().load(loaded_dict, config, strict)
        self._set_learning_rates()
        return result

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        config = deepcopy(cfg)
        algorithm = config["algorithm"]
        for key in ("class_name", "share_cnn_encoders"):
            algorithm.pop(key, None)
        groups = config["obs_groups"]

        def model_kwargs(name):
            return deepcopy(
                {
                    k: v
                    for k, v in config[name].items()
                    if k != "class_name" and v is not None
                }
            )

        actor = PhpStudentModel(
            obs, groups, "actor", env.num_actions, **model_kwargs("actor")
        ).to(device)
        critic = MLPModel(obs, groups, "critic", 1, **model_kwargs("critic")).to(device)
        teacher = PhpExpertRouter(
            [
                MLPModel(
                    obs, groups, "teacher", env.num_actions, **model_kwargs("teacher")
                )
                for _ in range(5)
            ],
            env.num_actions,
        ).to(device)
        template = obs.clone(recurse=False)
        template["teacher_actions"] = torch.zeros(
            env.num_envs, env.num_actions, device=device
        )
        template["ppo_gate"] = torch.ones(env.num_envs, 1, device=device)
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            config["num_steps_per_env"],
            template,
            [env.num_actions],
            device,
        )
        motion = env.unwrapped.command_manager.get_term("motion").motion
        signature = [
            (p.parent.name, p.name, int(length))
            for p, length in zip(motion.paths, motion.motion_lengths)
        ]
        return PhpDaggerPPO(
            actor,
            critic,
            teacher,
            storage,
            motion.bin_clips,
            signature,
            device=device,
            multi_gpu_cfg=config["multi_gpu"],
            **algorithm,
        )
