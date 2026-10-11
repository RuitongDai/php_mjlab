"""Independent PHP hybrid distillation configuration."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class PhpDaggerPPOCfg(RslRlPpoAlgorithmCfg):
    optimizer: Literal["adamw"] = "adamw"
    class_name: str = "src.tasks.php.rl:PhpDaggerPPO"
    dagger_loss_coef: float = 10.0
    ppo_start_iteration: int = 0
    dagger_end_iteration: int = 10_000
    gate_alpha: float = 0.001
    gate_warmup_samples: int = 500_000
    gate_low: float = 0.5
    depth_weight_decay: float = 0.01


@dataclass
class PhpStudentRunnerCfg(RslRlOnPolicyRunnerCfg):
    teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    teacher_checkpoint_files: tuple[str, ...] = ()
    """Order: Locomotion, Low-Step, High-Step, Low-Climb, High-Climb; all loaded on every rank."""


def php_student_runner_cfg():
    return PhpStudentRunnerCfg(
        actor=RslRlModelCfg(
            class_name="src.tasks.php.rl.models:PhpStudentModel",
            hidden_dims=(2048, 1024, 512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.01,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=False
        ),
        teacher=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        algorithm=PhpDaggerPPOCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=2,
            num_mini_batches=96,
            learning_rate=3e-4,
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            schedule="adaptive",
        ),
        obs_groups={
            "actor": ("student_proprio", "student_depth"),
            "critic": ("critic",),
            "teacher": ("teacher",),
        },
        experiment_name="g1_php_student",
        num_steps_per_env=24,
        max_iterations=30_000,
        save_interval=500,
        logger="wandb",
        wandb_project="php-student",
        upload_model=False,
    )
