"""Register E1 PHP teachers without requiring retargeted datasets at import time."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from ...dataset import BUNDLES
from .env_cfg import e1_php_teacher_env_cfg
from .rl_cfg import e1_php_teacher_runner_cfg

for skill in BUNDLES:
    register_mjlab_task(
        task_id=f"E1-PHP-{skill}-Teacher",
        env_cfg=e1_php_teacher_env_cfg(skill),
        play_env_cfg=e1_php_teacher_env_cfg(skill, play=True),
        rl_cfg=e1_php_teacher_runner_cfg(skill),
        runner_cls=MjlabOnPolicyRunner,
    )
