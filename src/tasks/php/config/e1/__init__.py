"""Register E1 PHP teachers and student without reading datasets at import time."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from ...dataset import BUNDLES
from .env_cfg import e1_php_teacher_env_cfg
from .rl_cfg import e1_php_teacher_runner_cfg
from .student_env_cfg import e1_php_student_env_cfg
from .student_rl_cfg import e1_php_student_runner_cfg
from ...rl import PhpStudentRunner

for skill in BUNDLES:
    register_mjlab_task(
        task_id=f"E1-PHP-{skill}-Teacher",
        env_cfg=e1_php_teacher_env_cfg(skill),
        play_env_cfg=e1_php_teacher_env_cfg(skill, play=True),
        rl_cfg=e1_php_teacher_runner_cfg(skill),
        runner_cls=MjlabOnPolicyRunner,
    )

register_mjlab_task(
    task_id="E1-PHP-Student",
    env_cfg=e1_php_student_env_cfg(),
    play_env_cfg=e1_php_student_env_cfg(play=True),
    rl_cfg=e1_php_student_runner_cfg(),
    runner_cls=PhpStudentRunner,
)
