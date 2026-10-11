"""Register one PHP teacher per skill bundle."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from ...dataset import BUNDLES
from .env_cfg import php_teacher_env_cfg
from .rl_cfg import php_teacher_runner_cfg
from .student_env_cfg import php_student_env_cfg
from .student_rl_cfg import php_student_runner_cfg
from ...rl import PhpStudentRunner

for skill in BUNDLES:
    register_mjlab_task(
        task_id=f"Unitree-G1-PHP-{skill}-Teacher",
        env_cfg=php_teacher_env_cfg(skill),
        play_env_cfg=php_teacher_env_cfg(skill, play=True),
        rl_cfg=php_teacher_runner_cfg(skill),
        runner_cls=MjlabOnPolicyRunner,
    )

register_mjlab_task(
    task_id="Unitree-G1-PHP-Student",
    env_cfg=php_student_env_cfg(),
    play_env_cfg=php_student_env_cfg(play=True),
    rl_cfg=php_student_runner_cfg(),
    runner_cls=PhpStudentRunner,
)
