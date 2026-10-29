from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def e1_php_teacher_runner_cfg(skill="High-Climb"):
    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.actor.obs_normalization = cfg.critic.obs_normalization = False
    cfg.experiment_name = "e1_php_" + skill.lower().replace("-", "_") + "_teacher"
    cfg.max_iterations = 20_000
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg
