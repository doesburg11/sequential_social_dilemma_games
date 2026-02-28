"""PPO-specific RLlib new stack configuration helpers.

This file is the manual PPO dashboard: all defaults are hardcoded numeric values.
"""

# Edit these values directly for tuning.
config_ppo = {
    "max_iters": 10000,
    # Core learning
    "lr": 0.0001,
    "gamma": 0.99,
    "lambda_": 1.0,
    "train_batch_size_per_learner": 10240,
    "minibatch_size": 2048,
    "num_epochs": 4,
    "entropy_coeff": 0.005,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    "grad_clip": 10.0,
    "batch_mode": "complete_episodes",
    "use_kl_loss": False,
    # Resources
    "num_learners": 0,
    "num_env_runners": 24,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 1.0,
    "num_cpus_for_main_process": 1,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 60.0,
    "rollout_fragment_length": "auto",
    # KL / exploration
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    # Original project PPO override.
    "vf_share_layers": True,
}


def resolve_config_ppo(args, train_batch_size, ppo_overrides=None):
    del train_batch_size  # Dashboard values are authoritative.
    resolved = dict(config_ppo)
    if ppo_overrides:
        resolved.update(ppo_overrides)

    # Optional explicit CLI override for quick experiments.
    if args.ppo_sgd_minibatch_size is not None:
        resolved["minibatch_size"] = int(args.ppo_sgd_minibatch_size)

    return {
        "max_iters": int(resolved["max_iters"]),
        "lr": float(resolved["lr"]),
        "gamma": float(resolved["gamma"]),
        "lambda_": float(resolved["lambda_"]),
        "train_batch_size_per_learner": int(resolved["train_batch_size_per_learner"]),
        "minibatch_size": int(resolved["minibatch_size"]),
        "num_epochs": int(resolved["num_epochs"]),
        "entropy_coeff": float(resolved["entropy_coeff"]),
        "vf_loss_coeff": float(resolved["vf_loss_coeff"]),
        "clip_param": float(resolved["clip_param"]),
        "grad_clip": float(resolved["grad_clip"]),
        "batch_mode": str(resolved["batch_mode"]),
        "use_kl_loss": bool(resolved["use_kl_loss"]),
        "num_learners": int(resolved["num_learners"]),
        "num_env_runners": int(resolved["num_env_runners"]),
        "num_envs_per_env_runner": int(resolved["num_envs_per_env_runner"]),
        "num_gpus_per_learner": float(resolved["num_gpus_per_learner"]),
        "num_cpus_for_main_process": int(resolved["num_cpus_for_main_process"]),
        "num_cpus_per_env_runner": int(resolved["num_cpus_per_env_runner"]),
        "sample_timeout_s": float(resolved["sample_timeout_s"]),
        "rollout_fragment_length": resolved["rollout_fragment_length"],
        "kl_coeff": float(resolved["kl_coeff"]),
        "kl_target": float(resolved["kl_target"]),
        "vf_share_layers": bool(resolved["vf_share_layers"]),
    }


def apply_ppo_training_config(algorithm_config, args, train_batch_size, ppo_overrides=None):
    runtime_ppo = resolve_config_ppo(
        args=args,
        train_batch_size=train_batch_size,
        ppo_overrides=ppo_overrides,
    )

    algorithm_config = algorithm_config.resources(
        num_cpus_for_main_process=runtime_ppo["num_cpus_for_main_process"],
    )
    algorithm_config = algorithm_config.env_runners(
        num_env_runners=runtime_ppo["num_env_runners"],
        num_envs_per_env_runner=runtime_ppo["num_envs_per_env_runner"],
        rollout_fragment_length=runtime_ppo["rollout_fragment_length"],
        num_cpus_per_env_runner=runtime_ppo["num_cpus_per_env_runner"],
        sample_timeout_s=runtime_ppo["sample_timeout_s"],
        batch_mode=runtime_ppo["batch_mode"],
    )
    algorithm_config = algorithm_config.learners(
        num_learners=runtime_ppo["num_learners"],
        num_gpus_per_learner=runtime_ppo["num_gpus_per_learner"],
    )
    return algorithm_config.training(
        lr=runtime_ppo["lr"],
        gamma=runtime_ppo["gamma"],
        lambda_=runtime_ppo["lambda_"],
        train_batch_size_per_learner=runtime_ppo["train_batch_size_per_learner"],
        entropy_coeff=runtime_ppo["entropy_coeff"],
        num_epochs=runtime_ppo["num_epochs"],
        minibatch_size=runtime_ppo["minibatch_size"],
        vf_loss_coeff=runtime_ppo["vf_loss_coeff"],
        clip_param=runtime_ppo["clip_param"],
        grad_clip=runtime_ppo["grad_clip"],
        use_kl_loss=runtime_ppo["use_kl_loss"],
        kl_coeff=runtime_ppo["kl_coeff"],
        kl_target=runtime_ppo["kl_target"],
        vf_share_layers=runtime_ppo["vf_share_layers"],
    )
