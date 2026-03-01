"""DQN-specific RLlib new stack configuration helpers.

This file is the manual DQN dashboard: all defaults are hardcoded numeric values.
"""

# Edit these values directly for tuning.
config_dqn = {
    "max_iters": 1000,
    # Core learning
    "lr": 0.0001,
    "gamma": 0.99,
    "train_batch_size": 32000,
    "grad_clip": 40.0,
    # Replay buffer
    "replay_buffer_type": "MultiAgentEpisodeReplayBuffer",
    # Must exceed one add() wave (num_env_runners * num_envs_per_env_runner * rollout_fragment_length)
    # to avoid first-add eviction edge cases in RLlib's multi-agent episode replay buffer.
    "replay_buffer_capacity": 500000,
    # Resources
    "num_learners": 0,
    "num_env_runners": 24,
    "num_envs_per_env_runner": 8,
    "num_gpus_per_learner": 1.0,
    "num_cpus_for_main_process": 0,
    "num_cpus_per_env_runner": 1,
    "num_gpus_per_env_runner": 0.0,
    "sample_timeout_s": 60.0,
    "rollout_fragment_length": 1000,
    "batch_mode": "truncate_episodes",
}


def resolve_config_dqn(args, train_batch_size, dqn_overrides=None):
    del args  # Dashboard values are authoritative.
    del train_batch_size  # Dashboard values are authoritative.
    resolved = dict(config_dqn)
    if dqn_overrides:
        resolved.update(dqn_overrides)

    return {
        "max_iters": int(resolved["max_iters"]),
        "lr": float(resolved["lr"]),
        "gamma": float(resolved["gamma"]),
        "train_batch_size": int(resolved["train_batch_size"]),
        "grad_clip": float(resolved["grad_clip"]),
        "replay_buffer_type": str(resolved["replay_buffer_type"]),
        "replay_buffer_capacity": int(resolved["replay_buffer_capacity"]),
        "num_learners": int(resolved["num_learners"]),
        "num_env_runners": int(resolved["num_env_runners"]),
        "num_envs_per_env_runner": int(resolved["num_envs_per_env_runner"]),
        "num_gpus_per_learner": float(resolved["num_gpus_per_learner"]),
        "num_cpus_for_main_process": int(resolved["num_cpus_for_main_process"]),
        "num_cpus_per_env_runner": int(resolved["num_cpus_per_env_runner"]),
        "num_gpus_per_env_runner": float(resolved["num_gpus_per_env_runner"]),
        "sample_timeout_s": float(resolved["sample_timeout_s"]),
        "rollout_fragment_length": resolved["rollout_fragment_length"],
        "batch_mode": str(resolved["batch_mode"]),
    }


def apply_dqn_training_config(algorithm_config, args, train_batch_size, dqn_overrides=None):
    runtime_dqn = resolve_config_dqn(
        args=args,
        train_batch_size=train_batch_size,
        dqn_overrides=dqn_overrides,
    )

    algorithm_config = algorithm_config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    algorithm_config = algorithm_config.resources(
        num_cpus_for_main_process=runtime_dqn["num_cpus_for_main_process"],
    )
    algorithm_config = algorithm_config.env_runners(
        num_env_runners=runtime_dqn["num_env_runners"],
        num_envs_per_env_runner=runtime_dqn["num_envs_per_env_runner"],
        rollout_fragment_length=runtime_dqn["rollout_fragment_length"],
        num_cpus_per_env_runner=runtime_dqn["num_cpus_per_env_runner"],
        num_gpus_per_env_runner=runtime_dqn["num_gpus_per_env_runner"],
        sample_timeout_s=runtime_dqn["sample_timeout_s"],
        batch_mode=runtime_dqn["batch_mode"],
    )
    algorithm_config = algorithm_config.learners(
        num_learners=runtime_dqn["num_learners"],
        num_gpus_per_learner=runtime_dqn["num_gpus_per_learner"],
    )
    return algorithm_config.training(
        gamma=runtime_dqn["gamma"],
        lr=runtime_dqn["lr"],
        train_batch_size=runtime_dqn["train_batch_size"],
        grad_clip=runtime_dqn["grad_clip"],
        replay_buffer_config={
            "type": runtime_dqn["replay_buffer_type"],
            "capacity": runtime_dqn["replay_buffer_capacity"],
        },
    )
