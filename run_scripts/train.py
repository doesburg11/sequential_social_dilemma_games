# ruff: noqa: E402
import argparse
import inspect
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import ray
from gymnasium.spaces import Box
from ray import tune
from ray.rllib.algorithms.impala import IMPALA, IMPALAConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

# Allow running this script directly (e.g. VS Code "Run Python File")
# without requiring `PYTHONPATH=.` in the shell.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from social_dilemmas.config.default_args import add_default_args
from social_dilemmas.envs.env_creator import get_env_creator
from run_scripts.config.ppo_config import apply_ppo_training_config, resolve_config_ppo
from utility_funcs import update_nested_dict

parser = argparse.ArgumentParser()
add_default_args(parser)


def get_algorithm_trainable(algorithm_name):
    algo = str(algorithm_name).upper()
    if algo == "PPO":
        return PPO
    if algo == "IMPALA":
        return IMPALA
    raise ValueError(f"Unsupported algorithm '{algorithm_name}'. Supported values: PPO, IMPALA.")


def get_algorithm_config_builder(algorithm_name):
    algo = str(algorithm_name).upper()
    if algo == "PPO":
        return PPOConfig()
    if algo == "IMPALA":
        return IMPALAConfig()
    raise ValueError(f"Unsupported algorithm '{algorithm_name}'. Supported values: PPO, IMPALA.")


class _RLlibEnvAdapter(MultiAgentEnv):
    """Adapter for legacy MapEnv API to RLlib's expected Gymnasium-style API."""

    def __init__(self, env, max_episode_steps=None):
        self._env = env
        super().__init__()
        self._max_episode_steps = (
            int(max_episode_steps) if max_episode_steps is not None and max_episode_steps > 0 else None
        )
        self._elapsed_steps = 0
        self._agent_obs_key = None
        base_observation_space = env.observation_space
        if hasattr(base_observation_space, "spaces") and "curr_obs" in base_observation_space.spaces:
            self._agent_obs_key = "curr_obs"
            base_observation_space = base_observation_space.spaces["curr_obs"]
        if isinstance(base_observation_space, Box) and base_observation_space.dtype != np.float32:
            base_observation_space = Box(
                low=np.asarray(base_observation_space.low, dtype=np.float32),
                high=np.asarray(base_observation_space.high, dtype=np.float32),
                shape=base_observation_space.shape,
                dtype=np.float32,
            )
        self.possible_agents = [f"agent-{i}" for i in range(int(getattr(env, "num_agents", 0)))]
        self.agents = list(self.possible_agents)
        # RLlib New API Stack expects per-agent spaces on multi-agent envs.
        self.observation_spaces = {
            agent_id: base_observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {agent_id: env.action_space for agent_id in self.possible_agents}
        # MultiAgentEnv in modern RLlib defines these as None by default.
        # Set them explicitly so policy specs have concrete spaces.
        self.observation_space = base_observation_space
        self.action_space = env.action_space

    def _convert_obs(self, obs):
        def _convert_single(agent_obs):
            if self._agent_obs_key is not None and isinstance(agent_obs, dict):
                agent_obs = agent_obs.get(self._agent_obs_key, agent_obs)
            if isinstance(agent_obs, np.ndarray) and agent_obs.dtype != np.float32:
                agent_obs = agent_obs.astype(np.float32, copy=False)
            return agent_obs

        if isinstance(obs, dict):
            return {agent_id: _convert_single(agent_obs) for agent_id, agent_obs in obs.items()}
        return _convert_single(obs)

    def reset(self, *, seed=None, options=None):
        self._elapsed_steps = 0
        obs = self._env.reset(seed=seed, options=options)
        if isinstance(obs, tuple) and len(obs) == 2:
            return self._convert_obs(obs[0]), obs[1]
        obs = self._convert_obs(obs)
        infos = {agent_id: {} for agent_id in obs.keys()} if isinstance(obs, dict) else {}
        return obs, infos

    def step(self, action_dict):
        out = self._env.step(action_dict)
        self._elapsed_steps += 1
        if isinstance(out, tuple) and len(out) == 5:
            obs, rewards, terminations, truncations, infos = out
            obs = self._convert_obs(obs)
            if self._max_episode_steps is not None and self._elapsed_steps >= self._max_episode_steps:
                terminations = dict(terminations)
                truncations = dict(truncations)
                agent_ids = set(obs.keys()) | set(rewards.keys()) | {
                    agent_id for agent_id in terminations.keys() if agent_id != "__all__"
                }
                for agent_id in agent_ids:
                    terminations.setdefault(agent_id, False)
                    truncations[agent_id] = truncations.get(agent_id, False) or (
                        not terminations[agent_id]
                    )
                terminations.setdefault("__all__", False)
                truncations["__all__"] = True
            return obs, rewards, terminations, truncations, infos
        obs, rewards, dones, infos = out
        obs = self._convert_obs(obs)
        all_done = bool(dones.get("__all__", False))
        terminations = {agent_id: done for agent_id, done in dones.items() if agent_id != "__all__"}
        truncations = {agent_id: False for agent_id in terminations.keys()}
        horizon_reached = (
            self._max_episode_steps is not None and self._elapsed_steps >= self._max_episode_steps
        )
        if all_done:
            terminations = {agent_id: True for agent_id in terminations.keys()}
        if horizon_reached and not all_done:
            for agent_id in terminations.keys():
                if not terminations[agent_id]:
                    truncations[agent_id] = True
        terminations["__all__"] = all_done
        truncations["__all__"] = horizon_reached and not all_done
        return obs, rewards, terminations, truncations, infos

    def __getattr__(self, item):
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(item)
        return getattr(env, item)


class _ResultNaNSanitizerCallback(RLlibCallback):
    """Sanitize non-informative NaN/Inf metrics before Tune/TensorBoard logging."""

    _NAN_SAFE_KEYS = {
        "env_reset_timer",
        "connector_pipeline_timer",
        "num_trainable_parameters",
        "num_non_trainable_parameters",
    }

    @staticmethod
    def _is_non_finite(value):
        if isinstance(value, (float, np.floating)):
            return not np.isfinite(value)
        return False

    @classmethod
    def _should_replace(cls, path_parts):
        if not path_parts:
            return False
        leaf = path_parts[-1]
        if leaf in cls._NAN_SAFE_KEYS:
            return True
        return "timers" in path_parts

    @classmethod
    def _sanitize_in_place(cls, obj, path_parts=()):
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = cls._sanitize_in_place(value, path_parts + (str(key),))
            return obj
        if isinstance(obj, list):
            for idx, value in enumerate(obj):
                obj[idx] = cls._sanitize_in_place(value, path_parts + (str(idx),))
            return obj
        if cls._is_non_finite(obj) and cls._should_replace(path_parts):
            return 0.0
        return obj

    def on_train_result(self, *, algorithm, result, **kwargs):
        self._sanitize_in_place(result)


def build_experiment_config_dict(args):
    """
    Create a config dict for a single Experiment object.
    :param args: The parsed arguments.
    :return: An Experiment config dict.
    """
    if args.model != "baseline":
        raise ValueError(
            f"Unsupported model '{args.model}' for the Ray RLlib 2.40+ stack. "
            "Only 'baseline' is supported."
        )
    if args.eager_mode:
        raise ValueError(
            "--eager_mode is not supported in this torch-only RLlib New API Stack setup."
        )

    episode_horizon = 1000
    base_env_creator = get_env_creator(
        env=args.env,
        num_agents=args.num_agents,
        return_agent_actions=False,
        use_collective_reward=args.use_collective_reward,
        num_switches=args.num_switches,
    )

    def env_creator(env_config):
        max_episode_steps = episode_horizon
        if isinstance(env_config, dict):
            max_episode_steps = env_config.get("max_episode_steps", episode_horizon)
        return _RLlibEnvAdapter(base_env_creator(env_config), max_episode_steps=max_episode_steps)

    env_name = args.env + "_env"
    register_env(env_name, env_creator)

    single_env = env_creator({"max_episode_steps": episode_horizon})
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Create 1 distinct policy spec per agent (independent policies).
    policy_specs = {}
    for i in range(args.num_agents):
        policy_specs[f"agent-{i}"] = PolicySpec(
            observation_space=obs_space,
            action_space=act_space,
            config={},
        )

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id

    env_config = {
        "func_create": base_env_creator,
        "env_name": env_name,
        "max_episode_steps": episode_horizon,
    }
    if env_name == "switch_env":
        env_config["num_switches"] = args.num_switches

    conv_filters = [[6, [3, 3], 1]]
    fcnet_hiddens = [32, 32]

    train_batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else max(1, args.num_workers) * args.num_envs_per_worker * args.rollout_fragment_length
    )

    lr_schedule = (
        list(zip(args.lr_schedule_steps, args.lr_schedule_weights))
        if args.lr_schedule_steps is not None and args.lr_schedule_weights is not None
        else None
    )

    algorithm_config = (
        get_algorithm_config_builder(args.algorithm)
        .framework("torch")
        .environment(
            env=env_name,
            env_config=env_config,
        )
        .resources(
            num_cpus_for_main_process=args.cpus_for_driver,
            num_gpus=args.gpus_for_driver,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
            num_cpus_per_env_runner=args.cpus_per_worker,
            num_gpus_per_env_runner=args.gpus_per_worker,
        )
        .learners(
            num_learners=0,
            # New API Stack trains on the Learner; any value >0 enables GPU training
            # on the local Learner when num_learners=0.
            num_gpus_per_learner=args.gpus_for_driver,
        )
        .multi_agent(
            policies=policy_specs,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            gamma=0.99,
            lr=args.lr,
            lr_schedule=lr_schedule,
            train_batch_size=train_batch_size,
            entropy_coeff=args.entropy_coeff,
            grad_clip=args.grad_clip,
        )
        .rl_module(
            model_config={
                "use_lstm": False,
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
            }
        )
    )
    callbacks_class = single_env.get_environment_callbacks()
    callback_classes = [_ResultNaNSanitizerCallback]
    if callbacks_class is not None:
        if isinstance(callbacks_class, list):
            callback_classes = callbacks_class + callback_classes
        else:
            callback_classes.insert(0, callbacks_class)
    algorithm_config = algorithm_config.callbacks(callbacks_class=callback_classes)

    if args.algorithm.upper() == "PPO":
        algorithm_config = apply_ppo_training_config(
            algorithm_config=algorithm_config,
            args=args,
            train_batch_size=train_batch_size,
        )
    elif args.algorithm.upper() == "IMPALA":
        algorithm_config = algorithm_config.training(vf_loss_coeff=0.1)
    else:
        raise ValueError("Unsupported algorithm. Supported algorithms for this stack: PPO, IMPALA.")

    config_dict = algorithm_config.to_dict()
    if args.tune_hparams:
        tune_dict = create_hparam_tune_dict(model=args.model, is_config=True)
        update_nested_dict(config_dict, tune_dict)

    return config_dict


def get_trainer(args):
    """
    Creates a trainer depending on what args are specified.
    :param args: The parsed arguments.
    :return: A new trainer.
    """
    if args.model != "baseline":
        raise ValueError(
            f"Unsupported model '{args.model}' for the Ray RLlib 2.40+ stack. "
            "Only 'baseline' is supported."
        )
    return get_algorithm_trainable(args.algorithm)


def initialize_ray(args):
    """
    Initialize ray and automatically turn on local mode when debugging.
    :param args: The parsed arguments.
    """
    if sys.gettrace() is not None:
        print(
            "Debug mode detected through sys.gettrace(), turning on ray local mode. Saving"
            " experiment under ray_results/debug_experiment"
        )
        args.local_mode = True
    if args.multi_node and args.local_mode:
        sys.exit("You cannot have both local mode and multi node on at the same time")
    init_kwargs = {"address": args.address, "local_mode": args.local_mode}
    if args.memory is not None:
        init_kwargs["memory"] = args.memory
    if args.object_store_memory is not None:
        init_kwargs["object_store_memory"] = args.object_store_memory
    if args.redis_max_memory is not None:
        init_kwargs["redis_max_memory"] = args.redis_max_memory

    # Ray 2.x renamed include_webui -> include_dashboard.
    init_kwargs["include_dashboard"] = False

    ray.init(**init_kwargs)


def _adjust_gpu_config_to_cluster(config):
    """Keep RLlib GPU settings aligned with available cluster GPUs."""
    cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))
    # Use Ray's accelerator managers to detect physically available local GPUs.
    # This guards against clusters reporting logical GPU resources on CPU-only hosts.
    from ray._private.accelerators import (
        AMDGPUAcceleratorManager,
        IntelGPUAcceleratorManager,
        MetaxGPUAcceleratorManager,
        NvidiaGPUAcceleratorManager,
    )

    local_gpus = float(
        max(
            NvidiaGPUAcceleratorManager.get_current_node_num_accelerators(),
            IntelGPUAcceleratorManager.get_current_node_num_accelerators(),
            AMDGPUAcceleratorManager.get_current_node_num_accelerators(),
            MetaxGPUAcceleratorManager.get_current_node_num_accelerators(),
        )
    )

    framework = str(config.get("framework", "torch")).lower()
    if framework not in ("", "torch"):
        raise ValueError(
            f"Unsupported framework '{framework}'. This stack supports only 'torch'."
        )

    import torch

    framework_visible_gpus = float(torch.cuda.device_count())

    effective_gpus = min(cluster_gpus, local_gpus) if local_gpus >= 0 else cluster_gpus
    if framework_visible_gpus is not None:
        effective_gpus = min(effective_gpus, framework_visible_gpus)

    gpu_keys = (
        "num_gpus",
        "num_gpus_per_env_runner",
        "num_gpus_per_learner",
        "num_gpus_per_offline_eval_runner",
    )

    requested = {k: float(config.get(k, 0) or 0.0) for k in gpu_keys}
    if effective_gpus <= 0:
        if any(value > 0 for value in requested.values()):
            print(
                "No usable local GPUs detected. Overriding GPU config to CPU-only: "
                f"(cluster_gpus={cluster_gpus}, local_gpus={local_gpus}) "
                + (
                    ""
                    if framework_visible_gpus is None
                    else f"(framework_visible_gpus={framework_visible_gpus}) "
                )
                + f"{ {k: v for k, v in requested.items() if v > 0} }"
            )
        config["num_gpus"] = 0
        config["num_gpus_per_env_runner"] = 0
        config["num_gpus_per_learner"] = 0
        config["num_gpus_per_offline_eval_runner"] = 0
        return config

    # Cap driver GPU request to available resources.
    if requested["num_gpus"] > effective_gpus:
        print(
            f"Requested num_gpus={requested['num_gpus']} but usable GPUs are {effective_gpus} "
            f"(cluster_gpus={cluster_gpus}, local_gpus={local_gpus}). "
            f"Capping num_gpus to {effective_gpus}."
        )
        config["num_gpus"] = effective_gpus

    # Cap learner GPU request to available resources.
    if requested["num_gpus_per_learner"] > effective_gpus:
        print(
            "Requested "
            f"num_gpus_per_learner={requested['num_gpus_per_learner']} but usable GPUs are "
            f"{effective_gpus} (cluster_gpus={cluster_gpus}, local_gpus={local_gpus}). "
            f"Capping num_gpus_per_learner to {effective_gpus}."
        )
        config["num_gpus_per_learner"] = effective_gpus

    return config


def get_experiment_name(args):
    """
    Build an experiment name based on environment, model and algorithm.
    :param args: The parsed arguments.
    :return: The experiment name.
    """
    if sys.gettrace() is not None:
        exp_name = "debug_experiment"
    elif args.exp_name is None:
        exp_name = args.env + "_" + args.model + "_" + args.algorithm
    else:
        exp_name = args.exp_name
    return exp_name


def build_experiment_dict(args, experiment_name, trainer, config):
    """
    Creates all parameters needed to create an Experiment object and puts them into a dict.
    :param args: The parsed arguments .
    :param experiment_name: The experiment name.
    :param trainer: The trainer used for the experiment.
    :param config: The config dict with experiment parameters.
    :return: A dict that can be unpacked to create an Experiment object.
    """
    experiment_dict = {
        "name": experiment_name,
        "run": trainer,
        "stop": {},
        "checkpoint_freq": args.checkpoint_frequency,
        "checkpoint_at_end": True,
        "config": config,
        "num_samples": args.num_samples,
        "max_failures": -1,
    }
    if args.stop_at_episode_reward_min is not None:
        experiment_dict["stop"]["episode_reward_min"] = args.stop_at_episode_reward_min
    if args.stop_at_timesteps_total is not None:
        # RLlib New Stack reports this counter instead of legacy `timesteps_total`.
        experiment_dict["stop"]["num_env_steps_sampled_lifetime"] = args.stop_at_timesteps_total
    training_iteration_stop = args.stop_at_training_iteration
    if args.algorithm.upper() == "PPO":
        ppo_train_batch_size = (
            args.train_batch_size
            if args.train_batch_size is not None
            else max(1, args.num_workers) * args.num_envs_per_worker * args.rollout_fragment_length
        )
        training_iteration_stop = resolve_config_ppo(
            args=args,
            train_batch_size=ppo_train_batch_size,
        )["max_iters"]
    if training_iteration_stop is not None:
        experiment_dict["stop"]["training_iteration"] = training_iteration_stop

    if args.use_s3:
        date = datetime.now(tz=pytz.utc)
        date = date.astimezone(pytz.timezone("US/Pacific")).strftime("%m-%d-%Y")
        s3_string = "s3://ssd-reproduce/" + date + "/" + experiment_name
        experiment_dict["upload_dir"] = s3_string

    return experiment_dict


def create_experiment(args):
    """
    Create a single experiment from arguments.
    :param args: The parsed arguments.
    :return: A new experiment with its own trainer.
    """
    experiment_name = get_experiment_name(args)
    config = build_experiment_config_dict(args)
    trainer = get_trainer(args=args)
    experiment_dict = build_experiment_dict(args, experiment_name, trainer, config)
    return experiment_dict


def create_hparam_tune_dict(model, is_config=False):
    """
    Create a hyperparameter tuning dict for population-based training.
    :param is_config: Whether these hyperparameters are being used in the config dict or not.
    When used for the config dict, all hyperparameter-generating functions need to be wrapped with
    tune.sample_from, so we do this automatically here.
    When it is not used for the config dict, it is for PBT initialization, where a lambda is needed
    as a function wrapper.
    :return: The hyperparameter tune dict.
    """

    def wrapper(fn):
        if is_config:
            return tune.sample_from(lambda spec: fn)
        else:
            return lambda: fn

    if model != "baseline":
        raise ValueError(
            f"Unsupported model '{model}' for the Ray RLlib 2.40+ stack. "
            "Only 'baseline' is supported."
        )

    return {
        "entropy_coeff": wrapper(np.random.exponential(1 / 1000)),
        "lr": wrapper(np.random.uniform(0.00001, 0.01)),
    }


def create_pbt_scheduler(model):
    """
    Create a population-based training (PBT) scheduler.
    :return: A new PBT scheduler.
    """
    hyperparam_mutations = create_hparam_tune_dict(model=model, is_config=False)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        metric="episode_reward_mean",
        mode="max",
        hyperparam_mutations=hyperparam_mutations,
    )
    return pbt


def run(args, experiments):
    """
    Run one or more experiments, with ray settings contained in args.
    :param args: The args to initialize ray with
    :param experiments: A list of experiments to run
    """
    initialize_ray(args)
    experiments["config"] = _adjust_gpu_config_to_cluster(experiments["config"])
    gpu_cfg_keys = (
        "num_gpus",
        "num_gpus_per_env_runner",
        "num_gpus_per_learner",
        "num_gpus_per_offline_eval_runner",
    )
    resolved_gpu_cfg = {k: experiments["config"].get(k, None) for k in gpu_cfg_keys}
    print(f"Resolved RLlib GPU config: {resolved_gpu_cfg}")
    scheduler = create_pbt_scheduler(args.model) if args.tune_hparams else None
    run_kwargs = {
        "name": experiments["name"],
        "stop": experiments["stop"],
        "checkpoint_freq": experiments["checkpoint_freq"],
        "config": experiments["config"],
        "num_samples": experiments["num_samples"],
        "max_failures": experiments["max_failures"],
        "resume": args.resume,
        "scheduler": scheduler,
        "reuse_actors": args.tune_hparams,
    }

    supported_args = set(inspect.signature(tune.run).parameters.keys())
    if "checkpoint_at_end" in supported_args:
        run_kwargs["checkpoint_at_end"] = experiments.get("checkpoint_at_end", True)
    if "queue_trials" in supported_args:
        run_kwargs["queue_trials"] = args.use_s3
    if "upload_dir" in supported_args and experiments.get("upload_dir") is not None:
        run_kwargs["upload_dir"] = experiments.get("upload_dir")

    tune.run(experiments["run"], **run_kwargs)


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    experiment = create_experiment(parsed_args)
    run(parsed_args, experiment)
