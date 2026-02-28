import argparse
import copy
import inspect
import sys
from datetime import datetime
from pathlib import Path

# Allow running this script directly (e.g. VS Code "Run Python File")
# without requiring `PYTHONPATH=.` in the shell.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytz
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from social_dilemmas.config.default_args import add_default_args
from social_dilemmas.envs.env_creator import get_env_creator
from utility_funcs import update_nested_dict

try:
    from ray.rllib.agents.registry import get_agent_class as _legacy_get_agent_class
except ImportError:  # pragma: no cover - Ray >=2.x
    _legacy_get_agent_class = None
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - fallback for legacy Ray versions
    from ray.rllib.env import MultiAgentEnv

parser = argparse.ArgumentParser()
add_default_args(parser)


def get_agent_class(algorithm_name):
    """Return RLlib algorithm class without triggering global trainable registration."""
    if _legacy_get_agent_class is not None:
        return _legacy_get_agent_class(algorithm_name)

    algo = str(algorithm_name).upper()
    if algo == "PPO":
        from ray.rllib.algorithms.ppo import PPO

        return PPO
    if algo == "IMPALA":
        from ray.rllib.algorithms.impala import IMPALA

        return IMPALA
    if algo == "A3C":
        from ray.rllib.algorithms.a3c import A3C

        return A3C

    # Fallback for non-standard algorithm names.
    from ray.tune.registry import get_trainable_cls

    return get_trainable_cls(algorithm_name)


def _maybe_register_custom_model(model_key, model_name):
    try:
        if model_key == "scm":
            from models.scm_model import SocialCuriosityModule

            ModelCatalog.register_custom_model(model_name, SocialCuriosityModule)
        elif model_key == "moa":
            from models.moa_model import MOAModel

            ModelCatalog.register_custom_model(model_name, MOAModel)
        elif model_key == "baseline":
            from models.baseline_model import BaselineModel

            ModelCatalog.register_custom_model(model_name, BaselineModel)
        return True
    except ImportError:
        return False


def _default_algorithm_config(agent_cls):
    if hasattr(agent_cls, "_default_config"):
        return copy.deepcopy(agent_cls._default_config)
    if hasattr(agent_cls, "get_default_config"):
        config = agent_cls.get_default_config()
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        return copy.deepcopy(config)
    return {}


class _RLlibEnvAdapter(MultiAgentEnv):
    """Adapter for legacy MapEnv API to RLlib's expected Gymnasium-style API."""

    def __init__(self, env, max_episode_steps=None):
        self._env = env
        self._max_episode_steps = (
            int(max_episode_steps) if max_episode_steps is not None and max_episode_steps > 0 else None
        )
        self._elapsed_steps = 0
        # MultiAgentEnv in modern RLlib defines these as None by default.
        # Set them explicitly so policy specs have concrete spaces.
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        self._elapsed_steps = 0
        try:
            obs = self._env.reset(seed=seed, options=options)
        except TypeError:
            obs = self._env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs
        infos = {agent_id: {} for agent_id in obs.keys()} if isinstance(obs, dict) else {}
        return obs, infos

    def step(self, action_dict):
        out = self._env.step(action_dict)
        self._elapsed_steps += 1
        if isinstance(out, tuple) and len(out) == 5:
            obs, rewards, terminations, truncations, infos = out
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
        return getattr(self._env, item)


def _normalize_builtin_rllib_config(config):
    # RLlib 2.54+ renamed worker concepts to env runner concepts.
    renames = {
        "num_workers": "num_env_runners",
        "num_envs_per_worker": "num_envs_per_env_runner",
        "num_cpus_per_worker": "num_cpus_per_env_runner",
        "num_gpus_per_worker": "num_gpus_per_env_runner",
    }
    for old_key, new_key in renames.items():
        if old_key in config:
            # Prefer explicit legacy CLI-sourced values over defaults in new keys.
            config[new_key] = config[old_key]
            config.pop(old_key, None)

    if "num_cpus_for_driver" in config:
        config["num_cpus_for_main_process"] = config.pop("num_cpus_for_driver")

    if "callbacks" in config and "callbacks_class" not in config:
        config["callbacks_class"] = config.pop("callbacks")

    # Ray can drop an empty main-process bundle from the placement group
    # when this is 0, which breaks env-runner bundle indexing.
    if (
        config.get("num_env_runners", 0) > 0
        and config.get("num_cpus_for_main_process", 0) <= 0
        and config.get("num_gpus", 0) <= 0
    ):
        config["num_cpus_for_main_process"] = 1

    # Keep only the modern key for built-in PPO config.
    if "sgd_minibatch_size" in config and "minibatch_size" in config:
        config.pop("sgd_minibatch_size", None)

    model_config = config.get("model", {})
    if isinstance(model_config, dict) and model_config.get("custom_model") in (None, ""):
        model_config.pop("custom_options", None)

    return config


def build_experiment_config_dict(args):
    """
    Create a config dict for a single Experiment object.
    :param args: The parsed arguments.
    :return: An Experiment config dict.
    """
    episode_horizon = 1000
    base_env_creator = get_env_creator(
        env=args.env,
        num_agents=args.num_agents,
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

    model_name = args.model + "_lstm"
    use_custom_model = _maybe_register_custom_model(args.model, model_name)

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        if use_custom_model:
            return None, obs_space, act_space, {"custom_model": model_name}
        return None, obs_space, act_space, {}

    # Create 1 distinct policy per agent
    policy_graphs = {}
    for i in range(args.num_agents):
        policy_graphs["agent-" + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id

    agent_cls = get_agent_class(args.algorithm)
    config = _default_algorithm_config(agent_cls)

    config["env"] = env_name
    config["eager"] = args.eager_mode
    # Keep RLlib on the old API stack for this legacy codebase.
    config["enable_rl_module_and_learner"] = False
    config["enable_env_runner_and_connector_v2"] = False

    # information for replay
    config.setdefault("env_config", {})
    config["env_config"]["func_create"] = base_env_creator
    config["env_config"]["env_name"] = env_name
    config["env_config"]["max_episode_steps"] = episode_horizon
    if env_name == "switch_env":
        config["env_config"]["num_switches"] = args.num_switches

    conv_filters = [[6, [3, 3], 1]]
    fcnet_hiddens = [32, 32]
    lstm_cell_size = 128

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

    # hyperparams
    update_nested_dict(
        config,
        {
            "horizon": episode_horizon,
            "gamma": 0.99,
            "lr": args.lr,
            "lr_schedule": lr_schedule,
            "rollout_fragment_length": args.rollout_fragment_length,
            "train_batch_size": train_batch_size,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "num_gpus": args.gpus_for_driver,  # The number of GPUs for the driver
            "num_cpus_for_driver": args.cpus_for_driver,
            "num_gpus_per_worker": args.gpus_per_worker,  # Can be a fraction
            "num_gpus_per_env_runner": args.gpus_per_worker,
            "num_gpus_per_learner": 0,
            "num_gpus_per_offline_eval_runner": 0,
            "num_cpus_per_worker": args.cpus_per_worker,  # Can be a fraction
            "entropy_coeff": args.entropy_coeff,
            "grad_clip": args.grad_clip,
            "multiagent": {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn},
            "callbacks": single_env.get_environment_callbacks(),
            "model": {
                "use_lstm": False,
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
                "custom_options": {
                    "cell_size": lstm_cell_size,
                    "num_other_agents": args.num_agents - 1,
                },
            },
        },
    )

    if use_custom_model:
        config["model"]["custom_model"] = model_name

    if args.model != "baseline":
        config["model"]["custom_options"].update(
            {
                "moa_loss_weight": args.moa_loss_weight,
                "influence_reward_clip": 10,
                "influence_reward_weight": args.influence_reward_weight,
                "influence_reward_schedule_steps": args.influence_reward_schedule_steps,
                "influence_reward_schedule_weights": args.influence_reward_schedule_weights,
                "return_agent_actions": True,
                "influence_divergence_measure": "kl",
                "train_moa_only_when_visible": True,
                "influence_only_when_visible": True,
            }
        )

    if args.model == "scm":
        config["model"]["custom_options"].update(
            {
                "scm_loss_weight": args.scm_loss_weight,
                "curiosity_reward_clip": 10,
                "curiosity_reward_weight": args.curiosity_reward_weight,
                "curiosity_reward_schedule_steps": args.curiosity_reward_schedule_steps,
                "curiosity_reward_schedule_weights": args.curiosity_reward_schedule_weights,
                "scm_forward_vs_inverse_loss_weight": args.scm_forward_vs_inverse_loss_weight,
            }
        )

    if args.tune_hparams:
        tune_dict = create_hparam_tune_dict(model=args.model, is_config=True)
        update_nested_dict(config, tune_dict)

    if args.algorithm == "PPO":
        config.update(
            {
                "num_sgd_iter": 10,
                "sgd_minibatch_size": args.ppo_sgd_minibatch_size
                if args.ppo_sgd_minibatch_size is not None
                else int(train_batch_size / 4),
                "minibatch_size": args.ppo_sgd_minibatch_size
                if args.ppo_sgd_minibatch_size is not None
                else int(train_batch_size / 4),
                "vf_loss_coeff": 1e-4,
                "vf_share_layers": True,
            }
        )
    elif args.algorithm == "A3C" or args.algorithm == "IMPALA":
        config.update({"vf_loss_coeff": 0.1})
    else:
        sys.exit("The only available algorithms are A3C, PPO and IMPALA")

    return config


def get_trainer(args, config):
    """
    Creates a trainer depending on what args are specified.
    :param args: The parsed arguments.
    :param config: The config dict that is provided to the trainer.
    :return: A new trainer.
    """
    trainer = None
    if args.model == "baseline":
        if args.algorithm == "A3C":
            try:
                from algorithms.a3c_baseline import build_a3c_baseline_trainer

                trainer = build_a3c_baseline_trainer(config)
            except ImportError:
                trainer = get_agent_class("A3C")
        if args.algorithm == "PPO":
            try:
                from algorithms.ppo_baseline import build_ppo_baseline_trainer

                trainer = build_ppo_baseline_trainer(config)
            except ImportError:
                trainer = get_agent_class("PPO")
        if args.algorithm == "IMPALA":
            try:
                from algorithms.impala_baseline import build_impala_baseline_trainer

                trainer = build_impala_baseline_trainer(config)
            except ImportError:
                trainer = get_agent_class("IMPALA")
    elif args.model == "moa":
        if args.algorithm == "A3C":
            from algorithms.a3c_moa import build_a3c_moa_trainer

            trainer = build_a3c_moa_trainer(config)
        if args.algorithm == "PPO":
            from algorithms.ppo_moa import build_ppo_moa_trainer

            trainer = build_ppo_moa_trainer(config)
        if args.algorithm == "IMPALA":
            from algorithms.impala_moa import build_impala_moa_trainer

            trainer = build_impala_moa_trainer(config)
    elif args.model == "scm":
        if args.algorithm == "A3C":
            # trainer = build_a3c_scm_trainer(config)
            raise NotImplementedError
        if args.algorithm == "PPO":
            from algorithms.ppo_scm import build_ppo_scm_trainer

            trainer = build_ppo_scm_trainer(config)
        if args.algorithm == "IMPALA":
            # trainer = build_impala_scm_trainer(config)
            raise NotImplementedError
    if trainer is None:
        raise NotImplementedError("The provided combination of model and algorithm was not found.")
    return trainer


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

    try:
        ray.init(**init_kwargs)
    except TypeError:
        init_kwargs.pop("memory", None)
        init_kwargs.pop("redis_max_memory", None)
        ray.init(**init_kwargs)


def _adjust_gpu_config_to_cluster(config):
    """Keep RLlib GPU settings aligned with available cluster GPUs."""
    cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))
    local_gpus = 0.0
    framework_visible_gpus = None
    try:
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
    except Exception:
        local_gpus = 0.0

    framework = str(config.get("framework", "")).lower()
    if framework == "torch":
        try:
            import torch

            framework_visible_gpus = float(torch.cuda.device_count())
        except Exception:
            framework_visible_gpus = 0.0
    elif framework in ("tf", "tf2"):
        try:
            import tensorflow as tf

            framework_visible_gpus = float(len(tf.config.list_physical_devices("GPU")))
        except Exception:
            framework_visible_gpus = 0.0

    effective_gpus = min(cluster_gpus, local_gpus) if local_gpus >= 0 else cluster_gpus
    if framework_visible_gpus is not None:
        effective_gpus = min(effective_gpus, framework_visible_gpus)

    # Keep legacy and modern keys consistent.
    if "num_gpus_per_worker" in config and "num_gpus_per_env_runner" not in config:
        config["num_gpus_per_env_runner"] = config["num_gpus_per_worker"]

    gpu_keys = (
        "num_gpus",
        "num_gpus_per_worker",
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
        config.pop("num_gpus_per_worker", None)
        return config

    # Cap driver GPU request to available resources.
    if requested["num_gpus"] > effective_gpus:
        print(
            f"Requested num_gpus={requested['num_gpus']} but usable GPUs are {effective_gpus} "
            f"(cluster_gpus={cluster_gpus}, local_gpus={local_gpus}). "
            f"Capping num_gpus to {effective_gpus}."
        )
        config["num_gpus"] = effective_gpus

    # Keep only the modern env-runner key; old key raises on modern RLlib.
    if "num_gpus_per_env_runner" in config:
        config.pop("num_gpus_per_worker", None)

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
        experiment_dict["stop"]["timesteps_total"] = args.stop_at_timesteps_total

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
    trainer = get_trainer(args=args, config=config)
    is_builtin_rllib_trainer = isinstance(trainer, str) or (
        inspect.isclass(trainer) and str(getattr(trainer, "__module__", "")).startswith("ray.rllib.")
    )
    if is_builtin_rllib_trainer:
        config = _normalize_builtin_rllib_config(config)
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

    baseline_options = {}
    model_options = {}
    if model == "baseline":
        baseline_options = {
            "entropy_coeff": wrapper(np.random.exponential(1 / 1000)),
            "lr": wrapper(np.random.uniform(0.00001, 0.01)),
        }
    if model == "moa":
        model_options = {
            "moa_loss_weight": wrapper(np.random.exponential(1 / 15)),
            "influence_reward_weight": wrapper(np.random.exponential(1)),
        }
    elif model == "scm":
        model_options = {
            "scm_loss_weight": wrapper(np.random.exponential(1 / 2)),
            "curiosity_reward_weight": wrapper(np.random.exponential(1)),
            "scm_forward_vs_inverse_loss_weight": wrapper(np.random.uniform(0, 1)),
        }

    hparam_dict = {
        **baseline_options,
        "model": {"custom_options": model_options},
    }
    return hparam_dict


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
