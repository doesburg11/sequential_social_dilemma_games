#!/usr/bin/env python
import argparse
import collections
import copy
import importlib
import json
import os
import pickle
import shelve
import shutil
import sys
import time
from pathlib import Path

try:
    gym = importlib.import_module("gymnasium")
except Exception:  # pragma: no cover - fallback for legacy gym installs
    gym = importlib.import_module("gym")
import numpy as np
import ray
import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - fallback for legacy Ray versions
    from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
try:
    deprecation_warning = getattr(
        importlib.import_module("ray._common.deprecation"), "deprecation_warning"
    )
except Exception:  # pragma: no cover - fallback for older Ray versions
    deprecation_warning = getattr(
        importlib.import_module("ray.rllib.utils.deprecation"), "deprecation_warning"
    )
try:
    flatten_to_single_ndarray = getattr(
        importlib.import_module("ray.rllib.utils.spaces.space_utils"), "flatten_to_single_ndarray"
    )
except Exception:  # pragma: no cover - fallback for older Ray versions
    flatten_to_single_ndarray = getattr(
        importlib.import_module("ray.rllib.utils.space_utils"), "flatten_to_single_ndarray"
    )
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.utils import merge_dicts

import utility_funcs

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

DEFAULT_DEBUG_CHECKPOINT = (
    "/home/doesburg/ray_results/harvest_baseline_PPO/"
    "PPO_harvest_env_77ba5_00000_0_2026-02-28_22-14-29/checkpoint_000005"
)
DEFAULT_DEBUG_ARGS = [
    "--run",
    "PPO",
    "--env",
    "harvest_env",
    "--episodes",
    "1",
    "--render-delay-ms",
    "10",
    "--random-action-prob",
    "0.02",
]

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))


class RolloutSaver:
    """Utility class for storing rollouts.

    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]

    If outfile is None, this class does nothing.
    """

    def __init__(
        self,
        outfile=None,
        use_shelve=False,
        write_update_file=False,
        target_steps=None,
        target_episodes=None,
        save_info=False,
    ):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = self._get_tmp_progress_filename().open(mode="w")
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            pickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes, self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps, self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append([obs, action, next_obs, reward, done, info])
            else:
                self._current_rollout.append([obs, action, next_obs, reward, done])
        self._total_steps += 1


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent " "given a checkpoint.",
        epilog=EXAMPLE_USAGE,
    )

    parser.add_argument("checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.",
    )
    required_named.add_argument("--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Suppress rendering of the environment.",
    )
    parser.add_argument(
        "--monitor",
        default=False,
        action="store_true",
        help="Wrap environment in gym Monitor to record video. NOTE: This "
        "option is deprecated: Use `--video-dir [some dir]` instead.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Specifies the directory into which videos of all episode " "rollouts will be stored.",
    )
    parser.add_argument(
        "--video-filename",
        type=str,
        default=None,
        help="Specifies the filename that the video will be saved under.",
    )
    parser.add_argument(
        "--steps", default=1000, help="Number of timesteps to roll out (overwritten by --episodes)."
    )
    parser.add_argument(
        "--episodes", default=0, help="Number of complete episodes to roll out (overrides --steps)."
    )
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Gets merged with loaded configuration from checkpoint file and "
        "`evaluation_config` settings therein.",
    )
    parser.add_argument(
        "--save-info",
        default=False,
        action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.",
    )
    parser.add_argument(
        "--use-shelve",
        default=False,
        action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
        "as it is generated). An output filename must be set using --out.",
    )
    parser.add_argument(
        "--track-progress",
        default=False,
        action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.",
    )
    parser.add_argument(
        "--random-action-prob",
        type=float,
        default=0.0,
        help="With this probability per action decision, override policy output "
        "with a random env action (useful for avoiding static videos).",
    )
    parser.add_argument(
        "--render-delay-ms",
        type=float,
        default=0.0,
        help="Delay in milliseconds between live rendered frames.",
    )
    parser.add_argument(
        "--deterministic-actions",
        default=False,
        action="store_true",
        help="Use argmax actions (greedy). Default is stochastic sampling for live visualization.",
    )
    parser.add_argument(
        "--step-log-interval",
        type=int,
        default=100,
        help="Print rollout progress every N environment steps (0 disables).",
    )
    return parser


def run(args, parser):
    def _register_optional_custom_models():
        # Some historical models require optional TF modules. Only register
        # models that can be imported in the current environment.
        model_specs = [
            ("baseline_lstm", "models.baseline_model", "BaselineModel"),
            ("moa_lstm", "models.moa_model", "MOAModel"),
            ("scm_lstm", "models.scm_model", "SocialCuriosityModule"),
        ]
        for model_name, module_name, class_name in model_specs:
            try:
                module = importlib.import_module(module_name)
                model_cls = getattr(module, class_name)
            except Exception:
                continue
            ModelCatalog.register_custom_model(model_name, model_cls)

    def _register_env_for_checkpoint(env_name, merged_config):
        env_config = merged_config.get("env_config", {})
        env_creator = env_config.get("func_create")
        if not callable(env_creator):
            return

        max_episode_steps = env_config.get("max_episode_steps")

        # Reuse the same adapter used by training when available.
        try:
            from run_scripts.train import _RLlibEnvAdapter
        except Exception:
            _RLlibEnvAdapter = None

        if _RLlibEnvAdapter is None:
            register_env(env_name, env_creator)
            return

        def wrapped_env_creator(env_cfg):
            horizon = max_episode_steps
            if isinstance(env_cfg, dict):
                horizon = env_cfg.get("max_episode_steps", max_episode_steps)
            return _RLlibEnvAdapter(env_creator(env_cfg), max_episode_steps=horizon)

        register_env(env_name, wrapped_env_creator)

    def _evaluate_and_print(agent):
        if (
            hasattr(agent, "env_runner")
            and getattr(agent.env_runner, "env", None) is None
            and hasattr(agent.env_runner, "make_env")
        ):
            agent.env_runner.make_env()
        result = agent.evaluate()
        env_stats = result.get("env_runners", {})
        print("evaluation/episode_return_mean:", env_stats.get("episode_return_mean"))
        print("evaluation/episode_len_mean:", env_stats.get("episode_len_mean"))
        print("evaluation/num_episodes:", env_stats.get("num_episodes"))

    def _load_rl_modules_from_checkpoint(checkpoint_path):
        rl_module_dir = os.path.join(
            checkpoint_path, "learner_group", "learner", "rl_module"
        )
        if not os.path.isdir(rl_module_dir):
            return {}

        modules = {}
        for policy_id in sorted(os.listdir(rl_module_dir)):
            policy_path = os.path.join(rl_module_dir, policy_id)
            if os.path.isdir(policy_path):
                modules[policy_id] = RLModule.from_checkpoint(policy_path)
        return modules

    class _DirectRLModuleAgent:
        def __init__(self, merged_config, modules):
            self.config = merged_config
            self._modules = modules

        def get_module(self, policy_id):
            if policy_id in self._modules:
                return self._modules[policy_id]
            if self._modules:
                # Fall back to first module to avoid hard crash on unexpected
                # policy mapping ids in legacy checkpoints.
                return self._modules[next(iter(self._modules))]
            raise KeyError(f"No RLModules loaded for policy '{policy_id}'")

    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!"
            )

    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    # Keep visualization/evaluation lightweight.
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    if "num_env_runners" in config:
        config["num_env_runners"] = min(2, int(config["num_env_runners"]))
    if "num_envs_per_env_runner" in config:
        config["num_envs_per_env_runner"] = 1
    if "evaluation_num_env_runners" in config:
        config["evaluation_num_env_runners"] = 0

    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config") or {})
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings.
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    is_new_api_stack = bool(
        config.get("enable_rl_module_and_learner", False)
        and config.get("enable_env_runner_and_connector_v2", False)
    )
    # For new stack runs, respect CLI episode/step limits during evaluate().
    if is_new_api_stack:
        if int(args.episodes) > 0:
            config["evaluation_duration_unit"] = "episodes"
            config["evaluation_duration"] = int(args.episodes)
        elif int(args.steps) > 0:
            config["evaluation_duration_unit"] = "timesteps"
            config["evaluation_duration"] = int(args.steps)

    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    # Determine the video output directory.
    # Deprecated way: Use (--out|~/ray_results) + "/monitor" as dir.
    video_dir = None
    if args.monitor:
        video_dir = os.path.join(
            os.path.dirname(args.out or "") or os.path.expanduser("~/ray_results/"), "monitor"
        )
    # New way: Allow user to specify a video output path.
    elif args.video_dir:
        video_dir = os.path.expanduser(args.video_dir)
    video_name = args.video_filename

    # Prefer direct RLModule checkpoint playback (PredPreyGrass-style) for
    # stability and to avoid Ray actor/raylet failures during visualization.
    direct_modules = _load_rl_modules_from_checkpoint(args.checkpoint)
    if direct_modules and is_new_api_stack:
        print(
            "Using direct RLModule checkpoint playback (no ray.init); "
            f"loaded policies: {list(direct_modules.keys())}"
        )
        direct_agent = _DirectRLModuleAgent(config, direct_modules)
        with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info,
        ) as saver:
            rollout(
                direct_agent,
                args.env,
                num_steps,
                num_episodes,
                saver,
                args.no_render,
                video_dir,
                video_name,
                random_action_prob=float(args.random_action_prob),
                render_delay_s=max(0.0, float(args.render_delay_ms) / 1000.0),
                deterministic_actions=bool(args.deterministic_actions),
                step_log_interval=max(0, int(args.step_log_interval)),
            )
            if not args.no_render and not video_dir:
                try:
                    import matplotlib.pyplot as plt

                    if plt.get_fignums():
                        plt.show(block=True)
                except Exception:
                    pass
        return

    ray.init(ignore_reinit_error=True)
    _register_optional_custom_models()
    _register_env_for_checkpoint(args.env, config)

    agent = None
    try:
        # Create the Trainer from config.
        cls = get_trainable_cls(args.run)
        agent = cls(env=args.env, config=config)
        # Load state from checkpoint.
        agent.restore(args.checkpoint)

        # New API stack-friendly checkpoint evaluation path.
        if is_new_api_stack and args.no_render and not video_dir and args.out is None:
            _evaluate_and_print(agent)
            return

        # Do the actual rollout.
        with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info,
        ) as saver:
            rollout(
                agent,
                args.env,
                num_steps,
                num_episodes,
                saver,
                args.no_render,
                video_dir,
                video_name,
                random_action_prob=float(args.random_action_prob),
                render_delay_s=max(0.0, float(args.render_delay_ms) / 1000.0),
                deterministic_actions=bool(args.deterministic_actions),
                step_log_interval=max(0, int(args.step_log_interval)),
            )
            if not args.no_render and not video_dir:
                try:
                    import matplotlib.pyplot as plt

                    if plt.get_fignums():
                        plt.show(block=True)
                except Exception:
                    pass
    finally:
        if agent is not None:
            try:
                agent.stop()
            except Exception:
                pass
        if ray.is_initialized():
            ray.shutdown()


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def rollout(
    agent,
    env_name,
    num_steps,
    num_episodes=0,
    saver=None,
    no_render=True,
    video_dir=None,
    video_name=None,
    random_action_prob=0.0,
    render_delay_s=0.0,
    deterministic_actions=False,
    step_log_interval=100,
):
    policy_agent_mapping = default_policy_agent_mapping
    use_module_inference = False
    policy_action_spaces = {}

    if saver is None:
        saver = RolloutSaver()

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        policy_action_spaces = {p: m.action_space for p, m in policy_map.items()}
    else:
        env_config = agent.config.get("env_config", {})
        env_creator = env_config.get("func_create")

        # Prefer direct env construction for this project so live matplotlib
        # rendering uses the underlying MapEnv implementation.
        if callable(env_creator):
            max_episode_steps = env_config.get("max_episode_steps")
            try:
                from run_scripts.train import _RLlibEnvAdapter

                env = _RLlibEnvAdapter(
                    env_creator(env_config),
                    max_episode_steps=max_episode_steps,
                )
            except Exception:
                env = env_creator(env_config)

            multiagent = isinstance(env, MultiAgentEnv)
            multiagent_config = agent.config.get("multiagent", {})
            policy_specs = multiagent_config.get("policies", {})
            if callable(multiagent_config.get("policy_mapping_fn")):
                policy_agent_mapping = multiagent_config["policy_mapping_fn"]
            if isinstance(policy_specs, dict) and policy_specs:
                policy_ids = list(policy_specs.keys())
            elif hasattr(env, "possible_agents") and env.possible_agents:
                policy_ids = list(env.possible_agents)
            else:
                policy_ids = [DEFAULT_POLICY_ID]
            policy_map = {policy_id: None for policy_id in policy_ids}
            state_init = {policy_id: [] for policy_id in policy_map.keys()}
            use_lstm = {policy_id: len(state) > 0 for policy_id, state in state_init.items()}
            policy_action_spaces = {
                policy_id: getattr(env, "action_space", None) for policy_id in policy_map.keys()
            }
            use_module_inference = True
        else:
            env = gym.make(env_name)
            multiagent = False
            try:
                policy_map = {DEFAULT_POLICY_ID: agent.policy}
            except AttributeError:
                raise AttributeError(
                    "Agent ({}) does not have a `policy` property! This is needed "
                    "for performing (trained) agent rollouts.".format(agent)
                )
            use_lstm = {DEFAULT_POLICY_ID: False}
            policy_action_spaces = {
                DEFAULT_POLICY_ID: getattr(agent.policy, "action_space", env.action_space)
            }

    def _coerce_action(raw_action, action_space):
        if hasattr(action_space, "n"):
            if isinstance(raw_action, np.ndarray):
                return int(raw_action.item())
            return int(raw_action)
        return flatten_to_single_ndarray(raw_action)

    def _compute_action_from_module(policy_id, observation, state_in=None):
        module = agent.get_module(policy_id)
        module_input = {
            Columns.OBS: torch.as_tensor(np.asarray(observation)[None], dtype=torch.float32)
        }
        if state_in:
            module_input[Columns.STATE_IN] = [
                torch.as_tensor(np.asarray(s)[None], dtype=torch.float32) for s in state_in
            ]

        module_out = module.forward_inference(module_input)

        if Columns.ACTION_DIST_INPUTS in module_out:
            dist_inputs = module_out[Columns.ACTION_DIST_INPUTS]
            action_space = policy_action_spaces.get(policy_id)
            if hasattr(action_space, "n"):
                if deterministic_actions:
                    action_batch = torch.argmax(dist_inputs, dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=dist_inputs)
                    action_batch = dist.sample()
            else:
                action_batch = dist_inputs
        elif Columns.ACTIONS in module_out:
            action_batch = module_out[Columns.ACTIONS]
        else:
            raise RuntimeError(
                f"Module inference output for policy '{policy_id}' did not include action keys."
            )

        if torch.is_tensor(action_batch):
            action_value = action_batch.detach().cpu().numpy()[0]
        else:
            action_value = np.asarray(action_batch)[0]

        state_out = None
        if Columns.STATE_OUT in module_out:
            state_out = []
            for state_tensor in module_out[Columns.STATE_OUT]:
                if torch.is_tensor(state_tensor):
                    state_tensor = state_tensor.detach().cpu().numpy()
                state_arr = np.asarray(state_tensor)
                state_out.append(state_arr[0] if state_arr.ndim > 0 else state_arr)

        return action_value, state_out

    action_init = {}
    for policy_id, policy in policy_map.items():
        action_space = policy_action_spaces.get(policy_id)
        if action_space is None:
            action_space = getattr(policy, "action_space", None)
        if action_space is None:
            action_space = getattr(env, "action_space", None)
        if action_space is None:
            action_init[policy_id] = 0
        else:
            action_init[policy_id] = _coerce_action(action_space.sample(), action_space)

    def _get_render_env(e):
        # PredPreyGrass-style evaluation uses a direct renderer object.
        # Here we unwrap RLlib adapters so rendering reaches the project env.
        if hasattr(e, "_env"):
            return e._env
        return e

    render_env = _get_render_env(env)

    # If rendering, create an array to store observations
    if video_dir:
        shape = render_env.base_map.shape
        horizon = agent.config.get("horizon") or agent.config.get("env_config", {}).get(
            "max_episode_steps", 1000
        )
        total_num_steps = max(int(num_steps or 0), int(num_episodes or 0) * int(horizon))
        if total_num_steps <= 0:
            total_num_steps = int(horizon)
        all_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8) for _ in range(total_num_steps)]

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out
        agent_states = DefaultMapping(lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        reward_total = 0.0
        episode_steps = 0
        while not done and keep_going(steps, num_steps, episodes, num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(agent_id, policy_agent_mapping(agent_id))
                    if policy_id not in use_lstm:
                        # Prefer per-agent trained policies when available.
                        if agent_id in use_lstm:
                            policy_id = agent_id
                        else:
                            policy_id = next(iter(use_lstm.keys()))
                        mapping_cache[agent_id] = policy_id
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        if use_module_inference:
                            a_action, p_state = _compute_action_from_module(
                                policy_id, a_obs, state_in=agent_states[agent_id]
                            )
                        else:
                            try:
                                action_out = agent.compute_action(
                                    a_obs,
                                    state=agent_states[agent_id],
                                    prev_action=prev_actions[agent_id],
                                    prev_reward=prev_rewards[agent_id],
                                    policy_id=policy_id,
                                    explore=not deterministic_actions,
                                )
                            except TypeError:
                                action_out = agent.compute_action(
                                    a_obs,
                                    state=agent_states[agent_id],
                                    prev_action=prev_actions[agent_id],
                                    prev_reward=prev_rewards[agent_id],
                                    policy_id=policy_id,
                                )
                            if isinstance(action_out, tuple):
                                a_action = action_out[0]
                                p_state = action_out[1] if len(action_out) > 1 else agent_states[agent_id]
                            else:
                                a_action = action_out
                                p_state = agent_states[agent_id]
                        agent_states[agent_id] = p_state
                    else:
                        if use_module_inference:
                            a_action, _ = _compute_action_from_module(policy_id, a_obs)
                        else:
                            try:
                                action_out = agent.compute_action(
                                    a_obs,
                                    prev_action=prev_actions[agent_id],
                                    prev_reward=prev_rewards[agent_id],
                                    policy_id=policy_id,
                                    explore=not deterministic_actions,
                                )
                            except TypeError:
                                action_out = agent.compute_action(
                                    a_obs,
                                    prev_action=prev_actions[agent_id],
                                    prev_reward=prev_rewards[agent_id],
                                    policy_id=policy_id,
                                )
                            a_action = action_out[0] if isinstance(action_out, tuple) else action_out
                    a_action = _coerce_action(a_action, policy_action_spaces.get(policy_id))
                    if random_action_prob > 0.0:
                        action_space = policy_action_spaces.get(policy_id) or getattr(
                            env, "action_space", None
                        )
                        if action_space is not None and np.random.random() < random_action_prob:
                            a_action = _coerce_action(action_space.sample(), action_space)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminations, truncations, info = step_out
                if multiagent:
                    done = bool(
                        terminations.get("__all__", False) or truncations.get("__all__", False)
                    )
                else:
                    done = bool(terminations) or bool(truncations)
            else:
                next_obs, reward, dones, info = step_out
                if multiagent and isinstance(dones, dict):
                    done = bool(dones.get("__all__", False))
                else:
                    done = bool(dones)
            if multiagent:
                if isinstance(reward, dict):
                    for agent_id, r in reward.items():
                        prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                reward_total += sum(reward.values()) if isinstance(reward, dict) else float(reward)
            else:
                reward_total += reward
            if not no_render:
                if video_dir:
                    rgb_arr = render_env.full_map_to_colors()
                    all_obs[steps] = rgb_arr.astype(np.uint8)
                else:
                    render_env.render()
                    if render_delay_s > 0.0:
                        time.sleep(render_delay_s)
            saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            episode_steps += 1
            if step_log_interval > 0 and (episode_steps % step_log_interval == 0):
                print(
                    f"[episode {episodes}] step={episode_steps} "
                    f"global_steps={steps} reward_so_far={reward_total:.3f}"
                )
            obs = next_obs
        saver.end_rollout()
        print("Episode #{}: reward: {}".format(episodes, reward_total))
        if done:
            episodes += 1

    # Render video from observations
    if video_dir:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        images_path = video_dir + "/images/"
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        height, width, _ = all_obs[0].shape
        # Upscale to be more legible
        width *= 20
        height *= 20
        utility_funcs.make_video_from_rgb_imgs(
            all_obs, video_dir, video_name=video_name, resize=(width, height)
        )

        # Clean up images
        shutil.rmtree(images_path)


if __name__ == "__main__":
    parser = create_parser()
    if len(sys.argv) == 1:
        if not os.path.exists(DEFAULT_DEBUG_CHECKPOINT):
            parser.error(
                "No CLI args given and default debug checkpoint was not found at: "
                + DEFAULT_DEBUG_CHECKPOINT
            )
        args = parser.parse_args([DEFAULT_DEBUG_CHECKPOINT, *DEFAULT_DEBUG_ARGS])
        print(
            "No CLI args provided; using default debug checkpoint run "
            f"(checkpoint={DEFAULT_DEBUG_CHECKPOINT}, video_dir={args.video_dir})."
        )
    else:
        args = parser.parse_args()

    # Old option: monitor, use video-dir instead.
    if args.monitor:
        deprecation_warning("--monitor", "--video-dir=[some dir]")
    # User tries to record videos, but no-render is set: Error.
    if (args.monitor or args.video_dir) and args.no_render:
        raise ValueError(
            "You have --no-render set, but are trying to record rollout videos"
            " (via options --video-dir/--monitor)! "
            "Either unset --no-render or do not use --video-dir/--monitor."
        )
    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via " "--out as well!"
        )
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via " "--out as well!"
        )

    run(args, parser)
