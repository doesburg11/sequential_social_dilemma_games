#!/usr/bin/env python
import argparse
import collections
import copy
import importlib
import json
import os
import pickle
import re
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

DEFAULT_DEBUG_CHECKPOINT = "/home/doesburg/Projects/SequentialSocialDilemmaGames/ray_results/gathering_baseline_DQN/DQN_gathering_env_254ce_00000_0_2026-03-01_15-58-45/checkpoint_000005"
DEFAULT_CHECKPOINT_SEARCH_DIRS = [
    str(Path(__file__).resolve().parents[1] / "ray_results"),
    str(Path.home() / "ray_results"),
]
DEFAULT_CHECKPOINT_DISPLAY_PREFIX = (
    str(Path(__file__).resolve().parents[1] / "ray_results") + os.sep
)
DEFAULT_DEBUG_ARGS = [
    "--run","PPO",
    "--env","harvest_env",
    "--episodes","1",
    "--render-delay-ms","10",
    "--random-action-prob","0.02",
    "--leibo-eval",
    "--leibo-out", str(Path(__file__).resolve().parents[1] / "output" / "leibo_eval.json"),
    # "--video-dir", str(Path(__file__).resolve().parents[1] / "output"),
    "--video-filename", "rollout.mp4",
]


def _find_all_checkpoints(search_roots=None):
    roots = search_roots or DEFAULT_CHECKPOINT_SEARCH_DIRS
    found = []
    seen = set()

    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            continue
        for checkpoint_path in root_path.rglob("checkpoint_*"):
            if not checkpoint_path.is_dir():
                continue
            checkpoint_name = checkpoint_path.name
            if not checkpoint_name.startswith("checkpoint_"):
                continue
            checkpoint_str = str(checkpoint_path)
            if checkpoint_str in seen:
                continue
            try:
                mtime = checkpoint_path.stat().st_mtime
            except OSError:
                continue
            found.append((mtime, checkpoint_str))
            seen.add(checkpoint_str)

    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in found]


def _hardcode_default_debug_checkpoint(selected_checkpoint):
    script_path = Path(__file__).resolve()
    script_text = script_path.read_text(encoding="utf-8")
    escaped_path = selected_checkpoint.replace("\\", "\\\\").replace('"', '\\"')
    replacement = f'DEFAULT_DEBUG_CHECKPOINT = "{escaped_path}"'
    updated_text, num_subs = re.subn(
        r'^DEFAULT_DEBUG_CHECKPOINT\s*=\s*".*"$',
        replacement,
        script_text,
        count=1,
        flags=re.MULTILINE,
    )
    if num_subs != 1:
        raise RuntimeError(
            "Could not update DEFAULT_DEBUG_CHECKPOINT in visualizer_rllib.py"
        )
    if updated_text != script_text:
        script_path.write_text(updated_text, encoding="utf-8")


def _select_checkpoint_gui(search_roots=None):
    checkpoints = _find_all_checkpoints(search_roots=search_roots)
    if not checkpoints:
        return None, False

    try:
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import messagebox
    except Exception:
        # Fallback to the newest checkpoint when Tk is unavailable.
        return checkpoints[0], False

    result = {"checkpoint": None, "hardcode": True}
    filtered_paths = list(checkpoints)
    display_index_to_path = {}

    root = tk.Tk()
    root.title("Select RLlib Checkpoint")
    root.geometry("1700x760")
    base_font = tkfont.nametofont("TkDefaultFont")
    base_size = max(1, int(base_font.cget("size")))
    big_size = max(12, base_size * 2)
    big_font_bold = (base_font.cget("family"), big_size, "bold")
    big_bold_font = (base_font.cget("family"), big_size, "bold")
    list_font = big_font_bold

    title = tk.Label(root, text="Checkpoint Picker (latest first)", font=big_bold_font)
    title.pack(padx=10, pady=(10, 4), anchor="w")

    info = tk.Label(
        root,
        text="Set a display prefix to shorten option text, then filter/click a checkpoint.",
        justify="left",
        font=big_font_bold,
    )
    info.pack(padx=10, pady=(0, 8), anchor="w")

    prefix_label = tk.Label(root, text="Display Prefix (stripped from options):", font=big_font_bold)
    prefix_label.pack(padx=10, pady=(0, 4), anchor="w")
    prefix_var = tk.StringVar(value=DEFAULT_CHECKPOINT_DISPLAY_PREFIX)
    prefix_entry = tk.Entry(root, textvariable=prefix_var, font=big_font_bold)
    prefix_entry.pack(fill="x", padx=10, pady=(0, 8))

    filter_label = tk.Label(root, text="Filter:", font=big_font_bold)
    filter_label.pack(padx=10, pady=(0, 4), anchor="w")
    filter_var = tk.StringVar()
    filter_entry = tk.Entry(root, textvariable=filter_var, font=big_font_bold)
    filter_entry.pack(fill="x", padx=10, pady=(0, 8))
    filter_entry.focus_set()

    list_frame = tk.Frame(root)
    list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    y_scroll = tk.Scrollbar(list_frame, orient="vertical")
    y_scroll.pack(side="right", fill="y")
    x_scroll = tk.Scrollbar(list_frame, orient="horizontal")
    x_scroll.pack(side="bottom", fill="x")

    checkpoint_list = tk.Listbox(
        list_frame,
        yscrollcommand=y_scroll.set,
        xscrollcommand=x_scroll.set,
        selectmode=tk.SINGLE,
        font=list_font,
        bg="#f6f8fc",
        fg="#0f172a",
        selectbackground="#1d4ed8",
        selectforeground="#ffffff",
        highlightthickness=1,
        highlightbackground="#94a3b8",
        activestyle="none",
        exportselection=False,
        width=180,
        height=28,
    )
    checkpoint_list.pack(side="left", fill="both", expand=True)
    y_scroll.config(command=checkpoint_list.yview)
    x_scroll.config(command=checkpoint_list.xview)

    hardcode_var = tk.BooleanVar(value=True)
    hardcode_check = tk.Checkbutton(
        root,
        text="Hardcode selected checkpoint into DEFAULT_DEBUG_CHECKPOINT in this script",
        variable=hardcode_var,
        font=big_font_bold,
    )
    hardcode_check.pack(padx=10, pady=(0, 8), anchor="w")

    selected_label_var = tk.StringVar(value="Selected: none")
    selected_label = tk.Label(
        root,
        textvariable=selected_label_var,
        justify="left",
        font=big_font_bold,
    )
    selected_label.pack(padx=10, pady=(0, 8), anchor="w")

    def _get_selected_path():
        sel = checkpoint_list.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx in display_index_to_path:
            return display_index_to_path[idx]

        # If user selected spacer row, snap to nearest real option row.
        for probe in (idx - 1, idx + 1):
            if probe in display_index_to_path:
                checkpoint_list.selection_clear(0, tk.END)
                checkpoint_list.selection_set(probe)
                checkpoint_list.activate(probe)
                checkpoint_list.see(probe)
                return display_index_to_path[probe]
        return None

    def refresh_list(*_):
        nonlocal filtered_paths, display_index_to_path
        query = filter_var.get().strip().lower()
        display_prefix = os.path.expanduser(prefix_var.get().strip())
        if display_prefix and not display_prefix.endswith(os.sep):
            display_prefix = display_prefix + os.sep
        if query:
            filtered_paths = [p for p in checkpoints if query in p.lower()]
        else:
            filtered_paths = list(checkpoints)
        checkpoint_list.delete(0, tk.END)
        display_index_to_path = {}
        for path in filtered_paths:
            row_idx = checkpoint_list.size()
            display_path = path
            if display_prefix and path.startswith(display_prefix):
                display_path = path[len(display_prefix) :]
            checkpoint_list.insert(tk.END, display_path)
            display_index_to_path[row_idx] = path
            # Spacer row to make each option more distinguishable.
            checkpoint_list.insert(tk.END, "")
        if filtered_paths:
            checkpoint_list.selection_clear(0, tk.END)
            checkpoint_list.selection_set(0)
            checkpoint_list.activate(0)
            checkpoint_list.see(0)
            selected_label_var.set(f"Selected: {filtered_paths[0]}")
        else:
            selected_label_var.set("Selected: none")

    def update_selected_label(_event=None):
        selected_path = _get_selected_path()
        if not selected_path:
            selected_label_var.set("Selected: none")
            return
        selected_label_var.set(f"Selected: {selected_path}")

    def submit():
        selected_path = _get_selected_path()
        if not selected_path:
            messagebox.showwarning("Checkpoint Picker", "Please select a checkpoint.")
            return
        result["checkpoint"] = selected_path
        result["hardcode"] = bool(hardcode_var.get())
        root.destroy()

    def cancel():
        root.destroy()

    button_row = tk.Frame(root)
    button_row.pack(fill="x", padx=10, pady=(0, 10))

    run_button = tk.Button(
        button_row, text="Run With Selected", command=submit, font=big_font_bold
    )
    run_button.pack(side="left")
    cancel_button = tk.Button(button_row, text="Cancel", command=cancel, font=big_font_bold)
    cancel_button.pack(side="left", padx=(8, 0))

    prefix_var.trace_add("write", refresh_list)
    filter_var.trace_add("write", refresh_list)
    checkpoint_list.bind("<<ListboxSelect>>", update_selected_label)
    checkpoint_list.bind("<Double-Button-1>", lambda _event: submit())
    root.bind("<Return>", lambda _event: submit())
    root.bind("<Escape>", lambda _event: cancel())

    refresh_list()
    root.mainloop()

    return result["checkpoint"], result["hardcode"]

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
    parser.add_argument(
        "--leibo-eval",
        default=False,
        action="store_true",
        help="Collect Leibo-style SSD comparison metrics during rollout.",
    )
    parser.add_argument(
        "--leibo-out",
        type=str,
        default=None,
        help="Optional JSON file path for Leibo-eval metrics output.",
    )
    parser.add_argument(
        "--leibo-fire-threshold",
        type=float,
        default=0.02,
        help="FIRE-rate threshold for classifying an agent as cooperative (C) vs defective (D).",
    )
    parser.add_argument(
        "--show-outcome-matrix",
        default=False,
        action="store_true",
        help="For 2-agent runs, print a live C/D joint-outcome matrix during rollout.",
    )
    parser.add_argument(
        "--outcome-matrix-interval",
        type=int,
        default=200,
        help="Print interval (steps) for the live 2-agent outcome matrix.",
    )
    parser.add_argument(
        "--outcome-matrix-window",
        type=int,
        default=200,
        help="Rolling window size (steps) used for the live 2-agent outcome matrix.",
    )
    return parser


def run(args, parser):
    def _handle_leibo_report(report):
        if not report:
            return
        summary = report.get("summary", {})
        print("leibo_eval/episodes:", summary.get("episodes"))
        print("leibo_eval/mean_social_welfare:", summary.get("mean_social_welfare"))
        print("leibo_eval/mean_episode_len:", summary.get("mean_episode_len"))
        print("leibo_eval/class_profile_counts:", summary.get("class_profile_counts"))
        payoff = report.get("two_agent_payoff_estimates", {})
        if payoff.get("enabled"):
            payoffs = payoff.get("payoffs", {})
            print("leibo_eval/payoffs:", payoffs)
            print("leibo_eval/inequality_checks:", payoff.get("inequality_checks"))
            matrix = report.get("two_agent_step_matrix_aggregate")
            if matrix:
                print("leibo_eval/two_agent_step_matrix_aggregate:", matrix)
        elif payoff:
            print("leibo_eval/payoff_estimation:", payoff.get("reason"))
        if args.leibo_out:
            out_path = Path(args.leibo_out).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, sort_keys=True)
            print(f"Leibo eval report saved to: {out_path}")

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
            leibo_report = rollout(
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
                leibo_eval=bool(args.leibo_eval or args.leibo_out or args.show_outcome_matrix),
                leibo_fire_threshold=float(args.leibo_fire_threshold),
                show_outcome_matrix=bool(args.show_outcome_matrix),
                outcome_matrix_interval=max(1, int(args.outcome_matrix_interval)),
                outcome_matrix_window=max(1, int(args.outcome_matrix_window)),
            )
            _handle_leibo_report(leibo_report)
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
        if (
            is_new_api_stack
            and args.no_render
            and not video_dir
            and args.out is None
            and not (args.leibo_eval or args.leibo_out or args.show_outcome_matrix)
        ):
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
            leibo_report = rollout(
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
                leibo_eval=bool(args.leibo_eval or args.leibo_out or args.show_outcome_matrix),
                leibo_fire_threshold=float(args.leibo_fire_threshold),
                show_outcome_matrix=bool(args.show_outcome_matrix),
                outcome_matrix_interval=max(1, int(args.outcome_matrix_interval)),
                outcome_matrix_window=max(1, int(args.outcome_matrix_window)),
            )
            _handle_leibo_report(leibo_report)
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


def _safe_mean(values):
    if not values:
        return None
    return float(np.mean(values))


class LeiboMetricsCollector:
    """Collect Leibo-style SSD comparison metrics from rollout episodes.

    Notes:
    - For 2-agent runs, this estimates R/S/T/P by classifying each agent as
      cooperative or defective using FIRE-action rate.
    - For >2 agents, only aggregated behavior metrics are reported.
    """

    _JOINT_CELLS = ("CC", "CD", "DC", "DD")

    def __init__(
        self,
        fire_rate_threshold=0.02,
        show_outcome_matrix=False,
        outcome_matrix_interval=200,
        outcome_matrix_window=200,
    ):
        self.fire_rate_threshold = float(fire_rate_threshold)
        self.show_outcome_matrix = bool(show_outcome_matrix)
        self.outcome_matrix_interval = max(1, int(outcome_matrix_interval))
        self.outcome_matrix_window = max(1, int(outcome_matrix_window))
        self.episodes = []
        self._agent_ids = []
        self._episode_index = 0
        self._step_count = 0
        self._returns = {}
        self._fire_counts = {}
        self._action_counts = {}
        self._tagged_steps = {}
        self._tag_events = {}
        self._prev_tagged = {}
        self._ordered_pair = ()
        self._pending_joint_cell = None
        self._window_joint_rows = collections.deque(maxlen=self.outcome_matrix_window)
        self._joint_stats_cumulative = self._new_joint_stats()

    @classmethod
    def _new_joint_stats(cls):
        return {
            cell: {"n": 0, "r0_sum": 0.0, "r1_sum": 0.0} for cell in cls._JOINT_CELLS
        }

    @staticmethod
    def _joint_mean_or_none(sum_value, count):
        if count <= 0:
            return None
        return float(sum_value) / float(count)

    @classmethod
    def _joint_stats_to_report(cls, stats):
        out = {}
        for cell in cls._JOINT_CELLS:
            count = int(stats[cell]["n"])
            out[cell] = {
                "n": count,
                "mean_r0": cls._joint_mean_or_none(stats[cell]["r0_sum"], count),
                "mean_r1": cls._joint_mean_or_none(stats[cell]["r1_sum"], count),
            }
        return out

    def _format_joint_cell(self, stats, cell):
        count = int(stats[cell]["n"])
        if count <= 0:
            return "n=0"
        mean_r0 = self._joint_mean_or_none(stats[cell]["r0_sum"], count)
        mean_r1 = self._joint_mean_or_none(stats[cell]["r1_sum"], count)
        return f"n={count:<4} r0={mean_r0:+.3f} r1={mean_r1:+.3f}"

    def _format_outcome_matrix(self, stats, window_len):
        a0, a1 = self._ordered_pair
        header = (
            f"[episode {self._episode_index}] step={self._step_count} "
            f"2-agent outcome matrix (window={window_len})"
        )
        labels = f"({a0}=rows, {a1}=cols; C=non-FIRE, D=FIRE)"
        row_header = f"{'':10} {'P1:C':<30} {'P1:D':<30}"
        row_c = (
            f"{'P0:C':<10} {self._format_joint_cell(stats, 'CC'):<30} "
            f"{self._format_joint_cell(stats, 'CD'):<30}"
        )
        row_d = (
            f"{'P0:D':<10} {self._format_joint_cell(stats, 'DC'):<30} "
            f"{self._format_joint_cell(stats, 'DD'):<30}"
        )
        return "\n".join([header, labels, row_header, row_c, row_d])

    @staticmethod
    def _is_fire_action(render_env, agent_id, action):
        try:
            agent = render_env.agents[agent_id]
            return agent.action_map(int(action)) == "FIRE"
        except Exception:
            return False

    @staticmethod
    def _get_tagged_state(render_env, agent_ids):
        tagged = {}
        agents = getattr(render_env, "agents", {})
        for agent_id in agent_ids:
            agent = agents.get(agent_id)
            tagged[agent_id] = bool(getattr(agent, "is_tagged_out", False)) if agent is not None else False
        return tagged

    def begin_episode(self, agent_ids, episode_index=0):
        self._agent_ids = sorted(agent_ids)
        self._episode_index = int(episode_index)
        self._step_count = 0
        self._returns = {agent_id: 0.0 for agent_id in self._agent_ids}
        self._fire_counts = {agent_id: 0 for agent_id in self._agent_ids}
        self._action_counts = {agent_id: 0 for agent_id in self._agent_ids}
        self._tagged_steps = {agent_id: 0 for agent_id in self._agent_ids}
        self._tag_events = {agent_id: 0 for agent_id in self._agent_ids}
        self._prev_tagged = {agent_id: False for agent_id in self._agent_ids}
        self._ordered_pair = tuple(self._agent_ids[:2]) if len(self._agent_ids) == 2 else ()
        self._pending_joint_cell = None
        self._window_joint_rows.clear()
        self._joint_stats_cumulative = self._new_joint_stats()

    def on_actions(self, action_dict, render_env):
        for agent_id, action in action_dict.items():
            if agent_id not in self._action_counts:
                continue
            self._action_counts[agent_id] += 1
            if self._is_fire_action(render_env, agent_id, action):
                self._fire_counts[agent_id] += 1

        if not self._ordered_pair:
            self._pending_joint_cell = None
            return

        agent0_id, agent1_id = self._ordered_pair
        if agent0_id not in action_dict or agent1_id not in action_dict:
            self._pending_joint_cell = None
            return
        a0_cls = "D" if self._is_fire_action(render_env, agent0_id, action_dict[agent0_id]) else "C"
        a1_cls = "D" if self._is_fire_action(render_env, agent1_id, action_dict[agent1_id]) else "C"
        self._pending_joint_cell = a0_cls + a1_cls

    def on_step(self, reward_dict, render_env):
        self._step_count += 1
        if isinstance(reward_dict, dict):
            for agent_id, reward in reward_dict.items():
                if agent_id in self._returns:
                    self._returns[agent_id] += float(reward)
        tagged_now = self._get_tagged_state(render_env, self._agent_ids)
        for agent_id in self._agent_ids:
            if tagged_now.get(agent_id, False):
                self._tagged_steps[agent_id] += 1
            if tagged_now.get(agent_id, False) and not self._prev_tagged.get(agent_id, False):
                self._tag_events[agent_id] += 1
        self._prev_tagged = tagged_now

        if (
            self._ordered_pair
            and self._pending_joint_cell in self._JOINT_CELLS
            and isinstance(reward_dict, dict)
        ):
            agent0_id, agent1_id = self._ordered_pair
            r0 = float(reward_dict.get(agent0_id, 0.0))
            r1 = float(reward_dict.get(agent1_id, 0.0))
            cell = self._pending_joint_cell
            self._joint_stats_cumulative[cell]["n"] += 1
            self._joint_stats_cumulative[cell]["r0_sum"] += r0
            self._joint_stats_cumulative[cell]["r1_sum"] += r1
            self._window_joint_rows.append((cell, r0, r1))

            if self.show_outcome_matrix and (self._step_count % self.outcome_matrix_interval == 0):
                window_stats = self._new_joint_stats()
                for w_cell, w_r0, w_r1 in self._window_joint_rows:
                    window_stats[w_cell]["n"] += 1
                    window_stats[w_cell]["r0_sum"] += float(w_r0)
                    window_stats[w_cell]["r1_sum"] += float(w_r1)
                print(self._format_outcome_matrix(window_stats, len(self._window_joint_rows)))

    def end_episode(self, episode_index):
        fire_rates = {}
        tagged_fraction = {}
        agent_class = {}
        for agent_id in self._agent_ids:
            action_n = max(1, int(self._action_counts.get(agent_id, 0)))
            fire_rate = float(self._fire_counts.get(agent_id, 0)) / float(action_n)
            fire_rates[agent_id] = fire_rate
            tagged_fraction[agent_id] = (
                float(self._tagged_steps.get(agent_id, 0)) / float(max(1, self._step_count))
            )
            agent_class[agent_id] = (
                "C" if fire_rate <= self.fire_rate_threshold else "D"
            )

        unique_classes = set(agent_class.values())
        if unique_classes == {"C"}:
            profile = "CC"
        elif unique_classes == {"D"}:
            profile = "DD"
        else:
            profile = "mixed"

        episode_data = {
            "episode_index": int(episode_index),
            "episode_len": int(self._step_count),
            "agent_returns": {k: float(v) for k, v in self._returns.items()},
            "social_welfare": float(sum(self._returns.values())),
            "fire_counts": {k: int(v) for k, v in self._fire_counts.items()},
            "fire_rates": fire_rates,
            "tag_events": {k: int(v) for k, v in self._tag_events.items()},
            "tagged_fraction": tagged_fraction,
            "agent_classification": agent_class,
            "class_profile": profile,
        }
        if self._ordered_pair:
            episode_data["two_agent_step_matrix_cumulative"] = self._joint_stats_to_report(
                self._joint_stats_cumulative
            )
        self.episodes.append(episode_data)
        return episode_data

    def _build_two_agent_payoff_estimates(self):
        if not self.episodes:
            return {}
        first = self.episodes[0]
        agent_ids = sorted(first["agent_returns"].keys())
        if len(agent_ids) != 2:
            return {
                "enabled": False,
                "reason": "R/S/T/P estimation is only defined in this report for 2-agent runs.",
            }

        a0, a1 = agent_ids
        r_samples = []
        p_samples = []
        s_samples = []
        t_samples = []
        for episode in self.episodes:
            classes = episode["agent_classification"]
            returns = episode["agent_returns"]
            c0, c1 = classes[a0], classes[a1]
            r0, r1 = float(returns[a0]), float(returns[a1])
            if c0 == "C" and c1 == "C":
                r_samples.extend([r0, r1])
            elif c0 == "D" and c1 == "D":
                p_samples.extend([r0, r1])
            elif c0 == "C" and c1 == "D":
                s_samples.append(r0)
                t_samples.append(r1)
            elif c0 == "D" and c1 == "C":
                t_samples.append(r0)
                s_samples.append(r1)

        r_val = _safe_mean(r_samples)
        p_val = _safe_mean(p_samples)
        s_val = _safe_mean(s_samples)
        t_val = _safe_mean(t_samples)

        inequality_checks = None
        if all(v is not None for v in (r_val, p_val, s_val, t_val)):
            inequality_checks = {
                "R_gt_P": bool(r_val > p_val),
                "R_gt_S": bool(r_val > s_val),
                "twoR_gt_T_plus_S": bool((2.0 * r_val) > (t_val + s_val)),
                "greed_T_gt_R": bool(t_val > r_val),
                "fear_P_gt_S": bool(p_val > s_val),
                "condition_4_greed_or_fear": bool((t_val > r_val) or (p_val > s_val)),
            }

        return {
            "enabled": True,
            "agent_ids": agent_ids,
            "samples": {
                "R_n": len(r_samples),
                "P_n": len(p_samples),
                "S_n": len(s_samples),
                "T_n": len(t_samples),
            },
            "payoffs": {"R": r_val, "P": p_val, "S": s_val, "T": t_val},
            "inequality_checks": inequality_checks,
        }

    def build_report(self):
        if not self.episodes:
            return {}
        episode_lens = [ep["episode_len"] for ep in self.episodes]
        social_welfare = [ep["social_welfare"] for ep in self.episodes]
        reward_std = [float(np.std(list(ep["agent_returns"].values()))) for ep in self.episodes]
        class_profiles = collections.Counter(ep["class_profile"] for ep in self.episodes)

        mean_fire_rate = {}
        mean_tagged_fraction = {}
        all_agent_ids = sorted(self.episodes[0]["agent_returns"].keys())
        for agent_id in all_agent_ids:
            mean_fire_rate[agent_id] = _safe_mean(
                [ep["fire_rates"][agent_id] for ep in self.episodes]
            )
            mean_tagged_fraction[agent_id] = _safe_mean(
                [ep["tagged_fraction"][agent_id] for ep in self.episodes]
            )

        report = {
            "summary": {
                "episodes": len(self.episodes),
                "mean_episode_len": _safe_mean(episode_lens),
                "mean_social_welfare": _safe_mean(social_welfare),
                "mean_return_std_across_agents": _safe_mean(reward_std),
                "class_profile_counts": dict(class_profiles),
                "mean_fire_rate": mean_fire_rate,
                "mean_tagged_fraction": mean_tagged_fraction,
                "fire_rate_threshold_for_C": self.fire_rate_threshold,
            },
            "two_agent_payoff_estimates": self._build_two_agent_payoff_estimates(),
            "episodes": self.episodes,
        }
        if self.episodes and len(self.episodes[0].get("agent_returns", {})) == 2:
            agg = self._new_joint_stats()
            for ep in self.episodes:
                matrix = ep.get("two_agent_step_matrix_cumulative", {})
                for cell in self._JOINT_CELLS:
                    cell_row = matrix.get(cell, {})
                    count = int(cell_row.get("n", 0))
                    mean_r0 = cell_row.get("mean_r0")
                    mean_r1 = cell_row.get("mean_r1")
                    agg[cell]["n"] += count
                    if mean_r0 is not None:
                        agg[cell]["r0_sum"] += float(mean_r0) * count
                    if mean_r1 is not None:
                        agg[cell]["r1_sum"] += float(mean_r1) * count
            report["two_agent_step_matrix_aggregate"] = self._joint_stats_to_report(agg)
        return report


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
    leibo_eval=False,
    leibo_fire_threshold=0.02,
    show_outcome_matrix=False,
    outcome_matrix_interval=200,
    outcome_matrix_window=200,
):
    policy_agent_mapping = default_policy_agent_mapping
    use_module_inference = False
    policy_action_spaces = {}

    if saver is None:
        saver = RolloutSaver()
    leibo_collector = LeiboMetricsCollector(
        fire_rate_threshold=leibo_fire_threshold,
        show_outcome_matrix=show_outcome_matrix,
        outcome_matrix_interval=outcome_matrix_interval,
        outcome_matrix_window=outcome_matrix_window,
    ) if leibo_eval else None

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
        if leibo_collector:
            if multiagent and isinstance(obs, dict):
                leibo_collector.begin_episode(list(obs.keys()), episode_index=episodes)
            else:
                leibo_collector.begin_episode([_DUMMY_AGENT_ID], episode_index=episodes)
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
            if leibo_collector:
                action_view = action if isinstance(action, dict) else {_DUMMY_AGENT_ID: action}
                leibo_collector.on_actions(action_view, render_env)
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
            if leibo_collector:
                reward_view = reward if isinstance(reward, dict) else {_DUMMY_AGENT_ID: reward}
                leibo_collector.on_step(reward_view, render_env)
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
        if leibo_collector:
            leibo_collector.end_episode(episodes)
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

    return leibo_collector.build_report() if leibo_collector else None


if __name__ == "__main__":
    parser = create_parser()
    if len(sys.argv) == 1:
        selected_checkpoint, hardcode_selected = _select_checkpoint_gui(
            search_roots=DEFAULT_CHECKPOINT_SEARCH_DIRS
        )
        checkpoint_to_use = selected_checkpoint or DEFAULT_DEBUG_CHECKPOINT
        if not os.path.exists(checkpoint_to_use):
            parser.error(
                "No CLI args given and no valid checkpoint was selected/found. "
                f"Missing path: {checkpoint_to_use}"
            )
        if selected_checkpoint and hardcode_selected:
            _hardcode_default_debug_checkpoint(selected_checkpoint)
            print(f"Hardcoded DEFAULT_DEBUG_CHECKPOINT to: {selected_checkpoint}")
        args = parser.parse_args([checkpoint_to_use, *DEFAULT_DEBUG_ARGS])
        print(
            "No CLI args provided; using default debug checkpoint run "
            f"(checkpoint={checkpoint_to_use}, video_dir={args.video_dir})."
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
