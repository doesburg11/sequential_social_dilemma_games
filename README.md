[![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.54.0-blue)](https://docs.ray.io/en/latest/rllib/)

# Sequential Social Dilemma Games
This repository is a Ray RLlib[[1]](#references) new API stack[[2]](#references) update (`Ray RLlib 2.54.0`) of the original `sequential_social_dilemma_games` codebase by Eugene Vinitsky and collaborators [[3]](#references) (`Ray RLlib 0.8.5`).

It provides an open-source implementation of DeepMind's Sequential Social Dilemma (SSD) multi-agent environments [[4]](#reference) [[5]](#reference) [[6]](#reference). SSDs are spatially and temporally extended Prisoner's Dilemma-like games where individually optimal short-term behavior can harm long-term group outcomes.

## What this repository does
- Implements the **Cleanup**, **Harvest**, and **Gathering** SSD environments.
- Exposes environment interfaces for **Gymnasium**, **PettingZoo**, and **RLlib MultiAgentEnv** workflows.
- Includes RLlib training entry points via `run_scripts/train.py`.
- Provides tests and visualization utilities for inspecting multi-agent behavior.

## Implemented Games

* **Cleanup**: A public goods dilemma in which agents get a reward for consuming apples, but must use a cleaning beam to clean a river in order for apples to grow. While an agent is cleaning the river, other agents can exploit it by consuming the apples that appear.

<img src="images/cleanup.png" alt="Image of the cleanup game" width="170" height="246"/>

* **Harvest**: A tragedy-of-the-commons dilemma in which apples regrow at a rate that depends on the amount of nearby apples. If individual agents employ an exploitative strategy by greedily consuming too many apples, the collective reward of all agents is reduced.

<img src="images/harvest.png" alt="Image of the Harvest game" width="483" height="187"/>

* **Gathering**: A two-player competitive social dilemma where agents collect apples for reward and can use beams to temporarily tag opponents out of the game.

<img src="images/schelling.png" alt="Schelling diagrams for Harvest and Cleanup" width="953" height="352"/>

The above plot shows the empirical Schelling diagrams for both Cleanup (A) and Harvest (B) (from [[2]](https://arxiv.org/abs/1803.08884)). These diagrams show the payoff that an individual agent can expect if it follows a defecting/exploitative strategy (red) vs a cooperative strategy (blue), given the number of other agents that are cooperating.  We can see that an individual agent can almost always greedily benefit from detecting, but the more agents that defect, the worse the outcomes for all agents.  

# Setup instructions
To install the SSD environments:
### Anaconda/miniconda (repo-local env at `./.conda`)
```bash
git clone -b master https://github.com/eugenevinitsky/sequential_social_dilemma_games
cd sequential_social_dilemma_games
conda env create --prefix ./.conda --file environment.yml
conda activate "$(pwd)/.conda"
# or:
./run_scripts/create_local_conda_env.sh
```
###
```bash
git clone -b master https://github.com/eugenevinitsky/sequential_social_dilemma_games
cd sequential_social_dilemma_games
python3 -m venv venv # Create a Python virtual environment
. venv/bin/activate
pip3 install --upgrade pip setuptools wheel
python3 setup.py develop
pip3 install -r requirements.txt
```

To install RLlib requirements for learning:
```bash
pip3 install social-dilemmas[rllib]
```

If using RLlib:
```

```

After the setup, you can run experiments like so:
- To train with default parameters (baseline model cleanup with 2 agents):
```bash
python3 run_scripts/train.py
```

- To train IMPALA on Cleanup with 5 agents:
```bash
python3 run_scripts/train.py --algorithm IMPALA --model baseline --num_agents 5
```

`run_scripts/train.py` now uses RLlib's true New API Stack (`PPOConfig`/`IMPALAConfig` with RLModule/Learner and EnvRunner v2) and is pinned to `--model baseline` with `--algorithm` in `{PPO, IMPALA}`.

Many more options are available which can be found in [default_args.py](config/default_args.py). A collection of preconfigured training scripts can be found in [run_scripts](run_scripts). 

Note that the RLlib initialization time can be rather high (up to 5 minutes) the more agents you use, and the more complex your used model is.

## Leibo-style evaluation (via `visualizer_rllib.py`)

You can run rollout evaluation and produce a Leibo-comparison metrics report
directly from `visualization/visualizer_rllib.py`.

- Run evaluation and print summary metrics:
```bash
PYTHONPATH=. ./.conda/bin/python visualization/visualizer_rllib.py \
  <checkpoint_path> --run DQN --env gathering_env --episodes 100 --no-render \
  --leibo-eval
```

- Save full metrics report to JSON:
```bash
PYTHONPATH=. ./.conda/bin/python visualization/visualizer_rllib.py \
  <checkpoint_path> --run DQN --env gathering_env --episodes 100 --no-render \
  --leibo-eval --leibo-out output/leibo_eval.json
```

For 2-agent runs, the JSON includes estimated `R,S,T,P` payoffs and checks the
Leibo inequality conditions. For more than 2 agents, it reports aggregate SSD
behavior metrics (returns, fire rates, tagging pressure, and class-profile counts).

- Show a live 2-agent per-step joint outcome matrix (rolling window):
```bash
PYTHONPATH=. ./.conda/bin/python visualization/visualizer_rllib.py \
  <checkpoint_path> --run DQN --env gathering_env --episodes 10 --no-render \
  --show-outcome-matrix --outcome-matrix-interval 200 --outcome-matrix-window 200
```

# CUDA, cuDNN and tensorflow-gpu

If you run into any cuda errors, make sure you've got a [compatible set](https://www.tensorflow.org/install/source#tested_build_configurations) of cuda/cudnn/tensorflow versions installed. However, beware of the following:
>The compatibility table given in the tensorflow site does not contain specific minor versions for cuda and cuDNN. However, if the specific versions are not met, there will be an error when you try to use tensorflow. [source](https://stackoverflow.com/a/53727997)

# Tests
Tests are located in the test folder and can be run individually or run by running `python -m pytest`. Many of the less obviously defined rules for the games can be understood by reading the tests, each of which outline some aspect of the game. 

# Constructing new environments
Every environment that subclasses MapEnv probably needs to implement the following methods:

```python
class NewMapEnv(MapEnv):
    ...
    
    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn apples"""
        pass

    def custom_action(self, agent, action):
        """Execute any custom, non-move actions that may be defined, like fire or clean"""
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError
```

## PPO Results

The below graphs display results for cleanup/harvest using un-tuned PPO in RLlib. As of yet, A3C remains untested.

**Collective cleanup reward**:

<img src="images/cleanup_collective_reward.svg" alt="Collective reward plot of cleanup" width="460.8" height="345.6"/>

**Collective harvest reward**:

<img src="images/harvest_collective_reward.svg" alt="Collective reward plot of harvest" width="460.8" height="345.6"/>


## References

1. Ray Team. [RLlib (Ray Reinforcement Learning Library) documentation](https://docs.ray.io/en/latest/rllib/).

2. Ray Team. [What's the New API Stack?](https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html#what-s-the-new-api-stack).

3. Vinitsky, E., Jaques, N., Leibo, J., Castañeda, A., Hughes, E., et al. [Sequential Social Dilemma Games (original repository)](https://github.com/eugenevinitsky/sequential_social_dilemma_games).

4. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).

5.  Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzman, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K., Koster, R., Tina Zhu, Roff, H., Graepel, T. (2018). [Inequity aversion improves cooperation in intertemporal social dilemmas](https://arxiv.org/abs/1803.08884). In Advances in Neural Information Processing Systems (pp. 3330-3340).

6. Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z. & de Freitas, N. (2018). [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647). arXiv preprint arXiv:1810.08647.

      
# Citation

If you want to cite this repository, please use the following citation:

```
@misc{SSDOpenSource,
author = {[Van Doesburg, Peter]},
title = {An Open Source Updated Implementation of Sequential Social Dilemma Games},
year = {2026},
publisher = {GitHub},
note = {GitHub repository},
```
