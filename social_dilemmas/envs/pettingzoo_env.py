from functools import lru_cache

try:
    from gymnasium.utils import EzPickle
except ImportError:  # pragma: no cover - fallback for old environments
    from gym.utils import EzPickle

from pettingzoo.utils import wrappers

try:
    from pettingzoo.utils import ParallelEnv
except ImportError:  # pragma: no cover - backwards compatibility
    from pettingzoo.utils.env import ParallelEnv

try:
    from pettingzoo.utils.conversions import parallel_to_aec
except ImportError:  # pragma: no cover - backwards compatibility
    from pettingzoo.utils.conversions import from_parallel_wrapper

    parallel_to_aec = None

from social_dilemmas.envs.env_creator import get_env_creator

MAX_CYCLES = 1000


def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
    return _parallel_env(max_cycles, **ssd_args)


def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
    parallel = parallel_env(max_cycles, **ssd_args)
    if parallel_to_aec is not None:
        return parallel_to_aec(parallel)
    return from_parallel_wrapper(parallel)


def env(max_cycles=MAX_CYCLES, **ssd_args):
    aec_env = raw_env(max_cycles, **ssd_args)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


class ssd_parallel_env(ParallelEnv):
    metadata = {"name": "social_dilemmas_parallel_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(self, env, max_cycles):
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

    @property
    def num_agents(self):
        return len(self.possible_agents)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        observations = self.ssd_env.reset()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def seed(self, seed=None):
        return self.ssd_env.seed(seed)

    def render(self, mode="human"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]
        obss, rews, dones, infos = self.ssd_env.step(actions)
        dones = dict(dones)
        dones.pop("__all__", None)
        self.num_cycles += 1

        terminations = {agent: bool(dones.get(agent, False)) for agent in current_agents}
        truncations = {agent: False for agent in current_agents}
        if self.num_cycles >= self.max_cycles:
            truncations = {agent: True for agent in current_agents}

        observations = {agent: obss[agent] for agent in current_agents if agent in obss}
        rewards = {agent: rews.get(agent, 0.0) for agent in current_agents}
        infos = {agent: infos.get(agent, {}) for agent in current_agents}

        self.terminations = terminations
        self.truncations = truncations
        self.agents = [
            agent
            for agent in current_agents
            if not (terminations.get(agent, False) or truncations.get(agent, False))
        ]
        return observations, rewards, terminations, truncations, infos


class _parallel_env(ssd_parallel_env, EzPickle):
    def __init__(self, max_cycles, **ssd_args):
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
        super().__init__(env, max_cycles)
