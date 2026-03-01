import numpy as np
try:
    from gymnasium.spaces import Box, Dict
except ImportError:  # pragma: no cover - fallback for legacy gym installs
    from gym.spaces import Box, Dict

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import GATHERING_MAP

_GATHERING_ACTIONS = {"FIRE": 5}
# Leibo et al. (2017) use a 16x21 egocentric RGB observation in Gathering.
# We implement it as: 15 cells ahead + current row, and 10 cells to each side.
GATHERING_OBS_AHEAD = 15
GATHERING_OBS_SIDE = 10
GATHERING_OBS_SHAPE = (GATHERING_OBS_AHEAD + 1, 2 * GATHERING_OBS_SIDE + 1, 3)
GATHERING_VIEW_PADDING = max(GATHERING_OBS_AHEAD, GATHERING_OBS_SIDE)
SELF_AGENT_COLOR = np.array([0, 0, 255], dtype=np.uint8)
OTHER_AGENT_COLOR = np.array([255, 0, 0], dtype=np.uint8)


class GatheringAgent(HarvestAgent):
    def __init__(self, agent_id, start_pos, start_orientation, full_map, view_len, tagged_frames):
        super().__init__(agent_id, start_pos, start_orientation, full_map, view_len)
        self._tagged_frames = int(tagged_frames)
        self._beam_hit_count = 0
        self._tagged_timer = 0
        self.collected_apple_positions = []

    @property
    def is_tagged_out(self):
        return self._tagged_timer > 0

    def reset_tag_state(self):
        self._beam_hit_count = 0
        self._tagged_timer = 0
        self.collected_apple_positions = []

    def tick_tag_timer(self):
        if self._tagged_timer <= 0:
            return False
        self._tagged_timer -= 1
        return self._tagged_timer == 0

    def hit(self, char):
        if char != b"F" or self.is_tagged_out:
            return
        self._beam_hit_count += 1
        if self._beam_hit_count >= 2:
            self._beam_hit_count = 0
            self._tagged_timer = self._tagged_frames

    def fire_beam(self, char):
        # Gathering uses tagging as a strategic action. Keep beam reward-neutral.
        return

    def consume(self, char):
        if self.is_tagged_out:
            return char
        if char == b"A":
            self.reward_this_turn += 1
            self.collected_apple_positions.append((int(self.pos[0]), int(self.pos[1])))
            return b" "
        return char


class GatheringEnv(MapEnv):
    @staticmethod
    def _normalize_ascii_map_rows(ascii_map):
        rows = []
        for row in ascii_map:
            if isinstance(row, str):
                rows.append(row)
                continue
            if isinstance(row, (bytes, bytearray)):
                rows.append(row.decode("ascii"))
                continue
            rows.append(
                "".join(
                    cell.decode("ascii")
                    if isinstance(cell, (bytes, bytearray, np.bytes_))
                    else str(cell)
                    for cell in row
                )
            )
        return rows

    @classmethod
    def _ensure_spawn_capacity(cls, ascii_map, num_agents):
        rows = cls._normalize_ascii_map_rows(ascii_map)
        if not rows:
            raise ValueError("Gathering map is empty.")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("Gathering map rows must all have equal length.")

        grid = [list(row) for row in rows]
        spawn_points = []
        empty_cells = []

        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == "P":
                    spawn_points.append((r, c))
                elif cell == " ":
                    empty_cells.append((r, c))

        if len(spawn_points) >= num_agents:
            return rows

        while len(spawn_points) < num_agents:
            if not empty_cells:
                raise ValueError(
                    f"Gathering map does not have enough free cells to place {num_agents} agents."
                )
            if spawn_points:
                # Place additional spawn points far from existing ones for better dispersion.
                dist_scores = [
                    min(abs(r - sr) + abs(c - sc) for sr, sc in spawn_points)
                    for r, c in empty_cells
                ]
                selected_idx = int(np.argmax(dist_scores))
            else:
                selected_idx = 0
            r, c = empty_cells.pop(selected_idx)
            grid[r][c] = "P"
            spawn_points.append((r, c))

        return ["".join(row) for row in grid]

    def __init__(
        self,
        ascii_map=GATHERING_MAP,
        num_agents=2,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
        apple_respawn_delay=3,
        tagged_respawn_delay=25,
        num_initial_apples=6,
    ):
        num_agents = int(num_agents)
        if num_agents < 1:
            raise ValueError(f"GatheringEnv requires at least one agent; got num_agents={num_agents}.")
        # Agent IDs are encoded as single ASCII chars in this codebase ("1".."9").
        if num_agents > 9:
            raise ValueError(
                f"GatheringEnv supports up to 9 agents in this implementation; got num_agents={num_agents}."
            )
        ascii_map = self._ensure_spawn_capacity(ascii_map, num_agents)
        self.apple_respawn_delay = int(apple_respawn_delay)
        self.tagged_respawn_delay = int(tagged_respawn_delay)
        self.num_initial_apples = int(num_initial_apples)
        super().__init__(
            ascii_map,
            _GATHERING_ACTIONS,
            GATHERING_VIEW_PADDING,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )
        self.apple_points = []
        self.apple_timers = {}
        self.active_apple_points = set()

        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append((row, col))

    @property
    def action_space(self):
        return DiscreteWithDType(8, dtype=np.uint8)

    @property
    def observation_space(self):
        obs_space = {
            "curr_obs": Box(
                low=0,
                high=255,
                shape=GATHERING_OBS_SHAPE,
                dtype=np.uint8,
            )
        }
        if self.return_agent_actions:
            obs_space = {
                **obs_space,
                "other_agent_actions": Box(
                    low=0,
                    high=len(self.all_actions),
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "visible_agents": Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "prev_visible_agents": Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
            }
        obs_space = Dict(obs_space)
        obs_space.dtype = np.uint8
        return obs_space

    @property
    def agent_pos(self):
        return [agent.pos.tolist() for agent in self.agents.values() if not agent.is_tagged_out]

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()
        for i in range(self.num_agents):
            agent_id = f"agent-{i}"
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            agent = GatheringAgent(
                agent_id,
                spawn_point,
                rotation,
                map_with_agents,
                view_len=self.view_len,
                tagged_frames=self.tagged_respawn_delay,
            )
            self.agents[agent_id] = agent

    def spawn_point(self):
        """Respawn to free spawn points considering only active agents."""
        spawn_index = 0
        is_free_cell = False
        curr_agent_pos = [
            agent.pos.tolist()
            for agent in self.agents.values()
            if not getattr(agent, "is_tagged_out", False)
        ]
        np.random.shuffle(self.spawn_points)
        for i, spawn_point in enumerate(self.spawn_points):
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                spawn_index = i
                is_free_cell = True
        assert is_free_cell, "There are not enough spawn points! Check your map?"
        return np.array(self.spawn_points[spawn_index])

    def custom_reset(self):
        self.apple_timers = {}
        self.active_apple_points = set()
        for row, col in self.apple_points:
            self.single_update_map(row, col, b" ")
        if self.apple_points:
            initial = min(max(self.num_initial_apples, 0), len(self.apple_points))
            indices = np.random.choice(len(self.apple_points), size=initial, replace=False)
            for idx in np.atleast_1d(indices):
                row, col = self.apple_points[int(idx)]
                self.active_apple_points.add((row, col))
                self.single_update_map(row, col, b"A")
        for agent in self.agents.values():
            agent.reset_tag_state()

    def custom_action(self, agent, action):
        if agent.is_tagged_out or action != "FIRE":
            return []
        agent.fire_beam(b"F")
        return self.update_map_fire(
            agent.pos.tolist(),
            agent.get_orientation(),
            self.all_actions["FIRE"],
            fire_char=b"F",
            beam_width=1,
        )

    def color_view(self, agent):
        row, col = agent.pos[0], agent.pos[1]
        # Build an oriented character patch first so we can recolor self/opponent
        # consistently from each agent's own perspective.
        map_with_agents = self.get_map_with_agents()
        h, w = map_with_agents.shape
        pad = self.map_padding
        padded_map = np.full((h + 2 * pad, w + 2 * pad), b"0", dtype="c")
        padded_map[pad : pad + h, pad : pad + w] = map_with_agents

        radius = self.view_len
        view_slice = padded_map[
            row + pad - radius : row + pad + radius + 1,
            col + pad - radius : col + pad + radius + 1,
        ]
        if agent.orientation == "UP":
            rotated_view = view_slice
        elif agent.orientation == "LEFT":
            rotated_view = np.rot90(view_slice)
        elif agent.orientation == "DOWN":
            rotated_view = np.rot90(view_slice, k=2)
        elif agent.orientation == "RIGHT":
            rotated_view = np.rot90(view_slice, k=1, axes=(1, 0))
        else:  # pragma: no cover - defensive branch
            rotated_view = view_slice

        center = radius
        row_start = center - GATHERING_OBS_AHEAD
        row_end = center + 1
        col_start = center - GATHERING_OBS_SIDE
        col_end = center + GATHERING_OBS_SIDE + 1
        char_crop = rotated_view[row_start:row_end, col_start:col_end]

        color_map = self.color_map.copy()
        for i in range(self.num_agents):
            color_map[bytes(str(i + 1), encoding="ascii")] = OTHER_AGENT_COLOR
        color_map[agent.get_char_id()] = SELF_AGENT_COLOR

        rgb_arr = np.zeros((*GATHERING_OBS_SHAPE[:2], 3), dtype=np.uint8)
        return self.map_to_colors(char_crop, color_map, rgb_arr, orientation="UP")

    def custom_map_update(self):
        # First update existing timers.
        respawn = []
        for pos, timer in list(self.apple_timers.items()):
            next_timer = timer - 1
            if next_timer <= 0:
                respawn.append((pos[0], pos[1], b"A"))
                self.active_apple_points.add(pos)
                del self.apple_timers[pos]
            else:
                self.apple_timers[pos] = next_timer

        if respawn:
            self.update_map(respawn)

        # Then register newly consumed apples (so full delay starts next step).
        for agent in self.agents.values():
            for pos in agent.collected_apple_positions:
                if pos not in self.apple_timers:
                    self.apple_timers[pos] = self.apple_respawn_delay
                self.active_apple_points.discard(pos)
            agent.collected_apple_positions = []

    def _respawn_tagged_agents(self):
        for agent in self.agents.values():
            if not agent.is_tagged_out:
                continue
            if agent.tick_tag_timer():
                agent.update_agent_pos(self.spawn_point())
                agent.update_agent_rot(self.spawn_rotation())

    def find_visible_agents(self, agent_id):
        agent = self.agents[agent_id]
        if agent.is_tagged_out:
            return np.zeros((self.num_agents - 1,), dtype=np.uint8)

        upper_lim = int(agent.pos[0] + agent.row_size)
        lower_lim = int(agent.pos[0] - agent.row_size)
        left_lim = int(agent.pos[1] - agent.col_size)
        right_lim = int(agent.pos[1] + agent.col_size)

        visible = []
        for other_id in sorted(self.agents.keys()):
            if other_id == agent_id:
                continue
            other = self.agents[other_id]
            if other.is_tagged_out:
                visible.append(0)
                continue
            in_view = (
                lower_lim <= other.pos[0] <= upper_lim and left_lim <= other.pos[1] <= right_lim
            )
            visible.append(1 if in_view else 0)
        return np.asarray(visible, dtype=np.uint8)

    def get_map_with_agents(self):
        grid = np.copy(self.world_map)
        for agent in self.agents.values():
            if agent.is_tagged_out:
                continue
            if not (0 <= agent.pos[0] < grid.shape[0] and 0 <= agent.pos[1] < grid.shape[1]):
                continue
            grid[agent.pos[0], agent.pos[1]] = agent.get_char_id()
        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]
        return grid

    def step(self, actions):
        self.beam_pos = []
        # Decrement respawn timers from previous steps before applying this step's actions.
        self._respawn_tagged_agents()

        # Ensure action dict keys exist for every known agent.
        for agent_id in self.agents.keys():
            actions.setdefault(agent_id, 4)  # STAY

        active_ids = [agent_id for agent_id, agent in self.agents.items() if not agent.is_tagged_out]
        active_actions = {
            agent_id: self.agents[agent_id].action_map(actions[agent_id]) for agent_id in active_ids
        }

        for agent_id in active_ids:
            agent = self.agents[agent_id]
            row, col = agent.pos[0], agent.pos[1]
            self.single_update_world_color_map(row, col, self.world_map[row, col])

        self.update_moves(active_actions)

        for agent_id in active_ids:
            agent = self.agents[agent_id]
            pos = agent.pos
            new_char = agent.consume(self.world_map[pos[0], pos[1]])
            self.single_update_map(pos[0], pos[1], new_char)

        self.update_custom_moves(active_actions)
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        for agent_id in active_ids:
            agent = self.agents[agent_id]
            row, col = agent.pos[0], agent.pos[1]
            if self.world_map[row, col] not in [b"F", b"C"]:
                self.single_update_world_color_map(row, col, agent.get_char_id())

        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        zero_obs = np.zeros(GATHERING_OBS_SHAPE, dtype=np.uint8)

        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = zero_obs if agent.is_tagged_out else self.color_view(agent)
            if self.return_agent_actions:
                prev_actions = np.array(
                    [actions[key] for key in sorted(actions.keys()) if key != agent.agent_id],
                    dtype=np.uint8,
                )
                visible_agents = self.find_visible_agents(agent.agent_id)
                if agent.prev_visible_agents is None:
                    prev_visible = visible_agents
                else:
                    prev_visible = agent.prev_visible_agents
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "other_agent_actions": prev_actions,
                    "visible_agents": visible_agents,
                    "prev_visible_agents": prev_visible,
                }
                agent.prev_visible_agents = visible_agents
            else:
                observations[agent.agent_id] = {"curr_obs": rgb_arr}

            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()
            infos[agent.agent_id] = {}

        if self.use_collective_reward:
            collective_reward = sum(rewards.values())
            for agent_id in rewards.keys():
                rewards[agent_id] = collective_reward
        if self.inequity_averse_reward:
            assert self.num_agents > 1, "Cannot use inequity aversion with only one agent!"
            temp_rewards = rewards.copy()
            for agent_id in rewards.keys():
                diff = np.array([r - rewards[agent_id] for r in rewards.values()])
                dis_inequity = self.alpha * sum(diff[diff > 0])
                adv_inequity = self.beta * sum(diff[diff < 0])
                temp_rewards[agent_id] -= (dis_inequity + adv_inequity) / (self.num_agents - 1)
            rewards = temp_rewards

        dones["__all__"] = np.any(list(dones.values()))
        return observations, rewards, dones, infos

    def spawn_rotation(self):
        # Original Gathering implementations initialize agents facing right.
        return "RIGHT"
