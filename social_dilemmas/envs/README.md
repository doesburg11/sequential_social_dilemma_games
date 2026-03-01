# Environment Guide (`social_dilemmas/envs`)

This document explains how the core SSD environments in this repository work:

- `harvest.py`
- `cleanup.py`
- `gathering.py`

It focuses on environment logic, reward dynamics, and what happens during each `step()`.

## Social Dilemma Definition (Leibo et al., 2017)

Leibo et al. define a social dilemma using payoff inequalities over outcomes from
cooperator/defector policy sets. Using standard matrix notation:

- `R`: payoff from mutual cooperation (`C` vs `C`)
- `P`: payoff from mutual defection (`D` vs `D`)
- `S`: cooperator payoff when facing a defector (`C` vs `D`, "sucker" payoff)
- `T`: defector payoff when facing a cooperator (`D` vs `C`, "temptation" payoff)

The social-dilemma conditions are:

1. `R > P`
2. `R > S`
3. `2R > T + S`
4. either `T > R` (greed) or `P > S` (fear)

Important: In Sequential Social Dilemmas (SSD), these are defined over expected
returns from policy interactions over time, not just one-step rewards.

### Why these environments are SSDs under this definition

- `Harvest`:
  - Cooperative behavior: restrained harvesting that preserves future regrowth.
  - Defective behavior: over-harvesting for short-term gain.
  - Structure: one-shot over-consumption can increase short-term individual gain
    (`T` pressure), but if everyone defects, regrowth collapses and long-run
    returns drop (`P`), making sustained cooperation preferable (`R > P`).
- `Cleanup`:
  - Cooperative behavior: spending effort to clean waste and sustain apple growth.
  - Defective behavior: free-riding (harvest without cleaning).
  - Structure: if both defect, waste accumulates and apple production collapses
    (`P` low). If one cooperates while the other defects, the cooperator bears
    the cleaning burden and is exploitable (`S` low). Mutual cooperation sustains
    the resource (`R` high).
- `Gathering`:
  - Cooperative behavior: peaceful foraging (limited/no tagging).
  - Defective behavior: aggressive tagging/area denial to monopolize apples.
  - Structure: unilateral aggression can create a temptation payoff (`T`), but
    mutual aggression reduces overall productivity via repeated conflict and
    downtime (`P`), while mutual restraint supports better average outcomes (`R`).

So, all three environments are intended to instantiate Leibo-style SSDs at the
policy-interaction level (`R,P,S,T`), even though exact numeric values depend on
the selected policy sets, training regime, and horizon.

## Shared Base Logic (`MapEnv`)

`HarvestEnv` and `CleanupEnv` inherit `MapEnv.step()` directly. `GatheringEnv` overrides `step()`, but keeps most of the same structure.

### Shared map model

- Grid is stored as ASCII-like bytes (`self.world_map`), e.g. walls `@`, apples `A`, waste `H`.
- Agent state is stored in agent objects (`self.agents`), not directly in `self.world_map`.
- `get_map_with_agents()` overlays agent IDs (`"1"`, `"2"`, ...) and active beam cells on top of `self.world_map` for observations/rendering.

### Shared movement/actions

Base discrete actions (`agent.py`):

- `0`: `MOVE_LEFT`
- `1`: `MOVE_RIGHT`
- `2`: `MOVE_UP`
- `3`: `MOVE_DOWN`
- `4`: `STAY`
- `5`: `TURN_CLOCKWISE`
- `6`: `TURN_COUNTERCLOCKWISE`

Environment-specific actions are appended:

- Harvest: `7 = FIRE`
- Cleanup: `7 = FIRE`, `8 = CLEAN`
- Gathering: `7 = FIRE`

### Shared observation format

`HarvestEnv` and `CleanupEnv` provide a centered square RGB crop:

- key: `"curr_obs"`
- shape: `(2 * view_len + 1, 2 * view_len + 1, 3)`
- dtype: `uint8`

`GatheringEnv` uses the paper-style forward-biased crop:

- key: `"curr_obs"`
- shape: `(16, 21, 3)`
- semantics: 15 cells ahead + current row, and 10 cells to each side (egocentric orientation)
- own avatar and opponent avatar are recolored from the observer perspective (self=blue, other=red)

If `return_agent_actions=True`, observations also include:

- `"other_agent_actions"`
- `"visible_agents"`
- `"prev_visible_agents"`

### Shared `MapEnv.step()` sequence (Harvest/Cleanup)

1. Clear previous beam markers.
2. Convert integer actions to action strings with each agent’s `action_map`.
3. Remove agent colors from `world_map_color` before movement.
4. Resolve movement/turn actions with conflict handling (`update_moves`).
5. Call `agent.consume(...)` on each agent’s current tile (e.g., collect apples).
6. Execute non-move custom actions (`FIRE`, `CLEAN`) via `update_custom_moves`.
7. Run environment dynamics (`custom_map_update`), e.g., apple/waste spawning.
8. Rebuild map overlay and color map.
9. Build observations, rewards, done flags.
10. Optional reward post-processing:
    - collective reward mode
    - inequity aversion mode

`dones["__all__"]` is true if any agent signals done. In these envs agents normally never terminate by themselves (`get_done() -> False`).

## Harvest

File: `harvest.py`

### Core idea

Harvest is a tragedy-of-the-commons environment:

- agents consume apples for reward;
- apple regrowth is local-density dependent;
- over-harvesting reduces future regrowth.

### State elements

- Apple spawn points come from `A` cells in the base map.
- `APPLE_RADIUS = 2`
- `SPAWN_PROB = [0, 0.005, 0.02, 0.05]`

At each candidate apple spawn location:

1. Count nearby apples within radius-2 disk.
2. Map count to spawn probability via `SPAWN_PROB[min(num_apples, 3)]`.
3. Sample Bernoulli spawn, unless blocked by existing apple/agent.

### Actions

- 8 discrete actions total.
- `FIRE` shoots a beam (length 5, width 3 pattern from `MapEnv.update_map_fire`).

### Reward logic (`HarvestAgent`)

- Consume apple (`A`): `+1`
- Fire beam (`FIRE`): `-1`
- Hit by beam (`F`): `-50`

### Custom per-step updates

- `custom_action`: executes beam fire.
- `custom_map_update`: executes `spawn_apples()`.

## Cleanup

File: `cleanup.py`

### Core idea

Cleanup is a public-goods social dilemma:

- Apples grow in designated `B` zones.
- River/waste system controls whether apples can regrow.
- Agents can spend actions cleaning waste to restore apple growth.

### State elements

Base-map symbols used by logic:

- `B`: apple candidate spawn cell
- `H`: waste cell
- `R`: river cell
- `S`: stream cell

Important parameters:

- `thresholdDepletion = 0.4`
- `thresholdRestoration = 0.0`
- `wasteSpawnProbability = 0.5`
- `appleRespawnProbability = 0.05`

Derived values:

- `potential_waste_area = count(H) + count(R)` from base map
- `waste_density = 1 - permitted_area / potential_waste_area`

Spawn control:

- If `waste_density >= thresholdDepletion`:
  - apple spawn prob = `0`
  - waste spawn prob = `0`
- Else:
  - waste spawn prob = `0.5`
  - apple spawn prob interpolates from `0.05` down to `0` as waste density rises.

### Actions

- 9 discrete actions total.
- `FIRE`: beam attack.
- `CLEAN`: beam that converts `H -> R` cells along the beam path.

### Reward logic (`CleanupAgent`)

- Consume apple (`A`): `+1`
- Fire beam (`FIRE`): `-1`
- Hit by beam (`F`): `-50`
- `CLEAN` itself does not directly change reward; value comes from enabling regrowth.

### Custom per-step updates

- `custom_action`:
  - `FIRE` uses regular beam.
  - `CLEAN` beam transforms waste cells (`H`) into river cells (`R`).
- `custom_map_update`:
  - recompute spawn probabilities from current waste density;
  - spawn apples and (at most one) waste tile each step.

## Gathering

File: `gathering.py`

This implementation follows the research-style Gathering setup in spirit:

- Multi-agent support (`1..9` agents in this implementation).
- Competitive apple collection.
- Beam tagging temporarily removes opponents from play.
- Map footprint uses a wide `33x11` layout matching the canonical DeepMind `pycolab` Gathering layout (two top orchard patches).
- Beam uses a single forward line (width 1).
- Spawn orientation defaults to facing right.

Reference used for constants/layout alignment:

- DeepMind `pycolab` Gathering (`lp-rnn.py`, "Gathering game described in Leibo et al. 2017"):
  https://raw.githubusercontent.com/deepmind/pycolab/master/pycolab/examples/research/lp-rnn/lp-rnn.py

### Important constraints

- Canonical Leibo setup is 2 players; this repo also supports more agents.
- If the map has fewer `P` spawn markers than `num_agents`, extra spawn points are
  added on empty cells automatically.
- Current implementation limit is `9` agents (agent IDs are single-char `1..9`).
- Default delays:
  - `apple_respawn_delay = 3`
  - `tagged_respawn_delay = 25`
- Default active apples on reset:
  - `num_initial_apples = 6`

### Agent state (`GatheringAgent`)

Each agent tracks:

- `_beam_hit_count`: number of consecutive beam hits accumulated
- `_tagged_timer`: frames remaining while tagged out
- `collected_apple_positions`: consumed apple cells pending respawn timer setup

Tagging rule:

- When an active agent is hit by beam `F`, increment hit count.
- On second hit, agent is tagged out (`_tagged_timer` set).
- Tagged-out agents:
  - are not shown on map;
  - cannot move/fire/consume;
  - receive zero-image observations;
  - respawn after timer expiration at a spawn point facing right.

### Reward logic

- Consume apple (`A`): `+1`
- `FIRE`: reward-neutral in this implementation
- Being hit by beam: no direct reward penalty (effect is temporary removal)

### Apple respawn logic

- When apple is consumed, record its position and start a fixed timer.
- Once timer reaches zero, apple respawns.
- On reset, apples are sampled from candidate map positions so only `num_initial_apples` are active.

### Custom `step()` flow (override)

Gathering overrides `MapEnv.step()` to support tagged-out players cleanly:

1. Fill missing actions with `STAY`.
2. Build action dict only for active (not tagged-out) agents.
3. Move active agents, resolve consumption, fire beams, apply map updates.
4. Advance apple respawn timers and tagged-out respawn timers.
5. Build observations/rewards for all agents:
   - active: normal local RGB observation
   - tagged-out: zero observation tensor

## Beam model details (all envs)

`MapEnv.update_map_fire(...)` behavior:

- Beam starts in front and side-front offsets (width-3 pattern by default).
- Beam stops on wall `@`.
- If beam encounters an agent, it calls that agent’s `hit(...)` and stops in that direction.
- Optional cell transformations are supported (used by Cleanup `CLEAN`).

## Notes for training and interpretation

- None of these envs have natural terminal conditions; rollouts usually end by time horizon set in training.
- Collective reward and inequity-aversion wrappers are handled in `MapEnv.step()` and apply to all environments that use that path.
- Gathering’s overridden `step()` keeps the same optional reward wrappers for consistency.
