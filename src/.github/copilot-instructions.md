<!-- Copilot / AI helper instructions for contributors and agents -->
# Copilot instructions — HW03 (DQN / SAC implementations)

This project implements DQN and SAC training code for Gym environments. Use these notes to help you quickly make meaningful edits, find entry points, and run experiments.

- Big picture
  - Top-level packages: `agents/` (algorithm implementations), `configs/` (experiment factories), `infrastructure/` (utilities: replay buffers, pytorch helpers, logging), `networks/` (policy & critic networks), and `scripts/` (training entry points).
  - Data flow: `scripts/run_*.py` builds a `config` via YAML -> config factory in `configs/` -> creates env(s) and an `agent` (class in `agents/`) -> interacts with Gym, stores transitions in `infrastructure/replay_buffer.py`, samples batches, converts to tensors via `infrastructure/ptu.py`, and calls the agent `update(...)` APIs.

- Key entry points (open these first for context)
  - `scripts/run_dqn.py` — DQN training loop and example of using `MemoryEfficientReplayBuffer` for frame-stacked Atari.
  - `scripts/run_sac.py` — SAC training loop and continuous-action usage.
  - `configs/dqn_config.py`, `configs/sac_config.py` — provide `configs` registries (e.g. `dqn_basic`, `dqn_atari`, `sac`) and example factories for `make_env`, optimizers, and schedules.
  - `agents/dqn_agent.py`, `agents/sac_agent.py` — algorithm implementations. Look for methods: `get_action`, `update`, `update_critic`, `update_target_critic`.

- Project-specific conventions and gotchas
  - Run from the `src` directory so package imports (e.g. `from agents...`) resolve. Example: `cd src && python -m scripts.run_dqn --config_file path/to/config.yml` (or `python scripts/run_dqn.py ...`).
  - Config YAML shape: the script expects a YAML with `base_config` (a key in `configs.<file>.configs`) and other keyword args. See `make_config()` in `scripts/run_dqn.py`/`run_sac.py` which calls `configs[base_config](**kwargs)`.
  - Replay buffer semantics: `ReplayBuffer.sample(batch_size)` returns numpy arrays. Convert to torch tensors using `ptu.from_numpy(...)` before passing to agent update routines.
  - Atari/frames: `MemoryEfficientReplayBuffer` is used for stacked frames; call `on_reset()` as shown in `run_dqn.py`. Observations for this buffer must be uint8 single frames.
  - GPU/device handling: use `infrastructure/pytorch_util.py` (`ptu.init_gpu`, `ptu.device`, `ptu.from_numpy`, `ptu.to_numpy`) for consistent device placement.
  - Logging: use `infrastructure/log_utils.py`. Scripts set up WandB by default — pass `--wandb_project` / `--wandb_entity` or disable as needed.
  - Gym API differences: the code assumes `env.reset()` returns an ndarray (old Gym API). Some places assert the return is not a tuple — be cautious with Gym>=0.26 which returns (obs, info).

- How to run a quick experiment (example)
  - Create a YAML, e.g. `example_dqn.yaml`:
    ```yaml
    base_config: dqn_basic
    env_name: CartPole-v1
    total_steps: 100000
    exp_name: quicktest
    ```
  - From the `src` directory run:
    ```bash
    python -m scripts.run_dqn --config_file ../path/to/example_dqn.yaml --seed 0 --log_interval 1000
    ```
  - For Atari, ensure `gym[atari,accept-rom-license]` is installed and use `base_config: dqn_atari`.

- Common edits and where to make them
  - Add a new network: `networks/` (follow `DQNCritic` / `StateActionCritic` / `MLPPolicy` patterns). Prefer `ptu.build_mlp` for device placement.
  - Add a new optimizer/scheduler or hyperparameter: add to a `configs/` factory and expose via YAML.
  - Add a new RL algorithm: create a new module under `agents/` implementing `get_action()` and `update(...)`, and add a `scripts/run_<algo>.py` or reuse existing script with a new config.

- Quick code patterns to follow
  - Convert replay-batch to tensors: `batch = ptu.from_numpy(batch)` then pass fields to agent.
  - Agents store optimizers and LR schedulers (e.g., `self.critic_optimizer`, `self.lr_scheduler`) and call `.step()` as shown in agent files.
  - Use `logger.log(dict, step)` to emit metrics; `dump_log(agent, logger, args, path)` for checkpoints.

If anything here is unclear or you want more examples (e.g., a minimal YAML per algorithm or a short snippet showing how to add a new policy network), tell me which section to expand and I will iterate.
