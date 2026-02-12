"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from src.hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from src.hw1_imitation.model import build_policy, PolicyType
from src.hw1_imitation.evaluation import Logger
from src.hw1_imitation.evaluation import evaluate_policy


LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "flow"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 4

    batch_size: int = 256
    lr: float = 20e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 800
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    global_step = 0
    recent_losses: list[float] = []
    t_last = datetime.now()
    steps_last = 0

    evaluate_policy(
        model=model,
        normalizer=normalizer,
        device=device,
        chunk_size=config.chunk_size,
        video_size=config.video_size,
        num_video_episodes=config.num_video_episodes,
        flow_num_steps=config.flow_num_steps,
        step=global_step,  # step=0
        logger=logger,
    )
    
    
    print("Initial evaluation complete.\n")
    print(f"Starting training for {config.num_epochs} epochs...")

    for epoch in range(config.num_epochs):
        model.train()

        for state, action_chunk in loader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(state, action_chunk)
            loss.backward()
            optimizer.step()

            global_step += 1
            recent_losses.append(float(loss.item()))

            # Training logs
            if global_step % config.log_interval == 0:
                avg_loss = float(np.mean(recent_losses[-config.log_interval:]))
                dt = (datetime.now() - t_last).total_seconds()
                dsteps = global_step - steps_last
                steps_per_sec = float(dsteps / dt) if dt > 0 else 0.0
                samples_per_sec = float(steps_per_sec * config.batch_size)

                logger.log(
                    {
                        "train/loss": avg_loss,
                        "train/epoch": epoch,
                        "train/steps_per_sec": steps_per_sec,
                        "train/samples_per_sec": samples_per_sec,
                    },
                    step=global_step,
                )

                t_last = datetime.now()
                steps_last = global_step

            # Periodic evaluation
            if global_step % config.eval_interval == 0:
                print(f"\nEvaluating at step {global_step}...")
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=global_step,
                    logger=logger,
                )
                model.train()
                print("Evaluation complete.\n")

    # Final evaluation
    print("\nRunning final evaluation...")
    evaluate_policy(
        model=model,
        normalizer=normalizer,
        device=device,
        chunk_size=config.chunk_size,
        video_size=config.video_size,
        num_video_episodes=config.num_video_episodes,
        flow_num_steps=config.flow_num_steps,
        step=global_step,
        logger=logger,
    )

    logger.dump_for_grading()
    print("Training complete!")

def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
