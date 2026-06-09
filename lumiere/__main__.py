"""Provides the main entrypoints for starting or resuming runs."""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

from lumiere.persistence.clients import FileSystemStorageClient
from lumiere.training import Config, TrainingOrchestrator
from lumiere.training.artifact import ArtifactStore
from lumiere.training.checkpoint import CheckpointStore
from lumiere.training.event import EventStore
from lumiere.training.run import RunRepository
from lumiere.utils.device import get_device


LUMIERE_CONFIG_PATH = "../lumiere.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def _parse_subcommand():
    """Extract the command and remaining args from the command line.

    Usage: `python -m lumiere <command> [args]`
    Example: `python -m lumiere train --config ./configs/transformer.yaml`

    Returns a tuple of (command, remaining_args) where command is the first
    positional argument and remaining_args is the rest of argv to be parsed
    by the command-specific parser.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m lumiere <command> [args]", file=sys.stderr)
        sys.exit(1)
    return sys.argv[1], sys.argv[2:]


def _parse_train_command_args(args):
    parser = argparse.ArgumentParser(
        description="Train a Lumiére model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--spec",
        dest="spec",
        type=str,
        default=None,
        help="The path to the training configuration file",
    )

    parser.add_argument(
        "--template",
        dest="template",
        type=str,
        default=None,
        help="The name of the model template",
    )

    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        type=str,
        default=None,
        help="The name of the training run",
    )

    parser.add_argument(
        "-o",
        "--set",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help="Properties of either the spec or template to be replaced during the run",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-e", "--max-epochs", type=int, default=10)
    group.add_argument("--no-max-epochs", action="store_true", default=False)

    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "-p", "--max-epochs-without-improvement", dest="patience", type=int, default=5
    )
    group.add_argument(
        "--no-max-epochs-without-improvement",
        dest="no_patience",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--min-improvement", dest="stopping_threshold", type=float, default=0.001
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--no-logging", action="store_true", default=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint-interval", type=int, default=3)
    group.add_argument("--no-checkpoints", action="store_true", default=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gradient-clip-norm", type=float, default=1.0)
    group.add_argument("--no-gradient-clipping", action="store_true", default=False)

    parser.add_argument("-d", "--device", default="cpu")
    return parser.parse_args(args)


def _train_cmd_line_runner(  # This should call the start method.
    spec: str | None,
    template: str | None,
    name: str | None,
    overrides: str | None,
    max_epochs: int | None,
    no_max_epochs: bool,
    patience: int | None,
    no_patience: bool,
    stopping_threshold: float,
    log_interval: int,
    no_logging: bool,
    checkpoint_interval: int,
    no_checkpoints: bool,
    gradient_clip_norm: float,
    no_gradient_clipping: bool,
    # storage_locations: str,
    device: str,
):
    if spec:
        config = _load_spec(spec)
    elif template:
        config = _load_template(template)
    else:
        raise ValueError("Either a spec or template must be provided.")

    _overrides = _parse_overrides(overrides) if overrides else None
    _max_epochs = None if no_max_epochs else max_epochs
    _patience = None if no_patience else patience
    _log_interval = None if no_logging else log_interval
    _checkpoint_interval = None if no_checkpoints else checkpoint_interval
    _gradient_clip_norm = None if no_gradient_clipping else gradient_clip_norm

    _start(
        model=config,
        name=name,
        overrides=_overrides,
        max_epochs=_max_epochs,
        patience=_patience,
        stopping_threshold=stopping_threshold,
        gradient_clip_norm=_gradient_clip_norm,
        log_interval=_log_interval,
        checkpoint_interval=_checkpoint_interval,
        device=device,
    )


def _start(
    model: Config,
    name: str | None = None,
    overrides: Config | None = None,
    max_epochs: int | None = 100,
    patience: int | None = 5,
    stopping_threshold: float = 1e-3,
    gradient_clip_norm: float | None = 1e-6,
    log_interval: int | None = 50,
    checkpoint_interval: int | None = 3,
    # storage_locations: str = "filesystem:filesystem",
    device: str | None = None,
):
    if overrides:
        model = _apply_overrides(model, overrides)

    storage_client = _init_storage_client()
    run_store = RunRepository(storage_client)
    checkpoint_store = CheckpointStore(storage_client)
    artifact_store = ArtifactStore(storage_client)
    event_store = EventStore(storage_client)
    device = get_device() if device is None else device

    orchestrator = TrainingOrchestrator(
        run_repository=run_store,
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        event_store=event_store,
        max_epochs=max_epochs,
        patience=patience,
        stopping_threshold=stopping_threshold,
        gradient_clip_norm=gradient_clip_norm,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        device=device,
    )

    orchestrator.train(config=model, run_name=name)


def _load_spec(spec):
    cwd = Path(os.getcwd())
    spec_path = (cwd / spec).resolve()
    return Config.from_file(spec_path)


def _load_template(template):
    cwd = Path(__file__).parent
    template_path = (cwd / f"../templates/{template}.yaml").resolve()
    if not template_path.exists():
        print(f"Error: Template '{template}' does not exist.")
        sys.exit(1)

    return Config.from_file(template_path)


def _parse_overrides(overrides: str) -> Config:
    return Config(dict(item.split("=", 1) for item in overrides))


# TODO: Need to correctly type convert values from cmd line.
def _apply_overrides(config, overrides):
    config = copy.deepcopy(config)

    for param, value in overrides.data.items():
        config[param] = _infer_type(value)

    return config


def _infer_type(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _init_storage_client():
    return FileSystemStorageClient(Path(__file__).parent.parent)


class TrainCommandParser:
    pass


class TrainCommandExecutor:
    pass


parsers = {"train": TrainCommandParser}
executors = {"train": TrainCommandExecutor}


def main():
    """Main entry point for the training script."""
    subcommand, args = _parse_subcommand()

    if subcommand == "train":
        args = _parse_train_command_args(args)

        _train_cmd_line_runner(**vars(args))
    else:
        print(f"Error: Unrecognized subcommand '{subcommand}'.")


if __name__ == "__main__":
    main()
