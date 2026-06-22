"""Provides the main entrypoints for starting or resuming runs."""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .lumiere import resume, start
from .training.config import Config
from .training.orchestrator import TrainingArguments


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./.lumiere.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def _parse_start_command_args(args) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Start a new training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--spec",
        dest="spec",
        type=str,
        default=None,
        help="The path to the training configuration file",
    )
    group.add_argument(
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
    group.add_argument(
        "-e",
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=10,
        help="The maximum number of epochs to train for",
    )
    group.add_argument(
        "--no-max-epochs",
        dest="no_max_epochs",
        action="store_true",
        default=False,
        help="Train indefinitely, with no maximum epoch limit",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-p",
        "--max-epochs-without-improvement",
        dest="patience",
        type=int,
        default=5,
        help="The maximum number of consecutive epochs allowed without improvement. "
        "Exceeding this number halts the training run",
    )
    group.add_argument(
        "--no-max-epochs-without-improvement",
        dest="no_patience",
        action="store_true",
        default=False,
        help="Disable early stopping based on lack of improvement",
    )

    parser.add_argument(
        "--min-improvement",
        dest="stopping_threshold",
        type=float,
        default=0.001,
        help="The minimum change in the monitored metric to qualify as an improvement",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--log-interval",
        dest="log_interval",
        type=int,
        default=10,
        help="Number of steps between logging updates",
    )
    group.add_argument(
        "--no-logging",
        dest="no_logging",
        action="store_true",
        default=False,
        help="Disable logging entirely",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval",
        type=int,
        default=3,
        help="Number of epochs between saving checkpoints",
    )
    group.add_argument(
        "--no-checkpoints",
        dest="no_checkpoints",
        action="store_true",
        default=False,
        help="Disable checkpoint saving entirely",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gradient-clip-norm",
        dest="gradient_clip_norm",
        type=float,
        default=1.0,
        help="Maximum norm for gradient clipping",
    )
    group.add_argument(
        "--no-gradient-clipping",
        dest="no_gradient_clipping",
        action="store_true",
        default=False,
        help="Disable gradient clipping entirely",
    )

    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="The device to train on",
    )

    return vars(parser.parse_args(args))


def _execute_start_command(
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
    device: str,
):
    if spec:
        config = _load_spec(spec)
    elif template:
        config = _load_template(template)

    if overrides:
        config = _apply_overrides(config, overrides)

    args = TrainingArguments(
        max_epochs=None if no_max_epochs else max_epochs,
        patience=None if no_patience else patience,
        stopping_threshold=stopping_threshold,
        log_interval=None if no_logging else log_interval,
        checkpoint_interval=None if no_checkpoints else checkpoint_interval,
        gradient_clip_norm=None if no_gradient_clipping else gradient_clip_norm,
        device=device,
    )

    start(config, args=args, name=name)


def _parse_resume_command_args(args) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="resume",
        description="Resume a training run from a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "name",
        help="The name of the training run to resume",
    )
    parser.add_argument(
        "-f",
        "--from",
        dest="tag",
        default="latest",
        help="The checkpoint to resume from",
    )

    return vars(parser.parse_args(args))


def _execute_resume_command(name: str, tag: str):
    resume(name=name, tag=tag)


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


def _apply_overrides(config, overrides):
    config = copy.deepcopy(config)

    for param, value in (item.split("=", 1) for item in overrides):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="lumiere")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start a training run")
    resume_parser = subparsers.add_parser("resume", help="Resume a training run")

    args, remaining = parser.parse_known_args()

    if args.command == "start":
        start_args = _parse_start_command_args(remaining)
        _execute_start_command(**start_args)
    elif args.command == "resume":
        resume_args = _parse_resume_command_args(remaining)
        _execute_resume_command(**resume_args)
