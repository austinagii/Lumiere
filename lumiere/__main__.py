"""Provides the main entrypoints for starting or resuming runs."""

import argparse
import logging
import sys
from argparse import Namespace

from lumiere.persistence.clients import (
    AzureBlobStorageClient,
    FileSystemStorageClient,
    StorageClientDemux,
)
from lumiere.training import Config, TrainingOrchestrator
from lumiere.training.artifact import ArtifactRepository
from lumiere.training.checkpoint import CheckpointRepository
from lumiere.training.run import RunRepository
from lumiere.utils.device import get_device


LUMIERE_CONFIG_PATH = "../lumiere.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def _parse_command():
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


class TrainCommandParser:
    @staticmethod
    def parse(args):
        """Parse the command and the specified args.

        Usage: `python -m lumiere <command> [args]`
        Example: `python -m lumiere train --config ./configs/transformer.yaml`
        """
        parser = argparse.ArgumentParser(
            description="Train a Lumiére model.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--config-path",
            dest="config_path",
            type=str,
            default=None,
            help="The path to the training configuration file",
        )

        parser.add_argument(
            "--run-id",
            dest="run_id",
            default=None,
            help="The ID of the training run to be resumed",
        )

        parser.add_argument(
            "--checkpoint-tag",
            dest="checkpoint_tag",
            default="best",
            help="The tag of the checkpoint to resume from (e.g., 'epoch:0001', 'best', 'final')",
        )

        return parser.parse_args(args)


storage_clients = {
    "filesystem": FileSystemStorageClient(),
    "azure-blob": AzureBlobStorageClient(),
}


class TrainCommandExecutor:
    @staticmethod
    def execute(global_config: Config, args: Namespace):
        train_config = None

        clients = [
            storage_clients[loc]
            for loc in global_config.get("training.persistence.locations")
        ]

        if len(clients) > 1:
            storage_client = StorageClientDemux(*clients)
        else:
            storage_client = clients[0]

        run_repository = RunRepository(storage_client)
        checkpoint_repository = CheckpointRepository(storage_client)
        artifact_repository = ArtifactRepository(storage_client)
        orchestrator = TrainingOrchestrator(
            run_repository=run_repository,
            checkpoint_repository=checkpoint_repository,
            artifact_repository=artifact_repository,
            device=get_device(),
        )

        TrainingOrchestrator.train(
            config=config,
            run_id=args.run_id,
            checkpoint_tag=args.checkpoint_tag,
        )


parsers = {"train": TrainCommandParser}
executors = {"train": TrainCommandExecutor}


def main():
    """Main entry point for the training script."""
    global_config_path = Path(__file__) / LUMIERE_CONFIG_PATH
    lumiere_config = Config.from_yaml(global_config_path)

    command, args = _parse_command()
    parser = parsers.get(command)
    if parser is None:
        raise ValueError(f"No parser found for command '{command}'.")

    command_args = parser.parse(args)
    command_executor = executors.get(command)
    if command_executor is None:
        raise RuntimeError()

    try:
        command_executor.execute(command_args)
    except Exception as e:
        logger.error(f"Fatal error: {e}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
