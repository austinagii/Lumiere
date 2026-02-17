#!/usr/bin/env python3
"""Lumiére CLI - Machine Learning Training and Evaluation Interface.

This script provides easy access to the Lumiére toolkit for training and evaluating
models.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    RESET = "\033[0m"


def log_info(message: str) -> None:
    """Log an informational message."""
    print(f"{Colors.GREEN}[ INFO ]{Colors.RESET} {message}")


def log_warning(message: str) -> None:
    """Log a warning message."""
    print(f"{Colors.YELLOW}[ WARNING ]{Colors.RESET} {message}")


def log_error(message: str) -> None:
    """Log an error message."""
    print(f"{Colors.RED}[ ERROR ]{Colors.RESET} {message}", file=sys.stderr)


def get_base_dir() -> Path:
    """Get the base directory of the project."""
    # Get the directory containing this script, then go up one level
    return Path(__file__).resolve().parent.parent


def run_with_python_env(script_path: Path, args: list[str]) -> int:
    """Run a Python script with the appropriate Python environment.

    Args:
        script_path: Path to the Python script to execute
        args: Additional command-line arguments to pass to the script

    Returns:
        Exit code from the script execution
    """
    base_dir = get_base_dir()

    # Build the command
    if shutil.which("pipenv"):
        log_info("Running with Pipenv environment...")
        cmd = ["pipenv", "run", "python", str(script_path)] + args
    else:
        log_info("Running with system Python...")
        cmd = [sys.executable, str(script_path)] + args

    # Set PYTHONPATH to include the project root
    import os

    env = os.environ.copy()
    pythonpath = str(base_dir)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    try:
        # Run the command and return its exit code
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        log_warning("Operation interrupted by user")
        return 130  # Standard exit code for SIGINT
    except FileNotFoundError as e:
        log_error(f"Failed to execute command: {e}")
        return 127  # Standard exit code for command not found
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return 1


def command_train(args: list[str]) -> int:
    """Execute the train command.

    Args:
        args: Command-line arguments to pass to the training script

    Returns:
        Exit code from the training script
    """
    base_dir = get_base_dir()
    script_path = base_dir / "scripts" / "train.py"

    if not script_path.exists():
        log_error(f"Training script not found at {script_path}")
        return 1

    log_info("Starting model training...")
    exit_code = run_with_python_env(script_path, args)

    if exit_code == 0:
        log_info("Training completed successfully")
    else:
        log_error(f"Training failed with exit code {exit_code}")

    return exit_code


def command_test(args: list[str]) -> int:
    """Execute the test/evaluation command.

    Args:
        args: Command-line arguments to pass to the evaluation script

    Returns:
        Exit code from the evaluation script
    """
    base_dir = get_base_dir()
    script_path = base_dir / "scripts" / "eval.py"

    if not script_path.exists():
        log_error(f"Evaluation script not found at {script_path}")
        return 1

    log_info("Starting model evaluation...")
    exit_code = run_with_python_env(script_path, args)

    if exit_code == 0:
        log_info("Evaluation completed successfully")
    else:
        log_error(f"Evaluation failed with exit code {exit_code}")

    return exit_code


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="lumi",
        description="Lumiére CLI - Machine Learning Training and Evaluation Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lumi train --config-path configs/transformer.yaml
  lumi train --run-id <run_id> --checkpoint-tag best
  lumi eval --run-id <run_id> --checkpoint-tag best
  lumi eval --run-id <run_id> --checkpoint-tag epoch:0001

        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=False,
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a machine learning model",
        description="Train a Lumiére transformer model with the specified configuration",
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained model",
        description="Evaluate a trained Lumiére model on test data",
    )

    return parser


def main() -> NoReturn:
    """Main entry point for the CLI."""
    parser = create_parser()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Parse only the command, then manually get the rest
    # Find the command position
    command_idx = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ["train", "eval"]:
            command_idx = i
            break

    if command_idx is None:
        # No valid command found, let argparse handle it
        args = parser.parse_args()
    else:
        # Parse up to and including the command
        args = parser.parse_args(sys.argv[1:command_idx + 1])
        # Everything after the command goes to the script
        script_args = sys.argv[command_idx + 1:]

    # Handle help command
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate command handler
    exit_code = 0

    if args.command == "train":
        exit_code = command_train(script_args)
    elif args.command == "eval":
        exit_code = command_test(script_args)
    else:
        log_error(f"Unknown command: {args.command}")
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
