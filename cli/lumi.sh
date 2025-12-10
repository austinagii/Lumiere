#!/usr/bin/env bash

# Lumiére CLI - Machine Learning Training and Chat Interface
# This script provides easy access to the Lumiére toolkit for training models and interactive chat.

set -e

readonly CMD_NAME="lumi"
# shellcheck disable=SC2155
readonly CMD_BASE_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"

# Color constants
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

log_info() {
    echo -e "${GREEN}[ INFO ]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[ WARNING ]${NC} $1"
}

log_error() {
    echo -e "${RED}[ ERROR ]${NC} $1"
}

show_usage() {
    echo "Usage: ${CMD_NAME} <command> [arguments...]"
    echo
    echo "Commands:"
    echo "  train       Train a machine learning model."
    echo "  eval        Evaluate a language model."
    echo "  info        Generate a model card."
    echo "  inference   Run inference with a language model."
    echo "  help        Show this help message."
    echo
    echo "Examples:"
    echo "  ${CMD_NAME} train transformer-large"
    echo "  ${CMD_NAME} inference transformer-large"
    echo "  ${CMD_NAME} eval transformer-large"
}

run_with_pipenv() {
    local script_path="$1"
    shift

    if command -v pipenv &>/dev/null; then
        log_info "Running with Pipenv environment..."
        pipenv run python "${script_path}" "$@"
    else
        log_info "Running with system Python..."
        python3 "${script_path}" "$@"
    fi
}

main() {
    # Handle help command or no arguments
    if [[ $# -eq 0 ]] || [[ "$1" == "help" ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi

    # Validate minimum argument count (need at least the command)
    if [[ $# -lt 1 ]]; then
        log_error "Invalid number of arguments"
        echo
        show_usage
        exit 1
    fi

    local command="$1"
    shift # Remove command from arguments, leaving all other args in "$@"

    # Validate command
    if [[ "$command" != "train" && "$command" != "eval" && "$command" != "info" ]]; then
        log_error "Unrecognized command: '$command'"
        log_error "Supported commands are: train, eval, info, help"
        echo
        show_usage
        exit 1
    fi

    # Allow the lumiere module to be discoverable
    export PYTHONPATH="${CMD_BASE_DIR}:${PYTHONPATH}"

    # Execute the appropriate command, passing all remaining arguments
    case "$command" in
    "train")
        log_info "Starting model training..."
        run_with_pipenv "${CMD_BASE_DIR}/scripts/train.py" "$@"
        log_info "Training completed"
        ;;
    "eval")
        log_info "Starting evaluation..."
        run_with_pipenv "${CMD_BASE_DIR}/scripts/eval.py" "$@"
        log_info "Evaluation completed"
        ;;
    "info")
        log_info "Starting info..."
        run_with_pipenv "${CMD_BASE_DIR}/scripts/info.py" "$@"
        log_info "Info completed"
        ;;
    esac
}

# Run main function with all arguments
main "$@"
