#!/usr/bin/env bash

# Lumiére CLI - Machine Learning Training and Chat Interface
# This script provides easy access to the Lumiére toolkit for training models and interactive chat.

set -e

readonly CMD_NAME="lumi"
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
    echo "Usage: ${CMD_NAME} <command> <model>"
    echo
    echo "Commands:"
    echo "  train    Train a machine learning model."
    echo "  chat     Start interactive chat session with a model"
    echo "  help     Show this help message"
    echo
    echo "Examples:"
    echo "  ${CMD_NAME} train transformer-large"
    echo "  ${CMD_NAME} chat transformer-large"
}

run_with_pipenv() {
    local script_path="$1"
    local model="$2"
    
    if command -v pipenv &>/dev/null; then
        log_info "Running with Pipenv environment..."
        pipenv run python "${script_path}" "${model}"
    else
        log_info "Running with system Python..."
        python3 "${script_path}" "${model}"
    fi
}

main() {
    # Handle help command or no arguments
    if [[ $# -eq 0 ]] || [[ "$1" == "help" ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    # Validate argument count
    if [[ $# -ne 2 ]]; then
        log_error "Invalid number of arguments"
        echo
        show_usage
        exit 1
    fi
    
    local command="$1"
    local model="$2"
    
    # Validate command
    if [[ "$command" != "train" && "$command" != "chat" ]]; then
        log_error "Unrecognized command: '$command'"
        log_error "Supported commands are: train, chat, help"
        echo
        show_usage
        exit 1
    fi
    
    # Allow the lumiere module to be discoverable
    export PYTHONPATH="${CMD_BASE_DIR}:${PYTHONPATH}"

    pipenv_installed=$(command -v pipenv &>/dev/null)
    if [[ $pipenv_installed ]]; then
        log_info "Pipenv detected: $(pipenv --version)"
    else
        log_warning "Pipenv not found - using system Python instead"
        log_warning "Consider installing Pipenv for better dependency management"
    fi

    # Execute the appropriate command.
    case "$command" in
        "train")
            log_info "Starting model training..."
            run_with_pipenv "${CMD_BASE_DIR}/scripts/train.py" "${model}"
            ;;
        "chat")
            log_info "Starting chat session..."
            run_with_pipenv "${CMD_BASE_DIR}/scripts/inference.py" "${model}"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
