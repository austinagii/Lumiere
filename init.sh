#!/usr/bin/env bash

# This script configures the 'lumi' command in the current environment to
# provide easy access to the Lumiére toolkit.

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CMD_NAME="lumi"
readonly SCRIPT_PATH="${PROJECT_DIR}/cli/${CMD_NAME}.py"
readonly CMD_BASE_DIR="/usr/local/bin"
readonly CMD_PATH="${CMD_BASE_DIR}/${CMD_NAME}"

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Return codes for actions
readonly RC_SUCCESS=0
readonly RC_FAILURE=1
readonly RC_PENDING=2

# Logging functions
log_info() {
    echo -e "${BLUE}[ INFO ]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[ SUCCESS ]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[ WARNING ]${NC} $1"
}

log_error() {
    echo -e "${RED}[ ERROR ]${NC} $1"
}

echo
echo "=================================================="
echo "              Lumiére - Installation              "
echo "=================================================="
echo

cleanup() {
    local exit_code=$?
    echo
    
    if [[ $exit_code -eq ${RC_SUCCESS} ]]; then
        log_success "The '${CMD_NAME}' has been installed successfully!"
        echo
        echo "You can now use the following commands:"
        echo -e "  ${GREEN}${CMD_NAME} train --config-path <config>${NC}  - Train a machine learning model"
        echo -e "  ${GREEN}${CMD_NAME} test --run-id <id>${NC}             - Evaluate a trained model"
        echo
        echo -e "See ${GREEN}${CMD_NAME} --help${NC} for more information and additional commands"
        echo
        echo "=================================================="
        echo "          Lumiére Installation Complete!          "
        echo "=================================================="
    elif [[ $exit_code -eq ${RC_PENDING} ]]; then
        log_info "Installation completed but requires terminal restart"
        log_info "Please restart your terminal or run: source ~/.bashrc"
        echo
        log_warning "After restart, you can use: ${GREEN}${CMD_NAME} help${NC} for more information"
        echo "=================================================="
        echo "        Lumiére Installation Pending Refresh      "
        echo "=================================================="
    else
        log_error "Installation encountered an error (exit code: $exit_code)"
        log_info "Please review the error messages above and try again"
        echo "=================================================="
        echo "          Lumiére Installation Failed             "
        echo "=================================================="
    fi
    echo
}
trap cleanup EXIT
set -e

#==================================================
#              Environment Diagnostics            #
#==================================================

log_info "Performing environment diagnostics..."

# Confirm Python 3 is installed.
if command -v python3 &>/dev/null; then
    log_info "Python 3 detected: $(python3 --version)"
else
    log_error "Python 3 is required but not found"
    log_error "Please install Python 3 and try again"
    exit 1
fi

# Confirm Pipenv is installed.
if command -v pipenv &>/dev/null; then
    log_info "Pipenv detected: $(pipenv --version)"
else
    log_warning "Pipenv not found - dependencies will need manual management"
fi

# Confirm Lumiére script is present and executable.
if [[ -f "${SCRIPT_PATH}" ]]; then
    log_info "Lumiére script located at: ${SCRIPT_PATH}"
    if [[ -x "${SCRIPT_PATH}" ]]; then
        log_info "Script has executable permissions"
    else
        log_warning "Script requires executable permissions"
        if chmod +x "${SCRIPT_PATH}"; then
            log_success "Executable permissions granted"
        else
            log_error "Failed to set executable permissions"
            log_error "Please check file permissions and try again"
            exit 1
        fi
    fi
else
    log_error "Lumiére script not found at: ${SCRIPT_PATH}"
    log_error "Please ensure the project structure is intact"
    exit 1
fi

log_success "Environment diagnostics completed successfully"
echo

#==================================================
#            Prepare to Install Command           #
#==================================================

log_info "Preparing Lumiére command installation..."

# Create command directory if needed.
if [[ -e "${CMD_BASE_DIR}" ]]; then
    if [[ ! -d "${CMD_BASE_DIR}" ]]; then
        log_error "Installation path '${CMD_BASE_DIR}' exists but is not a directory"
        exit 1
    fi

    if [[ -L "${CMD_PATH}" ]]; then
        if [ "$(realpath "${CMD_PATH}" 2>/dev/null || echo "${CMD_PATH}")" == "${SCRIPT_PATH}" ]; then
            log_success "Lumiére command is already properly installed"
            exit 0
        else
            log_warning "Existing '${CMD_NAME}' command found, but points to different location"
            log_warning "Current target: $(realpath "${CMD_PATH}" 2>/dev/null || echo "unknown")"
            log_warning "Required target: ${SCRIPT_PATH}"
            echo
            echo "The existing command will be replaced with the Lumiére installation."

            while true; do
                read -p "Continue with replacement? [y/N]: " -r RESPONSE
                case "${RESPONSE,,}" in
                y | yes)
                    if sudo rm "${CMD_PATH}"; then
                        log_info "Existing command removed successfully"
                        break
                    else
                        log_error "Failed to remove existing command"
                        exit 1
                    fi
                    ;;
                n | no | "")
                    log_info "Installation cancelled by user"
                    exit 0
                    ;;
                *)
                    echo "Please answer 'y' or 'n'"
                    ;;
                esac
            done
        fi
    fi
else 
    log_info "Creating installation directory: ${CMD_BASE_DIR}"
    if ! sudo mkdir -p "${CMD_BASE_DIR}"; then
        log_error "Failed to create installation directory"
        log_error "Please ensure you have appropriate system permissions"
        exit 1
    fi
    log_success "Installation directory created successfully"
fi
echo

#==================================================
#                 Create Command                  #
#==================================================

log_info "Installing Lumiére command..."

# Check if directory is writable.
if [[ -w "${CMD_BASE_DIR}" ]]; then
    ln -s "${SCRIPT_PATH}" "${CMD_PATH}"
else
    log_info "Requesting administrator privileges for installation"
    sudo ln -s "${SCRIPT_PATH}" "${CMD_PATH}"
fi

if [[ $? -eq 0 ]]; then
    log_success "Command symlink created at: ${CMD_PATH}"
else
    log_error "Failed to create command symlink"
    exit 1
fi
echo

#==================================================
#                 Verify Installation             #
#==================================================

log_info "Verifying installation..."

# Check if command is in PATH.
EXISTING_CMD_PATH="$(command -v "${CMD_NAME}" 2>/dev/null || true)"

# If command is not in PATH, exit with error.
if [[ -z "${EXISTING_CMD_PATH}" ]]; then
    log_warning "Command '${CMD_NAME}' not immediately available in PATH"
    log_warning "You may need to restart your terminal session"
    exit ${RC_PENDING}
fi

# If command is in PATH, verify it points to the correct script.
RESOLVED_CMD_PATH="$(realpath "${EXISTING_CMD_PATH}" 2>/dev/null || true)"
if [[ -z "${RESOLVED_CMD_PATH}" ]]; then
    log_error "Could not resolve command path for verification"
    log_error "Installation may be incomplete"
    exit ${RC_FAILURE}
fi

if [[ "${RESOLVED_CMD_PATH}" != "${SCRIPT_PATH}" ]]; then
    log_error "Installation verification failed"
    log_error "Command resolves to: ${RESOLVED_CMD_PATH}"
    log_error "Expected path: ${SCRIPT_PATH}"
    exit ${RC_FAILURE}
fi

log_success "Installation verified - command is ready to use"
log_success "Installation completed successfully"