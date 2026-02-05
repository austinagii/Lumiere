"""Test configuration for DeepScale tests."""

import os
from pathlib import Path

from dotenv import load_dotenv


def pytest_configure(config):
    """Load environment variables from .env file for all tests."""
    # Find the project root directory (where .env is located)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
    else:
        print(f"No .env file found at {env_file}")