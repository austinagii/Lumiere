"""Provides the main entrypoints for starting or resuming runs.

## Overview

This modules provides the the following methods for starting `DeepScale` runs.

 - [init_run][deepscale.init_run]: For starting a new run.
 - [resume_run][deepscale.resume_run]: For resuming a previous run.

As the main entrypoints for the `DeepScale` library, calling either method will lazily
initialize the `DeepScale` config (which is expected at
`<project_base_dir>/deepscale.yaml`).

In the event that no configuration is found the default configuration for the library
will be used.

On success, both functions will return a reference to a
[`RunManager`][deepscale.run.RunManager] instance configured to manage the newly
created / resumed run.

## Examples

Starting a new training run is pretty straightforward:

```python
import lumiere.deepscale as ds
from lumiere.deepscale import CheckpointType, Checkpoint

# Start a new run.
run_id, run_manager = ds.init_run(
     {"num_blocks": 24, "learning_rate": 0.0001, "max_epochs": 1000}
)

# Save a checkpoint after some training.
run_manager.save_checkpoint(
     CheckpointType.EPOCH,
     Checkpoint(model_params=model.state_dict(), loss=2.3231)
)
```

Resuming a previous training run is mostly the same:

```python
import lumiere.deepscale as ds
from lumiere.deepscale import CheckpointType, Checkpoint

# Resume the previous run.
run_config, checkpoint, run_manager = ds.resume_run("run-8wbcs7", "epoch:12")

# Save a new checkpoint some time after resuming training.
run_manager.save_checkpoint(
    CheckpointType.EPOCH,
    Checkpoint(model_params=model.state_dict(), loss=2.3231)
)
```
"""

import logging
from pathlib import Path
from typing import Any

import torch

from lumiere.training import Config, Checkpoint, RunManager


DS_CONFIG_PATH = "./deepscale.yaml"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def init_run(
    config: Config | None = None,
    config_file_paths: list[str | Path] | None = None,
    config_dir_path: str | Path | None = None,
) -> tuple[str, RunManager]:
    """Initializes a new training run with the specified configuration.

    This method provides a convenient way to initialize a new training run from a custom
    set of parameters or config file without manually creating a config, creating a
    run manager and initializing a run. It is equivalent to the following:
    ```
    run_manager = RunManager()
    train_config = TrainingConfiguration.from_files("model.yaml", "tokenizer.yaml")
    run_id = run_manager.initialize_run(train_config)
    ```

    One of `config`, `config_file_paths` or `config_dir` must be provided to initialize
    the training run. If either `config_file_paths` or `config_dir` is provided, then
    a configuration object created from the amalgamation of all config files present
    will be created and used to initialize the run.

    Arguments:
        run_config: The configuration to be used for the training run.

    Returns:
        The unique ID of the newly created run.
        The `RunManager` instance that manages the newly created run.
    """
    # Update method to accept directory of YAML files, compile them into a single file
    # and use that as the configuration for the training run. When saving this giant
    # config, use some pretty printing with ascii headers of the original file names
    # to display the config.
    if not Config.is_initialized():
        Config.from_yaml(DS_CONFIG_PATH)

    run_manager = RunManager.from_config(Config.get_instance())
    run_id = run_manager.init_run(run_config)
    return run_id, run_manager


def resume_run(
    run_id: str,
    checkpoint_tag: str | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[Any, Any], Checkpoint, RunManager]:
    """Resumes the specified training run.

    If a checkpoint tag is specified, then training is resumed from the matching
    checkpoint, else, training is resumed from the latest checkpoint.

    Args:
        run_id: The ID of the training run to be resumed.
        checkpoint_tag: The tag of the checkpoint to resume the specified training run
            from. Defaults to `None`.
        device: The device to load the checkpoint onto. Defaults to "cpu".

    Returns:
        A tuple containing the following:
            - The configuration used for the specified training run.
            - The checkpoint corresponding to the specified tag.
            - The run manager for the resumed training run.

    Raises:
        RunNotFoundError: If the specified run could not be found.
        CheckpointNotFoundError: If the specified checkpoint could not be found.
    """
    if not Config.is_initialized():
        Config.from_yaml(DS_CONFIG_PATH)

    # TODO: Consider saving the dsconfig used at the time of the training run. This
    # behavior could also be overriden.
    run_manager = RunManager.from_config(Config.get_instance())
    run_config, checkpoint = run_manager.resume_run(
        run_id, checkpoint_tag, device=device
    )
    return run_config, checkpoint, run_manager
