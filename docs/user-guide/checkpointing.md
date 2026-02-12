# Checkpointing

Lumière provides robust checkpointing with multiple storage backends.

## Overview

Checkpoints save:

- Model state (weights)
- Optimizer state
- Scheduler state
- Training configuration
- Epoch and step counters
- Best loss tracking

## Configuration

Configure checkpoint storage:

```yaml
runs:
  checkpoints:
    sources:
      - filesystem
      - azure-blob
    destinations:
      - filesystem
      - azure-blob

storage:
  clients:
    filesystem:
      basedir: ./runs
```

## Checkpoint Types

### Epoch Checkpoints

Saved automatically after each epoch:

- Tag format: `epoch:NNNN` (e.g., `epoch:0001`)
- Allows resuming from any epoch

### Best Checkpoint

Saved when validation loss improves:

- Tag: `best`
- Use for model selection

### Final Checkpoint

Saved at training completion:

- Tag: `final`
- Contains final model state

## Storage Backends

### Filesystem

Saves to local directory:

```yaml
storage:
  clients:
    filesystem:
      basedir: ./runs
```

### Azure Blob Storage

Saves to Azure cloud storage:

```yaml
runs:
  checkpoints:
    destinations:
      - azure-blob
```

Requires environment variable:

```bash
export AZURE_BLOB_CONNECTION_STRING="your-connection-string"
```

## Loading Checkpoints

Resume from checkpoint:

```bash
python scripts/train.py config.yaml \
  --run-id abc123 \
  --checkpoint-tag best
```

## Run Management

Each training run gets a unique 8-character ID. Runs store:

- Configuration
- Checkpoints
- Artifacts (tokenizer, etc.)

Find runs in the configured base directory:

```
./runs/
  runs/
    abc123/
      config.yaml
      checkpoints/
        epoch:0001.pkl
        best.pkl
      artifacts/
        tokenizer
```

## See Also

- [Configuration](configuration.md)
- [Training](training.md)
