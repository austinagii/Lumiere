# Training

Learn how to train transformer models with Lumière.

## Basic Training

The simplest way to start training:

```bash
python scripts/train.py config.yaml
```

## Training Process

The training script:

1. Loads configuration from YAML file
2. Initializes all components (model, tokenizer, data, etc.)
3. Creates a new training run with unique ID
4. Saves tokenizer and configuration artifacts
5. Runs training loop with automatic checkpointing
6. Saves final checkpoint on completion

## Training Parameters

Configure training behavior in the `training` section:

```yaml
training:
  max_epochs: 250              # Maximum number of epochs
  stopping_threshold: 0.0001   # Early stopping threshold
  patience: 5                  # Epochs without improvement before stopping
  gradient_clip_norm: 1.0      # Gradient clipping threshold
```

## Monitoring Training

Training progress is logged to:

- Console output (real-time)
- `training.log` file
- Checkpoint storage locations

## Resuming Training

Resume from a saved checkpoint:

```bash
python scripts/train.py config.yaml --run-id <RUN_ID> --checkpoint-tag best
```

Available checkpoint tags:

- `best` - Best performing checkpoint
- `final` - Final checkpoint from completed training
- `epoch:NNNN` - Checkpoint from specific epoch (e.g., `epoch:0010`)

## See Also

- [Configuration](configuration.md)
- [Checkpointing](checkpointing.md)
- [API Reference](../reference/)
