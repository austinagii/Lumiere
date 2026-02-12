# Quick Start

This guide will help you train your first transformer model with Lumière.

## Create a Configuration File

Create a `config.yaml` file:

```yaml
model:
  vocab_size: 4096
  context_size: 64
  embedding_size: 128
  num_blocks: 4
  embedding:
    type: embedding
    name: sinusoidal
    padding_id: 0
  block:
    type: block
    name: standard
    attention:
      type: attention
      name: multihead
      num_heads: 4
      d_key: 32
      d_value: 32
    feedforward:
      type: feedforward
      name: linear
      d_ff: 256
    normalization:
      type: normalization
      name: rms

data:
  datasets:
    - name: wikitext

tokenizer:
  name: bpe
  vocab_size: 4096
  min_frequency: 2

pipeline:
  name: text
  tokenizer: "@tokenizer"
  batch_size: 32
  context_size: 64
  pad_id: 2

optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01

training:
  max_epochs: 10
  patience: 5
  gradient_clip_norm: 1.0

runs:
  checkpoints:
    sources:
      - filesystem
    destinations:
      - filesystem

storage:
  clients:
    filesystem:
      basedir: ./runs
```

## Start Training

Run the training script:

```bash
pipenv run python scripts/train.py config.yaml
```

## Monitor Training

Watch the training progress in the console output. Checkpoints will be saved to `./runs/`.

## Resume from Checkpoint

To resume training from a checkpoint:

```bash
pipenv run python scripts/train.py config.yaml --run-id <RUN_ID> --checkpoint-tag best
```

## Next Steps

- Learn about [Configuration](../user-guide/configuration.md)
- Explore [Model Building](../user-guide/model-building.md)
- Set up [Checkpointing](../user-guide/checkpointing.md)
