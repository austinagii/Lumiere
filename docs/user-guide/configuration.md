# Configuration

Lumière uses YAML configuration files to define models, training parameters, and infrastructure settings.

## Configuration Structure

A complete configuration file has the following sections:

### Model Configuration

Define your transformer architecture:

```yaml
model:
  vocab_size: 4096
  context_size: 64
  embedding_size: 128
  num_blocks: 4
  embedding:
    type: embedding
    name: sinusoidal
  block:
    type: block
    name: standard
    # ... block configuration
```

### Data Configuration

Configure datasets and data loading:

```yaml
data:
  datasets:
    - name: wikitext
      split: "train:80%,validation:10%,test:10%"
```

### Tokenizer Configuration

Set up tokenization:

```yaml
tokenizer:
  name: bpe
  vocab_size: 4096
  min_frequency: 2
```

### Pipeline Configuration

Configure data preprocessing:

```yaml
pipeline:
  name: text
  tokenizer: "@tokenizer"
  batch_size: 32
  context_size: 64
  pad_id: 2
  sliding_window_size: 8
```

### Optimizer Configuration

Configure optimization:

```yaml
optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01
```

### Training Configuration

Set training parameters:

```yaml
training:
  max_epochs: 250
  stopping_threshold: 0.0001
  patience: 5
  gradient_clip_norm: 1.0
```

### Runs and Storage

Configure checkpointing:

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

## Dependency Injection

Use `@variable` syntax to inject dependencies:

```yaml
pipeline:
  tokenizer: "@tokenizer"  # Injects the tokenizer instance
```

## See Also

- [Model Building](model-building.md)
- [Training](training.md)
- [Checkpointing](checkpointing.md)
