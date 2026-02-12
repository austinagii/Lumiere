# Data Pipeline

Learn how to configure and use data pipelines in Lumière.

## Pipeline Components

The data pipeline consists of:

1. **Datasets**: Raw data sources
2. **DataLoader**: Manages multiple datasets
3. **Tokenizer**: Converts text to tokens
4. **Pipeline**: Processes and batches data
5. **Preprocessors**: Apply transformations

## Configuration

```yaml
data:
  datasets:
    - name: wikitext
      split: "train:80%,validation:10%,test:10%"

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
  sliding_window_size: 8
  preprocessors:
    - name: autoregressive
      device: mps
```

## Available Datasets

- `wikitext`: WikiText-2 dataset

## Text Pipeline

The text pipeline handles:

- Tokenization
- Batching
- Padding
- Sliding window
- Autoregressive preprocessing

## Custom Datasets

Create custom datasets by implementing the `Dataset` protocol:

```python
from lumiere.internal.registry import discover

@discover("dataset", "custom")
class CustomDataset:
    def __getitem__(self, split: str):
        # Return iterator for the split
        pass
```

## See Also

- [Configuration](configuration.md)
- [Training](training.md)
