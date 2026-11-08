# Data Pipeline

Learn how to configure and use data pipelines in Lumière.

## Pipeline Components

The data pipeline consists of:

1. **Datasets**: Raw data sources
2. **DataLoader**: Manages multiple datasets
3. **Tokenizer**: Converts text to tokens (text data only)
4. **Pipeline**: Processes and batches data
5. **Preprocessors**: Apply transformations

The `tokenizer` section is only required for text data. When it is omitted, no
tokenizer is trained or saved for the run.

## Configuration

### Text

Text data is tokenized and packed into fixed-length context windows:

```yaml
data:
  datasets:
    - name: wikitext
      split: "80:100:100"  # train:validation:test percentages

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

### Images

Labelled image data is batched and normalized without a tokenizer:

```yaml
data:
  datasets:
    - name: cifar10
      split: "100:100:0"  # train:validation:test percentages

pipeline:
  name: image
  batch_size: 128
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2470, 0.2435, 0.2616]
```

## Available Datasets

- `wikitext`: WikiText-2 dataset. Each sample is a complete Wikipedia article.
- `cifar10`: CIFAR-10 dataset. Each sample is a tuple of a `uint8` image of shape
  `(3, 32, 32)` and an integer class label in the range [0, 10). CIFAR-10 has no
  validation split, so the last 5,000 images of the original train split are held out
  for validation, leaving 45,000 training images.

All datasets accept a `split` of colon-separated percentages in the format
`"train:validation:test"`. Omitted values default to 100%, and a split set to 0 is
not loaded.

## Text Pipeline

The text pipeline handles:

- Tokenization
- Batching
- Padding
- Sliding window
- Autoregressive preprocessing

## Image Pipeline

The image pipeline handles:

- Batching `(image, label)` samples into `(images, labels)` batches, with images of
  shape `(batch_size, num_channels, height, width)` and labels of shape
  `(batch_size,)`
- Scaling integer pixel values to the range [0, 1]
- Per-channel normalization using `mean` and `std` (both optional)
- Optional preprocessors, applied to each `(images, labels)` batch in order

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
