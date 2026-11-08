# Templates

A template is a reusable training configuration that is referenced by name rather
than by path. Templates are useful for configurations you train often, such as a
baseline model that experiments are compared against.

Lumière does not ship with pre-built templates. You create your own.

## Creating a Template

Save a [configuration](configuration.md) file to the `templates/` directory at the
root of the repository. The file name, without the `.yaml` extension, becomes the
template name:

```text
lumiere/
├── lumiere/
├── templates/
│   └── lumiere-1m.yaml   # Template name: lumiere-1m
└── ...
```

## Using a Template

### From Python

Load a template with `load_template` and pass the resulting config to `start`:

```python
import lumiere

config = lumiere.load_template("lumiere-1m")
lumiere.start(config, name="baseline")
```

### From the Command Line

Pass the template name with `--template`:

```bash
python -m lumiere start --template lumiere-1m --name baseline
```

Use `--set KEY=VALUE` to replace individual values without editing the template.
The option can be repeated:

```bash
python -m lumiere start --template lumiere-1m \
  --set optimizer.lr=0.001 \
  --set pipeline.batch_size=64
```

!!! note
    `--template` and `--spec` are mutually exclusive. Use `--spec <path>` to train
    from a configuration file outside the `templates/` directory.

## Example: Lumière-1M

Lumière-1M is a small decoder-only transformer with roughly 1M parameters. It
trains on WikiText and is useful as a fast baseline for local experiments.

| Property          | Value                   |
| ----------------- | ----------------------- |
| Blocks            | 4                       |
| Embedding size    | 128                     |
| Attention heads   | 4 (`d_key = d_value = 32`) |
| Feed-forward size | 256                     |
| Context size      | 64 tokens               |
| Vocabulary        | 4096 (BPE)              |
| Normalization     | RMSNorm, pre-norm       |
| Positional encoding | Sinusoidal            |

To use it as a template, copy
[`examples/lumiere-1m.yaml`](https://github.com/austinagii/Lumiere/blob/main/examples/lumiere-1m.yaml)
into `templates/`:

```bash
mkdir -p templates
cp examples/lumiere-1m.yaml templates/
```

The full configuration:

```yaml
--8<-- "examples/lumiere-1m.yaml"
```

## See Also

- [Configuration](configuration.md)
- [Training](training.md)
