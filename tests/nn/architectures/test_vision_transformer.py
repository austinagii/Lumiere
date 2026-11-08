import pytest
import torch
from torch.nn import LayerNorm

from lumiere.nn.architectures import VisionTransformer
from lumiere.nn.builder import ModelBuilder
from lumiere.nn.components.attention import MultiHeadAttention
from lumiere.nn.components.blocks import StandardTransformerBlock
from lumiere.nn.components.embedding import PatchEmbedding
from lumiere.nn.components.feedforward import LinearFeedForward


IMAGE_SIZE = 16
PATCH_SIZE = 4
NUM_CHANNELS = 3
EMBEDDING_SIZE = 32
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 10
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2


@pytest.fixture
def model():
    return VisionTransformer(
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE,
        num_blocks=NUM_BLOCKS,
        embedding=lambda: PatchEmbedding(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_channels=NUM_CHANNELS,
            embedding_size=EMBEDDING_SIZE,
        ),
        block=lambda: StandardTransformerBlock(
            attention=lambda: MultiHeadAttention(
                num_heads=NUM_HEADS,
                embedding_size=EMBEDDING_SIZE,
                d_key=8,
                d_value=8,
                causal=False,
            ),
            feedforward=lambda: LinearFeedForward(EMBEDDING_SIZE, 64),
            normalization=lambda: LayerNorm(EMBEDDING_SIZE),
            dropout=0.0,
        ),
        normalization=lambda: LayerNorm(EMBEDDING_SIZE),
    )


class TestVisionTransformer:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_model_forward(self, model, batch_size):
        images = torch.randn(batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

        logits, attention_weights = model(images)

        assert logits.shape == (batch_size, NUM_CLASSES)
        assert attention_weights.shape == (
            batch_size,
            NUM_BLOCKS,
            NUM_HEADS,
            NUM_PATCHES + 1,
            NUM_PATCHES + 1,
        )

    def test_attention_is_bidirectional(self, model):
        images = torch.randn(2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

        _, attention_weights = model(images)

        # The class token must be able to attend to every patch.
        assert torch.all(attention_weights[:, :, :, 0, 1:] > 0)

    def test_gradients_flow_to_all_parameters(self, model):
        images = torch.randn(2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        targets = torch.randint(0, NUM_CLASSES, (2,))

        logits, _ = model(images)
        torch.nn.functional.cross_entropy(logits, targets).backward()

        assert all(p.grad is not None for p in model.parameters())

    def test_build_produces_a_vision_transformer_from_spec(self):
        spec = {
            "type": "vision_transformer",
            "num_classes": NUM_CLASSES,
            "embedding_size": EMBEDDING_SIZE,
            "num_blocks": NUM_BLOCKS,
            "embedding": {
                "type": "patch",
                "image_size": IMAGE_SIZE,
                "patch_size": PATCH_SIZE,
                "num_channels": NUM_CHANNELS,
            },
            "block": {
                "type": "standard",
                "attention": {
                    "type": "multihead",
                    "num_heads": NUM_HEADS,
                    "d_key": 8,
                    "d_value": 8,
                    "causal": False,
                },
                "feedforward": {"type": "linear", "d_ff": 64},
                "normalization": {"type": "layer", "normalized_shape": EMBEDDING_SIZE},
            },
            "normalization": {"type": "layer", "normalized_shape": EMBEDDING_SIZE},
        }

        model = ModelBuilder.build(spec)

        assert isinstance(model, VisionTransformer)
        assert isinstance(model.embedding, PatchEmbedding)
        assert model.embedding.embedding_size == EMBEDDING_SIZE
        assert len(model.blocks) == NUM_BLOCKS
        assert model.blocks[0].attention.causal is False

        logits, _ = model(torch.randn(2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        assert logits.shape == (2, NUM_CLASSES)

    @pytest.mark.parametrize(
        "args", [{"num_classes": 0}, {"embedding_size": 0}, {"num_blocks": 0}]
    )
    def test_raises_an_error_if_invalid_parameters(self, args):
        params = {
            "num_classes": NUM_CLASSES,
            "embedding_size": EMBEDDING_SIZE,
            "num_blocks": NUM_BLOCKS,
            "embedding": lambda: None,
            "block": lambda: None,
            "normalization": lambda: None,
            **args,
        }
        with pytest.raises(ValueError):
            VisionTransformer(**params)
