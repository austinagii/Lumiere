import pytest
import torch

from lumiere.nn.components.embedding import (
    PatchEmbedding,
    sinusoidal_positional_encoding,
)


class TestPatchEmbedding:
    @pytest.mark.parametrize("image_size,patch_size", [(32, 8), (32, 16), (28, 7)])
    @pytest.mark.parametrize("num_channels", [1, 3])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_patch_embeddings_have_correct_shape_and_dtype(
        self, image_size, patch_size, num_channels, batch_size
    ):
        embedding = PatchEmbedding(image_size, patch_size, num_channels, 16)
        images = torch.randn(batch_size, num_channels, image_size, image_size)

        output = embedding(images)

        num_patches = (image_size // patch_size) ** 2
        assert embedding.num_patches == num_patches
        assert output.shape == (batch_size, num_patches + 1, 16)
        assert output.dtype == torch.float32

    def test_class_token_is_prepended_to_each_sequence(self):
        embedding = PatchEmbedding(8, 4, 3, 8)
        images = torch.randn(2, 3, 8, 8)

        output = embedding(images)

        expected = (embedding.class_token + embedding.positional_embedding[:, :1])[0, 0]
        assert torch.allclose(output[0, 0], expected)
        assert torch.allclose(output[1, 0], expected)

    def test_patches_are_embedded_independently(self):
        embedding = PatchEmbedding(8, 4, 1, 8)
        images = torch.randn(1, 1, 8, 8)
        modified = images.clone()
        # Modify only the bottom right patch.
        modified[:, :, 4:, 4:] += 1.0

        output = embedding(images)
        modified_output = embedding(modified)

        assert torch.allclose(output[:, :4], modified_output[:, :4])
        assert not torch.allclose(output[:, 4], modified_output[:, 4])

    def test_learned_positional_embeddings_are_trainable(self):
        embedding = PatchEmbedding(8, 4, 3, 8)

        assert any(p is embedding.positional_embedding for p in embedding.parameters())

    def test_sinusoidal_positional_encodings_are_fixed(self):
        embedding = PatchEmbedding(8, 4, 3, 8, learned_positions=False)

        assert all(
            p is not embedding.positional_embedding for p in embedding.parameters()
        )
        assert torch.equal(
            embedding.positional_embedding[0], sinusoidal_positional_encoding(5, 8)
        )

    @pytest.mark.parametrize(
        "args",
        [
            {"image_size": 0},
            {"patch_size": 0},
            {"num_channels": 0},
            {"embedding_size": 0},
            {"image_size": 30, "patch_size": 8},
            {"dropout": 1.5},
            {"learned_positions": "yes"},
        ],
    )
    def test_raises_an_error_if_invalid_parameters(self, args):
        params = {
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "embedding_size": 16,
            **args,
        }
        with pytest.raises(ValueError):
            PatchEmbedding(**params)

    @pytest.mark.parametrize(
        "shape", [(2, 3, 16, 16), (2, 1, 32, 32), (3, 32, 32), (2, 3, 32, 16)]
    )
    def test_raises_an_error_if_images_have_incorrect_shape(self, shape):
        embedding = PatchEmbedding(32, 8, 3, 16)

        with pytest.raises(ValueError):
            embedding(torch.randn(*shape))
