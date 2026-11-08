import pytest
import torch

from lumiere.data.pipelines.image import ImagePipeline


def _samples(count, num_channels=3, size=4):
    return [
        (torch.full((num_channels, size, size), i % 256, dtype=torch.uint8), i % 10)
        for i in range(count)
    ]


class TestImagePipeline:
    @pytest.mark.parametrize("count,batch_size", [(8, 4), (10, 4), (3, 4)])
    def test_images_are_grouped_into_batches(self, count, batch_size):
        pipeline = ImagePipeline(batch_size=batch_size)

        batches = list(pipeline.batches(_samples(count)))

        sizes = [images.shape[0] for images, _ in batches]
        assert sum(sizes) == count
        assert all(size == batch_size for size in sizes[:-1])
        for images, labels in batches:
            assert images.shape[1:] == (3, 4, 4)
            assert images.dtype == torch.float32
            assert labels.shape == (images.shape[0],)
            assert labels.dtype == torch.long

    def test_labels_are_preserved_in_order(self):
        pipeline = ImagePipeline(batch_size=3)

        labels = torch.cat([y for _, y in pipeline.batches(_samples(7))])

        assert labels.tolist() == [i % 10 for i in range(7)]

    def test_integer_pixels_are_scaled_to_unit_range(self):
        pipeline = ImagePipeline(batch_size=1)
        image = torch.tensor([[[0, 255], [51, 102]]], dtype=torch.uint8)

        images, _ = next(pipeline.batches([(image, 0)]))

        assert torch.allclose(images, torch.tensor([[[[0.0, 1.0], [0.2, 0.4]]]]))

    def test_floating_point_pixels_are_not_rescaled(self):
        pipeline = ImagePipeline(batch_size=1)
        image = torch.full((1, 2, 2), 0.5)

        images, _ = next(pipeline.batches([(image, 0)]))

        assert torch.allclose(images, torch.full((1, 1, 2, 2), 0.5))

    def test_images_are_normalized_per_channel(self):
        pipeline = ImagePipeline(batch_size=1, mean=[0.0, 0.5], std=[1.0, 0.25])
        image = torch.full((2, 2, 2), 1.0)

        images, _ = next(pipeline.batches([(image, 0)]))

        assert torch.allclose(images[0, 0], torch.full((2, 2), 1.0))
        assert torch.allclose(images[0, 1], torch.full((2, 2), 2.0))

    def test_stops_after_specified_number_of_batches(self):
        pipeline = ImagePipeline(batch_size=2)

        batches = list(pipeline.batches(_samples(10), num_batches=2))

        assert len(batches) == 2

    def test_preprocessors_are_applied_in_order(self):
        def _add_one(images, labels):
            return images + 1, labels

        def _double(images, labels):
            return images * 2, labels

        pipeline = ImagePipeline(batch_size=1, preprocessors=[_add_one, _double])
        image = torch.zeros((1, 2, 2))

        images, _ = next(pipeline.batches([(image, 0)]))

        assert torch.allclose(images, torch.full((1, 1, 2, 2), 2.0))

    @pytest.mark.parametrize(
        "args",
        [
            {"batch_size": 0},
            {"std": [0.0, 1.0, 1.0]},
            {"mean": [0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        ],
    )
    def test_raises_an_error_if_invalid_parameters(self, args):
        with pytest.raises(ValueError):
            ImagePipeline(**{"batch_size": 2, **args})

    def test_raises_an_error_if_normalization_does_not_match_channels(self):
        pipeline = ImagePipeline(batch_size=1, mean=[0.5, 0.5])

        with pytest.raises(ValueError):
            next(pipeline.batches(_samples(1, num_channels=3)))
