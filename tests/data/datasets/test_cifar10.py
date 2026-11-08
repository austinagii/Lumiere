import datasets
import numpy as np
import pytest
import torch

from lumiere.data.datasets.cifar10 import CLASS_NAMES, CIFAR10Dataset


def _fake_split(num_samples: int, label_offset: int = 0) -> datasets.Dataset:
    rng = np.random.default_rng(0)
    return datasets.Dataset.from_dict(
        {
            "img": [
                rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                for _ in range(num_samples)
            ],
            "label": [(i + label_offset) % 10 for i in range(num_samples)],
        }
    )


@pytest.fixture
def mock_load_dataset(mocker):
    def _load_dataset(dataset_id, split, revision):
        return [_fake_split(4, label_offset=i) for i, _ in enumerate(split)]

    return mocker.patch(
        "lumiere.data.datasets.cifar10.datasets.load_dataset",
        side_effect=_load_dataset,
    )


class TestCIFAR10Dataset:
    def test_samples_are_channel_first_images_and_integer_labels(
        self, mock_load_dataset
    ):
        dataset = CIFAR10Dataset()

        samples = list(dataset["train"])

        assert len(samples) == 4
        for image, label in samples:
            assert image.shape == (3, 32, 32)
            assert image.dtype == torch.uint8
            assert isinstance(label, int)
            assert 0 <= label < len(CLASS_NAMES)

    def test_validation_split_is_held_out_from_the_end_of_the_train_split(
        self, mock_load_dataset
    ):
        CIFAR10Dataset()

        split = mock_load_dataset.call_args.kwargs["split"]
        assert split == ["train[:45000]", "train[45000:50000]", "test[:100%]"]

    def test_split_percentages_are_applied_to_each_split(self, mock_load_dataset):
        CIFAR10Dataset("10:50:20")

        split = mock_load_dataset.call_args.kwargs["split"]
        assert split == ["train[:4500]", "train[45000:47500]", "test[:20%]"]

    def test_omitted_splits_are_not_loaded(self, mock_load_dataset):
        dataset = CIFAR10Dataset("100:0:0")

        split = mock_load_dataset.call_args.kwargs["split"]
        assert split == ["train[:45000]"]
        with pytest.raises(KeyError):
            dataset["validation"]

    def test_raises_an_error_if_all_splits_are_empty(self, mock_load_dataset):
        with pytest.raises(ValueError):
            CIFAR10Dataset("0:0:0")

    def test_raises_an_error_if_split_is_not_available(self, mock_load_dataset):
        dataset = CIFAR10Dataset()

        with pytest.raises(KeyError):
            dataset["unknown"]
