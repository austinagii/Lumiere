import itertools
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import RMSNorm
from torch.optim import SGD

from lumiere.components.attention import MultiHeadAttention
from lumiere.components.block import TransformerBlock
from lumiere.components.embedding import Embedding
from lumiere.components.feedforward import LinearFeedForward
from lumiere.data import DataLoader
from lumiere.data.datasets import WikiText2Dataset
from lumiere.data.pipeline import TextPipeline
from lumiere.data.preprocessing.preprocessors import (
    AutoregressiveLanguageModellingPreprocessor,
)
from lumiere.data.preprocessing.tokenizer import SPECIAL_TOKENS
from lumiere.data.preprocessing.tokenizers import BPETokenizer
from lumiere.models.transformer import Transformer
from lumiere.training import Trainer
from lumiere.training.schedulers import cosine_annealing_lr_scheduler
from lumiere.utils.device import get_device
from lumiere.utils.testing.datasets import IdentityDataset
from lumiere.utils.testing.models import IdentityModel
from lumiere.utils.testing.pipelines import IdentityPipeline


VOCAB_SIZE = 512
EMBEDDING_SIZE = 16
CONTEXT_SIZE = 8
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def dataloader():
    return DataLoader.from_datasets([WikiText2Dataset("1:1:0")])


@pytest.fixture(scope="module")
def tokenizer(dataloader):
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, min_frequency=2)
    tokenizer.train(dataloader["train"])
    return tokenizer


@pytest.fixture(scope="module")
def pipeline(dataloader, tokenizer, device):
    preprocessors = [AutoregressiveLanguageModellingPreprocessor(device)]
    return TextPipeline(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE + 1,
        pad_id=SPECIAL_TOKENS["padding"].id,
        sliding_window_size=0,
        preprocessors=preprocessors,
    )


@pytest.fixture
def model():
    return Transformer(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        num_blocks=1,
        embedding_factory=lambda: Embedding(
            vocab_size=VOCAB_SIZE,
            context_size=CONTEXT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            padding_id=SPECIAL_TOKENS["padding"].id,
        ),
        block_factory=lambda: TransformerBlock(
            attention_factory=lambda: MultiHeadAttention(
                num_heads=1, embedding_size=EMBEDDING_SIZE, d_key=3, d_value=3
            ),
            feedforward_factory=lambda: LinearFeedForward(
                embedding_size=EMBEDDING_SIZE, d_ff=3, dropout=0
            ),
            normalization_factory=lambda: RMSNorm(EMBEDDING_SIZE),
            dropout=0,
            pre_norm=True,
            post_norm=False,
        ),
        normalization_factory=lambda: RMSNorm(EMBEDDING_SIZE),
    )


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def scheduler(optimizer):
    return cosine_annealing_lr_scheduler(
        optimizer, warmup_steps=10, max_epochs=100, epoch_steps=10
    )


@pytest.fixture
def loss_fn():
    def _cross_entropy(outputs, y):
        logits, _ = outputs
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=SPECIAL_TOKENS["padding"].id,
        )

    return _cross_entropy


@pytest.fixture(scope="module")
def device():
    return get_device()


@pytest.fixture
def trainer_builder(model, dataloader, pipeline, loss_fn, optimizer, scheduler, device):
    def _build_trainer(max_epochs=None, loss_fn=loss_fn, patience=None):
        return Trainer(
            model=model,
            dataloader=dataloader,
            pipeline=pipeline,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            stopping_threshold=1e-3,
            patience=patience,
            gradient_clip_norm=1e-3,
            device=device,
        )

    return _build_trainer


def capture_inputs(storage):
    def _capture_inputs(module, inputs, outputs):
        storage.append(inputs)

    return _capture_inputs


class TestTrainer:
    """Tests for the 'Trainer' class."""

    @pytest.mark.integration
    def test_fit_optimizes_model_parameters(self, trainer_builder, device):
        trainer = trainer_builder(max_epochs=3)
        optimizer = MagicMock(wraps=trainer.optimizer)
        scheduler = MagicMock(wraps=trainer.scheduler)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        pre_train_model_params = [
            param.data.clone().to(device) for param in trainer.model.parameters()
        ]

        metrics = trainer.train()

        post_train_model_params = [
            param.data.clone().to(device) for param in trainer.model.parameters()
        ]

        # Check that at least one parameter has changed as a result of gradient descent.
        assert any(
            not torch.equal(pre, post)
            for pre, post in zip(
                pre_train_model_params, post_train_model_params, strict=False
            )
        )

        assert optimizer.step.call_count == metrics.global_step
        assert scheduler.step.call_count == metrics.global_step

    def test_train_trains_model_on_all_training_data(
        self, trainer_builder, model, dataloader, pipeline, device
    ):
        actual_batches = []
        trainer = trainer_builder(max_epochs=1)

        def _capture_model_inputs(module, args) -> None:
            nonlocal actual_batches
            actual_batches.append(args)

        model.register_forward_pre_hook(_capture_model_inputs, prepend=True)

        trainer.train()

        # TODO: Improve readability. 'actual_batches' does not contain full batches
        # but instead contains a tuple of model inputs. For comparison, the model
        # inputs should be extracted from the pipeline batches and then compared.
        assert all(
            torch.equal(expected_batch[0][0].to(device), actual_batch[0].to(device))
            and torch.equal(expected_batch[0][1].to(device), actual_batch[1].to(device))
            for expected_batch, actual_batch in zip(
                pipeline.batches(dataloader["train"]), actual_batches, strict=False
            )
        )

    def test_train_executes_the_specified_number_of_epochs(
        self, trainer_builder, dataloader, pipeline, device
    ):
        trainer = trainer_builder(max_epochs=3)
        actual_inputs = []
        trainer.model.register_forward_hook(capture_inputs(actual_inputs))

        trainer.train()

        expected_inputs = itertools.chain(
            (inputs for inputs, _ in pipeline.batches(dataloader["train"])),
            (inputs for inputs, _ in pipeline.batches(dataloader["validation"])),
        )

        # This condition holds true only for the current implementation of pipeline
        # where batch data is deterministic.
        assert all(
            torch.equal(actual_input[0], expected_input[0].to(device))
            and torch.equal(actual_input[1], expected_input[1].to(device))
            for actual_input, expected_input in zip(
                actual_inputs, itertools.cycle(expected_inputs)
            )
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "epochs_with_improvement,epochs_without_improvement", [(2, 3), (3, 5), (1, 2)]
    )
    def test_train_stops_after_specified_epochs_without_improvement(
        self,
        trainer_builder,
        epochs_with_improvement,
        epochs_without_improvement,
        device,
    ):
        improving_losses = list(
            reversed([i / 42 for i in range(1, epochs_with_improvement + 1)])
        )
        fixed_losses = list(
            itertools.repeat(improving_losses[-1], epochs_without_improvement)
        )
        scores = improving_losses + fixed_losses

        def _make_loss_fn(i):
            def _loss_fn(y_pred, y):
                return torch.tensor(
                    scores[i],
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )

            return _loss_fn

        trainer = trainer_builder(patience=epochs_without_improvement)

        def _update_loss_fn(trainer):
            trainer.loss_fn = _make_loss_fn(trainer.state.current_epoch - 1)

        trainer.register_pre_epoch_hook(_update_loss_fn)

        trainer.train()

        assert (
            trainer.state.current_epoch
            == epochs_with_improvement + epochs_without_improvement
        )

    def test_train_calculates_loss_on_all_batches_and_optimizes_weights(self):
        pass

    @pytest.mark.slow
    @pytest.mark.parametrize("max_epochs", [0, 3, 5])
    def test_train_stops_after_specified_epoch(self, trainer_builder, max_epochs):
        trainer = trainer_builder(max_epochs=max_epochs)

        trainer.train()

        assert trainer.state.current_epoch == max_epochs

    def test_validation_scores_are_calculated_correctly(self):
        dataset = IdentityDataset(
            {
                "train": [
                    (
                        torch.tensor(
                            [
                                [0.4025, 0.7228, 0.4655, 0.6313, 0.4666],
                                [0.2656, 0.3988, 0.1337, 0.0449, 0.9695],
                                [0.8781, 0.1952, 0.7985, 0.1888, 0.5421],
                            ],
                            requires_grad=True,
                            dtype=torch.float32,
                        ),
                        None,
                    )
                ],
                "validation": [
                    (
                        torch.tensor(
                            [
                                [0.5100, 0.5910, 0.2581, 0.9480, 0.2824],
                                [0.2154, 0.8358, 0.9051, 0.2634, 0.0235],
                                [0.8681, 0.3576, 0.2209, 0.6767, 0.2125],
                            ],
                            requires_grad=True,
                            dtype=torch.float32,
                        ),
                        None,
                    )
                ],
            }
        )
        dataloader = DataLoader.from_datasets([dataset])
        expected_train_loss = torch.tensor([7.1039], dtype=torch.float32)
        expected_validation_loss = torch.tensor([7.1686], dtype=torch.float32)

        pipeline = IdentityPipeline()
        model = IdentityModel()
        optimizer = SGD(model.parameters())

        def sum_loss(outputs, y):
            return outputs.sum()

        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            pipeline=pipeline,
            loss_fn=sum_loss,
            optimizer=optimizer,
            scheduler=None,
            max_epochs=3,
            device=get_device(),
        )

        def assert_close_tensors(x, y):
            assert torch.allclose(x, y, atol=1e-3)

        trainer.register_post_fit_hook(
            lambda trainer, train_metrics: assert_close_tensors(
                train_metrics.avg_loss, expected_train_loss
            )
        )

        trainer.register_post_eval_hook(
            lambda trainer, eval_metrics: assert_close_tensors(
                eval_metrics.avg_loss, expected_validation_loss
            )
        )

        trainer.train()

    def test_train_raises_error_if_epoch_invalid(self):
        pass

    def test_train_optionally_clips_gradient(self):
        pass

    def test_train_learning_rate_is_updated(self, model):
        pass

    def test_train_gradients_are_clipped(self, mocker):
        pass

    def test_train_current_lr_value_accuracy(self, model):
        pass

    def test_train_time_taken_is_accurate(self):
        pass

    def test_progress_bar_displays_batch_stats(self):
        pass

    def test_train_zeros_gradients_after_each_batch(self, mocker, model):
        pass

    def test_fit_raises_an_error_if_pipeline_batches_are_incorrectly_formatted(self):
        pass

    def test_train_hooks_execute_at_correct_steps(self):
        pass

    def test_raises_an_error_if_invalid_parameters(self):
        pass

    def test_raises_error_if_required_args_not_provided(self):
        pass

    def test_epoch_in_state_is_accurate(self):
        pass
