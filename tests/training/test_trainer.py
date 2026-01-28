from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import RMSNorm

from lumiere.components.attention import MultiHeadAttention
from lumiere.components.block import TransformerBlock
from lumiere.components.embedding import Embedding
from lumiere.components.feedforward import LinearFeedForward
from lumiere.data import Pipeline
from lumiere.data.dataset import DataLoader
from lumiere.data.datasets import WikiText2Dataset
from lumiere.data.preprocessing.preprocessors import (
    AutoregressiveLanguageModellingPreprocessor,
)
from lumiere.data.preprocessing.tokenizer import SPECIAL_TOKENS
from lumiere.data.preprocessing.tokenizers import BPETokenizer
from lumiere.models.transformer import Transformer
from lumiere.training import Trainer
from lumiere.training.schedulers import cosine_annealing_lr_scheduler
from lumiere.utils.device import get_device


VOCAB_SIZE = 512
EMBEDDING_SIZE = 16
CONTEXT_SIZE = 8
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def dataloader():
    return DataLoader.from_datasets([WikiText2Dataset("1:0:0")])


@pytest.fixture(scope="module")
def tokenizer(dataloader):
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, min_frequency=2)
    tokenizer.train(dataloader["train"])
    return tokenizer


@pytest.fixture(scope="module")
def pipeline(dataloader, tokenizer):
    return Pipeline(
        dataloader=dataloader,
        split="train",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE + 1,
        pad_id=SPECIAL_TOKENS["padding"].id,
        sliding_window_size=0,
    )


class IdentityPreprocessor:
    def __call__(self, *args, **kwargs):
        return args, kwargs


@pytest.fixture(scope="module")
def preprocessors(device):
    return [AutoregressiveLanguageModellingPreprocessor(device)]


@pytest.fixture
def model():
    model = Transformer(
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
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.scheduler = cosine_annealing_lr_scheduler(
        model.optimizer, warmup_steps=10, max_epochs=100, epoch_steps=10
    )
    return model


@pytest.fixture
def loss_fn():
    def _cross_entropy(logits, y):
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
def trainer(model, pipeline, preprocessors, loss_fn, device):
    return Trainer(
        model=model,
        pipeline=pipeline,
        preprocessors=preprocessors,
        loss_fn=loss_fn,
        max_epochs=3,
        stopping_threshold=1e-3,
        patience=3,
        gradient_clip_norm=1e-3,
        device=device,
    )


class TestFit:
    """Tests for the 'fit' method."""

    @pytest.mark.integration
    def test_fit_optimizes_model_parameters(self, trainer, device):
        model = MagicMock(wraps=trainer.model)
        model.optimizer = MagicMock(wraps=model.optimizer)
        model.scheduler = MagicMock(wraps=model.scheduler)
        trainer.model = model
        pre_train_model_params = [
            param.data.clone().to(device) for param in model.parameters()
        ]

        metrics = trainer.train()

        post_train_model_params = [
            param.data.clone().to(device) for param in model.parameters()
        ]

        # Check that at least one parameter has changed as a result of gradient descent.
        assert any(
            not torch.equal(pre, post)
            for pre, post in zip(
                pre_train_model_params, post_train_model_params, strict=False
            )
        )

        assert model.optimizer.step.call_count == metrics.global_step
        assert model.scheduler.step.call_count == metrics.global_step

    # def test_train_learning_rate_is_updated(self, model, _call_train):
    #     initial_learning_rate = model.scheduler.get_last_lr()[0]
    #
    #     _call_train()
    #
    #     assert model.scheduler.get_last_lr()[0] != initial_learning_rate
    #

    # def test_fit_outputs_fit_metrics(self):
    #     pass

    # def test_train_loss_is_calculated(self, _call_train):
    #     metrics = _call_train()
    #
    #     assert metrics.avg_loss != 0.0
    #     assert metrics.avg_perplexity != 0.0

    # @pytest.mark.integration
    # def test_train_gradients_are_clipped(self, mocker, _call_train):
    #     # TODO: Revisit this test. We should be able to verify that the gradients
    #     # are clipped without having to spy on the clip_grad_norm_ function.
    #     mocker.spy(torch.nn.utils, "clip_grad_norm_")
    #
    #     _call_train()
    #
    #     assert torch.nn.utils.clip_grad_norm_.call_count == 1

    # def test_train_current_lr_value_accuracy(self, model, _call_train, training_state):
    #     initial_lr = model.scheduler.get_last_lr()[0]
    #
    #     _call_train()
    #
    #     # Verify current_lr matches what scheduler reports
    #     assert training_state.current_lr == model.scheduler.get_last_lr()[0]
    #     # Verify LR actually changed (for cosine annealing scheduler)
    #     assert training_state.current_lr != initial_lr

    # def test_train_time_taken_is_accurate(self, _call_train):
    #     start_time = time()
    #
    #     training_state = _call_train()
    #
    #     end_time = time()
    #     time_taken = end_time - start_time
    #
    #     assert (
    #         abs(time_taken - training_state.time_taken) < 0.01
    #     )  # Allow 10ms tolerance

    # def test_progress_bar_displays_batch_stats(self):
    #     pass

    # def test_train_calls_optimizer_zero_grad_after_each_batch(
    #     self, mocker, model, _call_train
    # ):
    #     mocker.spy(model.optimizer, "zero_grad")
    #
    #     _call_train(max_num_batches=3)
    #
    #     assert model.optimizer.zero_grad.call_count == 3
    #
    # def test_train_calls_model_forward_with_correct_inputs(
    #     self, mocker, model, _call_train
    # ):
    #     mocker.spy(model, "forward")
    #
    #     _call_train()
    #
    #     assert model.forward.call_count == 1
    #     # Verify the model was called with x and padding_mask
    #     call_args = model.forward.call_args
    #     x, padding_mask = call_args[0]
    #     assert (
    #         x.shape[1] == CONTEXT_SIZE - 1
    #     )  # sequence length is shifted for next-token prediction
    #     assert padding_mask.shape == x.shape  # padding mask should match input shape
    #
    # def test_fit_raises_an_error_if_pipeline_batches_are_incorrectly_formatted(self):
    #     pass
    #
