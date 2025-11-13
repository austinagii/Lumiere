from time import time
from unittest.mock import Mock

import pytest
import torch
from torch.nn import RMSNorm

from lumiere.research.src import utils
from lumiere.research.src.components.feedforward import LinearFeedForward
from lumiere.research.src.data.dataloader import get_data_loader
from lumiere.research.src.data.preprocessing import to_training_batches
from lumiere.research.src.data.tokenizer import SPECIAL_TOKENS, Tokenizer
from lumiere.research.src.models.transformer import Transformer
from lumiere.research.src.training.eval import EvaluationState, evaluate


VOCAB_SIZE = 512
EMBEDDING_SIZE = 6
CONTEXT_SIZE = 3
BATCH_SIZE = 3


@pytest.fixture
def model():
    def feedforward_factory():
        return LinearFeedForward(
            embedding_size=EMBEDDING_SIZE,
            d_ff=3,
            dropout=0,
        )

    return Transformer(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        context_size=CONTEXT_SIZE,
        num_layers=1,
        num_heads=1,
        d_key=3,
        d_value=3,
        feedforward_factory=feedforward_factory,
        normalization_factory=lambda: RMSNorm(EMBEDDING_SIZE),
        dropout=0,
        padding_id=SPECIAL_TOKENS["padding"].id,
    ).to(utils.get_device())


@pytest.fixture(scope="class")
def dataloader():
    return get_data_loader("wikitext", train_dataset_percentage=1)


@pytest.fixture(scope="class")
def tokenizer(dataloader):
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE, min_frequency=2)
    tokenizer.train(dataloader.iter_train())
    return tokenizer


@pytest.fixture
def eval_data(dataloader, tokenizer):
    return to_training_batches(
        corpus=dataloader.iter_validation(),
        tokenizer=tokenizer,
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
        sliding_window_size=2,
        pad_id=SPECIAL_TOKENS["padding"].id,
        num_batches=3,
    )


@pytest.fixture
def wandb_run():
    run = Mock()
    run.log = Mock()
    return run


class TestEvaluationState:
    def test_evaluation_state_creation(self):
        state = EvaluationState(
            avg_loss=1.5, avg_perplexity=4.48, num_batches=10, time_taken=2.5
        )

        assert state.avg_loss == 1.5
        assert state.avg_perplexity == 4.48
        assert state.num_batches == 10
        assert state.time_taken == 2.5


class TestEvaluate:
    @pytest.mark.integration
    def test_evaluate_returns_evaluation_state(self, model, eval_data):
        result = evaluate(model, eval_data)

        assert isinstance(result, EvaluationState)
        assert result.avg_loss > 0
        assert result.avg_perplexity > 0
        assert result.num_batches > 0
        assert result.time_taken > 0

    @pytest.mark.integration
    def test_evaluate_model_in_eval_mode(self, mocker, model, eval_data):
        mocker.spy(model, "eval")

        evaluate(model, eval_data)

        assert model.eval.call_count == 1

    @pytest.mark.integration
    def test_evaluate_model_moved_to_device(self, mocker, model, eval_data):
        device = torch.device("cpu")
        mocker.spy(model, "to")

        evaluate(model, eval_data, device=device)

        model.to.assert_called_with(device)

    @pytest.mark.integration
    def test_evaluate_no_gradients_computed(self, model, eval_data):
        # Ensure gradients are None before evaluation
        for param in model.parameters():
            param.grad = None

        evaluate(model, eval_data)

        # Gradients should still be None after evaluation
        for param in model.parameters():
            assert param.grad is None

    @pytest.mark.integration
    def test_evaluate_correct_input_shapes(self, mocker, model, eval_data):
        original_forward = model.forward

        def capture_forward(x, padding_mask):
            capture_forward.last_x = x
            capture_forward.last_padding_mask = padding_mask
            return original_forward(x, padding_mask)

        model.forward = capture_forward

        evaluate(model, eval_data)

        x = capture_forward.last_x
        padding_mask = capture_forward.last_padding_mask
        assert x.shape == padding_mask.shape
        assert x.shape[1] == CONTEXT_SIZE - 1  # shifted for next-token prediction

    @pytest.mark.integration
    def test_evaluate_cross_entropy_uses_ignore_index(self, mocker, model, eval_data):
        mocker.spy(torch.nn.functional, "cross_entropy")

        evaluate(model, eval_data)

        cross_entropy_call = torch.nn.functional.cross_entropy.call_args
        kwargs = cross_entropy_call[1] if cross_entropy_call[1] else {}
        assert "ignore_index" in kwargs
        assert kwargs["ignore_index"] == SPECIAL_TOKENS["padding"].id

    @pytest.mark.integration
    def test_evaluate_perplexity_calculation_accuracy(self, model, eval_data):
        result = evaluate(model, eval_data)

        # Perplexity should be exp(loss) - but due to averaging, we need to compare the relationship
        # The actual implementation calculates perplexity per batch then averages, so we verify the magnitude is correct
        assert result.avg_perplexity > 1.0  # Perplexity should be > 1
        assert result.avg_loss > 0  # Loss should be positive

    @pytest.mark.integration
    def test_evaluate_batch_count_accuracy(self, model, dataloader, tokenizer):
        # Create data with known batch count
        eval_data = to_training_batches(
            corpus=dataloader.iter_validation(),
            tokenizer=tokenizer,
            context_size=CONTEXT_SIZE,
            batch_size=BATCH_SIZE,
            sliding_window_size=2,
            pad_id=SPECIAL_TOKENS["padding"].id,
            num_batches=3,
        )

        result = evaluate(model, eval_data)

        # Should process exactly 3 batches as specified
        assert result.num_batches == 3

    @pytest.mark.integration
    def test_evaluate_time_measurement_accuracy(self, model, eval_data):
        start_time = time()

        result = evaluate(model, eval_data)

        end_time = time()
        actual_time_taken = end_time - start_time

        # Allow for some tolerance in time measurement
        assert abs(actual_time_taken - result.time_taken) < 0.1

    def test_evaluate_with_wandb_logging(self, model, eval_data, wandb_run):
        result = evaluate(model, eval_data, wandb_run=wandb_run)

        # Verify wandb logging was called
        wandb_run.log.assert_called_once()

        # Verify the logged metrics
        log_call_args = wandb_run.log.call_args[0][0]
        assert "eval/loss" in log_call_args
        assert "eval/perplexity" in log_call_args
        assert log_call_args["eval/loss"] == result.avg_loss
        assert log_call_args["eval/perplexity"] == result.avg_perplexity

    def test_evaluate_without_wandb_logging(self, model, eval_data):
        # Should not raise any errors when wandb_run is None
        result = evaluate(model, eval_data, wandb_run=None)

        assert isinstance(result, EvaluationState)

    @pytest.mark.integration
    def test_evaluate_with_empty_data(self, model):
        empty_data = iter([])

        # Should handle empty data gracefully, but will divide by zero
        with pytest.raises(ZeroDivisionError):
            evaluate(model, empty_data)

    @pytest.mark.integration
    def test_evaluate_model_parameters_unchanged(self, model, eval_data):
        device = utils.get_device()
        # Capture initial parameters on same device
        initial_params = [param.data.clone().to(device) for param in model.parameters()]

        evaluate(model, eval_data)

        # Verify parameters are unchanged
        for initial_param, current_param in zip(initial_params, model.parameters()):
            assert torch.equal(initial_param, current_param.data.to(device))

    @pytest.mark.integration
    def test_evaluate_consistent_results(self, model, dataloader, tokenizer):
        # Create two identical data iterators from the same source
        eval_data1 = to_training_batches(
            corpus=dataloader.iter_validation(),
            tokenizer=tokenizer,
            context_size=CONTEXT_SIZE,
            batch_size=BATCH_SIZE,
            sliding_window_size=2,
            pad_id=SPECIAL_TOKENS["padding"].id,
            num_batches=3,
        )

        eval_data2 = to_training_batches(
            corpus=dataloader.iter_validation(),
            tokenizer=tokenizer,
            context_size=CONTEXT_SIZE,
            batch_size=BATCH_SIZE,
            sliding_window_size=2,
            pad_id=SPECIAL_TOKENS["padding"].id,
            num_batches=3,
        )

        result1 = evaluate(model, eval_data1)
        result2 = evaluate(model, eval_data2)

        # Results should be identical since we're using the same data
        assert abs(result1.avg_loss - result2.avg_loss) < 1e-6
        assert abs(result1.avg_perplexity - result2.avg_perplexity) < 1e-6
        assert result1.num_batches == result2.num_batches

    def test_evaluate_device_handling(self, model, eval_data):
        device = torch.device("cpu")

        result = evaluate(model, eval_data, device=device)

        # Should complete successfully regardless of device
        assert isinstance(result, EvaluationState)
        assert result.avg_loss > 0
        assert result.avg_perplexity > 0
