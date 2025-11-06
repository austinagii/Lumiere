from time import time

import pytest
import torch

from lumiere.research.src import utils
from lumiere.research.src.components.feedforward import LinearFeedForward
from lumiere.research.src.data.dataloader import get_data_loader
from lumiere.research.src.data.preprocessing import to_training_batches
from lumiere.research.src.data.tokenizer import SPECIAL_TOKENS, Tokenizer
from lumiere.research.src.models.transformer import Transformer
from lumiere.research.src.training import train
from lumiere.research.src.training.schedulers import cosine_annealing_lr_scheduler


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
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture()
def scheduler(optimizer):
    return cosine_annealing_lr_scheduler(
        optimizer, warmup_steps=10, max_epochs=100, epoch_steps=10
    )


@pytest.fixture
def wandb_run(mocker):
    run = mocker.Mock()
    run.log = mocker.Mock()
    return run


@pytest.fixture
def _call_train(model, tokenizer, optimizer, scheduler, dataloader, wandb_run):
    def _train(data=None, max_num_batches=1, log_interval=50):
        batches = to_training_batches(
            corpus=data if data else dataloader.iter_train(),
            tokenizer=tokenizer,
            context_size=CONTEXT_SIZE,
            batch_size=BATCH_SIZE,
            sliding_window_size=2,
            pad_id=SPECIAL_TOKENS["padding"].id,
            num_batches=max_num_batches,
        )
        return train(
            wandb_run=wandb_run,
            model=model,
            data=batches,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=1.0,
            global_step=0,
            wandb_log_interval=log_interval,
        )

    return _train


class TestTrain:
    @pytest.mark.integration
    def test_train_iterates_through_entire_dataset_if_no_max_num_batches(
        self, dataloader, _call_train
    ):
        data = dataloader.iter_train()

        _call_train(data=data, max_num_batches=None)

        with pytest.raises(StopIteration):
            next(data)

    @pytest.mark.integration
    def test_train_updates_model_parameters(
        self, model, tokenizer, dataloader, _call_train
    ):
        device = utils.get_device()
        pre_train_params = [
            param.data.clone().to(device) for param in model.parameters()
        ]

        _call_train()

        post_train_params = [
            param.data.clone().to(device) for param in model.parameters()
        ]

        # Check that at least one parameter has changed as a result of training.
        assert any(
            [
                not torch.equal(pre_train_param, post_train_param)
                for pre_train_param, post_train_param in zip(
                    pre_train_params, post_train_params
                )
            ]
        )

    @pytest.mark.integration
    def test_train_gradients_are_clipped(self, mocker, _call_train):
        # TODO: Revisit this test. We should be able to verify that the gradients
        # are clipped without having to spy on the clip_grad_norm_ function.
        mocker.spy(torch.nn.utils, "clip_grad_norm_")

        _call_train()

        assert torch.nn.utils.clip_grad_norm_.call_count == 1

    def test_train_loss_is_calculated(self, _call_train):
        training_state = _call_train()

        assert training_state.avg_loss != 0.0
        assert training_state.avg_perplexity != 0.0

    def test_train_steps_are_updated_correctly(self, _call_train):
        training_state = _call_train(max_num_batches=50)

        assert training_state.global_step == 50
        assert training_state.num_batches == 50

    def test_train_learning_rate_is_updated(self, scheduler, _call_train):
        initial_learning_rate = scheduler.get_last_lr()[0]

        _call_train()

        assert scheduler.get_last_lr()[0] != initial_learning_rate

    def test_train_calls_optimizer_step(self, optimizer, _call_train):
        # Track calls by monkey-patching the instance method
        original_step = optimizer.step
        call_count = 0

        def counting_step(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_step(*args, **kwargs)

        optimizer.step = counting_step

        _call_train()

        assert call_count == 1

    def test_train_calls_scheduler_step(self, mocker, scheduler, _call_train):
        mocker.spy(scheduler, "step")

        _call_train()

        assert scheduler.step.call_count == 1

    def test_train_calls_optimizer_zero_grad_after_each_batch(
        self, mocker, optimizer, _call_train
    ):
        mocker.spy(optimizer, "zero_grad")

        _call_train(max_num_batches=3)

        assert optimizer.zero_grad.call_count == 3

    def test_train_calls_model_forward_with_correct_inputs(
        self, mocker, model, _call_train
    ):
        mocker.spy(model, "forward")

        _call_train()

        assert model.forward.call_count == 1
        # Verify the model was called with x and padding_mask
        call_args = model.forward.call_args
        x, padding_mask = call_args[0]
        assert (
            x.shape[1] == CONTEXT_SIZE - 1
        )  # sequence length is shifted for next-token prediction
        assert padding_mask.shape == x.shape  # padding mask should match input shape

    def test_train_cross_entropy_uses_ignore_index_correctly(self, mocker, _call_train):
        mocker.spy(torch.nn.functional, "cross_entropy")

        _call_train()

        cross_entropy_call = torch.nn.functional.cross_entropy.call_args
        kwargs = cross_entropy_call[1] if cross_entropy_call[1] else {}
        assert "ignore_index" in kwargs
        assert kwargs["ignore_index"] == SPECIAL_TOKENS["padding"].id

    def test_train_perplexity_calculation_accuracy(self, _call_train):
        training_state = _call_train()

        # Perplexity should be exp(loss)
        expected_perplexity = torch.exp(torch.tensor(training_state.avg_loss)).item()
        assert abs(training_state.avg_perplexity - expected_perplexity) < 0.001

    def test_train_input_output_flow_correctness(self, mocker, model, _call_train):
        original_forward = model.forward

        def capture_forward(x, padding_mask):
            # Capture the inputs to verify they're correct
            capture_forward.last_x = x
            capture_forward.last_padding_mask = padding_mask
            return original_forward(x, padding_mask)

        model.forward = capture_forward

        _call_train()

        # Verify x and padding_mask shapes are consistent
        x = capture_forward.last_x
        padding_mask = capture_forward.last_padding_mask
        assert x.shape == padding_mask.shape
        assert x.shape[0] <= BATCH_SIZE
        assert (
            x.shape[1] == CONTEXT_SIZE - 1
        )  # sequence length is shifted for next-token prediction

    def test_train_current_lr_value_accuracy(self, scheduler, _call_train):
        initial_lr = scheduler.get_last_lr()[0]

        training_state = _call_train()

        # Verify current_lr matches what scheduler reports
        assert training_state.current_lr == scheduler.get_last_lr()[0]
        # Verify LR actually changed (for cosine annealing scheduler)
        assert training_state.current_lr != initial_lr

    @pytest.mark.parametrize(
        ("log_interval", "num_epochs", "expected_log_count"),
        [
            (3, 2, 0),
            (3, 3, 1),
            (3, 5, 1),
            (3, 6, 2),
        ],
    )
    def test_train_logs_metrics_every_log_interval_steps(
        self,
        log_interval,
        num_epochs,
        expected_log_count,
        wandb_run,
        _call_train,
    ):
        _call_train(max_num_batches=num_epochs, log_interval=log_interval)

        assert wandb_run.log.call_count == expected_log_count

    def test_train_time_taken_is_accurate(self, _call_train):
        start_time = time()

        training_state = _call_train()

        end_time = time()
        time_taken = end_time - start_time

        assert (
            abs(time_taken - training_state.time_taken) < 0.01
        )  # Allow 10ms tolerance
