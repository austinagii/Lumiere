import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.nn import Flatten, Linear, Module, ReLU, Sequential
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import lumiere.deepscale as ds
from lumiere.deepscale.config import Config
from lumiere.deepscale.run import Checkpoint, generate_run_id


class Model(Module):
    def __init__(self, layer_sizes, learning_rate=0.001) -> None:
        super().__init__()

        # Ensure input is flattened before passing to linear layers.
        self.layers = Sequential(Flatten())

        # Add the linear layers based on the specified layer sizes.
        layer_sizes.insert(
            0, 784
        )  # Add the input size to ensure first layer is initialized correctly.
        for curr_layer_ix in range(1, len(layer_sizes)):
            prev_layer_ix = curr_layer_ix - 1

            self.layers.append(
                Linear(layer_sizes[prev_layer_ix], layer_sizes[curr_layer_ix])
            )
            if curr_layer_ix < len(layer_sizes) - 1:
                self.layers.append(ReLU())

        self.optimizer = SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.layers(x)

    def step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()


def train(model, data):
    num_batches = 0
    total_loss = 0

    for x, y in data:
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        model.step()
        num_batches += 1
        total_loss += loss.item()

    return total_loss / num_batches


@pytest.fixture
def ds_config_path(tmp_path):
    config = """
    runs:
        checkpoints:
            sources:
                - filesystem 
            destinations:
                - filesystem 
    """

    config_path = tmp_path / "deepscale.yaml"
    config_path.write_text(config)
    return config_path


@pytest.fixture
def run_config():
    config = """
    training:
        max_epochs: 10
    model:
        layer_sizes: [128, 128, 10]
    optimizer:
        learning_rate: 0.9
    """

    return yaml.safe_load(config)


@pytest.fixture
def data():
    return DataLoader(
        datasets.FashionMNIST(
            root="data", train=True, download=True, transform=ToTensor()
        ),
        batch_size=64,
    )


@pytest.fixture
def paused_run(tmp_path, run_config):
    run_id = generate_run_id()
    run_base_path = tmp_path / f"runs/{run_id}"
    run_base_path.mkdir(parents=True, exist_ok=True)

    # Save a run config..
    run_config_path = run_base_path / "config.yaml"
    run_config_path.write_text(yaml.dump(run_config))

    # Save a lightweight checkpoint.
    model = Model(
        layer_sizes=run_config["model"]["layer_sizes"],
        learning_rate=run_config["optimizer"]["learning_rate"],
    )
    checkpoint = ds.Checkpoint(
        current_epoch=3, model_state=model.state_dict(), loss=1.68
    )
    checkpoint_tag = "epoch:3"
    checkpoint_dir = run_base_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{checkpoint_tag}.pt"
    checkpoint_path.write_bytes(bytes(checkpoint))

    return run_id, checkpoint_tag


class TestDeepScale:
    # TODO: Add checkpoint run cleanup after each test.
    @pytest.mark.integration
    def test_new_run_can_be_started(self, ds_config_path, tmp_path, run_config, data):
        Config.from_yaml(ds_config_path, override=True)
        Config.get_instance()["storage.clients.filesystem.basedir"] = tmp_path

        _, run_manager = ds.init_run(run_config)
        model = Model(
            layer_sizes=run_config["model"]["layer_sizes"],
            learning_rate=run_config["optimizer"]["learning_rate"],
        )

        for epoch in range(10):
            avg_loss = train(model, data)

            checkpoint = Checkpoint(
                epoch=epoch, loss=avg_loss, model=model.state_dict()
            )
            run_manager.save_checkpoint(ds.CheckpointType.EPOCH, checkpoint)

    @pytest.mark.integration
    def test_previous_training_run_can_be_resumed(
        self, tmp_path, ds_config_path, paused_run, data
    ):
        Config.from_yaml(ds_config_path, override=True)
        Config.get_instance()["storage.clients.filesystem.basedir"] = tmp_path

        run_config, checkpoint, run_manager = ds.resume_run(*paused_run)

        model = Model(
            layer_sizes=run_config["model"]["layer_sizes"],
            learning_rate=run_config["optimizer"]["learning_rate"],
        )
        model.load_state_dict(checkpoint.model_state)

        for epoch in range(
            checkpoint.current_epoch, run_config["training"]["max_epochs"]
        ):
            avg_loss = train(model, data)

            checkpoint = Checkpoint(
                epoch=epoch, loss=avg_loss, model=model.state_dict()
            )
            run_manager.save_checkpoint(ds.CheckpointType.EPOCH, checkpoint)

    @pytest.mark.integration
    def test_previous_training_run_can_be_resumed_after_saving(
        self, tmp_path, ds_config_path, run_config, data
    ):
        Config.from_yaml(ds_config_path, override=True)
        Config.get_instance()["storage.clients.filesystem.basedir"] = tmp_path

        run_id, run_manager = ds.init_run(run_config)

        model = Model(
            layer_sizes=run_config["model"]["layer_sizes"],
            learning_rate=run_config["optimizer"]["learning_rate"],
        )

        # Do some initial training before we pause training.
        for epoch in range(5):
            avg_loss = train(model, data)

        checkpoint = Checkpoint(epoch=epoch, loss=avg_loss, model=model.state_dict())
        run_manager.save_checkpoint(ds.CheckpointType.EPOCH, checkpoint)

        # Now resume the saved run.
        loaded_run_config, loaded_checkpoint, run_manager = ds.resume_run(
            run_id, "epoch:0004"
        )

        loaded_model = Model(
            layer_sizes=loaded_run_config["model"]["layer_sizes"],
            learning_rate=loaded_run_config["optimizer"]["learning_rate"],
        )
        loaded_model.load_state_dict(loaded_checkpoint.model)

        assert run_config == loaded_run_config
        assert loaded_checkpoint.epoch == checkpoint.epoch
        assert loaded_checkpoint.loss == checkpoint.loss

        # Verify that model parameters are identical after loading
        for (name, original_param), (loaded_name, loaded_param) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name == loaded_name, (
                f"Parameter names don't match: {name} != {loaded_name}"
            )
            assert torch.allclose(original_param, loaded_param), (
                f"Parameter {name} values don't match"
            )

        for epoch in range(loaded_checkpoint.epoch, 10):
            avg_loss = train(model, data)

        checkpoint = Checkpoint(epoch=epoch, loss=avg_loss, model=model.state_dict())
        run_manager.save_checkpoint(ds.CheckpointType.FINAL, checkpoint)
