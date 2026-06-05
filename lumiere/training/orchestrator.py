import logging
import time
from typing import Any

from lumiere import Loader, register_dependency

from .artifact import ArtifactStore
from .checkpoint import Checkpoint, CheckpointStore, CheckpointTag
from .config import Config
from .errors import CheckpointNotFoundError, RunNotFoundError
from .event import EventStore
from .loss import cross_entropy_loss
from .run import Run, RunRepository, RunStatus
from .trainer import Trainer, TrainingState


logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates and executes training runs.

    The orchestrator's focus is on ensuring that all the components needed to
    execute a training run are present and that they are coordinated in a way to
    successfully complete the training run without issue, doing everything that they
    need to do at every step of the way.
    """

    def __init__(
        self,
        run_repository: RunRepository,
        checkpoint_store: CheckpointStore,
        artifact_store: ArtifactStore,
        event_store: EventStore,
        device="cpu",
        checkpoint_interval=3,
    ):
        """Initialize a TrainingOrchestrator.

        Args:
            run_repository: The repository for run information.
            artifact_store: The repository for training run artifacts.

        """
        self.run_repository = run_repository
        self.checkpoint_store = checkpoint_store
        self.artifact_store = artifact_store
        self.event_store = event_store
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self._buffer: dict[str, Any] = {}

    def train(
        self,
        config: Config | None = None,
        run_name: str | None = None,
        checkpoint_tag: str | None = None,
    ):
        """Orchestrate and execute a training run.

        If `run_id` is specified, then this method will attempt to resume the training
        run with the specified id from `checkpoint_tag` (or latest if not provided).
        Otherwise this method will attempt to start a new training run using `config`
        as the configuration for the new run. If `config` is not provided in this case
        an error will be raised.

        Args:
            config: The configuration to be used for the training run.
            run_name: The name of the training run to be resumed.
            checkpoint_tag: The checkpoint to resume training from.
        """
        if run_name is None:
            logger.info("Initializing a new training run...")
            if config is None:
                raise ValueError("No configuration provided for the training run.")

            run, trainer = self._init_training_run(config)
        else:
            if checkpoint_tag is None:
                logger.debug(
                    "No checkpoint specified as the restore point for training run"
                    + "'{run_name}'. Defaulting to 'latest'."
                )
                checkpoint_tag = CheckpointTag.LATEST

            logger.info(
                f"Resuming training run '{run_name}' from checkpoint '{checkpoint_tag}'..."  # NOQA: E501
            )
            run, trainer = self._load_training_run(run_name, checkpoint_tag)

        self._register_training_hooks(run, trainer)
        logger.info("Trainer initialized successfully")
        logger.info(f"Starting training run: '{run.name}'")
        train_metrics = trainer.train()
        logger.info(
            f"Training completed in {_format_hours(train_metrics.total_time_taken)}!"
        )

        run.status = RunStatus.COMPLETED
        run.updated_at = time.time_ns()
        self.run_repository.update(run)

        # self.checkpoint_store.mark_final()

        return run

    def _init_training_run(self, config: Config) -> tuple[Run, Trainer]:
        """Initialize components for a new training run.

        Args:
            run: The training run for which the components are to be intialized.
            device: The device to use for training.

        Returns:
            Tuple of (model, dataloader, pipeline, optimizer, scheduler, tokenizer)
        """
        # Validate early to avoid partially loading components.
        # TODO: Raise ValueError if required config fields are missing.
        # fmt: off
        assert config.get("tokenizer") is not None, "Config missing required 'tokenizer' section"   # NOQA: E501
        assert config.get("data") is not None, "Config missing required 'data' section"             # NOQA: E501
        assert config.get("pipeline") is not None, "Config missing required 'pipeline' section"     # NOQA: E501
        assert config.get("model") is not None, "Config missing required 'model' section"           # NOQA: E501
        assert config.get("optimizer") is not None, "Config missing required 'optimizer' section"   # NOQA: E501
        assert config.get("training") is not None, "Config missing required 'training' section"     # NOQA: E501
        # fmt: on

        run = Run(config)
        try:
            self.run_repository.insert(run)
        except Exception as e:
            logger.error(f"Failed to initialize training run '{run.name}'.")
            raise e
        logger.info(
            f"Successfully initialized training run '{run.name}' (ID: {run.id})."
        )

        try:
            logger.info("Loading dataset...")
            dataloader = Loader.data(run.config["data"])
            logger.info(
                f"Dataloader initialized with {len(dataloader.datasets)} dataset(s)"
            )

            logger.info("Loading tokenizer...")
            tokenizer = Loader.tokenizer(run.config["tokenizer"])
            register_dependency("tokenizer", tokenizer)
            logger.info("Tokenizer loaded successfully")

            logger.info("Training tokenizer on training data...")
            tokenizer.train(dataloader["train"])
            logger.info(
                f"Tokenizer trained successfully (vocab size: {tokenizer.vocab_size})"
            )

            logger.info("Saving tokenizer artifact...")
            self.artifact_store.add(run.name, "tokenizer", bytes(tokenizer))
            logger.info("Tokenizer artifact saved successfully")

            logger.info("Loading pipeline...")
            pipeline = Loader.pipeline(config["pipeline"])
            logger.info("Pipeline loaded successfully")

            logger.info("Building model...")
            model = Loader.model(config["model"]).to(self.device)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(
                f"Model built successfully - "
                f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
            )

            logger.info("Loading optimizer...")
            optimizer = Loader.optimizer(config["optimizer"], model.parameters())
            logger.info(f"Optimizer loaded: {type(optimizer).__name__}")

            scheduler = None
            if (scheduler_config := config.get("scheduler")) is not None:
                logger.info("Loading learning rate scheduler...")
                scheduler = Loader.scheduler(scheduler_config, optimizer)
                logger.info(f"Scheduler loaded: {type(scheduler).__name__}")

            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(
                "Failed to initialize training components for run '{run.name}'."
            )
            run.status = RunStatus.ERROR
            self.run_repository.update(run)
            raise e

        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            pipeline=pipeline,
            loss_fn=cross_entropy_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            **config["training"],
        )

        return run, trainer

    def _load_training_run(
        self, run_name: str, checkpoint_tag: CheckpointTag | str
    ) -> tuple[Run, Trainer]:
        """Load components from a previous training run.

        Args:
            config: The configuration from the previous training run.
            checkpoint: The checkpoint to resume from.
            device: The device to use for training.

        Returns:
            Tuple of (model, dataloader, pipeline, optimizer, scheduler, tokenizer)
        """
        logger.debug(f"Loading training run '{run_name}' from storage.")
        run = self.run_repository.get(run_name)
        if run is None:
            raise RunNotFoundError(run_name)

        logger.debug(f"Successfully retrieved run '{run_name}'.\n{run.to_dict()}")

        # fmt: off
        config = run.config
        assert config.get("tokenizer") is not None, "Config missing required 'tokenizer' section"   # NOQA: E501
        assert config.get("data") is not None, "Config missing required 'data' section"             # NOQA: E501
        assert config.get("pipeline") is not None, "Config missing required 'pipeline' section"     # NOQA: E501
        assert config.get("model") is not None, "Config missing required 'model' section"           # NOQA: E501
        assert config.get("optimizer") is not None, "Config missing required 'optimizer' section"   # NOQA: E501
        assert config.get("training") is not None, "Config missing required 'training' section"     # NOQA: E501
        # fmt: on

        checkpoint = self.checkpoint_store.get(run_name, checkpoint_tag)
        if checkpoint is None:
            raise CheckpointNotFoundError(run_name, checkpoint_tag)
        logger.debug(
            f"Successfully retrieved checkpoint '{checkpoint_tag}' from storage.\n"
            + f"{checkpoint.meta()}"
        )

        logger.info("Loading tokenizer...")
        tokenizer = Loader.tokenizer(config["tokenizer"], state=checkpoint.tokenizer)
        register_dependency("tokenizer", tokenizer)
        logger.info("Tokenizer loaded successfully")

        logger.info("Loading dataset...")
        dataloader = Loader.data(config["data"])
        logger.info(
            f"Dataloader initialized with {len(dataloader.datasets)} dataset(s)"
        )

        logger.info("Loading pipeline...")
        pipeline = Loader.pipeline(config["pipeline"])
        logger.info("Pipeline loaded successfully")

        logger.info("Building model...")
        model = Loader.model(config["model"]).to(self.device)
        assert "model_state_dict" in checkpoint, (
            "Model state could not be found in checkpoint"
        )
        model.load_state_dict(checkpoint.model_state_dict)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model loaded successfully - "
            f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
        )

        logger.info("Loading optimizer...")
        optimizer = Loader.optimizer(config["optimizer"], model.parameters())

        assert "optimizer_state_dict" in checkpoint, (
            "Optimizer state could not be found in checkpoint"
        )
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        logger.info(f"Optimizer loaded: {type(optimizer).__name__}")

        scheduler = None
        scheduler_config = config.get("scheduler")
        if scheduler_config is not None:
            logger.info("Loading scheduler...")
            scheduler = Loader.scheduler(scheduler_config, optimizer)

            assert "scheduler_state_dict" in checkpoint, (
                "Optimizer state could not be found in checkpoint"
            )
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)
            logger.info(f"Scheduler loaded: {type(scheduler).__name__}")

        logger.info("All components loaded successfully from checkpoint")

        training_state = TrainingState.from_dict(checkpoint.training_state)
        logger.info(
            f"Resuming run '{run_name}' from epoch {training_state.current_epoch}."
        )

        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            pipeline=pipeline,
            loss_fn=cross_entropy_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            state=training_state,
            **config["training"],
        )

        return run, trainer

    def _register_training_hooks(self, run: Run, trainer: Trainer) -> None:
        def _capture_train_loss(trainer: Trainer, train_metrics) -> None:
            self._buffer["train_loss"] = train_metrics.avg_loss.item()

        def _capture_val_loss(trainer: Trainer, eval_metrics) -> None:
            self._buffer["val_loss"] = eval_metrics.avg_loss.item()

        def _save_epoch_stats(trainer: Trainer) -> None:
            self.event_store.add(
                run.name,
                {
                    "train_loss": self._buffer.get("train_loss", 0.0),
                    "val_loss": self._buffer.get("val_loss", 0.0),
                    "lr": trainer.state.current_lr,
                    "epoch": trainer.state.current_epoch,
                    "global_step": trainer.state.global_step,
                },
            )

        def _save_checkpoint(trainer):
            """Save a checkpoint."""
            # Avoid creating a checkpoint if saving criteria aren't met.
            # TODO: Only skip creation of epoch checkpoints, best / latest should always be created.
            if (
                trainer.state.current_epoch % self.checkpoint_interval != 0
                and trainer.state.prev_loss != trainer.state.best_loss
            ):
                return

            checkpoint = Checkpoint(
                epoch=trainer.state.current_epoch,
                loss=trainer.state.prev_loss.item(),
                training_state=trainer.state_dict(),
                model_state_dict=trainer.model.state_dict(),
                optimizer_state_dict=trainer.optimizer.state_dict(),
            )
            if trainer.scheduler is not None:
                checkpoint.scheduler_state_dict = trainer.scheduler.state_dict()

            logging.info(f"Saving checkpoint '{checkpoint.id}'...")
            self.checkpoint_store.add(run.name, checkpoint)
            logger.info("Checkpoint saved successfully")

        def _update_run(trainer):
            run.current_epoch = trainer.state.current_epoch
            run.current_step = trainer.state.global_step
            run.current_loss = trainer.state.prev_loss.item()
            # TODO: Consider adding total training time as well.
            run.updated_at = time.time_ns()
            self.run_repository.update(run)

        trainer.register_post_fit_hook(_capture_train_loss)
        trainer.register_post_eval_hook(_capture_val_loss)
        trainer.register_post_epoch_hook(_save_epoch_stats)
        trainer.register_post_epoch_hook(_save_checkpoint)
        trainer.register_post_epoch_hook(_update_run)


def _format_hours(time: float) -> str:
    seconds = int(time)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}H {minutes}m {seconds}s"
