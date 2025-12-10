"""Records for tracking state within training runs."""

from dataclasses import dataclass


@dataclass
class TrainingState:
    """Persistent store of information between epochs."""

    total_time_taken: float = 0.0
    prev_loss: float = float("inf")
    best_loss: float = float("inf")
    best_perplexity: float = float("inf")
    global_step: int = 0
    current_epoch: int = 1
    patience: int = 0
    patience_counter: int = 0
    stopping_threshold: int = 0
    current_lr: int | None = None
