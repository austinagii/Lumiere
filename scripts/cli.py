from lumiere.training.persistence import Checkpoint, CheckpointType


def parse_checkpoint(arg: str) -> Checkpoint:
    if len(arg.strip()) == 0:
        return None

    checkpoint_parts = arg.split(":")
    try:
        checkpoint_type = CheckpointType(checkpoint_parts[0])
    except ValueError:
        return None

    checkpoint_value = None
    if len(checkpoint_parts) > 1:
        checkpoint_value = (
            int(checkpoint_parts[1])
            if checkpoint_type is CheckpointType.EPOCH
            else checkpoint_parts[1]
        )

    return Checkpoint(checkpoint_type, checkpoint_value)
