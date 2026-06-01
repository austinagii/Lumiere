class RunNotFoundError(Exception):
    def __init__(self, run_name: str):
        super().__init__(f"No run found with name '{run_name}'.")


class CheckpointNotFoundError(Exception):
    def __init__(self, run_name: str, checkpoint_tag: str):
        super().__init__(f"No run found with name '{run_name}'.")
