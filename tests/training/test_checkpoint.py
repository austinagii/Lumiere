from lumiere.training import Checkpoint


class TestCheckpoint:
    def test_checkpoint_can_be_instantiated_from_arbitrary_key_value_pairs(self):
        Checkpoint(
            random_key=1,
            another_random_key=0.1,
            yet_another_random_key="True",
        )

    def test_key_value_pairs_can_be_accessed_using_square_bracket_notation(self):
        checkpoint = Checkpoint(
            random_key=1,
            another_random_key=0.1,
            yet_another_random_key="True",
        )
        assert checkpoint["random_key"] == 1
        assert checkpoint["another_random_key"] == 0.1
        assert checkpoint["yet_another_random_key"] == "True"

    def test_key_value_pairs_can_be_accessed_using_dot_notation(self):
        checkpoint = Checkpoint(
            random_key=1,
            another_random_key=0.1,
            yet_another_random_key="True",
        )
        assert checkpoint.random_key == 1
        assert checkpoint.another_random_key == 0.1
        assert checkpoint.yet_another_random_key == "True"

    def test_checkpoint_can_be_converted_to_bytes(self):
        checkpoint = Checkpoint(
            random_key=1,
            another_random_key=0.1,
            yet_another_random_key="True",
        )
        assert isinstance(bytes(checkpoint), bytes)

    def test_checkpoint_can_be_constructed_from_bytes(self):
        checkpoint = Checkpoint(
            random_key=1,
            another_random_key=0.1,
            yet_another_random_key=True,
        )
        checkpoint_bytes = bytes(checkpoint)

        loaded_checkpoint = Checkpoint.from_bytes(checkpoint_bytes)

        assert isinstance(loaded_checkpoint, Checkpoint)
        assert loaded_checkpoint == checkpoint
