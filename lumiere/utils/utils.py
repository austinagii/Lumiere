
@contextmanager
def disable_tokenizer_parallelism():
    """Context manager to temporarily disable tokenizers parallelism.

    This prevents fork conflicts with the tokenizers library during blob operations.
    """
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        yield
    finally:
        # Restore original tokenizer parallelism setting
        if original_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
        else:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)


class RemoteStorageClient:
