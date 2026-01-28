class AutoregressiveLanguageModellingPreprocessor:
    def __init__(self, device):
        self.device = device

    def __call__(self, tokens, padding_mask):
        # Shift the input tokens to the left by one position to get the targets.
        next_tokens = tokens[:, 1:].to(self.device)
        # Shift the input tokens and its padding mask accordingly.
        tokens = tokens[:, :-1].to(self.device)
        padding_mask = padding_mask[:, :-1].to(self.device)
        return (tokens, padding_mask), next_tokens
