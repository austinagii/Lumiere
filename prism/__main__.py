from prism.tokenizer import Tokenizer
from prism.model import Model
from prism.exceptions import TrainingError
from prism.data import to_batches

import datasets


DATASET_NAME = "openwebtext"
TEXT_COLUMN_NAME = "text"
BATCH_SIZE = 64
CONTEXT_SIZE = 512
VOCAB_SIZE = 16_384
NUM_EPOCHS = 64
EMBEDDING_SIZE = 256


def main():
    try:
        dataset = datasets.load_dataset(DATASET_NAME)
    except Exception as e:
        raise TrainingError("An error occurred while loading the dataset", e) 

    train_dataset = dataset['train']

    tokenizer = Tokenizer().train(train_dataset, TEXT_COLUMN_NAME, BATCH_SIZE, VOCAB_SIZE)
    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, CONTEXT_SIZE)
    for epoch in range(NUM_EPOCHS):
        batches = to_batches(tokenizer, train_dataset, BATCH_SIZE, CONTEXT_SIZE)
        for batch in batches:
            y = model(batch)
            print(y)
            break
        break

main()