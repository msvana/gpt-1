import tarfile
from argparse import ArgumentParser

import tokenizers
from gpt_1 import config
from tokenizers import models, pre_tokenizers, trainers

args_parser = ArgumentParser(description="Train a tokenizer on the BookCorpus dataset.")
args_parser.add_argument(
    "-d",
    "--dataset_path",
    required=True,
    type=str,
    help="Path to a tar.bz2 file containing the BookCorpus dataset.",
)
args_parser.add_argument(
    "-o",
    "--output_path",
    required=True,
    type=str,
    help="Path to save the trained tokenizer model.",
)


def text_iterator(dataset_path, chunk_size: int):
    with tarfile.open(dataset_path, "r|bz2") as tar:
        for member in tar:
            file = tar.extractfile(member)
            assert file is not None
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk.decode("utf-8", errors="ignore")


def main():
    args = args_parser.parse_args()
    iterator = text_iterator(args.dataset_path, chunk_size=config.CHUNK_SIZE)
    tokenizer = tokenizers.Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # type: ignore
    trainer = trainers.BpeTrainer(vocab_size=config.VOCAB_SIZE, show_progress=True)
    tokenizer.train_from_iterator(iterator, trainer)
    tokenizer.save(args.output_path)
