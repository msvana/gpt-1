from argparse import ArgumentParser

import tokenizers
from gpt_1 import config, utils
from tokenizers import models, pre_tokenizers, trainers

args_parser = ArgumentParser(description="Train a tokenizer on the BookCorpus dataset.")
args_parser.add_argument(
    "-d",
    "--dataset-path",
    required=True,
    type=str,
    help="Path to a tar.bz2 file containing the BookCorpus dataset.",
)
args_parser.add_argument(
    "-o",
    "--output-path",
    required=True,
    type=str,
    help="Path to save the trained tokenizer model.",
)


def main():
    args = args_parser.parse_args()
    iterator = utils.text_iterator(args.dataset_path, chunk_size=config.CHUNK_SIZE)
    tokenizer = tokenizers.Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # type: ignore
    trainer = trainers.BpeTrainer(vocab_size=config.VOCAB_SIZE, show_progress=True)
    tokenizer.train_from_iterator(iterator, trainer)
    tokenizer.save(args.output_path)
