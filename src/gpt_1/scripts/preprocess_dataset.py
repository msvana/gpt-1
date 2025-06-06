import os
from argparse import ArgumentParser

from gpt_1 import config, utils
from tokenizers import Encoding, Tokenizer

arg_parser = ArgumentParser(
    description="Preprocess a dataset for training the GPT model."
)
arg_parser.add_argument(
    "-d",
    "--dataset-path",
    required=True,
    type=str,
    help="Path to the BookCorpus dataset tar.bz2 file.",
)
arg_parser.add_argument(
    "-o",
    "--output-path",
    required=True,
    type=str,
    help="Directory to save the preprocessed dataset.",
)
arg_parser.add_argument(
    "-t",
    "--tokenizer-path",
    required=True,
    type=str,
    help="Path to the tokenizer model file.",
)


def encode_chunk(index: int, chunk: str, tokenizer: Tokenizer, base_path: str):
    encoded: Encoding = tokenizer.encode(chunk)
    filename = f"{base_path}/chunk_{index}.txt"
    contents = ",".join(str(token_id) for token_id in encoded.ids)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(contents)


def main():
    args = arg_parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    iterator = utils.text_iterator(args.dataset_path, chunk_size=config.CHUNK_SIZE)
    os.makedirs(args.output_path, exist_ok=True)

    for index, chunk in enumerate(iterator):
        print(f"Processing chunk {index + 1}...")
        encode_chunk(index, chunk, tokenizer, args.output_path)
