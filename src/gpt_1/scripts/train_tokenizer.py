from argparse import ArgumentParser
from pathlib import Path


args_parser = ArgumentParser(description="Train a tokenizer on the BookCorpus dataset.")
args_parser.add_argument(
    "-d",
    "--dataset_path",
    required=True,
    type=Path,
    help="Path to a tar.bz2 file containing the BookCorpus dataset.",
)

def main():
    args = args_parser.parse_args()
