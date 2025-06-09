import os
import random
import tarfile

from torch.utils.data import Dataset
from tokenizers import Tokenizer


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


def prepad(tokenizer: Tokenizer, ids: list[int], length: int) -> list[int]:
    padding_ids = tokenizer.encode("[PAD]").ids
    return (
        padding_ids * (length - len(ids)) + ids if len(ids) < length else ids[-length:]
    )


class BookCorpusDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        samples_per_chunk: int,
        sequence_length: int,
        padding_id: int = 0,
    ):
        self._dataset_path = dataset_path
        self._num_files = len(os.listdir(dataset_path))
        self._samples_per_chunk = samples_per_chunk
        self._sequence_length = sequence_length
        self._padding_id = padding_id

    def __len__(self) -> int:
        return self._num_files * self._samples_per_chunk

    def __getitem__(self, index: int):
        file_index = index // self._samples_per_chunk
        filename = os.path.join(self._dataset_path, f"chunk_{file_index}.txt")
        with open(filename, "r", encoding="utf-8") as file:
            data = file.read()
            token_ids = list(map(int, data.split(",")))
            start_idx = random.randint(0, len(token_ids) - self._sequence_length - 1)
            end_idx = start_idx + self._sequence_length
            input_sequence = token_ids[start_idx:end_idx]
            should_pad = random.random() < 0.25

            if should_pad:
                how_many_to_pad = random.randint(1, 30)
                input_sequence[:how_many_to_pad] = [self._padding_id] * how_many_to_pad
            next_token = token_ids[end_idx]

        return input_sequence, next_token
