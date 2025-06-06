from argparse import ArgumentParser

import torch
from torch.distributions import Categorical
from gpt_1 import config, utils
from torch import nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

arg_parser = ArgumentParser(description="Train a GPT model on the BookCorpus dataset.")
arg_parser.add_argument(
    "-d",
    "--dataset-path",
    required=True,
    type=str,
    help="Path to the preprocessed dataset directory.",
)
arg_parser.add_argument(
    "-t",
    "--tokenizer-path",
    required=True,
    type=str,
    help="Path to the tokenizer model file.",
)


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        sequence_length: int,
        hidden_size: int,
        n_heads: int,
        n_transformers: int,
    ):
        super(GPTModel, self).__init__()
        self._embedding = nn.Embedding(vocab_size, embedding_size)
        self._position_embedding = nn.Embedding(sequence_length, embedding_size)
        self.transformers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_size,
                    nhead=n_heads,
                    dim_feedforward=hidden_size,
                )
                for _ in range(n_transformers)
            ]
        )
        self.fc_out = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_sequence: torch.Tensor):
        input_sequence_size = input_sequence.size(-1)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            input_sequence_size
        ).to(config.DEVICE)

        embedded = self._embedding(input_sequence)
        for transformer in self.transformers:
            embedded = transformer(embedded, is_causal=True, src_mask=mask)
        output = self.fc_out(embedded)
        return output


def collate_fn(batch):
    input_sequences, next_tokens = zip(*batch)
    input_sequences = torch.tensor(input_sequences, dtype=torch.long)
    next_tokens = torch.tensor(next_tokens, dtype=torch.long)
    return input_sequences, next_tokens


def generate_text(
    model: GPTModel, tokenizer: Tokenizer, prompt: str, max_length: int
) -> str:
    model.eval()
    text = prompt.lower()
    input_ids = tokenizer.encode(text).ids

    for _ in range(max_length):
        input_ids = input_ids[-config.SEQUENCE_LENGTH :]
        input_tensor = (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(config.DEVICE)
        )
        with torch.no_grad():
            output = model(input_tensor)

        distribution = Categorical(logits=output[:, -1, :])
        next_token_id = distribution.sample().item()
        input_ids.append(next_token_id)
        text += tokenizer.decode([next_token_id]).replace("Ġ", " ").replace("Ċ", "\n")

    return text


def train_model(
    model: GPTModel, data_loader: DataLoader, num_epochs: int, tokenizer: Tokenizer
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(num_epochs):
        average_loss = 0.0
        batches_processed = 0

        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (input_sequences, next_tokens) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}", end="       ")
            optimizer.zero_grad()
            input_sequences = input_sequences.to(config.DEVICE)
            next_tokens = next_tokens.to(config.DEVICE)
            outputs = model(input_sequences)[:, -1, :]
            loss = criterion(outputs, next_tokens)
            loss.backward()
            batches_processed += 1
            average_loss = (
                average_loss * (batches_processed - 1) + loss.item()
            ) / batches_processed

            optimizer.step()
            print(f"Loss: {average_loss:.4f}", end="\r")

            if (batch_idx + 1) % config.OUTPUT_FREQUENCY == 0:
                generated = generate_text(model, tokenizer, "Once upon a time", 32)
                print(f"\nGenerated text: {generated}")
                batches_processed = 0
                average_loss = 0.0
                model.train()


def main():
    args = arg_parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    dataset = utils.BookCorpusDataset(
        args.dataset_path,
        samples_per_chunk=config.SAMPLES_PER_CHUNK,
        sequence_length=config.SEQUENCE_LENGTH,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    model = GPTModel(
        config.VOCAB_SIZE,
        config.EMBEDDING_SIZE,
        config.HIDDEN_SIZE,
        config.N_HEADS,
        config.N_TRANSFORMERS,
    ).to(config.DEVICE)

    train_model(model, data_loader, config.EPOCHS, tokenizer)


if __name__ == "__main__":
    main()
