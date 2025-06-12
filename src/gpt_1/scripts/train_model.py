from argparse import ArgumentParser

import torch
from torch.distributions import Categorical
from gpt_1 import config, utils
from torch import nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import os

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
arg_parser.add_argument(
    "-m",
    "--model-path",
    required=False,
    type=str,
    help="Path to save the trained model state.",
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
        padding_idx: int = 2048,
    ):
        super(GPTModel, self).__init__()

        self._padding_idx = padding_idx

        self._embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx
        )
        self._position_embedding = nn.Embedding(sequence_length, embedding_size)
        self.transformers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_size,
                    nhead=n_heads,
                    dim_feedforward=hidden_size,
                    batch_first=True,
                )
                for _ in range(n_transformers)
            ]
        )
        self._fc_out = nn.Linear(embedding_size, vocab_size)

        self._position_ids = torch.arange(
            sequence_length, dtype=torch.long, device=config.DEVICE
        ).unsqueeze(0)

        self._mask = nn.Transformer.generate_square_subsequent_mask(sequence_length).to(
            config.DEVICE
        )

    def forward(self, input_sequence: torch.Tensor):
        src_padding_mask = (input_sequence == self._padding_idx)
        embedded = self._embedding(input_sequence)
        position_embedded = self._position_embedding(self._position_ids)
        embedded += position_embedded
        for transformer in self.transformers:
            embedded = transformer(
                embedded,
                is_causal=True,
                src_mask=self._mask,
                src_key_padding_mask=src_padding_mask,
            )
        output = self._fc_out(embedded)
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
    input_ids = utils.prepad(tokenizer, input_ids, config.SEQUENCE_LENGTH)

    for _ in range(max_length):
        input_ids = input_ids[-config.SEQUENCE_LENGTH :]
        input_tensor = (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(config.DEVICE)
        )
        with torch.no_grad():
            output = 1.5 * model(input_tensor)

        distribution = Categorical(logits=output[:, -1, :])
        next_token_id = int(distribution.sample().item())
        input_ids.append(next_token_id)
        text += tokenizer.decode([next_token_id]).replace("Ġ", " ").replace("Ċ", "\n")

    return text


def train_model(
    model: GPTModel,
    data_loader: DataLoader,
    num_epochs: int,
    tokenizer: Tokenizer,
    model_path: str | None = None,
):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(num_epochs):
        average_loss = 0.0
        batches_processed = 0

        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (input_sequences, output_sequences) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}", end="       ")
            optimizer.zero_grad()
            input_sequences = input_sequences.to(config.DEVICE)
            output_sequences = output_sequences.to(config.DEVICE)
            outputs = model(input_sequences)
            loss = criterion(
                outputs.view(-1, config.VOCAB_SIZE + 1), output_sequences.view(-1)
            )
            loss.backward()
            optimizer.step()
            batches_processed += 1
            average_loss = (
                average_loss * (batches_processed - 1) + loss.item()
            ) / batches_processed

            print(f"Loss: {average_loss:.4f}", end="\r")

            if (batch_idx + 1) % config.OUTPUT_FREQUENCY == 0:
                generated = generate_text(
                    model, tokenizer, "Once upon a time there lived", 50
                )
                print(f"\nGenerated text:")
                print("-----")
                print(generated)
                print("-----")

                batches_processed = 0
                average_loss = 0.0
                model.train()

            if (batch_idx + 1) % config.STORE_FREQUENCY == 0 and model_path:
                torch.save(model.state_dict(), model_path)
                print(f"\nModel state saved to {model_path}")


def main():
    args = arg_parser.parse_args()

    tokenizer: Tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.add_special_tokens(["[PAD]"])
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
        config.VOCAB_SIZE + 1,
        config.EMBEDDING_SIZE,
        config.SEQUENCE_LENGTH,
        config.HIDDEN_SIZE,
        config.N_HEADS,
        config.N_TRANSFORMERS,
        padding_idx=tokenizer.token_to_id("[PAD]"),
    ).to(config.DEVICE)

    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
        print(f"Model loaded from {args.model_path}")

    train_model(model, data_loader, config.EPOCHS, tokenizer, args.model_path)


if __name__ == "__main__":
    main()
