import torch

# Dataset configuration
CHUNK_SIZE = 32768 * 4
VOCAB_SIZE = 2048
SAMPLES_PER_CHUNK = 128

# Model configuration
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 384
N_HEADS = 8
N_TRANSFORMERS = 4
SEQUENCE_LENGTH = 128

# Training configuration
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = 8
OUTPUT_FREQUENCY = 250
STORE_FREQUENCY = 1000
