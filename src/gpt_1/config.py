import torch

# Dataset configuration
CHUNK_SIZE = 32768 * 4
VOCAB_SIZE = 2048
SAMPLES_PER_CHUNK = 128

# Model configuration
EMBEDDING_SIZE = 768
HIDDEN_SIZE = 3072
N_HEADS = 12
N_TRANSFORMERS = 12
SEQUENCE_LENGTH = 256

# Training configuration
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu" if torch.xpu.is_available() else "cpu"
)

EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = 6
OUTPUT_FREQUENCY = 250
STORE_FREQUENCY = 1000
