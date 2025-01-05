import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import wandb
import os
from model import build_transformer

# Initialize WandB
checkpoints_dir = "checkpoints"
wandb.init(
    project="spell-checker",
    name="Run 2 - Updated Dataset",
    config={
        "learning_rate": 0.0005,
        "epochs": 50,
        "batch_size": 128,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "d_ff": 1024,
    }
)

# Define vocabulary
character_vocab = {char: idx for idx, char in enumerate(
    ['<pad>', '<sos>', '<eos>', '<unk>'] + [chr(i) for i in range(97, 123)])}
vocab_size = len(character_vocab)
pad_idx = character_vocab['<pad>']
sos_idx = character_vocab['<sos>']
eos_idx = character_vocab['<eos>']

# Save tokenizer
def save_tokenizer(vocab, filepath="tokenizer.json"):
    with open(filepath, "w") as f:
        json.dump(vocab, f)
    print(f"Tokenizer saved to {filepath}")

# Dataset class
class SpellCheckDataset(Dataset):
    def __init__(self, jsonl_path, vocab, max_len=20):
        self.data = self.load_jsonl(jsonl_path)
        self.vocab = vocab
        self.max_len = max_len

    def load_jsonl(self, path):
        with open(path, "r") as f:
            return [json.loads(line.strip()) for line in f]

    def tokenize_word(self, word):
        return [sos_idx] + [self.vocab.get(c, self.vocab['<unk>']) for c in word] + [eos_idx]

    def pad_sequence(self, seq):
        seq = seq[:self.max_len]
        return seq + [pad_idx] * (self.max_len - len(seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        misspelled = self.data[idx]["misspelled"]
        correct = self.data[idx]["correct"]
        src = self.tokenize_word(misspelled)
        tgt = self.tokenize_word(correct)
        return torch.tensor(self.pad_sequence(src)), torch.tensor(self.pad_sequence(tgt))

# Hyperparameters
config = wandb.config
src_seq_len = tgt_seq_len = 20
d_model = config.d_model
num_layers = config.num_layers
num_heads = config.num_heads
dropout = config.dropout
d_ff = config.d_ff
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate

# Load dataset
dataset_path = "data/dataset.jsonl"
dataset = SpellCheckDataset(dataset_path, character_vocab)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = build_transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    src_seq_len=src_seq_len,
    tgt_seq_len=tgt_seq_len,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout=dropout,
    d_ff=d_ff
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Save tokenizer at the beginning
    tokenizer_save_path = os.path.join(checkpoints_dir, "tokenizer.json")
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_tokenizer(character_vocab, tokenizer_save_path)
    wandb.save(tokenizer_save_path)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
            tgt_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2)

            encoder_output = model.encode(src, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
            predictions = model.project(decoder_output)

            loss = criterion(predictions.reshape(-1, vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
            wandb.log({"batch_loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch_loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"spell_check_model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)
        print(f"Model checkpoint saved and logged to W&B as {checkpoint_path}!")

if __name__ == "__main__":
    train()