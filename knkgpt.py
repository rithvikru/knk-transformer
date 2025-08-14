import json
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import wandb
import math
from tqdm import tqdm

from nanogpt.model import GPT, GPTConfig
from nanogpt.trainer import Trainer, TrainerConfig
from nanogpt.tokenizer import encode, decode, token_to_id


class KNKDataset(Dataset):
    """Dataset for Knights and Knaves puzzles."""

    def __init__(self, jsonl_path, max_length=512, split='train', train_ratio=0.9):
        self.max_length = max_length
        self.block_size = max_length - 1
        self.data = []

        # Load the JSONL file
        print(f"Loading dataset from {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    entry = json.loads(line)
                    self.data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    print(line)
                    continue

        # Split data into train/val
        n_train = int(len(self.data) * train_ratio)
        if split == 'train':
            self.data = self.data[:n_train]
        else:
            self.data = self.data[n_train:]

        print(f"Loaded {len(self.data)} {split} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        puzzle = entry['puzzle']
        solution = entry['solution']

        # Encode as: <BOS> puzzle <SEP> solution <EOS>
        input_tokens = [token_to_id['<BOS>']] + encode(puzzle) + [token_to_id['<SEP>']] + encode(solution) + [token_to_id['<EOS>']]

        # Pad/truncate to max_length
        if len(input_tokens) > self.max_length:
            input_tokens = input_tokens[:self.max_length]
        else:
            input_tokens = input_tokens + [token_to_id['<PAD>']] * (self.max_length - len(input_tokens))

        # Autoregressive x/y shift
        x = torch.tensor(input_tokens[:-1], dtype=torch.long)
        y = torch.tensor(input_tokens[1:], dtype=torch.long)
        return x, y


def main():
    # CLI flag to disable wandb quickly
    use_wandb = '--no-wandb' not in sys.argv

    # Paths and basic hyperparameters
    dataset_path = Path('data/n_2v2mini.jsonl')  # mini set by default
    max_length = 256
    batch_size = 512 * 8
    learning_rate = 1e-3
    n_epochs = 2

    # Model size
    n_layer = 8
    n_head = 8
    n_embd = 512

    # Build datasets
    print(f"Preparing datasets from {dataset_path}")
    train_dataset = KNKDataset(dataset_path, max_length=max_length, split='train')
    val_dataset = KNKDataset(dataset_path, max_length=max_length, split='val')

    # Model configuration
    model_config = GPTConfig(
        vocab_size=len(token_to_id),
        block_size=max_length - 1,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {n_params:,} parameters ({n_params/1e6:.1f}M)")

    # Optional: load from checkpoint if exists
    checkpoint_path = 'knk_model.pt'
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location='cpu')
            raw_model = model.module if hasattr(model, 'module') else model
            raw_model.load_state_dict(state)
            print(f"Loaded checkpoint from {ckpt}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt}: {e}")

    # Scheduler tokens
    tokens_per_step = batch_size * (max_length - 1)
    warmup_steps = 10 if len(train_dataset) < 2000 else 200
    warmup_tokens = tokens_per_step * warmup_steps
    final_tokens = len(train_dataset) * (max_length - 1) * n_epochs

    # Trainer configuration
    train_config = TrainerConfig(
        max_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=True,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        num_workers=4,
        ckpt_path=checkpoint_path,
        weight_decay=0.1,
        grad_norm_clip=1.0,
        # wandb
        use_wandb=use_wandb,
        wandb_project='knk-transformer',
        wandb_run_name=f'knk-{dataset_path.stem}',
        wandb_watch=False,
        log_interval=10,
    )

    # Start training
    trainer = Trainer(model, train_dataset, val_dataset, train_config)
    print("Starting training...")
    trainer.train()
    print("Training complete!")


if __name__ == '__main__':
    main()