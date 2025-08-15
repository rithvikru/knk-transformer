import json
import math
import os
import random
import sys
from pathlib import Path
from array import array

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset
from tqdm import tqdm

from nanogpt.model import GPT, GPTConfig
from nanogpt.tokenizer import decode, encode, token_to_id
from nanogpt.trainer import Trainer, TrainerConfig


class KNKDataset(Dataset):
    """Streaming dataset for Knights and Knaves puzzles from a large JSONL.

    This does not load the entire file into memory. It builds an index of byte offsets
    for lines and seeks to them on demand. For the validation split, it samples a fixed-size
    subset to bound eval cost.
    """

    def __init__(self, jsonl_path, max_length=512, split='train', train_ratio=0.99, val_size=200_000, seed=42):
        self.max_length = max_length
        self.block_size = max_length - 1
        self.jsonl_path = str(jsonl_path)
        self.split = split
        self.seed = seed
        rng = random.Random(seed)

        # Build or load line offset index (memory-efficient, cached on disk)
        idx_path = str(jsonl_path) + '.idx'
        self.offsets: array
        if os.path.exists(idx_path):
            print(f"Loading index from {idx_path}")
            self.offsets = array('Q')
            with open(idx_path, 'rb') as idxf:
                self.offsets.frombytes(idxf.read())
        else:
            print(f"Indexing dataset from {jsonl_path} (one-time; writes {idx_path})")
            self.offsets = array('Q')
            with open(jsonl_path, 'rb') as f, open(idx_path, 'wb') as idxf:
                offset = 0
                for line in f:
                    if line.strip():
                        self.offsets.append(offset)
                    offset += len(line)
                idxf.write(self.offsets.tobytes())

        n_total = len(self.offsets)
        n_train = int(n_total * train_ratio)

        if split == 'train':
            self.indices = list(range(0, n_train))
        else:
            val_indices = list(range(n_train, n_total))
            if len(val_indices) > val_size:
                # sample a fixed subset for faster eval
                val_indices = rng.sample(val_indices, val_size)
            self.indices = sorted(val_indices)

        print(f"Prepared {len(self.indices)} {split} indices (total lines: {n_total})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        line_idx = self.indices[idx]
        with open(self.jsonl_path, 'rb') as f:
            f.seek(self.offsets[line_idx])
            raw = f.readline().decode('utf-8')

        entry = json.loads(raw)
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
    dataset_path = Path('data/n_2.jsonl')
    max_length = 256
    learning_rate = 3e-4
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
        pad_token_id=token_to_id['<PAD>'],
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
            if isinstance(state, dict) and 'model' in state:
                raw_model.load_state_dict(state['model'])
            else:
                raw_model.load_state_dict(state)
            print(f"Loaded checkpoint from {ckpt}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt}: {e}")

    # Scheduler tokens
    # With DDP + grad accumulation, we count tokens via trainer; provide a target token budget
    approx_tokens_per_sample = max_length - 1
    warmup_tokens = int(2e9)  # aggressive warmup with massive compute
    final_tokens = int(2e12)  # target total tokens for cosine schedule plateau

    # Set seeds for reproducibility
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Trainer configuration
    # Training configuration
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    # Use micro-batches per GPU + grad accumulation to reach a large global batch
    train_config = TrainerConfig(
        max_epochs=n_epochs,
        micro_batch_size=64,  # per-GPU micro batch, adjust based on memory
        global_batch_size= world_size * 64 * 8,  # target large global batch (8 accumulation steps)
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        lr_decay=True,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        num_workers=16,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=True,
        ckpt_path=checkpoint_path,
        weight_decay=0.1,
        grad_norm_clip=1.0,
        use_bf16=True,
        use_fp16=False,
        val_max_batches=500,
        ddp=True,
        # wandb
        use_wandb=use_wandb,
        wandb_project='knk-transformer',
        wandb_run_name=f'knk-{dataset_path.stem}',
        wandb_watch=False,
        log_interval=20,
    )

    # Start training
    trainer = Trainer(model, train_dataset, val_dataset, train_config)
    print("Starting training...")
    trainer.train()
    print("Training complete!")


if __name__ == '__main__':
    main()
