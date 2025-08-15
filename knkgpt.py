import json
import math
import os
import random
import sys
from pathlib import Path
from array import array
import time

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
        self._fh = None  # lazily-opened file handle per worker/process

        # Build or load line offset index (memory-efficient, cached on disk)
        idx_path = str(jsonl_path) + '.idx'
        self.offsets: array

        def idx_is_valid(path: str) -> bool:
            try:
                return os.path.exists(path) and os.path.getsize(path) > 0
            except Exception:
                return False

        rank_env = int(os.environ.get('RANK', '0'))
        world_size_env = int(os.environ.get('WORLD_SIZE', '1'))

        if not idx_is_valid(idx_path):
            if rank_env == 0:
                print(f"Indexing dataset from {jsonl_path} (one-time; writes {idx_path})")
                self.offsets = array('Q')
                tmp_path = idx_path + '.tmp'
                with open(jsonl_path, 'rb') as f, open(tmp_path, 'wb') as idxf:
                    offset = 0
                    for line in f:
                        if line.strip():
                            self.offsets.append(offset)
                        offset += len(line)
                    idxf.write(self.offsets.tobytes())
                    idxf.flush()
                    os.fsync(idxf.fileno())
                os.replace(tmp_path, idx_path)
            else:
                # Wait for rank 0 to build the index
                print(f"Waiting for index {idx_path} to be built by rank 0...")
                patience_sec = 3600
                start = time.time()
                while not idx_is_valid(idx_path) and (time.time() - start) < patience_sec:
                    time.sleep(1)
                if not idx_is_valid(idx_path):
                    raise RuntimeError(f"Timeout waiting for index file {idx_path}")

        print(f"Loading index from {idx_path}")
        self.offsets = array('Q')
        with open(idx_path, 'rb') as idxf:
            self.offsets.frombytes(idxf.read())

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
        if self._fh is None:
            self._fh = open(self.jsonl_path, 'rb', buffering=0)
        self._fh.seek(self.offsets[line_idx])
        raw = self._fh.readline().decode('utf-8')

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

    def __del__(self):
        try:
            if getattr(self, '_fh', None) is not None:
                self._fh.close()
        except Exception:
            pass


class PreTokenizedDataset(Dataset):
    """Memory-mapped pretokenized dataset (uint16 token stream + offsets).

    Files:
      - <jsonl>.tok.bin : concatenated uint16 token ids
      - <jsonl>.tok.idx : array('Q') of offsets length N+1
    """

    def __init__(self, tok_bin_path: str, tok_idx_path: str, max_length: int, split: str = 'train', train_ratio: float = 0.99, val_size: int = 200_000, seed: int = 42):
        self.max_length = max_length
        self.block_size = max_length - 1
        self.tok_bin_path = tok_bin_path
        self.tok_idx_path = tok_idx_path
        rng = random.Random(seed)

        # load offsets
        self.offsets = array('Q')
        with open(tok_idx_path, 'rb') as f:
            self.offsets.frombytes(f.read())
        self.n_samples = len(self.offsets) - 1

        # memmap tokens
        self.tokens = np.memmap(tok_bin_path, mode='r', dtype=np.uint16)

        n_train = int(self.n_samples * train_ratio)
        if split == 'train':
            self.indices = list(range(0, n_train))
        else:
            val_indices = list(range(n_train, self.n_samples))
            if len(val_indices) > val_size:
                val_indices = rng.sample(val_indices, val_size)
            self.indices = sorted(val_indices)

        print(f"Pretokenized {split} examples: {len(self.indices)} of {self.n_samples} total")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        start = self.offsets[i]
        end = self.offsets[i + 1]
        seq = self.tokens[start:end]
        # Pad/truncate to max_length
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        else:
            pad_len = self.max_length - len(seq)
            if pad_len:
                seq = np.pad(seq, (0, pad_len), constant_values=token_to_id['<PAD>'])
        # Autoregressive x/y shift
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def build_pretokenized(jsonl_path: str, tok_bin_path: str, tok_idx_path: str):
    """One-time pretokenization: JSONL -> uint16 token stream + offsets.

    The output consists of variable-length sequences; BOS/SEP/EOS are embedded.
    """
    print(f"Pretokenizing {jsonl_path} -> {tok_bin_path}, {tok_idx_path}")
    offsets = array('Q')
    offsets.append(0)
    total = 0
    tmp_bin = tok_bin_path + '.tmp'
    with open(jsonl_path, 'r', encoding='utf-8') as fin, open(tmp_bin, 'wb') as fb:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                puzzle = entry['puzzle']
                solution = entry['solution']
            except Exception:
                continue
            seq = [token_to_id['<BOS>']] + encode(puzzle) + [token_to_id['<SEP>']] + encode(solution) + [token_to_id['<EOS>']]
            arr = array('H', seq)  # uint16
            fb.write(arr.tobytes())
            total += len(seq)
            offsets.append(total)
            if (line_num % 1_000_000) == 0:
                print(f"  processed {line_num:,} lines...")

    # write idx atomically
    tmp_idx = tok_idx_path + '.tmp'
    with open(tmp_idx, 'wb') as fi:
        fi.write(offsets.tobytes())
        fi.flush()
        os.fsync(fi.fileno())
    os.replace(tmp_idx, tok_idx_path)
    os.replace(tmp_bin, tok_bin_path)
    print(f"Pretokenized {len(offsets)-1:,} samples; total tokens: {total:,}")


def main():
    # CLI flag to disable wandb quickly
    use_wandb = '--no-wandb' not in sys.argv

    # Paths and basic hyperparameters
    dataset_path = Path('data/n_2.jsonl')
    max_length = 256
    learning_rate = 3e-4
    # With massive data, one pass is sufficient with a token-based schedule
    n_epochs = 1

    # Model size
    n_layer = 8
    n_head = 8
    n_embd = 512

    # Build or load pretokenized dataset
    tok_bin = str(dataset_path) + '.tok.bin'
    tok_idx = str(dataset_path) + '.tok.idx'
    rank_env = int(os.environ.get('RANK', '0'))
    if not (os.path.exists(tok_bin) and os.path.exists(tok_idx)):
        if rank_env == 0:
            build_pretokenized(str(dataset_path), tok_bin, tok_idx)
        else:
            print(f"Waiting for pretokenized files to be built by rank 0...")
            while not (os.path.exists(tok_bin) and os.path.exists(tok_idx)):
                time.sleep(5)

    print(f"Preparing datasets from {dataset_path} (pretokenized)")
    train_dataset = PreTokenizedDataset(tok_bin, tok_idx, max_length=max_length, split='train')
    val_dataset = PreTokenizedDataset(tok_bin, tok_idx, max_length=max_length, split='val')

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
    # Compute per-rank token budgets tied to planned run length
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    micro_bs = 64  # keep in sync with TrainerConfig below
    accum_steps = 8  # target accumulation below
    tokens_per_seq = max_length - 1
    per_rank_tokens_per_step = micro_bs * tokens_per_seq * accum_steps
    global_batch = world_size * micro_bs * accum_steps
    steps_per_epoch = max(1, len(train_dataset) // global_batch)
    planned_tokens_per_rank = per_rank_tokens_per_step * steps_per_epoch * n_epochs
    warmup_tokens = max(int(0.02 * planned_tokens_per_rank), per_rank_tokens_per_step * 1500)
    warmup_tokens = min(warmup_tokens, max(per_rank_tokens_per_step, planned_tokens_per_rank // 3))
    final_tokens = max(per_rank_tokens_per_step, planned_tokens_per_rank)

    # Set seeds for reproducibility
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Trainer configuration
    # Training configuration
    # Use micro-batches per GPU + grad accumulation to reach a large global batch
    train_config = TrainerConfig(
        max_epochs=n_epochs,
        micro_batch_size=micro_bs,  # per-GPU micro batch, adjust based on memory
        global_batch_size= world_size * micro_bs * accum_steps,  # target large global batch
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        lr_decay=True,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
    num_workers=64,
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
