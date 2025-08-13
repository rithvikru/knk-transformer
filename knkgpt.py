import json
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
                    print(f"Line content: {line[:100]}...")
                    if line_num <= 5:  # Show first few problematic lines
                        print(f"Full line: {line}")
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
        
        # Encode the input: <BOS> puzzle <SEP> solution <EOS>
        input_tokens = [token_to_id['<BOS>']] + encode(puzzle) + [token_to_id['<SEP>']] + encode(solution) + [token_to_id['<EOS>']]
        
        # Pad or truncate to max_length
        if len(input_tokens) > self.max_length:
            input_tokens = input_tokens[:self.max_length]
        else:
            input_tokens = input_tokens + [token_to_id['<PAD>']] * (self.max_length - len(input_tokens))
        
        # Convert to tensors
        x = torch.tensor(input_tokens[:-1], dtype=torch.long)
        y = torch.tensor(input_tokens[1:], dtype=torch.long)
        
        return x, y


class WandbTrainer(Trainer):
    """Custom trainer with wandb logging."""
    
    def __init__(self, model, train_dataset, eval_dataset, config, use_wandb=True):
        super().__init__(model, train_dataset, eval_dataset, config)
        self.use_wandb = use_wandb
    
    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.eval_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # Log to wandb
                    if self.use_wandb and it % 10 == 0:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/lr': lr,
                            'train/epoch': epoch,
                            'train/iter': epoch * len(loader) + it,
                        })

                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                if self.use_wandb:
                    wandb.log({
                        'val/loss': test_loss,
                        'val/epoch': epoch,
                    })
                print(f"Validation loss: {test_loss:.5f}")
                return test_loss

        best_loss = float('inf')
        self.tokens = 0
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.eval_dataset is not None:
                test_loss = run_epoch('test')

            if self.config.ckpt_path is not None:
                if self.eval_dataset is None:
                    self.save_checkpoint()
                    continue
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()
                    if self.use_wandb:
                        wandb.log({'val/best_loss': best_loss})


def main():
    import sys
    
    # Check for --no-wandb flag
    use_wandb = '--no-wandb' not in sys.argv
    
    # Configuration - all hardcoded
    dataset_path = Path('data/n_2.jsonl')
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("\nTo generate the dataset:")
        print("  cd ../lean-knk")
        print("  lake exe knk --n 2 --count 100000 --out n_2")
        print("\nOr for a quick test:")
        print("  bash generate_test_data.sh")
        
        # Try test dataset
        test_path = Path('../lean-knk/data/n_2_test.jsonl')
        if test_path.exists():
            print(f"\n✅ Found test dataset at {test_path}")
            print("Using test dataset instead...")
            dataset_path = test_path
        else:
            return
    
    max_length = 256  # Most puzzles are < 200 tokens
    batch_size = 128   # Reduced for smaller GPUs
    learning_rate = 3e-4
    n_epochs = 50     # Reduced for testing
    
    # Smaller model for testing (GPT-2 small size)
    n_layer = 12
    n_head = 12
    n_embd = 768
    
    # Training configuration
    checkpoint_path = 'knk_model.pt'
    wandb_project = 'knk-transformer'
    wandb_run_name = 'knk-n2'
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                'dataset': str(dataset_path),
                'max_length': max_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'n_epochs': n_epochs,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'vocab_size': len(token_to_id),
            }
        )
    
    # Model configuration
    model_config = GPTConfig(
        vocab_size=len(token_to_id),
        block_size=max_length - 1,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1
    )
    
    # Create datasets
    train_dataset = KNKDataset(dataset_path, max_length=max_length, split='train')
    val_dataset = KNKDataset(dataset_path, max_length=max_length, split='val')
    
    # Initialize model
    model = GPT(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {n_params:,} parameters ({n_params/1e6:.1f}M)")
    
    if use_wandb:
        wandb.config.update({'n_params': n_params})
    
    # Training configuration
    train_config = TrainerConfig(
        max_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=n_epochs*len(train_dataset)*max_length,
        num_workers=4,
        ckpt_path=checkpoint_path,
        weight_decay=0.1,
        grad_norm_clip=1.0,
    )
    
    # Create trainer and train
    trainer = WandbTrainer(model, train_dataset, val_dataset, train_config, use_wandb=use_wandb)
    
    # Add some test samples to verify tokenization
    print("\nSample data:")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset.data[i]
        print(f"Puzzle: {sample['puzzle'][:50]}...")
        print(f"Solution: {sample['solution']}")
        print(f"Encoded length: {len(encode(sample['puzzle']))} tokens")
        print()
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete!")
    
    # Test the model with a sample
    model.eval()
    with torch.no_grad():
        # Get a sample from validation set
        sample = val_dataset.data[0]
        puzzle = sample['puzzle']
        
        # Prepare input
        input_ids = [token_to_id['<BOS>']] + encode(puzzle) + [token_to_id['<SEP>']]
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        
        # Generate solution
        generated = model.generate(x, max_new_tokens=10, temperature=0.8, do_sample=True)
        output = decode(generated[0].tolist())
        
        print(f"\nTest generation:")
        print(f"Input: {puzzle[:100]}...")
        print(f"Generated: {output}")
        print(f"Actual solution: {sample['solution']}")
    
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()