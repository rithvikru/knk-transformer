# Knights and Knaves Transformer Training

## Quick Start

1. **Generate dataset** (if not already done):
   ```bash
   # Option 1: Generate full dataset
   cd ../lean-knk
   lake exe knk --n 2 --count 100000 --out n_2
   
   # Option 2: Generate test dataset (smaller, faster)
   ./generate_test_data.sh
   ```

2. **Check dataset integrity**:
   ```bash
   python check_dataset.py ../lean-knk/data/n_2.jsonl
   ```

3. **Run training**:
   ```bash
   # With wandb logging
   uv run knk-transformer
   
   # Without wandb (for testing)
   uv run knk-transformer --no-wandb
   ```

## Common Issues

### JSON Decode Error
If you see `JSONDecodeError`, the dataset might be corrupted. Run the dataset checker:
```bash
python check_dataset.py ../lean-knk/data/n_2.jsonl
```

### Out of Memory
Reduce batch size in `knkgpt.py`:
```python
batch_size = 32  # or even 16
```

### Dataset Not Found
The trainer looks for the dataset at `../lean-knk/data/n_2.jsonl`. If your dataset is elsewhere, update the path in `knkgpt.py`.

## Configuration

Edit `knkgpt.py` to change:
- `max_length`: Maximum sequence length (default 256)
- `batch_size`: Batch size (default 64)
- `n_epochs`: Number of training epochs (default 20)
- `n_layer`, `n_head`, `n_embd`: Model architecture

## Model Sizes

- **Small** (default): 6 layers, 8 heads, 512 dim (~25M params)
- **Medium**: 12 layers, 12 heads, 768 dim (~124M params)
- **Large**: 24 layers, 16 heads, 1024 dim (~354M params)

