# KNK Transformer

A GPT-based model for solving Knights and Knaves logic puzzles.

## Setup

Using uv (recommended) [[memory:5713825]]:

```bash
# Install dependencies
uv sync

# Run training
uv run knk-transformer
```

## Dataset Format

The model expects JSONL files with the following format:
```json
{"puzzle": "says 0 (isKnight 1), says 1 (isKnave 0)", "solution": "KN"}
```

Where:
- `puzzle`: Logical formula representing the puzzle
- `solution`: String of K (Knight) or N (Knave) for each person

## Model Architecture

- GPT-based transformer model
- 6 layers, 8 attention heads, 512 embedding dimensions
- Custom tokenizer for logic formulas
- Trained with teacher forcing on puzzle-solution pairs

## Training

The training script will:
1. Load the dataset from `../lean-knk/data/n_2v2mini.jsonl`
2. Split into 90% train, 10% validation
3. Train for 50 epochs with learning rate decay
4. Save the best model to `knk_model.pt`

## Usage

After training, the model can generate solutions for new puzzles:
```python
# Generate solution for a puzzle
input_ids = encode_puzzle(puzzle_text)
solution = model.generate(input_ids)
```


