#!/bin/bash
# Generate test dataset for Knights and Knaves transformer

echo "Generating test dataset..."

# Create data directory
mkdir -p ../lean-knk/data

# Generate small n=2 dataset for testing
cd ../lean-knk
lake exe knk --n 2 --count 10000 --out n_2_test

echo "Test dataset generated at ../lean-knk/data/n_2_test.jsonl"

# Check the dataset
cd ../knk-transformer
python check_dataset.py ../lean-knk/data/n_2_test.jsonl
