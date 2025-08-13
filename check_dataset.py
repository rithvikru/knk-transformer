#!/usr/bin/env python3
"""
Check the Knights and Knaves dataset format and integrity.
"""
import json
import sys
from pathlib import Path

def check_dataset(file_path):
    """Check JSONL dataset for format issues."""
    print(f"Checking dataset: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    valid_lines = 0
    error_lines = []
    sample_entries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                entry = json.loads(line)
                
                # Check required fields
                if 'puzzle' not in entry:
                    error_lines.append((line_num, "Missing 'puzzle' field"))
                    continue
                if 'solution' not in entry:
                    error_lines.append((line_num, "Missing 'solution' field"))
                    continue
                
                # Validate solution format
                solution = entry['solution']
                if not all(c in ['K', 'N'] for c in solution):
                    error_lines.append((line_num, f"Invalid solution format: {solution}"))
                    continue
                
                valid_lines += 1
                
                # Save first few entries as samples
                if len(sample_entries) < 3:
                    sample_entries.append(entry)
                    
            except json.JSONDecodeError as e:
                error_lines.append((line_num, f"JSON parse error: {e}"))
                if line_num <= 3:
                    print(f"  Line {line_num} content: {line[:100]}...")
                continue
            except Exception as e:
                error_lines.append((line_num, f"Unexpected error: {e}"))
                continue
    
    # Report results
    print(f"\n✅ Valid lines: {valid_lines}")
    if error_lines:
        print(f"❌ Error lines: {len(error_lines)}")
        for line_num, error in error_lines[:10]:  # Show first 10 errors
            print(f"  Line {line_num}: {error}")
        if len(error_lines) > 10:
            print(f"  ... and {len(error_lines) - 10} more errors")
    
    if sample_entries:
        print("\n📋 Sample entries:")
        for i, entry in enumerate(sample_entries):
            print(f"\nEntry {i+1}:")
            print(f"  Puzzle: {entry['puzzle'][:80]}...")
            print(f"  Solution: {entry['solution']}")
            print(f"  Solution length (n): {len(entry['solution'])}")
    
    return len(error_lines) == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_dataset.py <path_to_jsonl>")
        print("Example: python check_dataset.py ../lean-knk/data/n_2.jsonl")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    success = check_dataset(dataset_path)
    sys.exit(0 if success else 1)
