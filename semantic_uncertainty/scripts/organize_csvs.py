#!/usr/bin/env python3
"""
Organize CSV files by dataset-model-entropy combinations.
"""

import os
import shutil
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

def extract_metadata_from_filename(filename):
    """
    Extract dataset, model, and entropy from filename.
    Expected format: detail_[dataset]__[model]_entropy_[value]_[category].csv
    """
    pattern = r'detail_(\w+)__(\w+)_entropy_([\d.]+)_(\w+)\.csv'
    match = re.match(pattern, filename)

    if match:
        dataset, model, entropy, category = match.groups()
        return {
            'dataset': dataset,
            'model': model,
            'entropy': entropy,
            'category': category,
            'filename': filename
        }
    return None

def get_model_from_csv(filepath):
    """
    Read the first row of CSV to get the model value.
    """
    try:
        df = pd.read_csv(filepath, nrows=1)
        if 'model' in df.columns:
            return df['model'].iloc[0]
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
    return None

def main():
    base_dir = Path('/home/kalashkala/Datasets/Semantic-Uncertainty/sample-data')

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return

    # Collect all CSV files
    csv_files = sorted(base_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files\n")

    # Track results
    organized_files = defaultdict(list)
    mismatched_files = []
    unmatched_files = []

    # Process each file
    for filepath in csv_files:
        filename = filepath.name
        metadata = extract_metadata_from_filename(filename)

        if not metadata:
            unmatched_files.append(filename)
            continue

        # Verify model matches between filename and CSV content
        csv_model = get_model_from_csv(filepath)

        if csv_model and csv_model != metadata['model']:
            mismatched_files.append({
                'filename': filename,
                'filename_model': metadata['model'],
                'csv_model': csv_model
            })
            continue

        # Create directory name
        dir_name = f"{metadata['dataset']}__{metadata['model']}__{metadata['entropy']}"
        organized_files[dir_name].append(filename)

    # Create directories and organize files
    print("Creating directories and organizing files...\n")
    created_dirs = []
    moved_files_count = 0

    for dir_name, files in sorted(organized_files.items()):
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        created_dirs.append(dir_name)

        for filename in files:
            src = base_dir / filename
            dst = dir_path / filename
            shutil.move(str(src), str(dst))
            moved_files_count += 1

    # Print summary
    print("=" * 70)
    print("ORGANIZATION COMPLETE")
    print("=" * 70)
    print(f"\nDirectories created: {len(created_dirs)}")
    for dir_name in sorted(created_dirs):
        file_count = len(organized_files[dir_name])
        print(f"  ✓ {dir_name}/ ({file_count} files)")

    print(f"\nTotal files moved: {moved_files_count}")

    if mismatched_files:
        print(f"\n⚠️  Files with model mismatch (SKIPPED): {len(mismatched_files)}")
        for item in mismatched_files:
            print(f"  - {item['filename']}")
            print(f"    Filename says: {item['filename_model']}, CSV says: {item['csv_model']}")

    if unmatched_files:
        print(f"\n⚠️  Files that don't match expected pattern (SKIPPED): {len(unmatched_files)}")
        for filename in unmatched_files:
            print(f"  - {filename}")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
