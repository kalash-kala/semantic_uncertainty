#!/usr/bin/env python3
"""
Create a CSV from combined_generations.jsonl and uncertainty_measures.jsonl files.
Matches the format of uncertainty_run_llama_answerable_math_combined.csv
"""

import json
import csv
import sys
from pathlib import Path


def load_jsonl(file_path):
    """Load a JSONL file and return a dictionary keyed by id."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                # Each line has one key-value pair where key is the id
                for key, value in obj.items():
                    data[key] = value
    return data


def format_list_string(items):
    """Format a list as a string like ['item1', 'item2']"""
    return str(items)


def main():
    if len(sys.argv) < 4:
        print("Usage: python create_csv_from_jsonl.py <combined_generations_path> <uncertainty_measures_path> <output_csv_path>")
        sys.exit(1)

    combined_gen_path = Path(sys.argv[1])
    uncertainty_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    # Load both JSONL files
    print(f"Loading combined generations from {combined_gen_path}...")
    combined_gen = load_jsonl(combined_gen_path)

    print(f"Loading uncertainty measures from {uncertainty_path}...")
    uncertainty = load_jsonl(uncertainty_path)

    # Get all unique IDs
    all_ids = set(combined_gen.keys()) | set(uncertainty.keys())
    print(f"Found {len(all_ids)} records")

    # Prepare CSV data
    rows = []
    for id_key in sorted(all_ids):
        gen_data = combined_gen.get(id_key, {})
        unc_data = uncertainty.get(id_key, {})

        # Extract fields
        ground_truth = gen_data.get('reference', {}).get('answers', {}).get('text', [])
        most_likely = gen_data.get('most_likely_answer', {})
        low_t_generation = most_likely.get('response', '')
        accuracy = most_likely.get('accuracy', 0.0)
        n_generations = len(gen_data.get('responses', []))
        cluster_entropy = unc_data.get('cluster_assignment_entropy', '')
        question = gen_data.get('question', '')
        p_true = gen_data.get('p_true', '')

        row = {
            'id': id_key,
            'ground_truth': format_list_string(ground_truth),
            'low_t_generation': low_t_generation,
            'accuracy': accuracy,
            'n_generations': n_generations,
            'cluster_assignment_entropy': cluster_entropy,
            'question': question,
            'p_true': p_true,
        }
        rows.append(row)

    # Write CSV
    print(f"Writing CSV to {output_path}...")
    fieldnames = ['id', 'ground_truth', 'low_t_generation', 'accuracy', 'n_generations',
                  'cluster_assignment_entropy', 'question', 'p_true']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Created {output_path} with {len(rows)} records")


if __name__ == '__main__':
    main()
