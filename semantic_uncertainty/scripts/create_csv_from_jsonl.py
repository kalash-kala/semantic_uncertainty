#!/usr/bin/env python3
"""
Create a CSV from combined_generations.jsonl and uncertainty_measures.jsonl files.
Matches the format of uncertainty_run_llama_answerable_math_combined.csv

This script searches for the required JSONL files in a specified directory
and combines them into a single CSV file with semantic uncertainty metrics.
"""

import json
import csv
import sys
import argparse
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


def find_jsonl_files(directory):
    """Find required JSONL files in the given directory."""
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    # Find combined_generations.jsonl
    combined_gen_files = list(dir_path.glob('**/combined_generations.jsonl'))
    if not combined_gen_files:
        raise FileNotFoundError(f"No 'combined_generations.jsonl' found in {directory}")
    combined_gen_path = combined_gen_files[0]

    # Find uncertainty_measures.jsonl
    uncertainty_files = list(dir_path.glob('**/uncertainty_measures.jsonl'))
    if not uncertainty_files:
        raise FileNotFoundError(f"No 'uncertainty_measures.jsonl' found in {directory}")
    uncertainty_path = uncertainty_files[0]

    return combined_gen_path, uncertainty_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate a CSV from semantic uncertainty JSONL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate CSV in the same directory with default name
  python create_csv_from_jsonl.py --directory path/to/uncertainty/run

  # Generate CSV with custom output filename
  python create_csv_from_jsonl.py --directory path/to/uncertainty/run --output my_results.csv

  # Verbose output with progress information
  python create_csv_from_jsonl.py --directory path/to/uncertainty/run --verbose
        '''
    )

    parser.add_argument(
        '-d', '--directory',
        type=str,
        required=True,
        metavar='PATH',
        help='directory containing combined_generations.jsonl and uncertainty_measures.jsonl files'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='combined_results.csv',
        metavar='FILENAME',
        help='output CSV filename (default: combined_results.csv)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='enable verbose output for debugging'
    )

    parser.add_argument(
        '--tag',
        type=str,
        default='',
        metavar='TAG',
        help='optional tag to add to the output filename (e.g., --tag qwen will output combined_results_qwen.csv)'
    )

    args = parser.parse_args()

    # Adjust output filename if tag is provided
    output_filename = args.output
    if args.tag:
        base_name = output_filename.rsplit('.', 1)[0]
        extension = output_filename.rsplit('.', 1)[1] if '.' in output_filename else 'csv'
        output_filename = f"{base_name}_{args.tag}.{extension}"

    try:
        # Find the JSONL files
        if args.verbose:
            print(f"[*] Searching for JSONL files in: {args.directory}")

        combined_gen_path, uncertainty_path = find_jsonl_files(args.directory)

        if args.verbose:
            print(f"[✓] Found combined_generations.jsonl: {combined_gen_path}")
            print(f"[✓] Found uncertainty_measures.jsonl: {uncertainty_path}")

        # Load both JSONL files
        if not args.verbose:
            print(f"Loading combined generations...")
        else:
            print(f"[*] Loading combined generations from {combined_gen_path}...")
        combined_gen = load_jsonl(combined_gen_path)

        if args.verbose:
            print(f"[✓] Loaded {len(combined_gen)} records from combined_generations.jsonl")

        if not args.verbose:
            print(f"Loading uncertainty measures...")
        else:
            print(f"[*] Loading uncertainty measures from {uncertainty_path}...")
        uncertainty = load_jsonl(uncertainty_path)

        if args.verbose:
            print(f"[✓] Loaded {len(uncertainty)} records from uncertainty_measures.jsonl")

        # Get all unique IDs
        all_ids = set(combined_gen.keys()) | set(uncertainty.keys())
        if args.verbose:
            print(f"[✓] Found {len(all_ids)} unique records to process")
        else:
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
            # Extract actual response texts from responses (first element of each tuple)
            n_generations = [resp[0] for resp in gen_data.get('responses', [])]
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
        output_path = Path(args.directory) / output_filename
        if args.verbose:
            print(f"[*] Writing CSV to {output_path}...")
        else:
            print(f"Writing CSV...")

        fieldnames = ['id', 'ground_truth', 'low_t_generation', 'accuracy', 'n_generations',
                      'cluster_assignment_entropy', 'question', 'p_true']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if args.verbose:
            print(f"[✓] Done! Created {output_path} with {len(rows)} records")
        else:
            print(f"Done! Created {output_path} with {len(rows)} records")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
