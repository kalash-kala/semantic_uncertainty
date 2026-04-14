#!/usr/bin/env python3
"""
Convert JSON files to CSV format with semantic uncertainty data.
Supports batch conversion of multiple JSON files.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any


def convert_json_to_csv(
    json_file: Path,
    csv_file: Path,
    model_name: str,
    label: str = None,
) -> None:
    """
    Convert a JSON file to CSV format.

    Args:
        json_file: Path to input JSON file
        csv_file: Path to output CSV file
        model_name: Model name to add to CSV
        label: Label to add to CSV (default: extracted from filename)
    """
    try:
        # Read JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        # Extract label from filename if not provided
        if label is None:
            # Extract label from filename pattern: *_LABEL.json
            filename = json_file.stem  # filename without extension
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                label = parts[1]
            else:
                label = 'Unknown'

        # CSV column headers
        fieldnames = [
            'model',
            'question',
            'ground_truth',
            'prediction',
            'sampled_answers',
            'semantic_entropy',
            'correct',
            'label'
        ]

        # Convert data and write to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in data:
                # Convert accuracy to boolean (0/0.0 -> False, 1/1.0 -> True)
                accuracy = item.get('accuracy', 0)
                correct_value = str(bool(accuracy))

                # Map JSON fields to CSV columns
                row = {
                    'model': model_name,
                    'question': item.get('question', ''),
                    'ground_truth': json.dumps(item.get('true_answer', '')),
                    'prediction': item.get('generated', ''),
                    'sampled_answers': json.dumps(item.get('temp_generations', [])),
                    'semantic_entropy': item.get('semantic_entropy', ''),
                    'correct': correct_value,
                    'label': label
                }
                writer.writerow(row)

        print(f"✓ Converted {json_file} → {csv_file}")
        print(f"  Rows written: {len(data)}")

    except FileNotFoundError:
        print(f"✗ Error: File not found: {json_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in {json_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error processing {json_file}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON files to CSV format with semantic uncertainty data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single JSON file
  python json_to_csv_converter.py input.json -o output.csv -m llama

  # Convert with auto-generated output filename
  python json_to_csv_converter.py input.json -m llama

  # Convert multiple files
  python json_to_csv_converter.py file1.json file2.json -m llama
        """
    )

    parser.add_argument(
        'input',
        nargs='+',
        type=Path,
        help='Input JSON file(s) to convert'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output CSV file path (default: input filename with .csv extension)'
    )

    parser.add_argument(
        '-m', '--model',
        required=True,
        help='Model name to include in CSV (e.g., llama, gpt4)'
    )

    parser.add_argument(
        '-l', '--label',
        help='Label for CSV (default: extracted from filename, e.g., "AH" from "*_AH.json")'
    )

    args = parser.parse_args()

    # Handle multiple input files
    if len(args.input) > 1:
        if args.output:
            print("✗ Error: Cannot specify -o/--output with multiple input files", file=sys.stderr)
            sys.exit(1)

        for json_file in args.input:
            csv_file = json_file.with_suffix('.csv')
            convert_json_to_csv(json_file, csv_file, args.model, args.label)
    else:
        json_file = args.input[0]
        csv_file = args.output if args.output else json_file.with_suffix('.csv')
        convert_json_to_csv(json_file, csv_file, args.model, args.label)


if __name__ == '__main__':
    main()
