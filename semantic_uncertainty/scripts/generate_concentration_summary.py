#!/usr/bin/env python3
"""
Generate summary tables from concentration output CSV files.
Processes LLM verdict concentration outputs and creates count summaries by category.
"""

import os
import glob
import re
import argparse
import pandas as pd

BASE_DIR = "/home/kalashkala/output"
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "summary")


def parse_filename(filepath):
    """Extract model and dataset name from concentration output filename pattern."""
    basename = os.path.basename(filepath)
    # Pattern: uncertainty_run_{model}_{dataset}_combined_llm_verdict_concentration_output.csv
    # with optional _combined_50K etc.
    match = re.match(r"uncertainty_run_(.+?)_(.+?)(?:_combined.*)?_llm_verdict_concentration_output\.csv", basename)
    if match:
        model = match.group(1)
        dataset = match.group(2)
        return model, dataset
    return None, None


def process_file(filepath):
    """Process a single concentration output CSV file and return summary row."""
    model, dataset = parse_filename(filepath)
    if model is None:
        print(f"Skipping unrecognized file: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Map LLM_verdict to correctness
    df["correct"] = df["LLM_verdict"] == True
    df["incorrect"] = df["LLM_verdict"] == False

    total = len(df)
    total_correct = int(df["correct"].sum())
    total_incorrect = int(df["incorrect"].sum())

    # Count each category
    category_counts = df["category"].value_counts().to_dict()

    # Safely get counts for each category (0 if not present)
    correct_high = int(category_counts.get("correct_high", 0))
    correct_low = int(category_counts.get("correct_low", 0))
    incorrect_high = int(category_counts.get("incorrect_high", 0))
    incorrect_low = int(category_counts.get("incorrect_low", 0))

    # Calculate percentages within their correctness buckets
    correct_high_pct = round(100 * correct_high / total_correct, 2) if total_correct > 0 else 0
    correct_low_pct = round(100 * correct_low / total_correct, 2) if total_correct > 0 else 0
    incorrect_high_pct = round(100 * incorrect_high / total_incorrect, 2) if total_incorrect > 0 else 0
    incorrect_low_pct = round(100 * incorrect_low / total_incorrect, 2) if total_incorrect > 0 else 0

    correct_pct = round(100 * total_correct / total, 2) if total > 0 else 0
    incorrect_pct = round(100 * total_incorrect / total, 2) if total > 0 else 0

    summary_row = {
        "model": model,
        "dataset": dataset,
        "total": total,
        "correct": total_correct,
        "correct_%": correct_pct,
        "incorrect": total_incorrect,
        "incorrect_%": incorrect_pct,
        "correct_high": correct_high,
        "correct_high_%": correct_high_pct,
        "correct_low": correct_low,
        "correct_low_%": correct_low_pct,
        "incorrect_high": incorrect_high,
        "incorrect_high_%": incorrect_high_pct,
        "incorrect_low": incorrect_low,
        "incorrect_low_%": incorrect_low_pct,
    }

    return summary_row


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary tables from concentration output CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default input and output directories
  python generate_concentration_summary.py

  # Specify custom input directory
  python generate_concentration_summary.py --input-dir /path/to/concentration/output

  # Specify custom output directory and enable verbose output
  python generate_concentration_summary.py --output-dir ./results --verbose

  # Process specific CSV files
  python generate_concentration_summary.py --csv-files file1.csv file2.csv
        '''
    )

    parser.add_argument(
        "-d", "--input-dir",
        type=str,
        default=BASE_DIR,
        metavar="PATH",
        help=f"Input directory containing concentration output CSV files (default: {BASE_DIR})"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help=f"Output directory for saving summary CSV (default: {DEFAULT_OUTPUT_DIR})"
    )

    parser.add_argument(
        "--csv-files",
        nargs="+",
        default=None,
        metavar="FILE",
        help="List of CSV file paths to process (space-separated). If not provided, searches input directory for all matching files."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Use provided CSV files or fall back to glob search in input directory
    if args.csv_files:
        csv_files = sorted(args.csv_files)
    else:
        csv_files = sorted(glob.glob(
            os.path.join(args.input_dir, "*_llm_verdict_concentration_output.csv")
        ))

    if not csv_files:
        print(f"No concentration output CSV files found in {args.input_dir}")
        return

    if args.verbose:
        print(f"[*] Found {len(csv_files)} CSV file(s) to process")
        for f in csv_files:
            print(f"    - {os.path.basename(f)}")

    summary_rows = []

    for filepath in csv_files:
        if args.verbose:
            print(f"[*] Processing: {filepath}")
        else:
            print(f"Processing: {os.path.basename(filepath)}")

        summary_row = process_file(filepath)
        if summary_row is None:
            continue

        summary_rows.append(summary_row)

        if args.verbose:
            model = summary_row["model"]
            dataset = summary_row["dataset"]
            total = summary_row["total"]
            correct = summary_row["correct"]
            incorrect = summary_row["incorrect"]
            c_high = summary_row["correct_high"]
            c_low = summary_row["correct_low"]
            ic_high = summary_row["incorrect_high"]
            ic_low = summary_row["incorrect_low"]
            print(f"  [✓] {model} / {dataset} — {total} samples "
                  f"(Correct: {correct} [high: {c_high}, low: {c_low}], "
                  f"Incorrect: {incorrect} [high: {ic_high}, low: {ic_low}])")
        else:
            print(f"  -> {summary_row['model']} / {summary_row['dataset']} — "
                  f"{summary_row['total']} samples")

    if not summary_rows:
        print("No files were successfully processed.")
        return

    # Create and save summary dataframe
    summary_df = pd.DataFrame(summary_rows)

    # Reorder columns for readability
    column_order = [
        "model", "dataset", "total",
        "correct", "correct_%", "incorrect", "incorrect_%",
        "correct_high", "correct_high_%",
        "correct_low", "correct_low_%",
        "incorrect_high", "incorrect_high_%",
        "incorrect_low", "incorrect_low_%",
    ]
    summary_df = summary_df[column_order]

    summary_path = os.path.join(output_dir, "concentration_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if args.verbose:
        print(f"\n[✓] Saved summary table: {summary_path}")
    else:
        print(f"\nSaved summary table: {summary_path}")

    print("\n=== Concentration Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
