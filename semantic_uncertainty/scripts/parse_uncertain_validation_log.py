import re
import json
import csv
from typing import List, Dict, Any, Optional

START_MARKER = "Starting with dataset_split validation."


def parse_log_metrics(log_file_path: str, output_jsonl_path: str, output_csv_path: str) -> List[Dict[str, Any]]:
    """
    Parse uncertainty_run_qwen.log starting only after the validation split begins.

    Extracts, for each validation example:
      - iteration
      - question
      - ground_truth / correct_answer
      - low_temperature_prediction
      - high_temperature_predictions (list)
      - first_high_temperature_prediction (convenience field)
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Start parsing only after the validation marker appears.
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if START_MARKER in line:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError(f"Could not find start marker: {START_MARKER!r}")

    results: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def finalize_current() -> None:
        nonlocal current
        if current is None:
            return
        current['first_high_temperature_prediction'] = (
            current['high_temperature_predictions'][0]
            if current['high_temperature_predictions'] else None
        )
        results.append(current)
        current = None

    for raw_line in lines[start_idx:]:
        line = raw_line.strip()

        # Detect the start of a new example.
        m_iter = re.search(r'Iteration\s+(\d+):', line)
        if m_iter:
            finalize_current()
            current = {
                'iteration': int(m_iter.group(1)),
                'question': None,
                'ground_truth': None,
                'low_temperature_prediction': None,
                'high_temperature_predictions': [],
            }
            continue

        if current is None:
            continue

        # Extract question.
        m_question = re.search(r'question:\s+(.*)$', line)
        if m_question:
            current['question'] = m_question.group(1).strip()
            continue

        # Extract ground truth / correct answer.
        m_correct = re.search(r'correct answer:\s+(.*)$', line)
        if m_correct:
            value = m_correct.group(1).strip()
            current['ground_truth'] = value
            continue

        # Extract low-temperature prediction.
        m_low = re.search(r'low-t prediction:\s+(.*)$', line)
        if m_low:
            current['low_temperature_prediction'] = m_low.group(1).strip()
            continue

        # Extract each high-temperature prediction.
        m_high = re.search(r'high-t prediction\s+\d+\s*:\s+(.*)$', line)
        if m_high:
            current['high_temperature_predictions'].append(m_high.group(1).strip())
            continue

    # Final example.
    finalize_current()

    # Write JSONL.
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Write CSV.
    csv_rows = []
    for item in results:
        csv_rows.append({
            'iteration': item['iteration'],
            'question': item['question'],
            'ground_truth': item['ground_truth'],
            'low_temperature_prediction': item['low_temperature_prediction'],
            'first_high_temperature_prediction': item['first_high_temperature_prediction'],
            'high_temperature_predictions': json.dumps(item['high_temperature_predictions'], ensure_ascii=False),
        })

    fieldnames = [
        'iteration',
        'question',
        'ground_truth',
        'low_temperature_prediction',
        'first_high_temperature_prediction',
        'high_temperature_predictions',
    ]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Parsed {len(results)} validation examples.")
    print(f"JSONL written to: {output_jsonl_path}")
    print(f"CSV written to: {output_csv_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('log_file_path', nargs='?', default='uncertainty_run_qwen.log')
    parser.add_argument('output_jsonl_path', nargs='?', default='parsed_validation_predictions.jsonl')
    parser.add_argument('output_csv_path', nargs='?', default='parsed_validation_predictions.csv')
    args = parser.parse_args()

    parse_log_metrics(
        args.log_file_path,
        args.output_jsonl_path,
        args.output_csv_path,
    )