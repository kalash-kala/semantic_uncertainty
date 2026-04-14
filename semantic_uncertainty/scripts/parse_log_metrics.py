import re
import json
import ast
import csv
import argparse
from pathlib import Path


def parse_log_metrics(log_file_path, output_path):
    items = []
    p_trues = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print("Using a list-based extraction to maintain ordered sequence.")

    for i in range(len(lines)):
        line = lines[i].strip()

        # Match new item
        m_item = re.search(r'NEW ITEM \d+ at id=`([^`]+)`\.', line)
        if m_item:
            item_data = {'id': m_item.group(1)}
            context = None
            question = None

            # Look ahead for attributes
            for j in range(i + 1, min(i + 60, len(lines))):
                if 'NEW ITEM' in lines[j]:
                    break

                if 'Context:' in lines[j]:
                    if j + 1 < len(lines):
                        m_val = re.search(r'INFO\s+(.*)', lines[j + 1].strip())
                        if m_val:
                            context = m_val.group(1).strip()

                if 'Question:' in lines[j]:
                    if j + 1 < len(lines):
                        m_val = re.search(r'INFO\s+(.*)', lines[j + 1].strip())
                        if m_val:
                            question = m_val.group(1).strip()

                if 'True Answers:' in lines[j]:
                    if j + 1 < len(lines):
                        m_val = re.search(r'INFO\s+(.*)', lines[j + 1].strip())
                        if m_val:
                            content = m_val.group(1).strip()
                            try:
                                ans_dict = ast.literal_eval(content)
                                if (
                                    isinstance(ans_dict, dict)
                                    and 'answers' in ans_dict
                                    and 'text' in ans_dict['answers']
                                ):
                                    item_data['ground_truth'] = ans_dict['answers']['text']
                                else:
                                    item_data['ground_truth'] = content
                            except (SyntaxError, ValueError):
                                item_data['ground_truth'] = content

                if 'Low Temperature Generation:' in lines[j] and 'Accuracy' not in lines[j]:
                    if j + 1 < len(lines):
                        m_val = re.search(r'INFO\s+(.*)', lines[j + 1].strip())
                        if m_val:
                            item_data['low_t_generation'] = m_val.group(1).strip().replace(',', '')

                if 'Low Temperature Generation Accuracy:' in lines[j]:
                    if j + 1 < len(lines):
                        m_val = re.search(r'INFO\s+([-\d\.]+)', lines[j + 1].strip())
                        if m_val:
                            try:
                                item_data['accuracy'] = float(m_val.group(1))
                            except ValueError:
                                pass

                if 'High Temp Generation:' in lines[j]:
                    if j + 1 < len(lines):
                        val_line = lines[j + 1].strip()
                        m_val = re.search(r'INFO\s+(.*)', val_line)
                        if m_val:
                            content = m_val.group(1).strip()
                            if content.startswith("['") or content.startswith('["'):
                                try:
                                    parsed = ast.literal_eval(content)
                                    item_data['n_generations'] = [s.replace(',', '') for s in parsed]
                                except Exception:
                                    item_data['n_generations'] = content
                            elif content.startswith('semantic_ids:'):
                                m_ent = re.search(r'cluster_assignment_entropy:([-\d\.]+)', content)
                                if m_ent:
                                    try:
                                        item_data['cluster_assignment_entropy'] = float(m_ent.group(1))
                                    except ValueError:
                                        pass

            # Combine context and question if both exist
            if context and question:
                item_data['question'] = context + '\n' + question
            elif question:
                item_data['question'] = question

            items.append(item_data)

        # Match p_true
        m_ptrue = re.search(r'p_true:\s+([-\d\.eE]+)', line)
        if m_ptrue:
            try:
                p_trues.append(float(m_ptrue.group(1)))
            except ValueError:
                pass

    print(f"Found {len(items)} items and {len(p_trues)} p_true values.")

    # Pair them up by order
    for idx, item in enumerate(items):
        if idx < len(p_trues):
            item['p_true'] = p_trues[idx]

    output_path = str(output_path)
    if output_path.endswith('.csv'):
        if not items:
            print('No items found. Nothing written.')
            return

        keys = []
        seen = set()
        for item in items:
            for key in item.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)

        if 'id' in keys:
            keys.insert(0, keys.pop(keys.index('id')))

        with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(items)
        print(f"Exported to {output_path}")
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Exported to {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description='Parse uncertainty log metrics into JSONL or CSV.'
    )
    parser.add_argument(
        '--input_log',
        help='Path to the input log file, e.g. uncertainty_run_qwen.log'
    )
    parser.add_argument(
        '--output_file',
        help='Path to the output file. Use .jsonl or .csv extension.'
    )
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_log)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f'Input log file not found: {input_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parse_log_metrics(str(input_path), str(output_path))
