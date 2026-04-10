import re
import json
import ast

def parse_log_metrics(log_file_path, output_path):
    results = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print("Using a list-based extraction to maintain ordered sequence.")
    p_trues = []
    items = []
    
    for i in range(len(lines)):
        line = lines[i].strip()
        
        # Match new item
        m_item = re.search(r'NEW ITEM \d+ at id=`([^`]+)`\.', line)
        if m_item:
            item_data = {'id': m_item.group(1)}
            # Look ahead for attributes
            for j in range(i+1, min(i+60, len(lines))):
                if 'NEW ITEM' in lines[j]:
                    break
                    
                if 'Question:' in lines[j]:
                    if j+1 < len(lines):
                        m_val = re.search(r'INFO\s+(.*)', lines[j+1].strip())
                        if m_val:
                            item_data['question'] = m_val.group(1).strip()
                            
                if 'True Answers:' in lines[j]:
                    if j+1 < len(lines):
                        m_val = re.search(r'INFO\s+(.*)', lines[j+1].strip())
                        if m_val:
                            content = m_val.group(1).strip()
                            try:
                                ans_dict = ast.literal_eval(content)
                                if isinstance(ans_dict, dict) and 'answers' in ans_dict and 'text' in ans_dict['answers']:
                                    item_data['ground_truth'] = ans_dict['answers']['text']
                                else:
                                    item_data['ground_truth'] = content
                            except (SyntaxError, ValueError):
                                item_data['ground_truth'] = content

                if 'Low Temperature Generation:' in lines[j] and 'Accuracy' not in lines[j]:
                    if j+1 < len(lines):
                        m_val = re.search(r'INFO\s+(.*)', lines[j+1].strip())
                        if m_val:
                            item_data['low_t_generation'] = m_val.group(1).strip()
                            
                if 'Low Temperature Generation Accuracy:' in lines[j]:
                    if j+1 < len(lines):
                        m_val = re.search(r'INFO\s+([\d\.]+)', lines[j+1].strip())
                        if m_val:
                            try:
                                item_data['accuracy'] = float(m_val.group(1))
                            except ValueError:
                                pass
                                
                if 'High Temp Generation:' in lines[j]:
                    if j+1 < len(lines):
                        val_line = lines[j+1].strip()
                        m_val = re.search(r'INFO\s+(.*)', val_line)
                        if m_val:
                            content = m_val.group(1).strip()
                            if content.startswith("['") or content.startswith('["'):
                                try:
                                    item_data['n_generations'] = ast.literal_eval(content)
                                except Exception:
                                    item_data['n_generations'] = content
                            elif content.startswith("semantic_ids:"):
                                m_ent = re.search(r'cluster_assignment_entropy:([-\d\.]+)', content)
                                if m_ent:
                                    try:
                                        item_data['cluster_assignment_entropy'] = float(m_ent.group(1))
                                    except ValueError:
                                        pass
                                        
            items.append(item_data)
            
        # Match p_true
        m_ptrue = re.search(r'p_true:\s+([-\d\.e]+)', line)
        if m_ptrue:
            try:
                p_trues.append(float(m_ptrue.group(1)))
            except ValueError:
                pass
                
    # Pair them up
    print(f"Found {len(items)} items and {len(p_trues)} p_true values.")
    for idx, item in enumerate(items):
        if idx < len(p_trues):
            item['p_true'] = p_trues[idx]
            
    if output_path.endswith('.csv'):
        if not items: return
        import csv
        # Order keys explicitly if we want question and ground truth to be early, but dict preserves insertion order since py3.7
        keys = set()
        for item in items:
            keys.update(item.keys())
        keys = list(keys)
        # Re-sort to put ID at start
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
                f.write(json.dumps(item) + '\n')
        print(f"Exported to {output_path}")
                
if __name__ == '__main__':
    parse_log_metrics('uncertainty_run_qwen.log', 'parsed_log_metrics.jsonl')
    parse_log_metrics('uncertainty_run_qwen.log', 'parsed_log_metrics.csv')
    print("Parsing complete.")
