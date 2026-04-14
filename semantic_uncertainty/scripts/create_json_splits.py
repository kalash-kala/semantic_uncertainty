import os
import pandas as pd
import json
import ast
import re
import string
from collections import Counter
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def normalize_text_custom(s):
    """
    Official SQuAD normalization + Porter Stemming for plural/singular robustness.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def stem_tokens(text):
        return " ".join([stemmer.stem(w) for w in text.split()])

    return white_space_fix(stem_tokens(remove_articles(remove_punc(lower(str(s))))))

def compute_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_text_custom(prediction).split()
    ground_truth_tokens = normalize_text_custom(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_squad_score(predict_text, truth_text):
    # Parse truth string into a list if necessary (SQuAD allows multiple aliases)
    truth_list = [truth_text]
    if isinstance(truth_text, str) and truth_text.startswith('['):
        try:
            parsed = ast.literal_eval(truth_text)
            if isinstance(parsed, list):
                truth_list = parsed
        except Exception:
            pass
    
    # Check if any ground truth alias matches above the 50% F1 threshold
    f1_scores = [compute_f1_score(predict_text, t) for t in truth_list]
    max_f1 = max(f1_scores) if f1_scores else 0
    return 1.0 if (max_f1 * 100.0 >= 50.0) else 0.0

def process_file(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    # Process rows to dict format
    def format_row(row):
        truth = row.get('ground_truth', '')
        if isinstance(truth, str) and truth.startswith('['):
            try:
                truth_list = ast.literal_eval(truth)
                if isinstance(truth_list, list) and len(truth_list) > 0:
                    truth = truth_list[0]
                else:
                    truth = str(truth_list)
            except Exception:
                pass
        
        n_gens = row.get('n_generations', '[]')
        try:
            temp_generations = ast.literal_eval(n_gens) if isinstance(n_gens, str) else []
        except Exception:
            temp_generations = []

        return {
            "id": row.get('id', ''),
            "question": row.get('question', ''),
            "prompt": row.get('question', ''),
            "true_answer": str(truth),
            "generated": str(row.get('low_t_generation', '')),
            "semantic_entropy": float(row.get('cluster_assignment_entropy', 0.0)),
            "accuracy": float(row.get('accuracy', 0.0)),
            "temp_generations": temp_generations
        }

    df['formatted'] = df.apply(format_row, axis=1)

    # Recompute accuracy using SQuAD metric as requested
    df['accuracy_squad'] = df.apply(lambda row: compute_squad_score(row['formatted']['generated'], row['ground_truth']), axis=1)

    # Filtering logic
    # certainty threshold is 0.5
    certain_mask = df['cluster_assignment_entropy'] < 0.5
    uncertain_mask = df['cluster_assignment_entropy'] >= 0.5
    correct_mask = df['accuracy_squad'] == 1.0
    wrong_mask = df['accuracy_squad'] == 0.0
    
    certain_preds = df[certain_mask & correct_mask]
    certain_mispreds = df[certain_mask & wrong_mask]
    uncertain_mispreds = df[uncertain_mask & wrong_mask]

    # sample exactly 100
    def get_100_samples(sub_df, sort_col=None, ascending=True):
        if sort_col:
            sub_df = sub_df.sort_values(by=sort_col, ascending=ascending)
            num_samples = min(100, len(sub_df))
            return sub_df.head(num_samples)['formatted'].tolist()
        else:
            num_samples = min(100, len(sub_df))
            return sub_df.sample(n=num_samples, random_state=42)['formatted'].tolist()
    
    cp_list = get_100_samples(certain_preds)
    cm_list = get_100_samples(certain_mispreds, sort_col='cluster_assignment_entropy', ascending=True)
    um_list = get_100_samples(uncertain_mispreds, sort_col='cluster_assignment_entropy', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "certain_predictions_pruned_100.json"), "w") as f:
        json.dump(cp_list, f, indent=4)
        
    with open(os.path.join(output_dir, "certain_mispredictions_pruned_100.json"), "w") as f:
        json.dump(cm_list, f, indent=4)
        
    with open(os.path.join(output_dir, "uncertain_mispredictions_pruned_100.json"), "w") as f:
        json.dump(um_list, f, indent=4)

    return len(cp_list), len(cm_list), len(um_list)

def main():
    base_dir = "/home/malay/Trust_me_Im_wrong/corpus"
    jsons_dir = os.path.join(base_dir, "jsons")
    os.makedirs(jsons_dir, exist_ok=True)

    datasets = ["triviaqa", "sciq", "svamp"]
    models = ["gemma", "llama", "mistral", "qwen"]

    for d in datasets:
        for m in models:
            csv_path = os.path.join(base_dir, d, f"uncertainty_run_{m}_{d}_combined.csv")
            if not os.path.exists(csv_path):
                print(f"File missing: {csv_path}")
                continue
            
            out_dir = os.path.join(jsons_dir, f"{m}_{d}")
            cp, cm, um = process_file(csv_path, out_dir)
            print(f"[{m}_{d}] -> CP: {cp}, CM: {cm}, UM: {um}")

if __name__ == "__main__":
    main()
