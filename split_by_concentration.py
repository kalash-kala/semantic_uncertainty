import pandas as pd
import ast
import json
import os

def calculate_c(greedy, samples):
    responses = [str(greedy).strip().lower()] + [str(s).strip().lower() for s in samples]
    N = len(responses)
    if N <= 1:
        return 1.0
    
    counts = {}
    for r in responses:
        counts[r] = counts.get(r, 0) + 1
        
    c = sum(n * (n - 1) for n in counts.values()) / (N * (N - 1))
    return c

def main():
    # Manually add file names to process here
    csv_files = [
        "/home/kalashkala/Datasets/Semantic-Uncertainty/triviaqa/uncertainty_run_llama_triviaqa_combined_50K_llm_verdict.csv",
        "/home/kalashkala/Datasets/Semantic-Uncertainty/triviaqa/uncertainty_run_mistral_triviaqa_combined_50K_llm_verdict.csv",
        "/home/kalashkala/Datasets/Semantic-Uncertainty/triviaqa/uncertainty_run_qwen_triviaqa_combined_50K_llm_verdict.csv",
    ]
    output_base_dir = "/home/kalashkala/output/non_inst_concentration_output"  # Change this to your desired output directory

    if not csv_files:
        print("No CSV files provided. Please add file paths to the csv_files list.")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        filename = os.path.basename(csv_file)

        # Remove .csv extension and add concentration_output suffix
        base_filename = filename.replace('.csv', '')
        output_filename = f"{base_filename}_concentration_output.csv"

        df = pd.read_csv(csv_file)

        all_records = []

        for idx, row in df.iterrows():
            greedy = row['low_t_generation']

            try:
                samples = ast.literal_eval(row['n_generations'])
            except:
                # Fallback if it's not a valid list string
                samples = []

            c_val = calculate_c(greedy, samples)

            # Determine correctness based on LLM_verdict
            is_correct = str(row['LLM_verdict']).strip().lower() in ['true', '1.0', '1']

            record = row.to_dict()

            category = ""
            if is_correct and c_val >= 0.5:
                category = 'correct_high'
            elif is_correct and c_val < 0.5:
                category = 'correct_low'
            elif not is_correct and c_val >= 0.5:
                category = 'incorrect_high'
            elif not is_correct and c_val < 0.5:
                category = 'incorrect_low'

            record['C_metric'] = c_val
            record['category'] = category

            all_records.append(record)

        # Write to CSV
        os.makedirs(output_base_dir, exist_ok=True)
        output_path = os.path.join(output_base_dir, output_filename)
        output_df = pd.DataFrame(all_records)
        output_df.to_csv(output_path, index=False)
        print(f"  Saved {len(all_records)} records to {output_path}")

    print("Done!")

if __name__ == "__main__":
    main()