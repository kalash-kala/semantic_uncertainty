from uncertainty.utils import utils
from compute_uncertainty_measures import main as main_compute

parser = utils.get_parser(stages=['generate', 'compute'])
args = parser.parse_args([
    '--dataset', 'answerable_math',
    '--model_name', 'Qwen/Qwen2.5-7B-Instruct',
    '--metric', 'math',
    '--reasoning',
    '--num_generations', '10',
    '--no-use_context',
    '--brief_prompt', 'cot',
    '--entailment_model', 'deberta',
    '--num_samples', '3000',
    '--no-compute_p_ik',
    '--no-compute_p_ik_answerable',
    '--no-compute_p_true_in_compute_stage',
    '--eval_wandb_runid', 'local',
])
main_compute(args)
