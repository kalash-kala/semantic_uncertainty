"""Compute overall performance metrics from predicted uncertainties."""
import argparse
import functools
import json
import logging
import os
import pickle

import numpy as np

from uncertainty.utils import utils
from uncertainty.utils.eval_utils import (
    bootstrap, compatible_bootstrap, auroc, accuracy_at_quantile,
    area_under_thresholded_accuracy)


utils.setup_logger()

result_dict = {}

UNC_MEAS = 'uncertainty_measures.pkl'


def analyze_run(
        run_id, answer_fractions_mode='default', **kwargs):
    """Analyze the uncertainty measures for a given local run directory."""
    out_dir = os.environ.get('SU_LOCAL_RUN_DIR', './root/uncertainty/local_run')
    os.makedirs(out_dir, exist_ok=True)
    logging.info('Analyzing run_id `%s` from local dir `%s`.', run_id, out_dir)

    # Set up evaluation metrics.
    if answer_fractions_mode == 'default':
        answer_fractions = [0.8, 0.9, 0.95, 1.0]
    elif answer_fractions_mode == 'finegrained':
        answer_fractions = [round(i, 3) for i in np.linspace(0, 1, 20+1)]
    else:
        raise ValueError

    rng = np.random.default_rng(41)
    eval_metrics = dict(zip(
        ['AUROC', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
        list(zip(
            [auroc, area_under_thresholded_accuracy, np.mean],
            [compatible_bootstrap, compatible_bootstrap, bootstrap]
        )),
    ))
    for answer_fraction in answer_fractions:
        key = f'accuracy_at_{answer_fraction}_answer_fraction'
        eval_metrics[key] = [
            functools.partial(accuracy_at_quantile, quantile=answer_fraction),
            compatible_bootstrap]

    # Load the results dictionary from a pickle file in the local run dir.
    unc_meas_path = os.path.join(out_dir, UNC_MEAS)
    with open(unc_meas_path, 'rb') as file:
        results_old = pickle.load(file)

    result_dict = {'performance': {}, 'uncertainty': {}}

    # First: Compute simple accuracy metrics for model predictions.
    all_accuracies = dict()
    all_accuracies['accuracy'] = 1 - np.array(results_old['validation_is_false'])

    for name, target in all_accuracies.items():
        result_dict['performance'][name] = {}
        result_dict['performance'][name]['mean'] = float(np.mean(target))
        result_dict['performance'][name]['bootstrap'] = bootstrap(np.mean, rng)(target)

    rum = results_old['uncertainty_measures']
    if 'p_false' in rum and 'p_false_fixed' not in rum:
        rum['p_false_fixed'] = [1 - np.exp(1 - x) for x in rum['p_false']]

    # Next: Uncertainty Measures.
    for measure_name, measure_values in rum.items():
        logging.info('Computing for uncertainty measure `%s`.', measure_name)

        validation_is_falses = [
            results_old['validation_is_false'],
            results_old['validation_unanswerable']
        ]
        logging_names = ['', '_UNANSWERABLE']

        for validation_is_false, logging_name in zip(validation_is_falses, logging_names):
            name = measure_name + logging_name
            result_dict['uncertainty'][name] = {}

            validation_is_false = np.array(validation_is_false)
            validation_accuracy = 1 - validation_is_false
            if len(measure_values) > len(validation_is_false):
                if 'p_false' not in measure_name:
                    raise ValueError
                logging.warning(
                    'More measure values for %s than in validation_is_false. Len(measure values): %d, Len(validation_is_false): %d',
                    measure_name, len(measure_values), len(validation_is_false))
                measure_values = measure_values[:len(validation_is_false)]

            fargs = {
                'AUROC': [validation_is_false, measure_values],
                'area_under_thresholded_accuracy': [validation_accuracy, measure_values],
                'mean_uncertainty': [measure_values]}

            for answer_fraction in answer_fractions:
                fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [validation_accuracy, measure_values]

            for fname, (function, bs_function) in eval_metrics.items():
                metric_i = function(*fargs[fname])
                result_dict['uncertainty'][name][fname] = {}
                result_dict['uncertainty'][name][fname]['mean'] = float(metric_i)
                logging.info("%s for measure name `%s`: %f", fname, name, metric_i)
                result_dict['uncertainty'][name][fname]['bootstrap'] = bs_function(
                    function, rng)(*fargs[fname])

    # Save results locally instead of wandb.log
    out_path = os.path.join(out_dir, 'analyze_results.jsonl')
    with open(out_path, 'w') as f:
        f.write(json.dumps({'run_id': run_id, **result_dict}) + '\n')

    logging.info(
        'Analysis for run_id `%s` finished. Results saved to `%s`. Full results dict: %s',
        run_id, out_path, result_dict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_ids', nargs='+', type=str,
                        help='Local run IDs (used as labels in output JSONL).')
    parser.add_argument('--answer_fractions_mode', type=str, default='default')

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    for run_id in (args.run_ids or ['local']):
        logging.info('Evaluating run_id `%s`.', run_id)
        analyze_run(run_id, args.answer_fractions_mode)
