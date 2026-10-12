"""Collect hashed run, numerical comparison and quality evidence."""
import argparse
import json
from pathlib import Path
import statistics
from delivery_utils import finite, sha256, write_json


ENVIRONMENT_KEYS = ('python', 'logical_device', 'ascend_home_path', 'cann_version',
                    'device_name', 'hostname')


def validate_environment(run):
    if type(run['logical_device']) is not int or run['logical_device'] < 0:
        raise ValueError('Invalid logical device')
    for key in ENVIRONMENT_KEYS:
        if key == 'logical_device':
            continue
        if not isinstance(run[key], str) or run[key].strip() in ('', 'not known', 'N/A'):
            raise ValueError(f'Missing NPU environment evidence: {key}')


def performance(run_dirs, reference):
    if len(run_dirs) < 3 or len({Path(p).resolve() for p in run_dirs}) != len(run_dirs):
        raise ValueError('At least three distinct process output directories required')
    validate_environment(reference)
    records, identities, medians = [], set(), []
    for path in run_dirs:
        path = Path(path)
        run = json.loads((path / 'run.json').read_text())
        if run['simulated'] or run['device'] != 'npu':
            raise ValueError('Real NPU performance evidence required')
        if not run['run_id'] or run['run_id'] in identities:
            raise ValueError('Duplicate or empty process run ID')
        identities.add(run['run_id'])
        validate_environment(run)
        for key in ENVIRONMENT_KEYS:
            if run[key] != reference[key]:
                raise ValueError(f'Performance environment differs: {key}')
        for key in ('manifest_sha256', 'checkpoint_sha256', 'backbone_sha256',
                    'upstream_commit', 'dtype', 'samples', 'proteins', 'warmup',
                    'runs', 'timing_scope', 'torch', 'torch_npu', 'visible_devices'):
            if run[key] != reference[key]:
                raise ValueError(f'Performance configuration differs: {key}')
        if run['predictions_sha256'] != sha256(path / 'predictions.jsonl'):
            raise ValueError('Performance predictions hash mismatch')
        seconds = [finite(value) for value in run['seconds']]
        if len(seconds) != run['runs'] or len(seconds) < 3 or min(seconds) <= 0:
            raise ValueError('Invalid performance durations')
        medians.append(statistics.median(seconds))
        records.append({'directory': str(path), 'run_sha256': sha256(path / 'run.json'), 'run': run})
    if reference['samples'] <= 0 or reference['proteins'] <= 0:
        raise ValueError('Positive sample and protein counts required')
    median = statistics.median(medians)
    return {'processes': len(records), 'process_median_seconds': medians,
            'median_seconds': median, 'mutations_per_second': reference['samples'] / median,
            'proteins_per_second': reference['proteins'] / median, 'evidence': records}


def collect(run_dir, comparison_path, metrics_path, performance_dirs=None):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / 'run.json').read_text())
    comparison = json.loads(Path(comparison_path).read_text())
    metrics = json.loads(Path(metrics_path).read_text())
    digest = sha256(run_dir / 'predictions.jsonl')
    if run['predictions_sha256'] != digest or metrics['predictions_sha256'] != digest:
        raise ValueError('Prediction evidence hash mismatch')
    if run['simulated'] or run['device'] != 'npu':
        raise ValueError('Real NPU run evidence required')
    if run['runs'] < 3 or len(run['seconds']) != run['runs']:
        raise ValueError('At least three measured runs required')
    if any(finite(value) <= 0 for value in run['seconds']):
        raise ValueError('Invalid run duration')
    if not comparison['passed']:
        raise ValueError('Numerical comparison failed')
    if comparison['candidate_sha256'] != digest:
        raise ValueError('Comparison candidate hash mismatch')
    if run['samples'] != comparison['samples'] or run['samples'] != metrics['samples']:
        raise ValueError('Evidence sample counts differ')
    pending = ['official split and label mapping review', 'public baseline comparability',
               'environment/fallback logs', 'clean-room and contribution gates']
    perf = None
    if performance_dirs:
        perf = performance(performance_dirs, run)
    else:
        pending.append('independent-process performance acceptance: at least three runs required')
    return {'status': 'NPU evidence collected; S3 requires official split/baseline review',
            'run': run, 'comparison': comparison, 'metrics': metrics,
            'performance': perf, 'pending': pending}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--performance-dirs', nargs='+', help='At least three independent NPU run directories')
    args = parser.parse_args()
    write_json(args.output, collect(args.run_dir, args.comparison, args.metrics, args.performance_dirs))


if __name__ == '__main__':
    main()
