"""Collect hashed run, numerical comparison and quality evidence."""
import argparse
import json
from pathlib import Path
from delivery_utils import finite, sha256, write_json


def collect(run_dir, comparison_path, metrics_path):
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
    return {'status': 'NPU evidence collected; S3 requires official split/baseline review',
            'run': run, 'comparison': comparison, 'metrics': metrics,
            'pending': ['official split and label mapping review', 'public baseline comparability',
                        'environment/fallback logs', 'clean-room and contribution gates']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write_json(args.output, collect(args.run_dir, args.comparison, args.metrics))


if __name__ == '__main__':
    main()
