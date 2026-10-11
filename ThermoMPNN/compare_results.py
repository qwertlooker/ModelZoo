"""Compare paired FP32 predictions; failures exit nonzero."""
import argparse
import json
from pathlib import Path
from delivery_utils import finite, read_jsonl, sha256, write_json


def compare(baseline, candidate, atol, rtol):
    if finite(atol) < 0 or finite(rtol) < 0:
        raise ValueError('Negative tolerance')
    left = {r['id']: r for r in read_jsonl(baseline)}
    right = {r['id']: r for r in read_jsonl(candidate)}
    if left.keys() != right.keys():
        raise ValueError('Sample ID sets differ')
    errors = []
    failed = []
    for key in left:
        for field in ('pdb_sha256', 'chain', 'position', 'wildtype', 'mutation'):
            if left[key][field] != right[key][field]:
                raise ValueError(f'Sample identity differs: {key}/{field}')
        if left[key].get('wt_sequence') != right[key].get('wt_sequence'):
            raise ValueError(f'WT sequence differs: {key}')
        if left[key].get('ddG_true') != right[key].get('ddG_true'):
            raise ValueError(f'Label differs: {key}')
        a, b = finite(left[key]['ddG_pred']), finite(right[key]['ddG_pred'])
        error = abs(a - b)
        errors.append(error)
        if error > atol + rtol * abs(a):
            failed.append(key)
    return {'samples': len(errors), 'max_abs_error': max(errors),
            'mean_abs_error': sum(errors) / len(errors), 'atol': atol,
            'rtol': rtol, 'failed_ids': failed, 'passed': not failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--atol', type=float, required=True)
    parser.add_argument('--rtol', type=float, required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.atol, args.rtol)
    report['baseline_sha256'] = sha256(args.baseline)
    report['candidate_sha256'] = sha256(args.candidate)
    report['status'] = 'numerical comparison only; not S3 certification'
    write_json(args.report, report)
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
