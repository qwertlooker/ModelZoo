"""Equivalent pooled ddG MSE/RMSE/R2/Pearson/Spearman evaluation."""
import argparse
import math
import numpy as np
from scipy.stats import pearsonr, spearmanr
from delivery_utils import finite, read_jsonl, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    truth = np.array([finite(r['ddG_true']) for r in rows])
    pred = np.array([finite(r['ddG_pred']) for r in rows])
    if len(rows) < 3 or np.var(truth) == 0 or np.var(pred) == 0:
        raise ValueError('At least three nonconstant labeled predictions required')
    mse = float(np.mean((pred - truth) ** 2))
    write_json(args.report, {
        'samples': len(rows), 'predictions_sha256': sha256(args.predictions),
        'mse': mse, 'rmse': math.sqrt(mse),
        'r2': finite(1 - np.sum((pred-truth)**2) / np.sum((truth-truth.mean())**2)),
        'pearson': finite(pearsonr(truth, pred).statistic),
        'spearman': finite(spearmanr(truth, pred).statistic),
        'aggregation': 'pooled mutations', 'ddG_convention': 'positive destabilizing kcal/mol',
        'official_split_verified': False,
    })


if __name__ == '__main__':
    main()
