"""Compare a fresh run of the detectors with the cached scores used in the paper.

    python scripts/62_compare_runs.py --new work_verify/results/probe

For every detector it reports the correlation of the scores, the largest absolute difference and the AUROC on
hand-written and MCPTox-style attacks in both runs. GPU inference in float16/bfloat16 is not bit-exact, so small
differences in the third decimal are expected; the conclusions of the paper should not change.
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def auroc(df, det, fam):
    out = []
    if det == 'tfidf_lr' and 'tfidf_train' in df:
        df = df[~df.tfidf_train.astype(bool)]
    for e, g in df.groupby('eco'):
        g = g[(g.y == 0) | (g.family == fam)]
        if g.y.nunique() == 2:
            out.append(roc_auc_score(g.y, g[det]))
    return np.mean(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default=os.path.join(ROOT, 'work', 'results', 'probe'))
    ap.add_argument('--new', required=True)
    a = ap.parse_args()
    A = pd.read_csv(os.path.join(a.old, 'judges_scores.csv'))
    B = pd.read_csv(os.path.join(a.new, 'judges_scores.csv'))
    key = [c for c in ('eco', 'y', 'family', 'group') if c in A]
    if len(A) != len(B) or not A[key].astype(str).equals(B[key].astype(str)):
        raise SystemExit('the two runs do not contain the same descriptions in the same order')
    dets = [c for c in A.columns if c.startswith('judge:') or c in ('pi_clf', 'tfidf_lr')]
    rows = []
    for d in dets:
        if d not in B:
            continue
        rows.append(dict(detector=d, corr=np.corrcoef(A[d], B[d])[0, 1], max_abs_diff=(A[d] - B[d]).abs().max(),
                         auroc_hand_old=auroc(A, d, 'hand'), auroc_hand_new=auroc(B, d, 'hand'),
                         auroc_mcptox_old=auroc(A, d, 'mcptox'), auroc_mcptox_new=auroc(B, d, 'mcptox')))
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(R.round(4).to_string(index=False))
    worst = (R.auroc_hand_old - R.auroc_hand_new).abs().max()
    print(f'\nlargest AUROC change on hand-written attacks: {worst:.4f}')
    print('OK: the runs agree' if worst < 0.005 and R['corr'].min() > 0.999 else 'CHECK: differences larger than expected')


if __name__ == '__main__':
    main()
