"""Leave-one-ecosystem-out few-shot calibration of ZERO-SHOT detectors (LLM judges, prompt-injection classifier) from
their probe scores; no model is run here (seconds on a CPU).

For each detector and target ecosystem T: source = benign scores of the other three ecosystems; k benign target
descriptions are sampled (k in {0, 5, 20, 50}; 100 seeds) and every calibration rule of tpc.calib sets the decision
z > 2 (nominal FPR 2.28%). FPR on the remaining target benign descriptions; recall on target poisoned descriptions of
each family. tfidf_lr is excluded (it is trained on MCP-curated and has its own protocol in 10_main.py).

    python scripts\\52_probe_calibration.py                      # uses probe/judges_scores.csv (or probe_scores.csv)
    python scripts\\52_probe_calibration.py --scores path\\to.csv
Outputs in TPC_WORK/results/probe/: calib_raw.csv, calib_fpr.csv, calib_recall.csv, calib_auroc.csv, calib_pairwise.csv
"""
import argparse, os, sys, zlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from tpc import calib, paths, data
from tpc.harness import wauc

KS = (0, 5, 20, 50)
METHODS = ['Global', 'Pooled', 'Src-Quantile', 'Quantile', 'Ours-A', 'Ours-B', 'Ours-B-tail', 'Ours-B-YJ']


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores'); ap.add_argument('--seeds', type=int, default=100)
    a = ap.parse_args()
    out = os.path.join(paths.WORK, 'results', 'probe'); os.makedirs(out, exist_ok=True)
    f = a.scores or next((os.path.join(out, n) for n in ('judges_scores.csv', 'probe_scores.csv') if os.path.exists(os.path.join(out, n))), None)
    if f is None: sys.exit('no score file found: run 50_semantic_probe.py or 51_llm_judges.py first')
    d = pd.read_csv(f); print(f'scores: {f}')
    dets = [c for c in d.columns if c in ('pi_clf', 'llm_judge') or c.startswith('judge:')]
    rows, aucs = [], []
    for det in dets:
        for T in data.ECOSYSTEMS:
            src = d[(d.eco != T) & (d.y == 0)]
            st = calib.SourceStats(src[det].values, src.eco.values); yj = calib.fit_yj(src[det].values, src.eco.values)
            ben = d[(d.eco == T) & (d.y == 0)][det].values
            pos = {fam: d[(d.eco == T) & (d.family == fam) & (d.split == 'test')][det].values for fam in data.FAMILIES}
            for fam, pv in pos.items():
                aucs.append(dict(detector=det, target=T, family=fam, auroc=wauc(np.r_[np.zeros(len(ben)), np.ones(len(pv))], np.r_[ben, pv], np.ones(len(ben) + len(pv)))))
            for k in KS:
                for seed in range(a.seeds if k else 1):
                    rng = np.random.RandomState(zlib.crc32(f'{det}|{T}|{k}|{seed}'.encode()))
                    sp = rng.choice(len(ben), k, replace=False) if k else np.array([], int)
                    neg = np.delete(ben, sp); shots = ben[sp]
                    dn = calib.decisions(neg, st, shots, yj=yj)
                    dp = {fam: calib.decisions(pv, st, shots, yj=yj) for fam, pv in pos.items()}
                    for m in dn:
                        r = dict(detector=det, target=T, k=k, seed=seed, method=m, fpr=float((dn[m][0] > calib.Z).mean()), w=dn[m][1], t=st.t)
                        for fam in pos: r[f'recall_{fam}'] = float((dp[fam][m][0] > calib.Z).mean())
                        rows.append(r)
    R = pd.DataFrame(rows); R['cal_err'] = (R.fpr - calib.ALPHA).abs(); R.to_csv(os.path.join(out, 'calib_raw.csv'), index=False)
    pd.DataFrame(aucs).pivot_table(index=['detector', 'family'], columns='target', values='auroc').round(3).to_csv(os.path.join(out, 'calib_auroc.csv'))
    F = (R.groupby(['detector', 'k', 'method', 'target']).fpr.mean().unstack() * 100)
    F['worst'] = F[list(data.ECOSYSTEMS)].max(axis=1); F['mean_abs_err'] = (F[list(data.ECOSYSTEMS)] - 100 * calib.ALPHA).abs().mean(axis=1)
    F = F.round(2); F.to_csv(os.path.join(out, 'calib_fpr.csv'))
    Rc = (R.groupby(['detector', 'k', 'method', 'target'])[['recall_mcptox', 'recall_hand']].mean() * 100).round(1)
    Rc.to_csv(os.path.join(out, 'calib_recall.csv'))
    # pre-specified comparison: Ours-B-tail vs each rule, k in {5, 20, 50}; blocks = detector x target x k (seed-averaged)
    pw = []
    for metric in ('cal_err', 'worst'):
        if metric == 'cal_err':
            P = R[R.k > 0].groupby(['detector', 'target', 'k', 'method']).cal_err.mean().unstack()
        else:
            P = R[R.k > 0].groupby(['detector', 'k', 'seed', 'method']).fpr.max().unstack().groupby(['detector', 'k']).mean()
        for m in METHODS:
            if m == 'Ours-B-tail' or m not in P: continue
            x = (P['Ours-B-tail'] - P[m]).dropna()
            if len(x) >= 6 and not np.allclose(x, 0):
                pw.append(dict(metric=metric, vs=m, mean_diff_pp=100 * x.mean(), tail_better=int((x < 0).sum()), blocks=len(x), p=wilcoxon(x).pvalue))
    W = pd.DataFrame(pw)
    if len(W): W['p_holm'] = holm(W.p)
    W.to_csv(os.path.join(out, 'calib_pairwise.csv'), index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 400)
    show = F.reset_index(); show = show[show.k.isin([0, 20, 50]) & show.method.isin(['Global', 'Src-Quantile', 'Quantile', 'Ours-A', 'Ours-B', 'Ours-B-tail'])]
    print(show.to_string(index=False)); print(W.round(4).to_string(index=False)); print(f'-> {out}')
