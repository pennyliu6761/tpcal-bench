"""Numbers behind Tables 4-8 and the statistical tests of the Applied Sciences manuscript.
Table 9 (contaminated shots) is written by 60_paper_figures.py (contam_summary.csv).

Inputs (default folder: work/results/probe):
  judges_scores.csv, judges_summary.csv, calib_raw.csv, calib_auroc.csv, calib_pairwise.csv
  optional: work/results/tables/audit_within.csv and audit_supervised.csv from 13_shortcut_audit.py (Table 4)
Outputs (default folder: results/paper_tables):
  table4_audit.csv          shortcut audit (only if the audit files exist)
  table5_detection.csv      AUROC on hand-written / MCPTox-style attacks per detector
  table6_source_stats.csv   variance ratio tau, tail threshold c_tail, EB weight w for k = 5, 20, 50
  table7_calibration.csv    mean FPR, calibration error, worst-ecosystem FPR and recall per rule and k
  table8_per_detector.csv   FPR and recall per detector (Global z k=0, Source quantile k=0, EB-Tail k=20)
  stats_friedman.csv        Friedman test and Nemenyi critical difference (k = 20, 50)
  stats_ranks.csv           mean ranks of the eight rules
  stats_pairwise.csv        Wilcoxon signed-rank tests EB-Tail vs every rule (Holm), copied from 52_probe_calibration.py
  stats_tail_vs_srcq.csv    EB-Tail vs Source quantile for each k

    python scripts/61_paper_tables.py
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
from tpc import calib  # noqa: E402

ECOS = ['MCP-curated', 'MCP-wild', 'Plugin', 'ToolEmu']
RULES = ['Global', 'Pooled', 'Src-Quantile', 'Quantile', 'Ours-A', 'Ours-B', 'Ours-B-YJ', 'Ours-B-tail']
NAME = {'Global': 'Global z', 'Pooled': 'Pooled z', 'Src-Quantile': 'Source quantile', 'Quantile': 'Shot quantile',
        'Ours-A': 'Shots-only z', 'Ours-B': 'EB z', 'Ours-B-YJ': 'EB z + YJ', 'Ours-B-tail': 'EB-Tail'}


def short(d):
    if d == 'pi_clf': return 'PI-DeBERTa'
    if d == 'tfidf_lr': return 'TF-IDF LR (supervised)'
    _, m, p = d.split(':')
    m = {'Qwen2.5-3B-Instruct': 'Qwen-3B', 'Qwen2.5-1.5B-Instruct': 'Qwen-1.5B', 'granite-3.3-2b-instruct': 'Granite-2B'}[m]
    return f'{m}/{p}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(ROOT, 'work', 'results', 'probe'))
    ap.add_argument('--audit', default=os.path.join(ROOT, 'work', 'results', 'tables'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'paper_tables'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rd = lambda n: pd.read_csv(os.path.join(a.results, n))
    S, J, R, AU, PW = rd('judges_scores.csv'), rd('judges_summary.csv'), rd('calib_raw.csv'), rd('calib_auroc.csv'), rd('calib_pairwise.csv')
    save = lambda df, n, **kw: (df.to_csv(os.path.join(a.out, n), **kw), print(f'-> {n}'))
    R['fpr100'] = 100 * R.fpr; R['ce100'] = 100 * R.cal_err
    dets = list(AU.detector.unique())

    # Table 4: shortcut audit
    fw, fs = os.path.join(a.audit, 'audit_within.csv'), os.path.join(a.audit, 'audit_supervised.csv')
    if os.path.exists(fw) and os.path.exists(fs):
        W = pd.read_csv(fw).pivot(index='cue', columns='cond', values='auroc')
        W.index = [f'{c}: AUROC' for c in W.index]
        L = pd.read_csv(fs).set_index(['setting', 'cond'])[['auroc', 'recall_at_0_5', 'fpr_at_0_5']].stack().unstack('cond')
        L.index = [f'TF-IDF LR, {st}: {m}' for st, m in L.index]
        save(pd.concat([W, L]).round(3), 'table4_audit.csv')
    else:
        print('Table 4 skipped: run scripts/13_shortcut_audit.py first (needs the source data in external/)')

    # Table 5: detection quality
    rows = []
    for d in ['tfidf_lr'] + dets:
        if d == 'tfidf_lr':
            h = J[(J.detector == d) & (J.family == 'hand')].set_index('target').auroc
            m = J[(J.detector == d) & (J.family == 'mcptox')].set_index('target').auroc
        else:
            h = AU[(AU.detector == d) & (AU.family == 'hand')].iloc[0][ECOS].astype(float)
            m = AU[(AU.detector == d) & (AU.family == 'mcptox')].iloc[0][ECOS].astype(float)
        jj = J[(J.detector == d) & (J.family == 'hand')]
        rows.append(dict(detector=short(d), hand_mean=h.mean(), hand_min=h.min(), hand_max=h.max(),
                         hand_len_matched=jj.auroc_lenmatched.mean(), tpr_at_1pct_fpr=100 * jj.tpr_at_1pct_fpr.mean(),
                         mcptox_mean=m.mean()))
    save(pd.DataFrame(rows).round(3), 'table5_detection.csv', index=False)

    # Table 6: source statistics (mean over the four folds)
    rows = []
    for d in dets:
        for T in ECOS:
            src = S[(S.eco != T) & (S.y == 0)]; st = calib.SourceStats(src[d].values, src.eco.values)
            rows.append(dict(detector=short(d), target=T, tau=st.t, c_tail=st.c_tail,
                             **{f'w_k{k}': k * st.t / (k * st.t + 1) for k in (5, 20, 50)}))
    ST = pd.DataFrame(rows)
    save(ST.groupby('detector', sort=False)[['tau', 'c_tail', 'w_k5', 'w_k20', 'w_k50']].mean().round(3), 'table6_source_stats.csv')

    # Table 7: calibration summary
    rows = []
    for (k, m), g in R.groupby(['k', 'method']):
        rows.append(dict(k=k, rule=NAME[m], mean_fpr=g.fpr100.mean(), cal_err=g.ce100.mean(),
                         worst_fpr=g.groupby(['detector', 'seed']).fpr100.max().mean(),
                         recall_hand=100 * g.recall_hand.mean(), recall_mcptox=100 * g.recall_mcptox.mean()))
    T7 = pd.DataFrame(rows); T7['_o'] = T7.rule.map({NAME[r]: i for i, r in enumerate(RULES)})
    save(T7.sort_values(['k', '_o']).drop(columns='_o').round(2), 'table7_calibration.csv', index=False)

    # Table 8: per detector
    g = R[((R.k == 0) & R.method.isin(['Global', 'Src-Quantile'])) | ((R.k == 20) & (R.method == 'Ours-B-tail'))]
    P = (g.groupby(['detector', 'method'])[['fpr', 'recall_hand']].mean() * 100).unstack()
    P.columns = [f'{NAME[b]} {a}' for a, b in P.columns]; P.index = [short(d) for d in P.index]
    save(P.round(1), 'table8_per_detector.csv')

    # Friedman / Nemenyi (blocks = detector x target, draw-averaged calibration error)
    fr, rk = [], {}
    for k in (20, 50):
        B = R[R.k == k].groupby(['detector', 'target', 'method']).cal_err.mean().unstack()[RULES]
        chi2, p = friedmanchisquare(*[B[c] for c in RULES])
        cd = studentized_range.ppf(0.95, len(RULES), 1e6) / np.sqrt(2) * np.sqrt(len(RULES) * (len(RULES) + 1) / (6 * len(B)))
        fr.append(dict(k=k, chi2=chi2, p=p, blocks=len(B), nemenyi_cd=cd))
        rk[f'k{k}'] = B.rank(axis=1).mean().rename(NAME)
    save(pd.DataFrame(fr), 'stats_friedman.csv', index=False)
    save(pd.DataFrame(rk).sort_values('k20').round(2), 'stats_ranks.csv')
    save(PW, 'stats_pairwise.csv', index=False)

    # EB-Tail vs Source quantile per k
    rows = []
    for k in (5, 20, 50):
        B = R[R.k == k].groupby(['detector', 'target', 'method']).cal_err.mean().unstack()
        x = B['Ours-B-tail'] - B['Src-Quantile']
        Wd = R[R.k == k].groupby(['detector', 'seed', 'method']).fpr.max().unstack().groupby('detector').mean()
        y = Wd['Ours-B-tail'] - Wd['Src-Quantile']
        rows.append(dict(k=k, cal_err_diff_pp=100 * x.mean(), cal_err_wins=int((x < 0).sum()), cal_err_p=wilcoxon(x).pvalue,
                         worst_fpr_diff_pp=100 * y.mean(), worst_fpr_wins=int((y < 0).sum()), worst_fpr_p=wilcoxon(y).pvalue))
    save(pd.DataFrame(rows).round(4), 'stats_tail_vs_srcq.csv', index=False)


if __name__ == '__main__':
    main()
