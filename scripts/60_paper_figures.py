"""Draw all figures of the Applied Sciences manuscript (Figures 1-8) and the
robustness (contaminated-shot) results used in Table 9 and Figure 8.

Inputs (searched recursively under --results, default: results/):
  (default input folder: work/results/probe)
  judges_scores.csv   written by scripts/51_llm_judges.py
  calib_raw.csv       written by scripts/52_probe_calibration.py
  calib_auroc.csv     written by scripts/52_probe_calibration.py
  calib_pairwise.csv  written by scripts/52_probe_calibration.py
Outputs (in --out, default: results/paper_figures/):
  Figure1.png ... Figure8.png (300 dpi), contam_raw.csv, contam_summary.csv

Run from the repository root:
  python scripts/60_paper_figures.py
"""
import argparse, os, sys, zlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tpc import calib  # noqa: E402

ECOS = ['MCP-curated', 'MCP-wild', 'Plugin', 'ToolEmu']
ECOL = dict(zip(ECOS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
EMK = dict(zip(ECOS, 'osD^'))
INK, INK2, GRID = '#1b1b1b', '#55544f', '#e6e5e0'
NOMINAL = 100 * calib.ALPHA
RULE = {  # name, colour, marker, line style
    'Global': ('Global z', '#8a8983', 'o', '--'),
    'Src-Quantile': ('Source quantile', '#eb6834', 's', '-'),
    'Quantile': ('Shot quantile', '#eda100', 'D', ':'),
    'Ours-A': ('Shots-only z', '#e87ba4', '^', ':'),
    'Ours-B': ('EB z', '#6250d6', 'P', '-.'),
    'Ours-B-YJ': ('EB z + YJ', '#1baf7a', 'X', '-.'),
    'Pooled': ('Pooled z', '#c9c8c2', 'v', '--'),
    'Ours-B-tail': ('EB-Tail (proposed)', '#2a78d6', 'o', '-'),
}


def short(d):
    if d == 'pi_clf':
        return 'PI-DeBERTa'
    _, m, p = d.split(':')
    m = {'Qwen2.5-3B-Instruct': 'Qwen-3B', 'Qwen2.5-1.5B-Instruct': 'Qwen-1.5B',
         'granite-3.3-2b-instruct': 'Granite-2B'}[m]
    return f'{m}/{p}'


def find(root, name):
    p = os.path.join(root, name)
    if os.path.isfile(p):
        return p
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if 'mock' not in x)
        if name in files:
            return os.path.join(d, name)
    raise FileNotFoundError(f'{name} not found under {os.path.abspath(root)}')


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
        'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False})


def save(fig, out, n):
    fig.savefig(os.path.join(out, f'Figure{n}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure{n}.png')


# ------------------------------------------------------------------ diagrams
def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12', fc=fc, ec='#b9b8b2', lw=0.9))
    ax.text(x + w / 2, y + h - 0.25, title, ha='center', va='top', fontsize=9, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h - 0.68, body, ha='center', va='top', fontsize=7.8, color=INK, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=13, color=INK2, lw=1.3))


def fig_framework(out):
    fig, ax = plt.subplots(figsize=(7.4, 1.6))
    ax.set_xlim(0, 16.4); ax.set_ylim(0, 2.4); ax.axis('off')
    steps = [('Data', '4 benign tool\necosystems;\n2 attack families', '#e3eefb'),
             ('Detectors', 'small LLM judges\nand a classifier;\nno training', '#e3eefb'),
             ('Transfer', 'set threshold on\n3 ecosystems;\ntest on the 4th', '#f3f2ee'),
             ('Calibrate', 'optionally use k\nbenign target\ndescriptions', '#dcf3ea'),
             ('Evaluate', 'FPR per\necosystem;\nworst case; recall', '#f3f2ee')]
    w, gap = 2.75, 0.62
    for i, (t, b, c) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.2, w, 2.05, t, b, c)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.05, 1.25, x + w + gap - 0.05, 1.25)
    save(fig, out, 1)


def fig_method(out):
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    ax.set_xlim(0, 16.4); ax.set_ylim(0, 3.7); ax.axis('off')
    box(ax, 0.1, 1.95, 5.0, 1.65, 'Source ecosystems', 'average benign centre and spread;\ntail threshold', '#e3eefb')
    box(ax, 0.1, 0.1, 5.0, 1.65, 'Target ecosystem', 'k benign descriptions:\ntheir centre and spread', '#dcf3ea')
    box(ax, 5.9, 0.6, 5.0, 2.5, 'Combine', 'weight w on the target;\nw grows with k and with\nhow much ecosystems differ', '#fcecc8')
    box(ax, 11.7, 0.6, 4.6, 2.5, 'Decide', 'standardize the score;\nflag it if it exceeds\nthe tail threshold', '#f3f2ee')
    arrow(ax, 5.15, 2.75, 5.85, 2.3); arrow(ax, 5.15, 0.95, 5.85, 1.4); arrow(ax, 10.95, 1.85, 11.65, 1.85)
    save(fig, out, 2)


# ------------------------------------------------------------------ data figures
def fig_prompts(au, out):
    h = au[au.family == 'hand'].set_index('detector')
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    models = [('Qwen2.5-3B-Instruct', 'Qwen-3B', '#2a78d6', 'o'), ('granite-3.3-2b-instruct', 'Granite-2B', '#1baf7a', 's'),
              ('Qwen2.5-1.5B-Instruct', 'Qwen-1.5B', '#eb6834', '^')]
    for j, (m, lab, c, mk) in enumerate(models):
        v = np.array([h.loc[f'judge:{m}:{p}', ECOS].astype(float).values for p in ('P1', 'P2', 'P3')])
        x = np.arange(3) + (j - 1) * 0.08
        ax.errorbar(x, v.mean(1), yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], color=c, marker=mk, ms=6,
                    lw=1.6, capsize=3, label=lab)
    pi = h.loc['pi_clf', ECOS].astype(float).mean()
    ax.axhline(pi, color=INK2, ls=':', lw=1); ax.text(-0.27, pi + 0.012, 'PI-DeBERTa', fontsize=7.5, color=INK2, ha='left')
    ax.axhline(0.5, color=INK2, ls='--', lw=0.9); ax.text(2.25, 0.512, 'chance', fontsize=7.5, color=INK2, ha='right')
    ax.set_xticks(range(3)); ax.set_xticklabels(['P1\nexplicit criteria', 'P2\nminimal', 'P3\nscope only'])
    ax.set_ylabel('AUROC, hand-written attacks'); ax.set_ylim(0.25, 1.0); ax.set_xlim(-0.3, 2.3)
    ax.legend(loc='lower left', fontsize=8)
    save(fig, out, 3)


def fig_transfer(raw, out):
    g = raw[(raw.k == 0) & (raw.method == 'Global')].pivot_table(index='detector', columns='target', values='fpr') * 100
    g = g.loc[g.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    y = np.arange(len(g))
    for yi, d in zip(y, g.index):
        ax.plot([g.loc[d].min(), g.loc[d].max()], [yi, yi], color='#c9c8c2', lw=1.4, zorder=1)
    for T in ECOS:
        ax.scatter(g[T], y, marker=EMK[T], s=30, color=ECOL[T], edgecolor='white', lw=0.5, zorder=3, label=T)
    ax.axvline(NOMINAL, color=INK, ls='--', lw=0.9)
    ax.text(NOMINAL + 0.3, -0.75, 'nominal 2.28%', fontsize=8, color=INK); ax.set_ylim(-1.1, len(g) - 0.5)
    ax.set_yticks(y); ax.set_yticklabels([short(d) for d in g.index]); ax.grid(axis='y', visible=False)
    ax.set_xlabel('Realized FPR on the held-out ecosystem (%)')
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.45, 1.0), fontsize=8)
    save(fig, out, 4)


def fig_tails(scores, dets, out):
    b = scores[scores.y == 0]
    rate = {}
    for d in dets:
        z = np.concatenate([(g[d] - g[d].mean()) / g[d].std(ddof=1) for _, g in b.groupby('eco')])
        rate[d] = 100 * (z > 2).mean()
    s = pd.Series(rate).sort_values()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    y = np.arange(len(s))
    ax.barh(y, s.values, color=['#2a78d6' if v > NOMINAL else '#a9c7ee' for v in s.values], height=0.6)
    for yi, v in zip(y, s.values):
        ax.text(v + 0.06, yi, f'{v:.1f}%', va='center', fontsize=7.5, color=INK)
    ax.axvline(NOMINAL, color=INK, ls='--', lw=0.9)
    ax.text(NOMINAL + 0.05, len(s) - 0.35, 'Gaussian 2.28%', fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([short(d) for d in s.index]); ax.grid(axis='y', visible=False)
    ax.set_xlabel('Benign descriptions with standardized score above 2 (%)')
    save(fig, out, 5)
    return s


def fig_k(raw, out):
    rules = ['Global', 'Src-Quantile', 'Ours-A', 'Ours-B-tail']
    rows = []
    for (k, m), g in raw[raw.method.isin(rules)].groupby(['k', 'method']):
        rows.append(dict(k=k, method=m, ce=100 * g.cal_err.mean(),
                         worst=100 * g.groupby(['detector', 'seed']).fpr.max().mean()))
    S = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(wspace=0.3))
    for ax, col, lab in zip(axs, ['ce', 'worst'], ['Calibration error (pp)', 'Worst-ecosystem FPR (%)']):
        for m in rules:
            g = S[S.method == m].sort_values('k'); n, c, mk, ls = RULE[m]
            ax.plot(g.k, g[col], color=c, marker=mk, ls=ls, lw=2.2 if m == 'Ours-B-tail' else 1.3, ms=5, label=n)
        ax.set_xticks([0, 5, 20, 50]); ax.set_xlabel('Benign target descriptions k'); ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    axs[1].axhline(NOMINAL, color=INK, ls=':', lw=0.8)
    axs[1].text(50, NOMINAL + 0.5, 'nominal 2.28%', fontsize=7.5, color=INK, ha='right', va='bottom')
    axs[0].set_title('(a)', loc='left', fontsize=9); axs[1].set_title('(b)', loc='left', fontsize=9)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=8)
    save(fig, out, 6)


def fig_paired(raw, pw, out):
    order = ['Src-Quantile', 'Ours-B-YJ', 'Ours-B', 'Global', 'Pooled', 'Quantile', 'Ours-A']
    rng = np.random.RandomState(0)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw=dict(wspace=0.12))
    P_ce = raw[raw.k > 0].groupby(['detector', 'target', 'k', 'method']).cal_err.mean().unstack() * 100
    P_w = raw[raw.k > 0].groupby(['detector', 'k', 'seed', 'method']).fpr.max().unstack().groupby(['detector', 'k']).mean() * 100
    for ax, P, metric, ttl in ((axs[0], P_ce, 'cal_err', '(a) Calibration error'), (axs[1], P_w, 'worst', '(b) Worst-ecosystem FPR')):
        for i, m in enumerate(order[::-1]):
            d = (P['Ours-B-tail'] - P[m]).dropna().values
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = pw[(pw.metric == metric) & (pw.vs == m)].p_holm.iloc[0]
            ax.plot([lo, hi], [i, i], color=INK, lw=1.8)
            ax.scatter(d.mean(), i, s=40, color='#2a78d6' if p < 0.05 else 'white', edgecolor=INK, lw=1, zorder=3)
            ax.text(1.2, i, 'p<0.001' if p < 0.001 else f'p={p:.3f}', fontsize=7.5, va='center', color=INK2)
        ax.axvline(0, color=INK, lw=0.9); ax.set_xlim(-11, 2.8); ax.grid(axis='y', visible=False)
        ax.set_title(ttl, loc='left', fontsize=9); ax.set_xlabel('EB-Tail minus rule (pp); < 0 favours EB-Tail')
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([RULE[m][0] for m in order[::-1]])
    save(fig, out, 7)


def contamination(scores, dets, out):
    path = os.path.join(out, 'contam_raw.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    rules = ['Src-Quantile', 'Quantile', 'Ours-A', 'Ours-B', 'Ours-B-tail']
    rows = []
    for det in dets:
        for T in ECOS:
            src = scores[(scores.eco != T) & (scores.y == 0)]
            st = calib.SourceStats(src[det].values, src.eco.values)
            ben = scores[(scores.eco == T) & (scores.y == 0)][det].values
            for fam in ('hand', 'mcptox'):
                pos = scores[(scores.eco == T) & (scores.family == fam)][det].values
                for k in (20, 50):
                    for c in (0, 1, 2):
                        for seed in range(100):
                            rng = np.random.RandomState(zlib.crc32(f'{det}|{T}|{fam}|{k}|{c}|{seed}'.encode()))
                            sb = rng.choice(len(ben), k - c, replace=False); sp = rng.choice(len(pos), c, replace=False)
                            shots = np.r_[ben[sb], pos[sp]]
                            dn = calib.decisions(np.delete(ben, sb), st, shots)
                            dp = calib.decisions(np.delete(pos, sp), st, shots)
                            for m in rules:
                                rows.append(dict(detector=det, target=T, family=fam, k=k, c=c, seed=seed, method=m,
                                                 fpr=float((dn[m][0] > calib.Z).mean()),
                                                 recall=float((dp[m][0] > calib.Z).mean())))
    R = pd.DataFrame(rows); R['ce'] = (R.fpr - calib.ALPHA).abs()
    R.to_csv(path, index=False)
    return R


def fig_contam(R, out):
    rules = ['Src-Quantile', 'Quantile', 'Ours-A', 'Ours-B-tail']
    G = R[R.k == 20]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(wspace=0.3))
    for ax, fam, ttl in ((axs[0], 'hand', '(a) Hand-written attacks'), (axs[1], 'mcptox', '(b) MCPTox-style attacks')):
        for m in rules:
            v = [100 * G[(G.family == fam) & (G.c == c) & (G.method == m)].recall.mean() for c in (0, 1, 2)]
            n, col, mk, ls = RULE[m]
            ax.plot([0, 1, 2], v, color=col, marker=mk, ls=ls, lw=2.2 if m == 'Ours-B-tail' else 1.3, ms=5, label=n)
        ax.set_xticks([0, 1, 2]); ax.set_xlabel('Poisoned descriptions among the 20 shots'); ax.set_ylabel('Recall (%)')
        ax.set_title(ttl, loc='left', fontsize=9)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=8)
    save(fig, out, 8)


def contam_table(R, out):
    rows = []
    for m in ['Src-Quantile', 'Quantile', 'Ours-A', 'Ours-B', 'Ours-B-tail']:
        r = {'rule': RULE[m][0]}
        for k in (20, 50):
            G = R[(R.k == k) & (R.method == m)]
            for fam in ('hand', 'mcptox'):
                a = 100 * G[(G.family == fam) & (G.c == 0)].recall.mean(); b = 100 * G[(G.family == fam) & (G.c == 2)].recall.mean()
                r[f'k{k}_{fam}_c0'] = a; r[f'k{k}_{fam}_c2'] = b; r[f'k{k}_{fam}_loss'] = a - b
            r[f'k{k}_ce_c0'] = 100 * G[G.c == 0].ce.mean(); r[f'k{k}_ce_c2'] = 100 * G[G.c == 2].ce.mean()
            r[f'k{k}_worst_c2'] = 100 * G[G.c == 2].groupby(['detector', 'family', 'seed']).fpr.max().mean()
        rows.append(r)
    T = pd.DataFrame(rows).round(2)
    T.to_csv(os.path.join(out, 'contam_summary.csv'), index=False)
    print(T.to_string())


def main():
    ap = argparse.ArgumentParser()
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    ap.add_argument('--results', default=os.path.join(root, 'work', 'results', 'probe'))
    ap.add_argument('--out', default=os.path.join(root, 'results', 'paper_figures'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); style()
    scores = pd.read_csv(find(a.results, 'judges_scores.csv'))
    raw = pd.read_csv(find(a.results, 'calib_raw.csv'))
    au = pd.read_csv(find(a.results, 'calib_auroc.csv'))
    pw = pd.read_csv(find(a.results, 'calib_pairwise.csv'))
    dets = [c for c in scores.columns if c == 'pi_clf' or c.startswith('judge:')]
    fig_framework(a.out); fig_method(a.out)
    fig_prompts(au, a.out); fig_transfer(raw, a.out)
    print(fig_tails(scores, dets, a.out).round(2).to_string())
    fig_k(raw, a.out); fig_paired(raw, pw, a.out)
    R = contamination(scores, dets, a.out)
    fig_contam(R, a.out); contam_table(R, a.out)


if __name__ == '__main__':
    main()
