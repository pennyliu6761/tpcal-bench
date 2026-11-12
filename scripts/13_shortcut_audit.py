"""Shortcut audit (Section 4.1, Table 2-3, Figures 3-4).
(a) audit_cues.csv       construction-cue profile of every source: boilerplate rate, length, cue-lexicon hits
(b) audit_within.csv     inside MCPTox (native benign vs native poisoned, server-held-out): AUROC of single cues
                         (length, boilerplate flag, cue lexicon) under C0 / C1 / C2
(c) audit_supervised.csv a char TF-IDF logistic regression trained on MCPTox train-half servers, evaluated on
                         MCPTox test-half servers (in-family), leave-one-template-out, and on the hand-written attacks
                         against MCP-wild benign tools (cross-family; cf. TP-Detect). Recall / FPR at probability 0.5.
"""
import os, zlib
import _common
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tpc import data, cues, paths, scores as S
from tpc.harness import wauc

if __name__ == '__main__':
    a = _common.parser(__doc__).parse_args(); sfx = '_quick' if a.quick else ''
    df = data.load_corpus(); df['words'] = cues.n_words(df.text.values)
    df['boiler'] = [cues.has_boilerplate(t) for t in df.text]; df['rule'] = S.fixed_scores(list(df.text))['rule']
    df['must'] = df.text.str.contains(r'\bmust\b', case=False, regex=True)
    nat = df[df.native]
    src = np.where(nat.y == 0, nat.eco + ' benign', np.where(nat.family == 'mcptox', 'MCPTox ' + nat.template, 'Hand ' + nat.batch))
    A = (nat.assign(source=src).groupby('source')
           .agg(n=('rid', 'size'), words_median=('words', 'median'), boilerplate=('boiler', 'mean'),
                rule_hits=('rule', 'mean'), must=('must', 'mean')).reset_index())
    A.to_csv(os.path.join(paths.TABLE_DIR, f'audit_cues{sfx}.csv'), index=False); print(A.round(3).to_string(index=False))

    m = df[(df.eco == 'MCP-curated') & df.native].copy()
    test_srv = m.group.map(lambda g: zlib.crc32(g.encode()) % 2 == 1)   # same split as data._split_of ('test' half)
    rows = []
    for cond in cues.CONDITIONS:
        t = cues.texts_for(cond, m.text.values); L = cues.n_words(t); y = m.y.values
        w = np.ones(len(m))
        if cond == 'C2':
            wb, wp = cues.common_support_weights(L[y == 0], L[y == 1]); w[y == 0] = wb; w[y == 1] = wp
        fx = S.fixed_scores(t); bp = np.array([cues.has_boilerplate(x) for x in t], float)
        for name, v in (('length', L), ('boilerplate flag', bp), ('cue lexicon (rule)', fx['rule'])):
            rows.append(dict(cond=cond, cue=name, auroc=wauc(y, v, w)))
    W = pd.DataFrame(rows); W.to_csv(os.path.join(paths.TABLE_DIR, f'audit_within{sfx}.csv'), index=False)
    print(W.pivot(index='cue', columns='cond', values='auroc').round(3))

    wild = df[(df.eco == 'MCP-wild') & df.native]
    rows = []
    for cond in cues.CONDITIONS:
        t_m = np.array(cues.texts_for(cond, m.text.values), dtype=object); t_w = np.array(cues.texts_for(cond, wild.text.values), dtype=object)
        y = m.y.values; tr = ~test_srv.values; te = test_srv.values

        def fit(mask):
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)
            X = vec.fit_transform(t_m[mask]); yy = y[mask]; w = np.ones(mask.sum())
            if cond == 'C2':
                L = cues.n_words(t_m[mask]); wb, wp = cues.common_support_weights(L[yy == 0], L[yy == 1]); w[yy == 0] = wb; w[yy == 1] = wp
            return vec, LogisticRegression(C=1.0, class_weight='balanced', max_iter=4000).fit(X, yy, sample_weight=w)

        def report(setting, vec, clf, texts, yy):
            s = clf.decision_function(vec.transform(texts))
            rows.append(dict(cond=cond, setting=setting, n_pos=int((yy == 1).sum()), n_neg=int((yy == 0).sum()),
                             auroc=wauc(yy, s, np.ones(len(yy))), recall_at_0_5=float((s[yy == 1] > 0).mean()),
                             fpr_at_0_5=float((s[yy == 0] > 0).mean())))
        vec, clf = fit(tr)
        report('in-family (MCPTox test-half servers)', vec, clf, t_m[te], y[te])
        report('cross-family (hand-written vs MCP-wild benign)', vec, clf, t_w, wild.y.values)
        for T in ('Template-1', 'Template-2', 'Template-3'):
            vec, clf = fit(tr & (m.template.values != T))
            sel = te & ((y == 0) | (m.template.values == T))
            report(f'leave-one-template-out ({T})', vec, clf, t_m[sel], y[sel])
    Sup = pd.DataFrame(rows); Sup.to_csv(os.path.join(paths.TABLE_DIR, f'audit_supervised{sfx}.csv'), index=False)
    print(Sup.round(3).to_string(index=False))
