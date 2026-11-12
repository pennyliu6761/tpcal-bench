"""Leave-one-ecosystem-out (LOEO) protocol.

Step 02 (bundles): for each condition and each target ecosystem T
  * trained scorers are fit on the source ecosystems (all except T): benign + mcptox-family TRAIN payloads;
  * target rows (all rows of T) are scored by these models;
  * source-benign reference scores are CROSS-FITTED: the benign rows of source ecosystem S are scored by models fit on
    the remaining source ecosystems (all except T and S), so they are out-of-ecosystem like the target;
  * kNN scores additionally keep the similarity of every target row to every target benign row so that k shots can be
    added to the reference set at evaluation time.
Step 10 (evaluate): sample k benign target shots (seeded), calibrate every score with every rule (calib.py), and record
FPR on the remaining target benign descriptions and recall on target poisoned descriptions of each family.
"""
import os, pickle, zlib
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from . import cues, calib, scores as S, paths
from .data import ECOSYSTEMS, FAMILIES

FUSIONS = {'fusion': ('tfidf_lr', 'charlm'),     # Ours-C: supervised + unsupervised score, each pooled-z, summed
           'fusion_nn': ('emb_lr', 'ppl')}       # neural counterpart (only when neural scores exist)


def train_mask(df, exclude):
    return (~df.eco.isin(exclude)) & ((df.y == 0) | ((df.family == 'mcptox') & (df.split == 'train')))


def _weights(df, idx, cond):
    if cond != 'C2':
        return np.ones(len(idx))
    L = cues.n_words(df.text_c.values[idx]); y = df.y.values[idx]; w = np.ones(len(idx))
    wb, wp = cues.common_support_weights(L[y == 0], L[y == 1]); w[y == 0] = wb; w[y == 1] = wp
    return w


def _fit(df, exclude, cond, emb, names):
    tr = np.where(train_mask(df, exclude))[0]; ref = np.where((~df.eco.isin(exclude)) & (df.y == 0))[0]
    return S.Trainer(list(df.text_c.values), emb, names).fit(tr, df.y.values[tr], _weights(df, tr, cond), ref), ref


def build_bundles(df, cond, neural=None, names=S.TRAINED, verbose=True):
    """neural: None or dict(ppl=array, pi_clf=array, emb=matrix) aligned with df rows (texts of this condition)."""
    df = df.copy(); df['text_c'] = cues.texts_for(cond, df.text.values)
    fixed = S.fixed_scores(list(df.text_c.values))
    emb = None
    if neural is not None:
        for k in S.NEURAL_FIXED:
            if k in neural: fixed[k] = np.asarray(neural[k], float)
        emb = neural.get('emb')
        names = tuple(names) + tuple(n for n in S.NEURAL_TRAINED if emb is not None)
    out = {}
    for T in ECOSYSTEMS:
        tgt = np.where(df.eco.values == T)[0]
        tben = np.where((df.eco.values == T) & (df.y.values == 0))[0]
        model, _ = _fit(df, [T], cond, emb, names)
        St = model.score(tgt)
        knn_sim = {}
        for kn in S.KNN_SCORES:
            if kn in St:
                X = model.kv.transform(list(df.text_c.values[tgt])) if kn == 'tfidf_knn' else emb[tgt]
                Bm = model.kv.transform(list(df.text_c.values[tben])) if kn == 'tfidf_knn' else emb[tben]
                sim = X @ Bm.T; sim = sim.toarray() if hasattr(sim, 'toarray') else np.asarray(sim)
                sim[tgt[:, None] == tben[None, :]] = -1.0
                knn_sim[kn] = (sim.astype(np.float32), tben)
        src_rows, src_eco, Ss = [], [], {n: [] for n in St}
        for Sx in [e for e in ECOSYSTEMS if e != T]:
            rows = np.where((df.eco.values == Sx) & (df.y.values == 0))[0]
            m2, _ = _fit(df, [T, Sx], cond, emb, names)
            sc = m2.score(rows)
            for n in St: Ss[n].append(sc[n])
            src_rows.append(rows); src_eco += [Sx] * len(rows)
        src_rows = np.concatenate(src_rows)
        S_src = {n: np.concatenate(v) for n, v in Ss.items()}
        for n, v in fixed.items():
            St[n] = v[tgt]; S_src[n] = v[src_rows]
        out[T] = dict(tgt=tgt, tben=tben, src_rows=src_rows, src_eco=np.array(src_eco), S_tgt=St, S_src=S_src, knn_sim=knn_sim)
        if verbose:
            print(f'   {cond} target={T:12s} target rows={len(tgt)} source benign={len(src_rows)} scores={sorted(St)}', flush=True)
    return out


def bundle_path(cond, tag=''):
    return os.path.join(paths.SCORE_DIR, f'bundles_{cond}{tag}.pkl')


def save_bundles(b, cond, tag=''):
    with open(bundle_path(cond, tag), 'wb') as f: pickle.dump(b, f)


def load_bundles(cond, tag=''):
    with open(bundle_path(cond, tag), 'rb') as f: return pickle.load(f)


# ------------------------------------------------------------------ evaluation
def _with_shots(B, name, shots_pos):
    """Target scores of a kNN scorer after adding target benign shots (positions within B['tben']) to the reference."""
    base = B['S_tgt'][name]
    if name not in B['knn_sim'] or len(shots_pos) == 0:
        return base
    sim, _ = B['knn_sim'][name]
    return np.minimum(base, 1 - sim[:, shots_pos].max(1))


def _fusion(B, names, comps):
    """Fused score = sum of the pooled-z of the components (pooled stats from the cross-fitted source benign)."""
    if not all(c in names for c in comps):
        return None
    zt, zs = 0, 0
    for c in comps:
        st = calib.SourceStats(B['S_src'][c], B['src_eco'])
        zt = zt + (B['S_tgt'][c] - st.mu0) / st.sd0; zs = zs + (B['S_src'][c] - st.mu0) / st.sd0
    return zt, zs


def wauc(y, s, w):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0 or w[pos].sum() == 0 or w[neg].sum() == 0:
        return np.nan
    o = np.argsort(s[neg]); sn, wn = s[neg][o], w[neg][o]; cw = np.r_[0, np.cumsum(wn)]
    lo, hi = np.searchsorted(sn, s[pos], 'left'), np.searchsorted(sn, s[pos], 'right')
    return float((w[pos] * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (w[pos].sum() * wn.sum()))


def evaluate(df, bundles, cond, ks, seeds, fusions=FUSIONS, eb_variants=None, score_names=None):
    """Returns a DataFrame with one row per (target, family, k, seed, score, method)."""
    rows = []
    L_all = cues.n_words(cues.texts_for(cond, df.text.values))
    for T, B in bundles.items():
        names = [n for n in B['S_tgt'] if score_names is None or n in score_names]
        S_tgt, S_src = dict(B['S_tgt']), dict(B['S_src'])
        base = list(B['S_tgt'])
        for fname, comps in fusions.items():
            fz = _fusion(B, base, comps)
            if fz is not None and (score_names is None or fname in score_names):
                S_tgt[fname], S_src[fname] = fz; names = names + [fname]
        tgt, tben = B['tgt'], B['tben']
        pos_in_tgt = {r: i for i, r in enumerate(tgt)}
        ben_pos = np.array([pos_in_tgt[r] for r in tben])
        stats = {n: calib.SourceStats(S_src[n], B['src_eco']) for n in names}
        yjs = {n: calib.fit_yj(S_src[n], B['src_eco']) for n in names}
        for fam in FAMILIES:
            P = np.array([pos_in_tgt[r] for r in tgt if df.y.values[r] == 1 and df.family.values[r] == fam
                          and df.split.values[r] == 'test'])
            if len(P) == 0:
                continue
            Lb, Lp = L_all[tgt[ben_pos]], L_all[tgt[P]]
            wb_all, wp = (cues.common_support_weights(Lb, Lp) if cond == 'C2' else (np.ones(len(Lb)), np.ones(len(Lp))))
            succ = df.n_success.values[tgt[P]].astype(float)
            for k in ks:
                if k > len(ben_pos) - 10:
                    continue
                for seed in range(seeds if k else 1):
                    rng = np.random.RandomState(zlib.crc32(f'{T}|{fam}|{k}|{seed}'.encode()))
                    sp = np.sort(rng.choice(len(ben_pos), k, replace=False)) if k else np.array([], int)
                    keep = np.setdiff1d(np.arange(len(ben_pos)), sp)
                    neg = ben_pos[keep]; wb = wb_all[keep]
                    test = np.r_[neg, P]; y = np.r_[np.zeros(len(neg)), np.ones(len(P))]; w = np.r_[wb, wp]
                    for n in names:
                        v = _with_shots(B, n, sp) if n in S.KNN_SCORES else S_tgt[n]
                        st = stats[n]
                        shots = v[ben_pos[sp]]; vt = v[test]
                        auc = wauc(y, vt, w)
                        ap = average_precision_score(y, vt, sample_weight=w) if w[y == 1].sum() > 0 else np.nan
                        if eb_variants:
                            dec = {}
                            for mode in eb_variants:
                                mu, sd, wt = calib.eb_params(st, shots, mode)
                                dec[mode] = ((vt - mu) / sd, wt)
                        else:
                            dec = calib.decisions(vt, st, shots, yj=yjs[n])
                        for meth, (z, wt) in dec.items():
                            f = z > calib.Z; fn, fp_ = f[y == 0], f[y == 1]
                            rows.append(dict(cond=cond, target=T, family=fam, k=k, seed=seed, score=n, method=meth,
                                             fpr=float(fn.mean()), n_neg=int(len(fn)), recall=float((wp * fp_).sum() / wp.sum()) if wp.sum() else np.nan,
                                             recall_raw=float(fp_.mean()),
                                             blocked=float((succ * fp_).sum() / succ.sum()) if succ.sum() else np.nan,
                                             auc=auc, ap=ap, w=wt, t=st.t))
    out = pd.DataFrame(rows)
    if len(out):
        out['cal_err'] = (out.fpr - calib.ALPHA).abs()
        out['fp_per_1000'] = out.fpr * 1000
    return out
