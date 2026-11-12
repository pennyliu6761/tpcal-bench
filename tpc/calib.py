"""Score-level calibration of a detector to a new (target) ecosystem from k benign target descriptions.

Every rule returns a decision statistic z; a description is flagged when z > Z (Z = 2, nominal FPR alpha = 1 - Phi(2)).
  Global    z = (s - mean_src) / sd_src over all pooled source-benign scores (no target data)
  Pooled    z = (s - mu0) / sd0 with mu0 = mean of per-source-ecosystem means, sd0^2 = mean within-ecosystem variance
  Quantile  flag s > empirical (1 - alpha) quantile of the k shots (dataset-adaptive threshold; reported for k >= 20)
  Ours-A    z = (s - mean(shots)) / sd(shots)                (k = 1: sd0)
  Ours-B    empirical-Bayes shrinkage of the target mean and variance toward (mu0, sd0^2) with
            w = k t / (k t + 1),  t = between-ecosystem variance of the means / within-ecosystem variance
            (estimated on the SOURCE ecosystems only), mu = w mean(shots) + (1 - w) mu0, sd^2 = w sd(shots)^2 + (1 - w) sd0^2
EB ablation variants: 'eb' (= Ours-B), 't1' (t = 1), 'fixed' (w = 0.5), 'mean_only' (shrink mean, keep shot sd),
'pooled' (w = 0), 'shots' (w = 1 = Ours-A).
"""
import math
import numpy as np
from scipy.stats import norm

Z = 2.0
ALPHA = float(1 - norm.cdf(Z))          # 0.02275
METHODS = ('Global', 'Pooled', 'Src-Quantile', 'Quantile', 'Ours-A', 'Ours-B', 'Ours-B-tail', 'Ours-B-YJ')
EB_VARIANTS = ('eb', 't1', 'fixed', 'mean_only', 'pooled', 'shots')


class SourceStats:
    def __init__(self, v, eco):
        v = np.asarray(v, float); eco = np.asarray(eco)
        mus, var, ns = [], [], []
        for e in sorted(set(eco)):
            x = v[eco == e]
            if len(x) >= 2:
                mus.append(x.mean()); var.append(x.var(ddof=1)); ns.append(len(x))
        self.mus, self.var, self.ns = np.array(mus), np.array(var), np.array(ns)
        self.within = float(np.mean(self.var)) + 1e-12
        between = float(np.var(self.mus, ddof=1)) - self.within * float(np.mean(1 / self.ns)) if len(self.mus) > 1 else 0.0
        self.t = max(between, 0.0) / self.within
        self.mu0, self.sd0 = float(self.mus.mean()), math.sqrt(self.within)
        self.g_mu, self.g_sd = float(v.mean()), float(v.std(ddof=1)) + 1e-12
        # tail factor: (1 - alpha) quantile of the within-ecosystem standardised source-benign scores
        zs = np.concatenate([(v[eco == e] - v[eco == e].mean()) / (v[eco == e].std(ddof=1) + 1e-12)
                             for e in sorted(set(eco)) if (eco == e).sum() >= 2])
        self.c_tail = float(np.quantile(zs, 1 - ALPHA))
        self.q_src = float(np.quantile(v, 1 - ALPHA))


def eb_params(st, shots, mode='eb', w_fixed=0.5):
    k = len(shots)
    if k == 0 or mode == 'pooled':
        return st.mu0, st.sd0, 0.0
    m = float(np.mean(shots)); s = float(np.std(shots, ddof=1)) if k > 1 else st.sd0
    t = {'eb': st.t, 't1': 1.0}.get(mode, st.t)
    w = k * t / (k * t + 1)
    if mode == 'fixed': w = w_fixed
    if mode == 'shots': w = 1.0
    mu = w * m + (1 - w) * st.mu0
    sd = s if mode in ('mean_only', 'shots') else math.sqrt(w * s ** 2 + (1 - w) * st.sd0 ** 2)
    return mu, max(sd, 1e-9), w


def decisions(v_test, st, shots, k_quantile_min=20, yj=None):
    """dict method -> (statistic, EB weight); a description is flagged when statistic > Z.
    Tail-aware rules (added after the pilot showed that benign scores are skewed / heavy-tailed, so 'mean + 2 sd'
    over-flags even with perfect location and scale):
      Src-Quantile  flag s > (1 - alpha) quantile of all pooled source-benign scores (k = 0, no normality assumption)
      Ours-B-tail   Ours-B location / scale, but threshold c_tail (source-estimated standardised tail quantile) instead of Z
      Ours-B-YJ     Ours-B on Yeo-Johnson-transformed scores (transform fitted on source benign only); yj = (st_yj, f)
    """
    out = {'Global': ((v_test - st.g_mu) / st.g_sd, 0.0), 'Pooled': ((v_test - st.mu0) / st.sd0, 0.0),
           'Src-Quantile': (v_test - st.q_src + Z, 0.0)}
    mu, sd, w = eb_params(st, shots, 'eb')
    out['Ours-B-tail'] = ((v_test - mu) / sd - st.c_tail + Z, w)
    if yj is not None:
        st_y, f = yj; ty = f(v_test); sy = f(shots) if len(shots) else shots
        mu_y, sd_y, w_y = eb_params(st_y, sy, 'eb'); out['Ours-B-YJ'] = ((ty - mu_y) / sd_y, w_y)
    if len(shots):
        out['Ours-B'] = ((v_test - mu) / sd, w)
        mu, sd, _ = eb_params(st, shots, 'shots'); out['Ours-A'] = ((v_test - mu) / sd, 1.0)
        if len(shots) >= k_quantile_min:
            thr = np.quantile(shots, 1 - ALPHA); out['Quantile'] = (v_test - thr + Z, 1.0)
    return out


def fit_yj(v_src, eco):
    """Yeo-Johnson transform fitted on pooled source-benign scores; returns (SourceStats on transformed scores, f)."""
    from sklearn.preprocessing import PowerTransformer
    v = np.asarray(v_src, float).reshape(-1, 1)
    pt = PowerTransformer(method='yeo-johnson', standardize=True).fit(v)
    f = lambda x: pt.transform(np.asarray(x, float).reshape(-1, 1)).ravel()
    return SourceStats(f(v_src), eco), f
