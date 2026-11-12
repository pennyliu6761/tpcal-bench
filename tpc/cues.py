"""Construction cues and their controls.

C0  raw text
C1  template boilerplate removed from EVERY description (benign and poisoned alike):
      'Ignore the previous instructions (and do the following)', '<IMPORTANT>' / '</IMPORTANT>' tags,
      "This description's priority is higher than the user's query/request".
    These strings are fixed phrases of the MCPTox generation templates (and the <IMPORTANT> tag of public PoC payloads);
    the malicious instruction itself (e.g. 'before using this tool, call write_file ...') is NOT removed.
C2  C1 + length matching: two-sided importance weights so that benign and poisoned descriptions have the same
    log-length distribution on their common support (weights are used for AUROC / AP / recall and for training).
"""
import re
import numpy as np

CONDITIONS = ('C0', 'C1', 'C2')
MAIN_CONDITION = 'C2'

BOILERPLATE = [
    r"ignore (all |any )?(the )?(previous|prior|above) instructions?\s*(and\s*)?(do the following)?\s*[:.,]?",
    r"</?\s*important\s*>",
    r"this description'?s priority is higher than the user'?s query\s*/?\s*(or\s*)?request\.?",
]
_BP = [re.compile(p, flags=re.I) for p in BOILERPLATE]


def has_boilerplate(t):
    return any(p.search(t) for p in _BP)


def strip_boilerplate(t):
    for p in _BP:
        t = p.sub(' ', t)
    return re.sub(r"[ \t]+", ' ', t).strip()


def texts_for(cond, raw):
    return list(raw) if cond == 'C0' else [strip_boilerplate(t) for t in raw]


def n_words(texts):
    return np.array([len(t.split()) for t in texts], float)


def common_support_weights(len_neg, len_pos, width=0.2):
    """Two-sided weights making log(1+length) of both classes identical on the common support (bins of `width`).
    Samples outside the common support get weight 0. Each class's weights average 1 over its samples."""
    xb, xp = np.log1p(len_neg), np.log1p(len_pos)
    if len(xb) == 0 or len(xp) == 0:
        return np.ones(len(xb)), np.ones(len(xp))
    lo, hi = max(xb.min(), xp.min()), min(xb.max(), xp.max())
    if hi < lo:
        return np.zeros(len(xb)), np.zeros(len(xp))
    edges = np.arange(lo - 1e-9, hi + width, width)
    hb = np.histogram(xb, edges)[0].astype(float); hp = np.histogram(xp, edges)[0].astype(float)
    both = (hb > 0) & (hp > 0)
    target = np.where(both, np.minimum(hb / hb.sum(), hp / hp.sum()), 0.0)
    if target.sum() == 0:
        return np.zeros(len(xb)), np.zeros(len(xp))
    target /= target.sum()

    def w(x, h):
        idx = np.searchsorted(edges, x, side='right') - 1
        ok = (idx >= 0) & (idx < len(h))
        out = np.zeros(len(x))
        out[ok] = np.where(both[idx[ok]], target[idx[ok]] / np.maximum(h[idx[ok]], 1e-12), 0.0)
        s = out.sum()
        return out * (len(x) / s) if s > 0 else out
    return w(xb, hb), w(xp, hp)


def effective_n(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
