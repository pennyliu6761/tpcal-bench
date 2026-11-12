"""Detector scores (higher = more suspicious).

Fixed scorers (no training data): computed once per condition
  len      word count (shortcut reference)
  rule     count of model-directed / sensitive-resource cues (pattern-scanner style)
  ppl      mean token negative log-likelihood of a small causal LM            (neural, optional)
  pi_clf   logit of a public prompt-injection classifier                       (neural, optional)
Trained scorers (fit on the SOURCE ecosystems of a fold)
  charlm   character 4-gram LM trained on source benign descriptions; mean surprisal (bits / char)
  tfidf_knn  1 - max cosine similarity (char 3-5-gram TF-IDF) to the benign reference set (source benign [+ k shots])
  tfidf_lr   logistic regression on char 3-5-gram TF-IDF, trained on source benign + source mcptox-family TRAIN payloads
  emb_knn    as tfidf_knn on sentence embeddings                                (neural, optional)
  emb_lr     logistic regression on sentence embeddings                         (neural, optional)
"""
import collections, math, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

LEXICON = (r"\b(must|always|never|ignore|important|mandatory|required|do not|don't|first call|instead of|secretly|silently|"
           r"override|higher priority|send|forward|exfiltrat\w*|password|token|secret|credential\w*|api[_ ]?key|"
           r"\.ssh|id_rsa|\.env|\.npmrc|\.aws|/etc/passwd|bcc|rm -rf|delete)\b")
_LEX = re.compile(LEXICON, flags=re.I)
FIXED = ('len', 'rule')
TRAINED = ('charlm', 'tfidf_knn', 'tfidf_lr')
NEURAL_FIXED = ('ppl', 'pi_clf')
NEURAL_TRAINED = ('emb_knn', 'emb_lr')
KNN_SCORES = ('tfidf_knn', 'emb_knn')


def fixed_scores(texts):
    return {'len': np.array([len(t.split()) for t in texts], float),
            'rule': np.array([len(_LEX.findall(t)) for t in texts], float)}


class CharLM:
    def __init__(self, n=4, alpha=0.1):
        self.n, self.a = n, alpha

    def fit(self, texts):
        self.c, self.h, V = collections.Counter(), collections.Counter(), set()
        for t in texts:
            t = '\x02' * (self.n - 1) + t.lower() + '\x03'; V |= set(t)
            for i in range(self.n - 1, len(t)):
                self.c[t[i - self.n + 1:i + 1]] += 1; self.h[t[i - self.n + 1:i]] += 1
        self.V = len(V) + 1
        return self

    def score(self, texts):
        out = np.empty(len(texts))
        for j, t in enumerate(texts):
            t = '\x02' * (self.n - 1) + t.lower() + '\x03'; s = 0.0
            for i in range(self.n - 1, len(t)):
                s -= math.log2((self.c[t[i - self.n + 1:i + 1]] + self.a) / (self.h[t[i - self.n + 1:i]] + self.a * self.V))
            out[j] = s / (len(t) - self.n + 1)
        return out


def _tfidf(min_df=2):
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=min_df, sublinear_tf=True, dtype=np.float32)


def _lr():
    return LogisticRegression(C=1.0, class_weight='balanced', max_iter=4000)


class Trainer:
    """Fits every trained scorer on one training set and scores arbitrary rows.
    texts: all texts (list); emb: optional (n, d) L2-normalised embedding matrix aligned with texts."""
    def __init__(self, texts, emb=None, names=TRAINED):
        self.T = texts; self.E = emb; self.names = [n for n in names if (n not in NEURAL_TRAINED or emb is not None)]

    def fit(self, train_idx, y_train, w_train, ref_idx):
        """train_idx/y_train/w_train: supervised training rows; ref_idx: benign reference rows (kNN / LM)."""
        T = self.T; self.models = {}
        if 'charlm' in self.names:
            self.models['charlm'] = CharLM().fit([T[i] for i in ref_idx])
        if 'tfidf_knn' in self.names or 'tfidf_lr' in self.names:
            vec = _tfidf().fit([T[i] for i in train_idx]); self.vec = vec
            if 'tfidf_lr' in self.names:
                self.models['tfidf_lr'] = _lr().fit(vec.transform([T[i] for i in train_idx]), y_train, sample_weight=w_train)
            if 'tfidf_knn' in self.names:
                kv = _tfidf(min_df=1).fit([T[i] for i in ref_idx]); self.kv = kv
                self.ref_tfidf = kv.transform([T[i] for i in ref_idx])
        if self.E is not None:
            if 'emb_lr' in self.names:
                self.models['emb_lr'] = LogisticRegression(C=1.0, class_weight='balanced', max_iter=4000).fit(
                    self.E[train_idx], y_train, sample_weight=w_train)
            if 'emb_knn' in self.names:
                self.ref_emb = self.E[ref_idx]
        return self

    def score(self, idx, extra_ref=None):
        """Scores for rows `idx`. extra_ref: additional benign reference rows (k target shots) for kNN scores;
        a row is never its own neighbour."""
        T = self.T; out = {}; idx = np.asarray(idx)
        if 'charlm' in self.names:
            out['charlm'] = self.models['charlm'].score([T[i] for i in idx])
        if 'tfidf_lr' in self.names:
            out['tfidf_lr'] = self.models['tfidf_lr'].decision_function(self.vec.transform([T[i] for i in idx]))
        if 'tfidf_knn' in self.names:
            Xq = self.kv.transform([T[i] for i in idx])
            out['tfidf_knn'] = 1 - _maxsim(Xq, self.ref_tfidf, idx, extra_ref, lambda j: self.kv.transform([T[i] for i in j]))
        if self.E is not None:
            if 'emb_lr' in self.names:
                out['emb_lr'] = self.models['emb_lr'].decision_function(self.E[idx])
            if 'emb_knn' in self.names:
                out['emb_knn'] = 1 - _maxsim(self.E[idx], self.ref_emb, idx, extra_ref, lambda j: self.E[j])
        return out


def _maxsim(Xq, R, idx, extra_ref, embed):
    s = Xq @ R.T
    s = s.toarray() if hasattr(s, 'toarray') else np.asarray(s)
    best = s.max(1) if s.shape[1] else np.full(len(idx), -1.0)
    if extra_ref is not None and len(extra_ref):
        e = Xq @ embed(extra_ref).T
        e = e.toarray() if hasattr(e, 'toarray') else np.asarray(e)
        e[np.asarray(idx)[:, None] == np.asarray(extra_ref)[None, :]] = -1.0
        best = np.maximum(best, e.max(1))
    return best
