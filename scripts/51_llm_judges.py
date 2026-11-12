"""Several zero-shot LLM judges x prompt variants on the probe set (robustness of the probe findings).

    python scripts\\51_llm_judges.py                 # 3 models x 3 prompts (about 1-2 h on an RTX 4060 incl. downloads)
    python scripts\\51_llm_judges.py --models Qwen/Qwen2.5-1.5B-Instruct --prompts P1
    python scripts\\51_llm_judges.py --mock          # random scores, checks the plumbing only

Every (model, prompt) result is cached in TPC_WORK/results/probe/judges/*.npy as soon as it is finished, so an
interrupted run continues where it stopped. Outputs: probe/judges_scores.csv (one column per detector) and
probe/judges_summary.csv (same metrics as 50_semantic_probe.py). Afterwards run scripts\\52_probe_calibration.py.
"""
import argparse, importlib.util, os, re, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, pandas as pd
from tpc import data, paths, judge

MODELS = ['Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'ibm-granite/granite-3.3-2b-instruct']
spec = importlib.util.spec_from_file_location('probe50', os.path.join(os.path.dirname(__file__), '50_semantic_probe.py'))
probe50 = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe50)


def slug(m):
    return re.sub(r'[^A-Za-z0-9.]+', '-', m.split('/')[-1])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=MODELS); p.add_argument('--prompts', nargs='+', default=list(judge.PROMPTS))
    p.add_argument('--device', default='auto'); p.add_argument('--dtype', default='auto')
    p.add_argument('--batch-size', type=int, default=4); p.add_argument('--max-desc-tokens', type=int, default=384)
    p.add_argument('--llm-4bit', action='store_true'); p.add_argument('--mock', action='store_true')
    a = p.parse_args()
    out_dir = os.path.join(paths.WORK, 'results', 'probe' + ('_mock' if a.mock else ''))   # mock never touches real results
    cache = os.path.join(out_dir, 'judges'); os.makedirs(cache, exist_ok=True)
    if not all(os.path.isdir(data.repo(r)) for r in data.REPOS):
        print('downloading the source repositories ...'); data.fetch()
    full = data.build_corpus(seed=0)
    df = full[(full.y == 0) | (full.split == 'test')].reset_index(drop=True)     # identical to 50_semantic_probe.py
    texts = list(df.text.values); print(f'{len(df)} descriptions')
    S = {}
    S['tfidf_lr'], train_mask = probe50.tfidf_reference(full, df)
    f = os.path.join(cache, 'pi_clf.npy')
    if os.path.exists(f):
        S['pi_clf'] = np.load(f)
    elif a.mock:
        S['pi_clf'] = np.random.RandomState(1).randn(len(df)) + df.y.values
    else:
        from tpc import neural as N
        S['pi_clf'] = N.injection_logit(texts, N.DEFAULTS['pi'], a.device, 16); np.save(f, S['pi_clf'])
    t_all = time.time()
    for m in a.models:
        todo = [pr for pr in a.prompts if not os.path.exists(os.path.join(cache, f'{slug(m)}__{pr}.npy'))]
        for pr in a.prompts:
            if pr not in todo: print(f'   cached: {slug(m)} {pr}')
        if todo:
            print(f'=== {m}: prompts {todo}', flush=True)
            if not a.mock:
                tok, model = judge.load(m, a.device, a.dtype, a.llm_4bit)
            for pr in todo:
                t0 = time.time()
                v = (np.random.RandomState(len(m) + int(pr[1])).randn(len(df)) + 2 * df.y.values) if a.mock else \
                    judge.score(texts, tok, model, pr, a.batch_size, a.max_desc_tokens, label=f'{slug(m)} {pr}')
                np.save(os.path.join(cache, f'{slug(m)}__{pr}.npy'), v)
                print(f'   done {slug(m)} {pr} in {time.time() - t0:.0f} s', flush=True)
            if not a.mock:
                judge.free(model)
        for pr in a.prompts:
            S[f'judge:{slug(m)}:{pr}'] = np.load(os.path.join(cache, f'{slug(m)}__{pr}.npy'))
    sc = df[['rid', 'eco', 'group', 'y', 'family', 'split', 'native', 'template', 'style', 'batch']].copy()
    for k, v in S.items(): sc[k] = v
    sc['tfidf_train'] = train_mask; sc.to_csv(os.path.join(out_dir, 'judges_scores.csv'), index=False)
    T = probe50.summarize(df, S, train_mask); T.to_csv(os.path.join(out_dir, 'judges_summary.csv'), index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300)
    print(T[T.family == 'hand'][['detector', 'target', 'auroc', 'tpr_at_1pct_fpr', 'fpr_source_thr']].round(3).to_string(index=False))
    print(f'\nall done in {time.time() - t_all:.0f} s -> {out_dir}' + ('\n(mock run: random scores, written to probe_mock; nothing real was changed)' if a.mock else '\nnext: python scripts\\52_probe_calibration.py'))
