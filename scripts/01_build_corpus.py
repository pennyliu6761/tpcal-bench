"""Build the unified corpus (4 benign ecosystems; mcptox and hand-written poisoned families; transplants) and the
dataset summary table (Table 1)."""
import os
import _common  # noqa
import numpy as np, pandas as pd
from tpc import data, paths, cues

if __name__ == '__main__':
    df = data.build_corpus(seed=0)
    p = data.save_corpus(df); print(f'corpus: {len(df)} descriptions -> {p}')
    df['words'] = cues.n_words(df.text.values); df['boiler'] = [cues.has_boilerplate(t) for t in df.text]
    T = (df.groupby(['eco', 'family', 'split', 'native'])
           .agg(n=('rid', 'size'), groups=('group', 'nunique'), words_median=('words', 'median'),
                words_p90=('words', lambda x: np.percentile(x, 90)), boilerplate=('boiler', 'mean')).reset_index())
    T.to_csv(os.path.join(paths.TABLE_DIR, 'data_summary.csv'), index=False)
    print(T.round(2).to_string(index=False))
