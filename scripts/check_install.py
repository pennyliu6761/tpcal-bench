"""Check that the code package and the downloaded data are complete.

    python scripts\\check_install.py            # check only (downloads nothing)
    python scripts\\check_install.py --fetch    # download missing data first, then check

Checks (1) every code file, (2) size and SHA-256 of the six data files the pipeline reads, (3) the corpus counts,
(4) optionally the Python packages and the GPU.
"""
import argparse, hashlib, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tpc import paths

CODE = ['README.md', 'requirements.txt', 'tpc/__init__.py', 'tpc/paths.py', 'tpc/data.py', 'tpc/cues.py',
        'tpc/scores.py', 'tpc/neural.py', 'tpc/calib.py', 'tpc/harness.py', 'tpc/judge.py', 'scripts/_common.py',
        'scripts/00_fetch_data.py', 'scripts/01_build_corpus.py', 'scripts/13_shortcut_audit.py',
        'scripts/50_semantic_probe.py', 'scripts/51_llm_judges.py', 'scripts/52_probe_calibration.py',
        'scripts/60_paper_figures.py', 'scripts/61_paper_tables.py', 'scripts/62_compare_runs.py']
DATA = {  # path inside TPC_DATA_ROOT: (bytes, sha256) at the pinned commits
    'MCPTox-Benchmark/response_all.json': (20666873, '79a90049be931c59e71446d6180b1d7f0d196d123d08a59bc155d142b5041c03'),
    'MetaTool/dataset/plugin_info.json': (326361, '864bf25bd5494feca861acb98f69d260f212d9c098a2dc68b4b2ee357d6fb2f5'),
    'tp-detect/data/wild_benign.json': (794239, 'bd30653fc93cab8d8fae812d9d71723a2908d185cfe1f9c1f7ba978b085cdb10'),
    'tp-detect/data/ood_attacks.json': (115317, 'a07957d55beae87789c3cf873a998d68c2126f913b2041941e7ae0fdc394af5f'),
    'tp-detect/data/ood2_attacks.json': (90158, 'a1458bb8f1bf4270ed6f01a3fd75d224f1d6be5bae456a0760c72593f8de65f4'),
    'ToolEmu/assets/all_toolkits.json': (541498, 'e21a8f70b1d5de4677d6d52642936a322655d79b17a72c84f600550384083a1e'),
}
CORPUS = {('MCP-curated', 'benign'): 326, ('MCP-curated', 'hand'): 157, ('MCP-curated', 'mcptox'): 1348,
          ('MCP-wild', 'benign'): 610, ('MCP-wild', 'hand'): 204, ('MCP-wild', 'mcptox'): 600,
          ('Plugin', 'benign'): 389, ('Plugin', 'hand'): 157, ('Plugin', 'mcptox'): 600,
          ('ToolEmu', 'benign'): 329, ('ToolEmu', 'hand'): 157, ('ToolEmu', 'mcptox'): 600}
OK, BAD, WARN = '  OK  ', ' FAIL ', ' WARN '

if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__); a.add_argument('--fetch', action='store_true'); a = a.parse_args()
    fails = 0
    print('1) code files'); miss = [f for f in CODE if not os.path.isfile(os.path.join(paths.PKG, f))]
    print(f'[{OK if not miss else BAD}] {len(CODE) - len(miss)}/{len(CODE)} present' + (f'; missing: {miss}' if miss else '')); fails += bool(miss)
    if a.fetch:
        from tpc import data; print('   downloading missing data ...'); data.fetch()
    print(f'2) data files in {paths.DATA_ROOT}')
    for rel, (size, sha) in DATA.items():
        p = os.path.join(paths.DATA_ROOT, rel)
        if not os.path.isfile(p):
            print(f'[{BAD}] {rel}: not found (run with --fetch, or see README, Data sources)'); fails += 1; continue
        b = open(p, 'rb').read(); h = hashlib.sha256(b).hexdigest()
        if h == sha:
            print(f'[{OK}] {rel} ({len(b):,} bytes)')
        elif hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest() == sha:
            print(f'[{WARN}] {rel}: Windows line endings, content identical')
        else:
            print(f'[{BAD}] {rel}: {len(b):,} bytes (expected {size:,}), checksum differs -> delete its folder and run --fetch'); fails += 1
    print('3) corpus')
    try:
        from tpc import data
        df = data.build_corpus(); got = df.groupby(['eco', 'family']).size().to_dict()
        bad = {k: (got.get(k, 0), v) for k, v in CORPUS.items() if got.get(k, 0) != v}
        print(f'[{OK if not bad else BAD}] {len(df):,} descriptions (expected 5,477)' + (f'; differences (got, expected): {bad}' if bad else ''))
        fails += bool(bad)
    except Exception as e:
        print(f'[{BAD}] corpus could not be built: {e}'); fails += 1
    print('4) packages')
    for m in ('numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'torch', 'transformers', 'accelerate', 'sentencepiece'):
        try:
            mod = __import__(m); print(f'[{OK}] {m} {getattr(mod, "__version__", "")}')
        except Exception:
            core = m in ('numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib')
            print(f'[{BAD if core else WARN}] {m} not installed' + ('' if core else ' (needed only for the neural / probe steps)')); fails += core
    deep = ['pandas._libs.parsers', 'sklearn.feature_extraction.text', 'sklearn.linear_model', 'sklearn.metrics',
            'sklearn.preprocessing', 'scipy.stats', 'matplotlib.pyplot']
    bad = []
    for m in deep:
        try:
            __import__(m)
        except Exception as e:
            bad.append(f'{m}: {e}')
    print(f'[{OK if not bad else BAD}] compiled modules used by the scripts' + ''.join(f'\n         {b}' for b in bad)); fails += bool(bad)
    try:
        import torch
        print(f'[{OK if torch.cuda.is_available() else WARN}] GPU: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU only; update the NVIDIA driver or install the CUDA build of torch)'))
    except Exception:
        pass
    print('\nALL CHECKS PASSED' if fails == 0 else f'\n{fails} check(s) failed')
