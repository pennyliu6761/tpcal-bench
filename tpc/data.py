"""Source repositories, loaders and the unified corpus.

Benign ecosystems (unit of the leave-one-ecosystem-out protocol)
  MCP-curated : benign tool descriptions of the 45 MCP servers in MCPTox (AAAI 2026)
  MCP-wild    : tools/list output of public PyPI / npm MCP servers collected by TP-Detect (MIT)
  Plugin      : 'description_for_model' of ChatGPT plugins in MetaTool (ICLR 2024, MIT)
  ToolEmu     : tool summaries of the ToolEmu toolkits (ICLR 2024, Apache-2.0)
Poisoned families
  mcptox : LLM-generated poisoned descriptions of MCPTox (3 templates). Split by server into a TRAIN half and a TEST
           half; detectors are trained on TRAIN-half payloads only and evaluated on TEST-half payloads only.
  hand   : author-written poisoned descriptions of TP-Detect (OOD-1, OOD-2; 15 styles). Never used for training.
In MCP-curated (mcptox) and MCP-wild (hand) the poisoned descriptions are native. In every other ecosystem a payload is
transplanted onto a benign host description of that ecosystem (host text + payload), which is how tool poisoning is
deployed in practice (a plausible description with a hidden instruction).
The data are NOT redistributed: scripts/00_fetch_data.py clones the repositories at the pinned commits.
"""
import difflib, json, os, re, subprocess, zlib
import numpy as np, pandas as pd
from . import paths

REPOS = {
    'MCPTox-Benchmark': ('https://github.com/zhiqiangwang4/MCPTox-Benchmark', 'f85189f9ad12504c197c7f920ab818a40657b1fa'),
    'MetaTool': ('https://github.com/HowieHwong/MetaTool', '35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb'),
    'tp-detect': ('https://github.com/amaneja21/tp-detect', '804a36bb76c090f0d5a3764cd381467d15faf943'),
    'ToolEmu': ('https://github.com/ryoungj/ToolEmu', 'ac4a7ab7ed8c7985d96231e214bd6b54304b7ddb'),
}
ECOSYSTEMS = ('MCP-curated', 'MCP-wild', 'Plugin', 'ToolEmu')
FAMILIES = ('mcptox', 'hand')
CAP_TRANSPLANT = 300          # transplanted poisoned descriptions per (ecosystem, family, split)


def repo(name):
    return os.path.join(paths.DATA_ROOT, name)


def _download_zip(name, url, commit):
    """Git-free download: fetch the GitHub archive of the pinned commit (https://codeload.github.com/...) and unpack it."""
    import io, shutil, urllib.request, zipfile
    owner_repo = url.rstrip('/').split('github.com/')[1]
    zurl = f'https://codeload.github.com/{owner_repo}/zip/{commit}'
    print(f'   downloading {zurl}', flush=True)
    req = urllib.request.Request(zurl, headers={'User-Agent': 'tpcal-bench'})
    with urllib.request.urlopen(req, timeout=300) as r:
        buf = io.BytesIO(r.read())
    tmp = os.path.join(paths.DATA_ROOT, f'_tmp_{name}')
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(buf) as z:
        z.extractall(tmp)
    inner = [os.path.join(tmp, d) for d in os.listdir(tmp)]
    src = inner[0] if len(inner) == 1 and os.path.isdir(inner[0]) else tmp
    shutil.move(src, repo(name)); shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(repo(name), 'PINNED_COMMIT.txt'), 'w') as f:
        f.write(commit + '\n')


def fetch(pin=True):
    """Download every source repository at its pinned commit: with git if it is installed, otherwise as a zip archive
    over HTTPS (no git needed)."""
    import shutil
    if shutil.which('git') is None:
        for name, (url, commit) in REPOS.items():
            if os.path.isdir(repo(name)):
                print(f'   {name:18s} already present'); continue
            _download_zip(name, url, commit); print(f'   {name:18s} {commit[:12]} (zip)')
        return
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    for name, (url, commit) in REPOS.items():
        if os.path.isdir(repo(name)) and not os.path.isdir(os.path.join(repo(name), '.git')):
            print(f'   {name:18s} already present (downloaded without git)'); continue
        d = repo(name)
        if not os.path.isdir(os.path.join(d, '.git')):
            subprocess.check_call(['git', 'clone', '--quiet', url, d], env=env)
        head = subprocess.check_output(['git', '-C', d, 'rev-parse', 'HEAD'], text=True).strip()
        if pin and head != commit:
            try:
                subprocess.check_call(['git', '-C', d, 'fetch', '--quiet', 'origin', commit], env=env)
                subprocess.check_call(['git', '-C', d, 'checkout', '--quiet', commit], env=env)
            except subprocess.CalledProcessError:
                print(f'   warning: could not check out pinned commit {commit[:10]} of {name}; using {head[:10]}')
        head = subprocess.check_output(['git', '-C', d, 'rev-parse', 'HEAD'], text=True).strip()
        print(f'   {name:18s} {head[:12]}')


# ------------------------------------------------------------------ loaders
def load_mcptox():
    r = json.load(open(os.path.join(repo('MCPTox-Benchmark'), 'response_all.json'), encoding='utf-8'))
    ben, poi, seen = [], [], set()
    for sname, x in r['servers'].items():
        for tname, d in re.findall(r"Tool: (.+?)\nDescription: (.*?)\nArguments:", x['clean_system_promot'], flags=re.S):
            d = d.strip()
            if d and (sname, d) not in seen:
                seen.add((sname, d)); ben.append(dict(group=sname, tool=tname.strip(), text=d, category=str(x['category_id'])))
        for inst in x['malicious_instance']:
            m = re.search(r"Description: (.*?)(\nArguments:|$)", inst['poisoned_tool'], flags=re.S)
            meta = inst.get('metadata') or {}
            labels = [v for d in inst['datas'] for v in d['label'].values()]
            poi.append(dict(group=sname, text=(m.group(1) if m else inst['poisoned_tool']).strip(), category=str(x['category_id']),
                            template=meta.get('paradigm', 'NA'), risk=meta.get('security risk', 'NA'),
                            n_success=sum(v == 'Success' for v in labels), n_judged=len(labels)))
    return ben, poi


def load_wild():
    W = json.load(open(os.path.join(repo('tp-detect'), 'data', 'wild_benign.json'), encoding='utf-8'))
    return [dict(group=w['server'], tool=w['tool'], text=w['description'].strip()) for w in W if w['description'].strip()]


def load_hand():
    out = []
    for f, batch in (('ood_attacks.json', 'OOD-1'), ('ood2_attacks.json', 'OOD-2')):
        for o in json.load(open(os.path.join(repo('tp-detect'), 'data', f), encoding='utf-8')):
            out.append(dict(group=o['server'], tool=o['tool'], text=o['description'], style=o.get('style', 'NA'),
                            goal=o.get('goal', 'NA'), batch=batch))
    return out


def load_plugin():
    info = json.load(open(os.path.join(repo('MetaTool'), 'dataset', 'plugin_info.json'), encoding='utf-8'))
    out, seen = [], set()
    for x in info:
        t = (x.get('description_for_model') or '').strip()
        if t and t not in seen:
            seen.add(t); out.append(dict(group=x.get('name_for_model', 'plugin'), tool=x.get('name_for_model', ''), text=t))
    return out


def load_toolemu():
    out, seen = [], set()
    for k in json.load(open(os.path.join(repo('ToolEmu'), 'assets', 'all_toolkits.json'), encoding='utf-8')):
        for t in k['tools']:
            s = (t.get('summary') or '').strip()
            if s and s not in seen:
                seen.add(s); out.append(dict(group=k['toolkit'], tool=t.get('name', ''), text=s))
    return out


def extract_payload(poisoned, carrier):
    """Text inserted into `carrier` to obtain `poisoned` (concatenation of inserted / replaced segments)."""
    sm = difflib.SequenceMatcher(None, carrier, poisoned, autojunk=False)
    ins = [poisoned[j1:j2] for op, i1, i2, j1, j2 in sm.get_opcodes() if op in ('insert', 'replace')]
    return re.sub(r"\s+", ' ', ' '.join(ins)).strip()


def _split_of(server):
    return 'train' if zlib.crc32(server.encode()) % 2 == 0 else 'test'


# ------------------------------------------------------------------ corpus
def build_corpus(seed=0):
    rng = np.random.RandomState(seed)
    mb, mp = load_mcptox(); wild = load_wild(); hand = load_hand(); plug = load_plugin(); emu = load_toolemu()
    mcp_texts = {b['text'] for b in mb}
    wild = [w for w in wild if w['text'] not in mcp_texts]          # no description in two ecosystems
    rows = []

    def add(eco, group, text, y, family, split='', native=True, host=-1, **kw):
        rows.append(dict(eco=eco, group=group, text=text, y=int(y), family=family, split=split, native=native, host=host, **kw))
    benign = {'MCP-curated': mb, 'MCP-wild': wild, 'Plugin': plug, 'ToolEmu': emu}
    for eco, B in benign.items():
        for b in B:
            add(eco, b['group'], b['text'], 0, 'benign')
    # mcptox family: native in MCP-curated (split by server), transplanted elsewhere
    for p in mp:
        add('MCP-curated', p['group'], p['text'], 1, 'mcptox', _split_of(p['group']), True, -1, template=p['template'],
            risk=p['risk'], n_success=p['n_success'], n_judged=p['n_judged'])
    df0 = pd.DataFrame(rows)
    host_idx = {eco: df0.index[(df0.eco == eco) & (df0.y == 0)].values for eco in benign}
    for eco in ('MCP-wild', 'Plugin', 'ToolEmu'):
        for split in ('train', 'test'):
            pool = [p for p in mp if _split_of(p['group']) == split]
            pick = rng.choice(len(pool), min(CAP_TRANSPLANT, len(pool)), replace=False)
            for i in pick:
                h = rng.choice(host_idx[eco]); p = pool[i]
                add(eco, df0.at[h, 'group'], df0.at[h, 'text'].rstrip() + ' ' + p['text'], 1, 'mcptox', split, False, int(h),
                    template=p['template'], risk=p['risk'])
    # hand family (test only): native in MCP-wild, payload transplanted elsewhere
    wild_by_tool = {(w['group'], w['tool']): w['text'] for w in wild}
    wild_by_tool.update({(w['group'], w['tool']): w['text'] for w in load_wild()})
    payloads = []
    for o in hand:
        add('MCP-wild', o['group'], o['text'], 1, 'hand', 'test', True, -1, style=o['style'], goal=o['goal'], batch=o['batch'])
        car = wild_by_tool.get((o['group'], o['tool']))
        pl = extract_payload(o['text'], car) if car else ''
        if len(pl.split()) >= 4:
            payloads.append((pl, o))
    for eco in ('MCP-curated', 'Plugin', 'ToolEmu'):
        for pl, o in payloads:
            h = rng.choice(host_idx[eco])
            add(eco, df0.at[h, 'group'], df0.at[h, 'text'].rstrip() + '\n' + pl, 1, 'hand', 'test', False, int(h),
                style=o['style'], goal=o['goal'], batch=o['batch'])
    df = pd.DataFrame(rows)
    for c in ('template', 'risk', 'style', 'goal', 'batch'):
        df[c] = df[c].fillna('')
    for c in ('n_success', 'n_judged'):
        df[c] = df[c].fillna(0).astype(int)
    df.insert(0, 'rid', np.arange(len(df)))
    return df


def corpus_path():
    return os.path.join(paths.CORPUS_DIR, 'corpus.parquet' if _has_parquet() else 'corpus.pkl')


def _has_parquet():
    try:
        import pyarrow  # noqa
        return True
    except Exception:
        return False


def save_corpus(df):
    p = corpus_path()
    df.to_parquet(p) if p.endswith('.parquet') else df.to_pickle(p)
    df.drop(columns=['text']).to_csv(os.path.join(paths.CORPUS_DIR, 'corpus_index.csv'), index=False)
    return p


def load_corpus():
    p = corpus_path()
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} not found: run scripts/01_build_corpus.py first')
    return pd.read_parquet(p) if p.endswith('.parquet') else pd.read_pickle(p)
