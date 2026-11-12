"""Where data are read from and results are written to.

TPC_DATA_ROOT : folder that holds the cloned source repositories (default: <package>/external)
TPC_WORK      : output folder (default: <package>/work)
"""
import os

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.environ.get('TPC_DATA_ROOT', os.path.join(PKG, 'external'))
WORK = os.environ.get('TPC_WORK', os.path.join(PKG, 'work'))
CORPUS_DIR = os.path.join(WORK, 'corpus')
SCORE_DIR = os.path.join(WORK, 'scores')
RAW_DIR = os.path.join(WORK, 'results', 'raw')
TABLE_DIR = os.path.join(WORK, 'results', 'tables')
FIG_DIR = os.path.join(WORK, 'figures')

for d in (DATA_ROOT, CORPUS_DIR, TABLE_DIR):
    os.makedirs(d, exist_ok=True)
