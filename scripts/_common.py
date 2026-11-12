"""Shared command-line options for the experiment scripts."""
import argparse, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# k = number of benign target-ecosystem descriptions available for calibration.
KS_PAPER = (0, 5, 20, 50)
KS_PRESPECIFIED = (5, 20, 50)
SEEDS = 10


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--quick', action='store_true', help='smoke test: 2 seeds, k in {0, 20}; results get the suffix _quick')
    p.add_argument('--seeds', type=int, default=SEEDS, help=f'shot-sampling seeds for k >= 1 (default {SEEDS})')
    p.add_argument('--ks', type=int, nargs='+', help='override the k values')
    return p


def settings(args, ks=KS_PAPER):
    if args.ks:
        ks = tuple(args.ks)
    if args.quick:
        return tuple(k for k in ks if k in (0, 20)) or ks[:2], 2, '_quick'
    return tuple(ks), args.seeds, ''


class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): print(f'   [{self.label}: {time.time() - self.t:.0f} s]', flush=True)
