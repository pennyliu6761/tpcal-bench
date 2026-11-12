"""Clone the four public source repositories (GitHub only) into TPC_DATA_ROOT at the pinned commits."""
import _common  # noqa
from tpc import data, paths

if __name__ == '__main__':
    print(f'data root: {paths.DATA_ROOT}')
    data.fetch()
