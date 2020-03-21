import time
from contextlib import contextmanager
import cloudpickle


@contextmanager
def timer(name='process'):
    """
    Usage:
        with timer('process train'):
            (Process)
    """
    print(f'\n[{name}] start\n')
    start_time = time.time()
    yield
    print(f'\n[{name}] done in {time.time() - start_time:.2f} sec\n')


def load(path):
    with open(path, 'rb') as f:
        obj = cloudpickle.loads(f.read())
    return obj


def dump(obj, path):
    with open(path, 'wb') as f:
        f.write(cloudpickle.dumps(obj))
