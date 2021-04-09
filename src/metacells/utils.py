import scanpy as sc
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

def load_data():
    p = get_data('sample_data.h5ad')
    return sc.read(p)