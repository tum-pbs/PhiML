import os

from ..math import Shape, batch, layout


def listdir(path, list_dim: Shape = batch('files')):
    files = os.listdir(path)
    if not list_dim.size:
        list_dim = list_dim.with_size(files)
    paths = [os.path.join(path, f) for f in files]
    return layout(paths, list_dim)
