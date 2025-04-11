"""
Stand-in replacement for `os.path` with vectorization support.
All functions accept `Tensor` as inputs, broadcasting the function call.
"""
from os import path as impl
from typing import Union, Sequence

from ..math import Tensor, broadcast, map


# --- Constants ---
supports_unicode_filenames: bool = impl.supports_unicode_filenames
# aliases (also in os)
curdir = impl.curdir
pardir = impl.pardir
sep = impl.sep
altsep = impl.altsep
extsep = impl.extsep
pathsep = impl.pathsep
defpath = impl.defpath
devnull = impl.devnull

# --- Functions ---
split = broadcast(impl.split, name=False)
splitdrive = broadcast(impl.splitdrive, name=False)
splitext = broadcast(impl.splitext, name=False)
join = broadcast(impl.join, name=False)
def commonpath(paths: Sequence[Union[str, Tensor]]):
    return map(lambda *p: impl.commonpath(p), *paths)
def commonprefix(list: Sequence[Union[str, Tensor]]):
    return map(lambda *p: impl.commonpath(p), *list)
relpath = broadcast(impl.relpath, name=False)
samefile = broadcast(impl.samefile, name=False)

basename = broadcast(impl.basename, name=False)
dirname = broadcast(impl.dirname, name=False)
realpath = broadcast(impl.realpath, name=False)
abspath = broadcast(impl.abspath, name=False)
expanduser = broadcast(impl.expanduser, name=False)
expandvars = broadcast(impl.expandvars, name=False)
normcase = broadcast(impl.normcase, name=False)
normpath = broadcast(impl.normpath, name=False)

exists = broadcast(impl.exists, name=False)
lexists = broadcast(impl.lexists, name=False)
isdir = broadcast(impl.isdir, name=False)
isfile = broadcast(impl.isfile, name=False)
isabs = broadcast(impl.isabs, name=False)
islink = broadcast(impl.islink, name=False)
ismount = broadcast(impl.ismount, name=False)

getsize = broadcast(impl.getsize, name=False)
getatime = broadcast(impl.getatime, name=False)
getctime = broadcast(impl.getctime, name=False)
getmtime = broadcast(impl.getmtime, name=False)

sameopenfile = broadcast(impl.sameopenfile, name=False)
samestat = broadcast(impl.samestat, name=False)
