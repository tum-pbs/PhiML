"""
Adding vectorization to `os` functions.

Example:
    >>> from phiml import os
    >>> dirs = list_directories('root', startswith='data_')
    >>> files = list_files(dirs, endswith='.h5')
    >>> which_exist = os.path.exists(files)
"""
import os as impl
from typing import Union, Callable

from ..math import Tensor, Shape, broadcast, batch, layout, map

# --- Submodules ---
from . import path

# --- Constants ---
from os import curdir, pardir, sep, pathsep, defpath, extsep, altsep, devnull


# --- Functions ---

def listdir(path: Union[str, Tensor], list_dim: Shape = batch('files'), file_filter: Callable[[str, str], bool] = None, full_paths=False) -> Tensor:
    """
    Returns a `Tensor` of all entries in the directory or directory batch given by `path`.

    Args:
        path: Single directory as `str` or multiple directories as string `Tensor`.
        list_dim: Dim along which to list entries.
        file_filter: (Optional) Function with signature `(directory: str, filename: str) -> bool` used for filtering the files.
        full_paths: Whether to return the full paths or just the file names.

    Returns:
        `Tensor` with all dims of `path` and `list_dim`. If directories contain different numbers of entries, the result `Tensor` will be non-uniform.
    """
    if isinstance(path, Tensor):
        assert path.shape.isdisjoint(list_dim), f"list_dim {list_dim} is already contained in path {path.shape}."
    def list_single(path: str):
        files = impl.listdir(path)
        if file_filter is not None:
            files = filter(lambda f: file_filter(path, f), files)
        if full_paths:
            files = [impl.path.join(path, f) for f in files]
        return layout(files, list_dim)
    return map(list_single, path)


def list_files(directory: Union[str, Tensor], list_dim: Shape = batch('files'), startswith: str = None, endswith: str = None, full_paths=True):
    """
    List all files contained directly in `directory`. Unlike `listdir`, only returns a list of files, excluding folders.

    Args:
        directory: Single directory as `str` or multiple directories as string `Tensor`.
        list_dim: Dim along which to list entries.
        startswith: (Optional) List only files whose names start with this string.
        endswith: (Optional) List only files whose names end with this string.
        full_paths: Whether to return the full paths or just the file names.

    Returns:
        `Tensor` with all dims of `path` and `list_dim`. If directories contain different numbers of entries, the result `Tensor` will be non-uniform.
    """
    def file_filter(dir: str, name: str):
        if startswith is not None and not name.startswith(startswith):
            return False
        if endswith is not None and not name.endswith(endswith):
            return False
        path = impl.path.join(dir, name)
        return impl.path.isfile(path)
    return listdir(directory, list_dim=list_dim, file_filter=file_filter, full_paths=full_paths)


def list_directories(directory: Union[str, Tensor], list_dim: Shape = batch('subdirs'), startswith: str = None, endswith: str = None, full_paths=True):
    """
    List all directories contained directly in `directory`. Unlike `listdir`, only returns a list of folders, excluding files.

    Args:
        directory: Single directory as `str` or multiple directories as string `Tensor`.
        list_dim: Dim along which to list entries.
        startswith: (Optional) List only directories whose names start with this string.
        endswith: (Optional) List only directories whose names end with this string.
        full_paths: Whether to return the full paths or just the file names.

    Returns:
        `Tensor` with all dims of `path` and `list_dim`. If directories contain different numbers of entries, the result `Tensor` will be non-uniform.
    """
    def dir_filter(dir: str, name: str):
        if startswith is not None and not name.startswith(startswith):
            return False
        if endswith is not None and not name.endswith(endswith):
            return False
        path = impl.path.join(dir, name)
        return impl.path.isdir(path)
    return listdir(directory, list_dim=list_dim, file_filter=dir_filter, full_paths=full_paths)


makedirs = broadcast(impl.makedirs, name=False)
mkdir = broadcast(impl.mkdir, name=False)
rmdir = broadcast(impl.rmdir, name=False)
remove = broadcast(impl.remove, name=False)
unlink = broadcast(impl.unlink, name=False)
rename = broadcast(impl.rename, name=False)
replace = broadcast(impl.replace, name=False)
chmod = broadcast(impl.chmod, name=False)
if hasattr(impl, 'chown'):
    chown = broadcast(impl.chown, name=False)
stat = broadcast(impl.stat, name=False)
lstat = broadcast(impl.lstat, name=False)
access = broadcast(impl.access, name=False)
symlink = broadcast(impl.symlink, name=False)
link = broadcast(impl.link, name=False)
readlink = broadcast(impl.readlink, name=False)

# --- Base functions, available on all systems ---
from os import (
    abort,
    chdir,
    close,
    closerange,
    cpu_count,
    device_encoding,
    dup,
    dup2,
    error,
    execl,
    execle,
    execlp,
    execlpe,
    execv,
    execve,
    execvp,
    execvpe,
    fdopen,
    fsdecode,
    fsencode,
    fspath,
    fstat,
    fsync,
    ftruncate,
    get_exec_path,
    get_inheritable,
    get_terminal_size,
    getcwd,
    getcwdb,
    getenv,
    getlogin,
    getpid,
    getppid,
    isatty,
    kill,
    lseek,
    open,
    pipe,
    popen,
    putenv,
    read,
    removedirs,
    renames,
    scandir,
    set_inheritable,
    spawnl,
    spawnle,
    spawnv,
    spawnve,
    stat_result,
    statvfs_result,
    strerror,
    system,
    terminal_size,
    times,
    times_result,
    truncate,
    umask,
    uname_result,
    urandom,
    utime,
    waitpid,
    walk,
    write,
)

# --- Additional functions depending on OS and Python version ---
for name in dir(impl):
    if name not in globals():
        globals()[name] = getattr(impl, name)
