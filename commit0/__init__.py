"""Commit0 Lib"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("commit0")
except PackageNotFoundError:
    __version__ = "0.0.0"
