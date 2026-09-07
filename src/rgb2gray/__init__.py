"""
rgb2gray -- batch-convert image datasets to 8-bit grayscale.

The folder tree, the empty directories and every paired ``<stem>.json``
annotation survive the conversion; see ``README.md``.

Nothing heavy is imported here on purpose.  Under ``spawn``/``forkserver``
every worker process re-imports the package that owns the worker function, so
a convenience re-export of :mod:`rgb2gray.app` would drag Tk into each child.
Import the pieces you need explicitly::

    from rgb2gray import gray_core       # the engine, GUI-free
    from rgb2gray.app import main        # the Tk front end
"""

#: The one place the version is written; ``pyproject.toml`` reads it from here.
__version__ = "0.2.0"

__all__ = ["__version__"]
