"""Single source of truth for the version.

The installer config and README download links must match this; a doc-sync
test enforces it, so bumping the version is a one-line change that CI refuses
to let drift.
"""

__version__ = "1.4.0"
