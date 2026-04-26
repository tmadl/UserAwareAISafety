"""Small helper for scripts that need raw publication data from a source
dataset that is not redistributed in this repository.

Use:

    from _raw_data_check import require_raw
    require_raw(PATH, "Costello", "https://osf.io/gdkb7/")
"""
import sys
from pathlib import Path


def require_raw(path: Path, source_name: str, source_url: str) -> None:
    """Exit with a friendly message if `path` does not exist."""
    if path.exists():
        return
    print(
        f"\nERROR: raw {source_name} publication data missing at:\n"
        f"  {path}\n\n"
        "This script requires raw participant-level fields from the source "
        "publication, which are governed by the source-dataset licence and "
        "are not redistributed here. Download the file from\n"
        f"  {source_url}\n"
        "and place it at the path above. See docs/DATA_PROVENANCE.md for the "
        "full integration walk-through.\n",
        file=sys.stderr,
    )
    sys.exit(2)
