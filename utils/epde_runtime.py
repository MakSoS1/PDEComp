"""Resolve one modern EPDE checkout for search and benchmark metrics."""
import os
import sys
from pathlib import Path

_CHECKOUT = None


def configure_epde(path=None):
    global _CHECKOUT
    candidate = path or _CHECKOUT or os.environ.get('EPDE_PATH')
    if candidate is None:
        candidate = Path(__file__).resolve().parents[1] / 'epde' / 'EPDE'
    checkout = Path(candidate).expanduser().resolve()
    if not (checkout / 'epde' / '__init__.py').is_file() or not (checkout / 'projects/pic/epde_bench/metrics.py').is_file():
        raise ValueError(f'Modern EPDE checkout required at {checkout}. Pass --epde-path or set EPDE_PATH; the legacy submodule does not provide the configuration API and current metrics.')
    loaded = sys.modules.get('epde')
    if loaded is not None and Path(loaded.__file__).resolve().parent != checkout / 'epde':
        raise ValueError('EPDE was already imported from a different checkout; start a fresh process with EPDE_PATH set.')
    sys.path.insert(0, str(checkout / 'projects/pic'))
    sys.path.insert(0, str(checkout))
    _CHECKOUT = checkout
    return checkout
