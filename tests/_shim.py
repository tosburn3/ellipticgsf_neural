"""
Minimal stand-in for the slice of pytest this suite uses.

The tests are written as ordinary pytest tests and should be run with pytest
where it is available. This shim only exists so the suite still runs in a bare
environment (`python -m tests.run`), which is useful on a
cluster login node or a fresh container.
"""

from __future__ import annotations


class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        names = [s.strip() for s in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def decorator(func):
            cases = []
            for values in argvalues:
                if not isinstance(values, (tuple, list)):
                    values = (values,)
                cases.append(dict(zip(names, values)))
            func._param_cases = cases
            return func

        return decorator


class _PytestShim:
    mark = _Mark()

    @staticmethod
    def approx(value, rel=1e-6, abs=0.0):  # noqa: A002
        raise NotImplementedError("approx is not used by this suite")


def get_pytest():
    """Return the real pytest if importable, otherwise the shim."""
    try:
        import pytest  # type: ignore

        return pytest
    except ImportError:
        return _PytestShim()
