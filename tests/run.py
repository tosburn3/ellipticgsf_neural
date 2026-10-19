"""
Run the suite without pytest, from the repository folder:  python -m tests.run

Prefer `pytest tests` where pytest is installed. This runner
expands the same parametrized cases and reports pass or fail per case; it
exists so the suite still runs on a cluster login node or a bare container.
"""

from __future__ import annotations

import traceback

from . import (
    test_assemble,
    test_effective_source_library,
    test_neural,
    test_problem,
    test_solve,
    test_source,
)

MODULES = (
    test_problem,
    test_assemble,
    test_source,
    test_effective_source_library,
    test_solve,
    test_neural,
)


def main() -> int:
    passed = failed = 0
    failures = []

    for module in MODULES:
        name = module.__name__.rsplit(".", 1)[-1]
        print(f"\n=== {name} ===")
        funcs = [
            (fname, obj)
            for fname, obj in sorted(vars(module).items())
            if fname.startswith("test_") and callable(obj)
        ]
        for fname, func in funcs:
            for kwargs in getattr(func, "_param_cases", [{}]):
                label = fname + (
                    f"[{', '.join(f'{k}={v}' for k, v in kwargs.items())}]" if kwargs else ""
                )
                try:
                    func(**kwargs)
                except Exception:
                    failed += 1
                    failures.append((f"{name}.{label}", traceback.format_exc()))
                    print(f"FAIL {label}")
                else:
                    passed += 1
                    print(f"ok   {label}")

    print(f"\n{passed} passed, {failed} failed")
    for label, tb in failures:
        print(f"\n--- {label} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
