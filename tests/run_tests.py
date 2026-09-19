"""Dependency-light test runner: `python tests/run_tests.py` (no pytest needed).

Only needs pandas/numpy. Broker and LLM SDKs are stubbed by the test modules,
so the whole suite runs offline.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODULES = [
    "tests.test_no_lookahead",
    "tests.test_trailing_stop",
    "tests.test_indicators",
    "tests.test_portfolio",
    "tests.test_backtest_stops",
    "tests.test_auto_execute",
]


def main() -> int:
    passed = failed = 0
    failures = []
    for modname in MODULES:
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception as exc:
            print(f"  ERROR importing {modname}: {exc}")
            failed += 1
            continue
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            try:
                getattr(mod, name)()
            except Exception:
                failed += 1
                failures.append((modname, name, traceback.format_exc()))
                print(f"  FAIL  {modname}.{name}")
            else:
                passed += 1
                print(f"  ok    {modname}.{name}")

    print(f"\n{passed} passed, {failed} failed")
    for modname, name, tb in failures:
        print(f"\n--- {modname}.{name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
