# tests/run_tests.py

import unittest
import sys
from pathlib import Path

def main():
    # Ensure project root is on sys.path so ra_compiler can be imported
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root_dir))

    unittest.main(module=None, argv=[
        "unittest", "discover", "-s", "tests", "-p", "test*.py", "-v", "-f"
    ])

if __name__ == "__main__":
    main()
