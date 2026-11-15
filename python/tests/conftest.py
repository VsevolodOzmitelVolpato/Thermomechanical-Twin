"""Makes the Python package of the project (python/twin/) importable from the tests."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))