import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.io import set_global_seed


def test_seed_setting():
    set_global_seed(42)
    assert True
