import os
import unittest
from pathlib import Path

import pandas as pd

from pymc_experimental.statespace import BayesianLocalLevel

ROOT = Path(__file__).parent.absolute()
nile = pd.read_csv(os.path.join(ROOT, "test_data/nile.csv"))
nile.index = pd.date_range(start="1871-01-01", end="1970-01-01", freq="AS-Jan")
nile.rename(columns={"x": "height"}, inplace=True)
nile = (nile - nile.mean()) / nile.std()


def test_local_level_model():
    mod = BayesianLocalLevel(data=nile.values)


if __name__ == "__main__":
    unittest.main()
