"""Backstory loading for the cognitive interview pipeline."""

import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BACKSTORY_FILES = {
    "short": DATA_DIR / "backstories_short.csv",
    "long": DATA_DIR / "backstories_long.csv",
}


def load_backstories(n: int, backstory_type: str = "short",
                     path: str = None, seed: int = None) -> list[str]:
    """Sample *n* backstories from ESS microdata CSV.

    Args:
        n: Number of backstories to sample (without replacement).
        backstory_type: ``"short"`` (~580 chars, demographics only) or
            ``"long"`` (~1500 chars, with attitudinal variables).
        path: Override path to a specific CSV file.  If ``None``, uses the
            default file for the given *backstory_type*.
        seed: Random seed for reproducible sampling.

    Returns:
        List of backstory strings.
    """
    if path is None:
        fpath = BACKSTORY_FILES.get(backstory_type)
        if fpath is None:
            raise ValueError(
                f"Unknown backstory_type: {backstory_type}. Use 'short' or 'long'."
            )
    else:
        fpath = Path(path)

    if not fpath.exists():
        raise FileNotFoundError(
            f"Backstory file not found: {fpath}\n"
            "Backstory CSV files should be in the data/ directory."
        )

    df = pd.read_csv(fpath)
    if "backstory" not in df.columns:
        raise ValueError(
            f"CSV must have a 'backstory' column. Found: {list(df.columns)}"
        )
    all_backstories = df["backstory"].dropna().tolist()

    if len(all_backstories) < n:
        raise ValueError(
            f"Backstory file has {len(all_backstories)} entries but {n} requested"
        )

    rng = random.Random(seed)
    return rng.sample(all_backstories, n)
