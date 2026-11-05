from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from sklearn.datasets import fetch_20newsgroups


def load_20newsgroups(
    subset: str = "train",
    categories: Optional[Sequence[str]] = None,
    remove: Iterable[str] = (),
    shuffle: bool = True,
    random_state: int = 42,
    data_home: Optional[Path] = None,
):
    """Fetch 20 Newsgroups and return modeling data.

    Args:
        subset: "train", "test", or "all".
        categories: Optional list of category names to filter.
        remove: Iterable of elements to remove from the text.
        shuffle: Whether to shuffle the data.
        random_state: Random seed for shuffling.
        data_home: Optional root directory for data storage.
            Defaults to project-local data/cache. The cache is ignored by Git.

    Returns:
        A sklearn.utils.Bunch with data, target, and metadata fields.
    """
    if data_home is None:
        data_home = Path(__file__).resolve().parents[1] / "data" / "cache"
    data_home.mkdir(parents=True, exist_ok=True)

    effective_remove = tuple(dict.fromkeys(remove))

    dataset = fetch_20newsgroups(
        subset=subset,
        categories=list(categories) if categories else None,
        remove=effective_remove,
        shuffle=shuffle,
        random_state=random_state,
        data_home=str(data_home),
    )

    return dataset
