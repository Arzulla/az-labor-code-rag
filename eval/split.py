"""Golden-set dev/test split: stratified by category, random with a fixed seed (ADR-006).

The split was assigned once, before any eval run. ``tests/test_golden.py`` re-derives it
from the union of both files and checks it matches, so the split stays reproducible
without a separate draft file.
"""

import random
from collections.abc import Sequence

from labor_code_rag.models import CATEGORIES, GoldenItem

SPLIT_SEED = 20261006
DEV_SHARE = 0.55


def assign_split(
    items: Sequence[GoldenItem], seed: int = SPLIT_SEED, dev_share: float = DEV_SHARE
) -> dict[str, str]:
    """Map item id -> "dev" | "test".

    Per category: sort ids, shuffle with ``random.Random(seed)``, the first
    ``round(dev_share * n)`` go to dev. Sorting first makes the result independent of
    file order; one RNG per category keeps categories independent of each other.
    """
    split: dict[str, str] = {}
    for category in CATEGORIES:
        ids = sorted(i.id for i in items if i.category == category)
        random.Random(f"{seed}:{category}").shuffle(ids)
        n_dev = round(dev_share * len(ids))
        split.update({item_id: "dev" for item_id in ids[:n_dev]})
        split.update({item_id: "test" for item_id in ids[n_dev:]})
    return split
