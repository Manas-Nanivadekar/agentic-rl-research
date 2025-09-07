from typing import List, Tuple


def train_val_test_split_by_index(n: int, train: int, val: int, test: int) -> List[Tuple[int, int]]:
    """Return list of (start_index, end_index) test segments for walk-forward.

    For each segment, the training window is the preceding `train`, then `val`, then `test` spans.
    """
    spans = train + val + test
    splits = []
    start = 0
    while start + spans <= n:
        splits.append((start + train + val, start + spans))  # define test slice (val end -> test end)
        start += test  # roll forward by a test window
    return splits

