import pandas as pd


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
):
    """Split a DataFrame into train and test by row order.

    NOT used for TweetEval. TweetEval ships official train, validation and
    test splits and its rows carry no timestamps, so a temporal split makes
    no sense there and every training script uses the official splits.

    This function keeps the first (1 - test_size) fraction of rows as train
    and the rest as test. It only produces a meaningful temporal split when
    df is already sorted by time, oldest first. Kept for datasets where that
    holds.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test
