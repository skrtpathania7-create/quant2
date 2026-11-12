import pytest

from quantbt import synthetic_ohlcv, walk_forward
from quantbt.strategies import ZScoreMeanReversion


@pytest.fixture(scope="module")
def result():
    data = synthetic_ohlcv(n=600, mean_reversion=0.05, seed=5)
    return walk_forward(
        data,
        ZScoreMeanReversion,
        {"lookback": [10, 20], "entry_z": [1.5, 2.0]},
        train_size=200,
        test_size=100,
    )


def test_folds_roll_forward_without_overlap(result):
    assert len(result.folds) == 4
    for prev, cur in zip(result.folds, result.folds[1:]):
        assert prev.test_end < cur.test_start
    for fold in result.folds:
        assert fold.train_end < fold.test_start  # never tested on training data


def test_out_of_sample_covers_only_test_windows(result):
    assert len(result.oos_returns) == 400
    assert result.oos_returns.index.is_monotonic_increasing
    assert result.oos_returns.index[0] == result.folds[0].test_start


def test_best_params_come_from_grid(result):
    for fold in result.folds:
        assert fold.best_params["lookback"] in (10, 20)
        assert fold.best_params["entry_z"] in (1.5, 2.0)


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        walk_forward(synthetic_ohlcv(n=100), ZScoreMeanReversion, {"lookback": [10]}, train_size=80, test_size=40)
