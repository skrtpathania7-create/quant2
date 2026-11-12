import warnings

import pandas as pd
import pytest

from conftest import make_bars
from quantbt import CachedSource, DataSource, load_prices, synthetic_ohlcv, validate_bars


class FakeSource(DataSource):
    name = "fake"

    def __init__(self):
        self.calls = []
        self.frame = synthetic_ohlcv(n=500, start="2020-01-01", seed=1)

    def fetch(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        return self.frame.loc[start:end]


def test_cache_serves_repeat_and_narrower_requests(tmp_path):
    fake = FakeSource()
    cached = CachedSource(fake, tmp_path)
    kwargs = dict(source=cached, validate=False)
    first = load_prices("X", "2020-03-01", "2020-12-31", **kwargs)["X"]
    again = load_prices("X", "2020-03-01", "2020-12-31", **kwargs)["X"]
    inner = load_prices("X", "2020-06-01", "2020-09-30", **kwargs)["X"]
    assert len(fake.calls) == 1
    pd.testing.assert_frame_equal(first, again, check_freq=False)
    assert inner.index[0] >= pd.Timestamp("2020-06-01") and inner.index[-1] <= pd.Timestamp("2020-09-30")


def test_cache_extends_when_range_grows(tmp_path):
    fake = FakeSource()
    cached = CachedSource(fake, tmp_path)
    load_prices("X", "2020-06-01", "2020-09-30", source=cached, validate=False)
    wider = load_prices("X", "2020-03-01", "2020-12-31", source=cached, validate=False)["X"]
    assert len(fake.calls) == 2
    assert fake.calls[1][1:] == (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"))
    assert wider.index[0] <= pd.Timestamp("2020-03-03")


def test_refresh_ignores_cache(tmp_path):
    fake = FakeSource()
    load_prices("X", "2020-03-01", "2020-06-01", source=CachedSource(fake, tmp_path), validate=False)
    load_prices("X", "2020-03-01", "2020-06-01", source=CachedSource(fake, tmp_path, refresh=True), validate=False)
    assert len(fake.calls) == 2


def test_clean_data_has_no_issues():
    assert validate_bars(synthetic_ohlcv(n=300, seed=2)).empty


def test_validation_flags_bad_bars():
    df = make_bars(
        [
            (100, 101, 99, 100, 1e6),
            (100, 99, 98, 100, 1e6),  # high below open/close
            (100, 101, 99, 100, 0),  # zero volume
            (100, 160, 99, 150, 1e6),  # +50% in a day
            (150, 151, 149, 150, 1e6),
        ]
    )
    df.index = df.index[:4].append(pd.DatetimeIndex([df.index[3] + pd.Timedelta(days=30)]))  # long gap
    found = set(validate_bars(df)["check"])
    assert found == {"high below open/close", "zero volume", "move over 25%", "gap over 7 days"}


def test_load_prices_warns_on_suspicious_data():
    class Bad(DataSource):
        def fetch(self, symbol, start, end):
            return make_bars([(100, 101, 99, 100, 1e6), (100, 101, 99, 100, 0)], start="2024-01-01")

    with pytest.warns(UserWarning, match="zero volume"):
        load_prices("X", "2024-01-01", "2024-01-31", source=Bad())


def test_empty_result_raises():
    class Empty(DataSource):
        def fetch(self, symbol, start, end):
            return make_bars([(1, 1, 1, 1, 1)]).iloc[0:0]

    with pytest.raises(ValueError):
        load_prices("X", "2024-01-01", "2024-01-31", source=Empty())
