from datetime import date, datetime
from enum import Enum
from typing import List, Tuple
from unittest import TestCase

import numpy as np
import pandas as pd

from sitemkt.exception import DimensionException


class Availability(Enum):
    AVAILABLE = 1.0
    RESERVED = 2.0
    WALKIN = 3.0
    NOT_AVAILABLE = 4.0
    UNKNOWN = np.nan


class SiteDateAvailable(object):
    def __init__(self, sites: List[int] = None, dates: List[date] = None, is_available: np.ndarray = None,
                 timestamps: Tuple[datetime, datetime] = None):
        if timestamps is None:
            timestamps = [datetime.now(), datetime.now()]
        self.timestamps = timestamps

        if sites is None:
            self.sites = []
            self.dates = []
            self.is_available = np.ones((0, 0))
            return

        n_sites = len(sites)
        n_dates = len(dates)
        rows, cols = is_available.shape
        if n_sites != rows or n_dates != cols:
            raise DimensionException('dimensions mismatch')
        if not issubclass(is_available.dtype.type, np.float):
            raise TypeError('is_available must be float type but it is {}'.format(is_available.dtype.type))
        self.sites = sites
        self.dates = dates
        self.is_available = is_available

    def to_series(self) -> pd.Series:
        x = pd.DataFrame(data=self.is_available, index=self.sites, columns=self.dates)
        x.columns.name = 'date'
        x.index.name = 'site'
        return x.stack('date').rename('is_available')

    def append(self, other):
        if self.isempty():
            return other
        elif other.isempty():
            return self

        t0 = min(self.timestamps[0], other.timestamps[0])
        t1 = max(self.timestamps[1], other.timestamps[1])
        z = pd.concat([self.to_series(), other.to_series()])
        z = z.loc[~z.isnull()].sort_values()
        z = z[~z.index.duplicated(keep='first')]
        z = z.unstack('date')
        return SiteDateAvailable(
            sites=z.index.values, dates=z.columns,
            is_available=z.values,
            timestamps=(t0, t1))

    def window_last_day(self):
        return self.dates[-1]

    def __str__(self):
        if self.isempty():
            return 'empty'

        n_reserved = (self.is_available == Availability.RESERVED.value).sum().sum()
        n_available = (self.is_available == Availability.AVAILABLE.value).sum().sum()
        n_walk_in = (self.is_available == Availability.WALKIN.value).sum().sum()
        n_not_available = (self.is_available == Availability.NOT_AVAILABLE.value).sum().sum()
        n_unknown = np.isnan(self.is_available).sum().sum()
        slots = len(self.sites) * len(self.dates)
        return \
            '''
{n_sites} sites
{n_dates} dates {first} - {last}
{n_not_available} not available {pct_not_available:.1f}%
{n_reserved} reserved {pct_reserved:.1f}%
{n_available} available {pct_available:.1f}%
{n_walk_in} walk in {pct_walk_in:.1f}%
{n_unknown} unknown in {pct_unknown:.1f}%
'''.format(
                n_sites=len(self.sites), n_dates=len(self.dates), first=self.dates[0], last=self.dates[-1],
                n_not_available=n_not_available, pct_not_available=n_not_available / slots * 100,
                n_reserved=n_reserved, pct_reserved=n_reserved / slots * 100,
                n_available=n_available, pct_available=n_available / slots * 100,
                n_walk_in=n_walk_in, pct_walk_in=n_walk_in / slots * 100,
                n_unknown=n_unknown, pct_unknown=n_unknown / slots * 100,
            )

    def isempty(self):
        return self.is_available.size == 0


class TestSiteDateAvailable(TestCase):
    def test_append_0(self):
        a = SiteDateAvailable([1,2], [date(2018, 1, 1), date(2018, 1, 2)], np.ones((2, 2)))
        b = SiteDateAvailable([1,2], [date(2018, 1, 3), date(2018, 1, 5), date(2018, 1, 6)],
                              np.ones((2, 3)))
        result = a.append(b)
        assert result.is_available.shape == (2, 5)

    def test_append_1(self):
        a = SiteDateAvailable()
        b = SiteDateAvailable([1,2], [date(2018, 1, 3), date(2018, 1, 5), date(2018, 1, 6)],
                              np.ones((2, 3)))
        result = a.append(b)
        assert result.is_available.shape == (2, 3)

    def test_append_2(self):
        a = SiteDateAvailable([1,2], [date(2018, 1, 1), date(2018, 1, 2)], np.ones((2, 2)))
        b = SiteDateAvailable([1,2], [date(2018, 1, 2), date(2018, 1, 5), date(2018, 1, 6)],
                              np.ones((2, 3)))
        result = a.append(b)
        assert result.is_available.shape == (2, 4)
1