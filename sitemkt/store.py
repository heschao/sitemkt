import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime, date, timedelta
from typing import List

import numpy as np
import pandas as pd
from craniutil.bulk import bulk_upload
from craniutil.dbtest.testdb import TestDb
from dateutil.relativedelta import relativedelta
from nose.tools import set_trace
from sqlalchemy import func
from sqlalchemy.orm import Session

from sitemkt.data import Availability, SiteDateAvailable
from sitemkt.model import RawAvailable, Base, SiteAvailable, Site, Campground

logger = logging.getLogger(__name__)

class Store(metaclass=ABCMeta):
    @abstractmethod
    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        pass

    @abstractmethod
    def get(self, timestamp: date) -> SiteDateAvailable:
        pass

    @abstractmethod
    def get_url(self) -> str:
        pass


class DbStore(Store):
    def get_url(self) -> str:
        return self.session.query(Campground).filter(Campground.park_id==self.park_id).first().url


    def __init__(self, session, park_id):
        self.session = session
        self.park_id = park_id

    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        x = a.to_series().reset_index().assign(t0=t0, t1=t1)
        x = x.assign(t0=t0, t1=t1)
        x = x.rename(columns={'site': 'site_number', 'is_available': 'availability'})
        x = self.map_to_site_id(x)
        cols = ['site_id', 't0', 't1', 'date', 'availability']
        data = [cols] + x[cols].values.tolist()

        self.session.query(RawAvailable).filter(RawAvailable.site_id.in_(x.site_id.drop_duplicates())).filter(
            RawAvailable.date.in_(x.date.drop_duplicates())
        ).delete(synchronize_session=False)

        bulk_upload(cls=RawAvailable, session=self.session, table_data=data)
        self.session.commit()

    def get(self, timestamp: date) -> SiteDateAvailable:
        for x in self.session.query(
                RawAvailable.site_id,
                func.max(RawAvailable.t1).label('t1')
        ).filter(
            RawAvailable.t1 <= timestamp + timedelta(days=1)
        ).group_by(
            RawAvailable.site_id
        ).all():
            q = self.session.query(
                RawAvailable.site_id,
                RawAvailable.date,
                RawAvailable.availability
            ).filter(
                (RawAvailable.site_id == x.site_id) & \
                (RawAvailable.t1 == x.t1)
            ).order_by(
                RawAvailable.site_id,
                RawAvailable.date,
            )
            y = pd.read_sql_query(q.statement, self.session.connection(), index_col=['site_id', 'date'])
            y = y.unstack('date').reset_index()
            y = self.map_to_site_number(y)

            return SiteDateAvailable(sites=y.site_number, dates=y.columns, is_available=y.availability)

    def map_to_site_id(self, x:pd.DataFrame) -> pd.DataFrame:
        m = self.get_site_map()
        u = x.merge(m,on='site_number',how='left')
        assert not u.site_id.isnull().any()
        return u

    def get_site_map(self):
        q = self.session.query(Site.id.label('site_id'), Site.number.label('site_number')).filter(
            Site.park_id == self.park_id)
        m = pd.read_sql_query(q.statement, self.session.connection())
        return m

    def map_to_site_number(self, y):
        m = self.get_site_map()
        u = y.merge(m,on='site_id',how='left')
        assert not u.site_number.isnull().any()
        return u


class TestDbStore(TestDb):
    @classmethod
    def setUpClass(cls):
        super(TestDbStore,cls).setUpClass()
        cls.session.add_all([
            Campground(park_id=1,name='Doe Point',state='OR'),
            Campground(park_id=2,name='Dowdy Lake',state='CO'),
        ])
        cls.session.flush()
        cls.session.add_all([
            Site(id='1-1', number=1, park_id=1),
            Site(id='1-2', number=2, park_id=1),
            Site(id='1-3', number=3, park_id=1),
            Site(id='1-4', number=4, park_id=1),
            Site(id='2-3', number=3, park_id=2),
            ]
        )
        cls.session.commit()


    @classmethod
    def tearDownClass(cls):
        super(TestDbStore,cls).tearDownClass()


    @classmethod
    def base(cls):
        return Base

    def test_put_0(self):
        try:
            instance = DbStore(self.session, park_id=1)
            sites = [1, 2, 3]
            dates = [date(2018, 1, 1), date(2018, 2, 1)]
            is_available = (np.ones((3, 2)) * Availability.WALKIN.value).astype(int)
            a = SiteDateAvailable(sites=sites, dates=dates, is_available=is_available)
            instance.put(a, datetime.utcnow(), datetime.utcnow())
            count = self.session.query(RawAvailable).count()
            assert count == 6
        finally:
            self.session.rollback()

    def test_put_1(self):
        try:
            instance = DbStore(self.session, park_id=1)
            sites = [1, 2, 3]
            dates = [date(2018, 1, 1), date(2018, 2, 1)]
            is_available = (np.ones((3, 2)) * Availability.WALKIN.value).astype(int)
            a = SiteDateAvailable(sites=sites, dates=dates, is_available=is_available)
            instance.put(a, datetime.utcnow(), datetime.utcnow())

            sites = [3, 4]
            dates = [date(2018, 1, 1), date(2018, 2, 1)]
            is_available = (np.ones((2, 2)) * Availability.RESERVED.value).astype(int)
            b = SiteDateAvailable(sites=sites, dates=dates, is_available=is_available)
            instance.put(b, datetime.utcnow(), datetime.utcnow())

            count = self.session.query(RawAvailable).count()
            assert count == 8

            result = self.session.query(RawAvailable).filter(RawAvailable.site_id == '1-3').first()
            assert result.availability == Availability.RESERVED.value, result
        finally:
            self.session.rollback()


class CompressedStore(Store):
    def get(self, d: date) -> SiteDateAvailable:
        z = []
        for x in self.session.query(
                SiteAvailable.site_id,
                func.max(SiteAvailable.t1).label('t1')
        ).filter(SiteAvailable.t1 <= d + timedelta(days=1)).all():
            q = self.session.query(SiteAvailable).filter(
                (SiteAvailable.site_id == x.site_id) & \
                (SiteAvailable.t1 == x.t1)
            )
            y = pd.read_sql_query(q.statement, self.session.connection())
            z.append(y)
        w = pd.concat(z)
        is_available = []
        sites = []
        dates = self.build_dates(date)
        for t in w.itertuples():
            sites.append(t.site_id)
            is_available.append(self.decode(t.t1, (t.c0, t.c1, t.c2, t.c3, t.c4, t.c5, t.c6,)))
        return SiteDateAvailable(sites, dates, np.array(is_available))

    def __init__(self, session):
        self.session = session

    def put(self, a: SiteDateAvailable, t0, t1):
        pass

    def build_dates(self, date) -> List[date]:
        dmax = date + relativedelta(months=6)
        d = date + relativedelta(days=1)
        result = []
        while d <= dmax:
            result.append(d)
        return result

    def decode(self, t1, codes):
        pass


class ConsoleStore(Store):
    def get_url(self) -> str:
        pass

    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        print(a)

    def get(self, timestamp: date) -> SiteDateAvailable:
        pass


class CampgroundStore(metaclass=ABCMeta):
    @abstractmethod
    def put(self,campgrounds:List[Campground]):
        pass


class CampgroundStoreDb(CampgroundStore):
    def __init__(self, session:Session):
        self.session = session

    def put(self, campgrounds: List[Campground]):
        logger.info('got {} campgrounds'.format(len(campgrounds)))
        existing = set([x.park_id for x in self.session.query(Campground.park_id).all()])
        n = 0
        for c in campgrounds:
            if c.park_id in existing:
                logger.warning('{} already exists'.format(c))
                continue
            self.session.add(c)
            n+=1
        self.session.commit()
        logger.info('added {} new campgrounds'.format(n))

