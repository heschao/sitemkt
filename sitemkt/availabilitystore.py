import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime, date, timedelta
from typing import List

import numpy as np
import pandas as pd
from craniutil.bulk import bulk_upload
from craniutil.dbtest.testdb import TestDb
from dateutil.relativedelta import relativedelta
from sqlalchemy import func
from sqlalchemy.orm import Session

from sitemkt.data import Availability, SiteDateAvailable
from sitemkt.model import RawAvailable, Base, SiteAvailable, Site, Campground

logger = logging.getLogger(__name__)


class AvailabilityStore(metaclass=ABCMeta):
    @abstractmethod
    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        pass

    @abstractmethod
    def get(self, timestamp: date) -> SiteDateAvailable:
        pass

    @abstractmethod
    def get_url(self) -> str:
        pass


class DbAvailabilityStore(AvailabilityStore):
    def get_url(self) -> str:
        return "https://recreation.gov" + self.session.query(Campground).filter(
            Campground.park_id == self.park_id).first().url

    def __init__(self, session, park_id):
        self.session = session
        self.park_id = park_id

    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        x = a.to_series().reset_index().assign(t0=t0, t1=t1)
        x = x.assign(t0=t0, t1=t1)
        x = x.rename(columns={'site': 'site_name', 'is_available': 'availability'})
        self.update_sites(site_names=x.site_name.values)

        x = self.map_to_site_id(x)
        x = x.assign(availability=x.availability.astype(int))
        cols = ['site_id', 't0', 't1', 'date', 'availability']
        data = [cols] + x[cols].values.tolist()

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
            y = self.map_to_site_name(y)

            return SiteDateAvailable(sites=y.site_name, dates=y.columns, is_available=y.availability)

    def map_to_site_id(self, x: pd.DataFrame) -> pd.DataFrame:
        m = self.get_site_map()
        u = x.merge(m, on='site_name', how='left')
        assert not u.site_id.isnull().any()
        return u

    def get_site_map(self):
        q = self.session.query(Site.id.label('site_id'), Site.name.label('site_name')).filter(
            Site.park_id == self.park_id)
        m = pd.read_sql_query(q.statement, self.session.connection())
        return m

    def map_to_site_name(self, y):
        m = self.get_site_map()
        u = y.merge(m, on='site_id', how='left')
        assert not u.site_name.isnull().any()
        return u

    def update_sites(self, site_names: List[str]):
        existing_site_names = [x.name for x in self.session.query(Site.name).filter(Site.park_id == self.park_id).all()]
        new_site_names = set(site_names).difference(set(existing_site_names))
        for site_name in new_site_names:
            self.session.add(Site(park_id=self.park_id, name=site_name))
        self.session.flush()


class TestDbStore(TestDb):
    @classmethod
    def setUpClass(cls):
        super(TestDbStore, cls).setUpClass()
        cls.session.add_all([
            Campground(park_id=1, name='Doe Point', state='OR'),
            Campground(park_id=2, name='Dowdy Lake', state='CO'),
        ])
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        super(TestDbStore, cls).tearDownClass()

    @classmethod
    def base(cls):
        return Base

    def test_put_0(self):
        try:
            instance = DbAvailabilityStore(self.session, park_id=1)
            sites = ['1', '2', '3']
            dates = [date(2018, 1, 1), date(2018, 2, 1)]
            is_available = (np.ones((3, 2)) * Availability.WALKIN.value)
            a = SiteDateAvailable(sites=sites, dates=dates, is_available=is_available)
            instance.put(a, datetime.utcnow(), datetime.utcnow())
            count = self.session.query(RawAvailable).count()
            assert count == 6
        finally:
            self.session.rollback()

    def test_put_1(self):
        try:
            instance = DbAvailabilityStore(self.session, park_id=1)
            sites = ['1', '2', '3']
            dates = [date(2018, 1, 1), date(2018, 2, 1)]
            is_available = (np.ones((3, 2)) * Availability.WALKIN.value)
            a = SiteDateAvailable(sites=sites, dates=dates, is_available=is_available)
            instance.put(a, datetime.utcnow(), datetime.utcnow())

            sites = ['3', '4']
            dates = [date(2018, 1, 1), date(2018, 2, 1)]
            is_available = (np.ones((2, 2)) * Availability.RESERVED.value)
            b = SiteDateAvailable(sites=sites, dates=dates, is_available=is_available)
            instance.put(b, datetime.utcnow(), datetime.utcnow())

            count = self.session.query(RawAvailable).count()
            assert count == 16, count
        finally:
            self.session.rollback()


class CompressedAvailabilityStore(AvailabilityStore):
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


class ConsoleAvailabilityStore(AvailabilityStore):
    def get_url(self) -> str:
        pass

    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        print(a)

    def get(self, timestamp: date) -> SiteDateAvailable:
        pass


class CampgroundStore(metaclass=ABCMeta):
    @abstractmethod
    def put(self, campgrounds: List[Campground]):
        pass

    @abstractmethod
    def get(self) -> List[Campground]:
        pass


class DbCampgroundStore(CampgroundStore):
    def get(self) -> List[Campground]:
        return self.session.query(Campground).all()

    def __init__(self, session: Session):
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
            n += 1
        self.session.commit()
        logger.info('added {} new campgrounds'.format(n))
