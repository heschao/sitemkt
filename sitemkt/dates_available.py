import asyncio
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from abc import abstractmethod, ABCMeta
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Tuple
from unittest import TestCase

import click
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from craniutil.bulk import bulk_upload
from craniutil.dbtest.testdb import TestDb
from dateutil.relativedelta import relativedelta
# noinspection PyUnresolvedReferences
from nose.tools import set_trace
from pyppeteer import launch
from sqlalchemy import func

from sitemkt import config
from sitemkt.model import get_session, SiteAvailable, RawAvailable, Base
from sitemkt.util import config_logging

URL = r'https://recreation.gov'
logger = logging.getLogger(__name__)


class Availability(Enum):
    AVAILABLE = 1
    RESERVED = 2
    WALKIN = 3
    NOT_AVAILABLE = 4


class SiteDateAvailable(object):
    def __init__(self, sites: List[str] = None, dates: List[date] = None, is_available: np.ndarray = None,
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
        # if is_available.dtype.type == np.int64:
        #     raise TypeError('is_available must be int64 type but it is {}'.format(is_available.dtype.type))
        self.sites = sites
        self.dates = dates
        self.is_available = is_available

    def to_series(self) -> pd.Series:
        x = pd.DataFrame(data=self.is_available, index=self.sites, columns=self.dates)
        x.columns.name = 'date'
        x.index.name = 'site'
        return x.stack('date').rename('is_available')

    def append(self, other):
        t0 = min(self.timestamps[0], other.timestamps[0])
        t1 = max(self.timestamps[1], other.timestamps[1])
        z = pd.concat([self.to_series(), other.to_series()])
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
        a = SiteDateAvailable(['1', '2'], [date(2018, 1, 1), date(2018, 1, 2)], np.ones((2, 2)))
        b = SiteDateAvailable(['1', '2'], [date(2018, 1, 3), date(2018, 1, 5), date(2018, 1, 6)], np.ones((2, 3)))
        result = a.append(b)
        assert result.is_available.shape == (2, 5)

    def test_append_1(self):
        a = SiteDateAvailable()
        b = SiteDateAvailable(['1', '2'], [date(2018, 1, 3), date(2018, 1, 5), date(2018, 1, 6)], np.ones((2, 3)))
        result = a.append(b)
        assert result.is_available.shape == (2, 3)


class Store(metaclass=ABCMeta):
    @abstractmethod
    def put(self, a:SiteDateAvailable, t0:datetime, t1:datetime):
        pass

    @abstractmethod
    def get(self, timestamp:date) -> SiteDateAvailable:
        pass


def load_test_html() -> str:
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'doe-point.html')
    with open(test_file, 'r') as f:
        return f.read()


class DimensionException(Exception):
    pass


def find_months(table):
    td = table.find(lambda tag: tag.name == 'td' and tag.has_attr('class') and \
                                'weeknav' in tag['class'] and 'month' in tag['class'])
    span = td.find(lambda tag: tag.name == 'span')
    name = span.contents[0]
    months = parse_month_name(name)
    return months


def parse_month_name(name) -> List[date]:
    m = re.compile(r'^(\w{3} \d{4})$').search(name)
    if m:
        return [datetime.strptime(name, '%b %Y').date()]
    m = re.compile(r'^(\w{3})-(\w{3}) (\d{4})$').search(name)
    assert m, name
    return [
        datetime.strptime('{} {}'.format(m.group(1), m.group(3)), '%b %Y').date(),
        datetime.strptime('{} {}'.format(m.group(2), m.group(3)), '%b %Y').date()
    ]


def decode_dates(months, days) -> List[date]:
    dates = []
    prev_day = -1
    for day in days:
        if day < prev_day:
            month = months[1]
        else:
            month = months[0]
        prev_day = day
        dates.append(month + timedelta(days=day - 1))
    return dates


def parse_site_number(cell):
    label = cell.find(lambda tag: tag.name == 'div' and tag.has_attr("class") and \
                                  'siteListLabel' in tag['class'])
    if not label:
        return None
    a = label.find(lambda tag: tag.name == 'a')
    return a.contents[0]


def parse_cell(cell) -> Availability:
    """
    Parse a cell to get whether it's available, reserved, or walk-in
    """
    if cell.find(lambda tag: tag.name == 'a'):
        return Availability.AVAILABLE
    text = cell.contents[0].strip().upper()
    if text == 'R':
        return Availability.RESERVED
    if text == 'W':
        return Availability.WALKIN
    if text == 'X':
        return Availability.NOT_AVAILABLE
    else:
        raise DimensionException('bad cell content {}'.format(cell))


def find_site_availabilities(table) -> (List[str], np.ndarray):
    tbody = table.find(lambda tag: tag.name == 'tbody')
    rows = tbody.findAll(lambda tag: tag.name == 'tr' and not tag.has_attr('class'))
    sites = []
    y = []
    for row in rows:
        x = []
        cells = row.findAll(lambda tag: tag.name == 'td')
        site_number = parse_site_number(cells[0])
        sites.append(site_number)
        if not site_number:
            continue
        for cell in cells[2:]:
            x.append(parse_cell(cell).value)
        y.append(x)
    is_available = np.array(y)
    return sites, is_available


def find_days(table) -> List[int]:
    x = []
    for th in table.findAll(lambda tag: tag.name == 'th' and tag.has_attr('class') and \
                                        'calendar' in tag['class']):
        div = th.find(lambda tag: tag.name == 'div' and tag.has_attr('class') and \
                                  'date' in tag['class'])
        x.append(int(div.contents[0]))
    return x


class FirstLast(Enum):
    FIRST = 0
    MIDDLE = 1
    LAST = 2


def page_first_last(html) -> FirstLast:
    next, prev = get_prev_next(html)
    if prev.has_attr('class') and 'disabled' in prev['class']:
        return FirstLast.FIRST
    elif next.has_attr('class') and 'disabled' in next['class']:
        return FirstLast.LAST
    else:
        return FirstLast.MIDDLE


def get_prev_next(html):
    soup = BeautifulSoup(html, 'html.parser')
    nav = soup.select('span.pagenav')[0]
    prev, next = nav.select('a')
    return next, prev


def test_page_first_last():
    html = load_test_html()
    result = page_first_last(html)
    assert result == FirstLast.FIRST


async def rewind(page, direction):
    assert direction in ('previous', 'next')
    nav = await page.querySelector('span.pagenav')
    for link in await nav.JJ('a'):
        text = await page.evaluate('(link)=>link.textContent', link)
        if direction in text.lower():
            await page.evaluate('(link)=>link.click()', link)
            await page.waitForNavigation()
            break


class UnexpectedValueException(Exception):
    pass


async def parse_availability(page, n=20) -> SiteDateAvailable:
    html = await page.evaluate('()=>document.body.innerHTML')
    page_order = page_first_last(html)

    k = 0
    while page_order == FirstLast.MIDDLE:
        if k > n:
            logger.error('reached max iterations {} and still did not reach first page'.format(k))
            break
        k += 1
        await rewind(page, 'previous')
        html = await page.evaluate('()=>document.body.innerHTML')
        page_order = page_first_last(html)

    availability = SiteDateAvailable()
    if page_order == FirstLast.FIRST:
        direction = 'next'
        last_expected_order = FirstLast.LAST
    elif page_order == FirstLast.LAST:
        direction = 'previous'
        last_expected_order = FirstLast.FIRST
    else:
        raise UnexpectedValueException('expecting page_order to be FIRST or LAST but it is {}'.format(page_order))

    k = 0
    while k <= n:
        k += 1
        html = await page.evaluate('()=>document.body.innerHTML')
        availability = availability.append(parse_subset_availability(html))
        page_order = page_first_last(html)
        if page_order == last_expected_order:
            break
        logger.info('wind {} ...'.format(direction))
        await rewind(page, direction)
    return availability


def parse_subset_availability(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find(lambda tag: tag.name == 'table' and tag.has_attr('id') \
                                  and tag['id'] == "calendar")

    months = find_months(table)
    days = find_days(table)
    dates = decode_dates(months, days)

    sites, is_available = find_site_availabilities(table)
    return SiteDateAvailable(sites=sites, dates=dates, is_available=is_available)


def test_parse_subset_availability():
    html = load_test_html()
    result = parse_subset_availability(html)
    assert result.is_available.shape == (25, 14), result.is_available.shape


def assign_availability_dates(site, dates):
    x = []
    for i in range(len(site.avaibilities)):
        x.append((
            site.number, dates[i], site.avaibilities[i]
        ))
    return x


class ContentNotFoundException(Exception):
    pass


def parse_season_last_day(html) -> date:
    soup = BeautifulSoup(html, 'html.parser')
    div = soup.select('#campgStatus')[0]
    for s in div.contents:
        if not isinstance(s, str):
            continue
        m = re.compile(r'^.*Open through (.+)$').search(s)
        if m:
            return datetime.strptime(m.group(1).strip(), '%a %b %d %Y').date()
        else:
            logger.error("did not find season end date; giving you 6 months from now")
            return (datetime.today() + timedelta(days=183)).date()


class DbStore(Store):
    def __init__(self, session):
        self.session = session

    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        x = a.to_series().reset_index().assign(t0=t0, t1=t1).rename(columns={'site': 'site_id'})
        x = x.assign(t0 = t0, t1= t1)
        x = x.rename(columns={'site':'site_id','is_available':'availability'})
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
                (RawAvailable.site_id==x.site_id) & \
                (RawAvailable.t1 == x.t1)
            ).order_by(
                RawAvailable.site_id,
                RawAvailable.date,
            )
            y = pd.read_sql_query(q.statement, self.session.connection(), index_col=['site_id','date'])
            y = y.unstack('date').reset_index()
            return SiteDateAvailable(sites=y.site_id, dates=y.columns, is_available=y.availability)


class TestDbStore(TestDb):

    @classmethod
    def base(cls):
        return Base

    def test_put(self):
        try:
            instance = DbStore(self.session)
            sites= ['1','2','3']
            dates = [date(2018,1,1),date(2018,2,1)]
            is_available = (np.ones((3,2)) * Availability.WALKIN.value).astype(int)
            a = SiteDateAvailable(sites=sites,dates=dates,is_available=is_available)
            instance.put(a,datetime.utcnow(),datetime.utcnow())
            count = self.session.query(RawAvailable).count()
            assert count == 6
        finally:
            self.session.rollback()

class CompressedStore(Store):
    def get(self, d: date) -> SiteDateAvailable:
        z = []
        for x in self.session.query(
                SiteAvailable.site_id,
                func.max(SiteAvailable.t1).label('t1')
        ).filter(SiteAvailable.t1<=d+timedelta(days=1)).all():
            q = self.session.query(SiteAvailable).filter(
                (SiteAvailable.site_id==x.site_id)&\
                (SiteAvailable.t1==x.t1)
            )
            y = pd.read_sql_query(q.statement, self.session.connection())
            z.append(y)
        w = pd.concat(z)
        is_available = []
        sites = []
        dates = self.build_dates(date)
        for t in w.itertuples():
            sites.append(t.site_id)
            is_available.append(self.decode(t.t1, (t.c0,t.c1,t.c2,t.c3,t.c4,t.c5, t.c6,)))
        return SiteDateAvailable(sites,dates,np.array(is_available))

    def __init__(self, session):
        self.session = session

    def put(self, a: SiteDateAvailable, t0, t1):
        pass

    def build_dates(self, date) -> List[date]:
        dmax = date+relativedelta(months=6)
        d = date+relativedelta(days=1)
        result = []
        while d<=dmax:
            result.append(d)
        return result

    def decode(self, t1, codes):
        pass


def get_next_url(html) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    return urllib.parse.urljoin(URL, soup.select('#nextWeek')[0]['href'])


async def browse_to_site(url, show_ui):
    logger.info('launch browser')
    browser = await launch(
        args=['--no-sandbox', '--disable-setuid-sandbox'],
        headless=not show_ui,
        devtools=False,
        slowMo=0.5)

    page = await browser.newPage()
    await page.setViewport(dict(width=1366, height=768))
    logger.info('navigate to url')
    await page.goto(url)
    return browser, page


def build_url(park_id) -> str:
    return r'https://www.recreation.gov/camping/Doe-Point-Campground/r/campsiteCalendar.do?page=calendar&search=site&contractCode=NRSO&parkId=119240'


class ConsoleStore(Store):
    def put(self, a: SiteDateAvailable, t0: datetime, t1: datetime):
        print(a)

    def get(self, timestamp: date) -> SiteDateAvailable:
        pass



async def get_availability(park_url, show_ui=False, n=9999, store:Store=ConsoleStore()):
    window_last_day = date.today()
    season_last_day = date(2050, 1, 1)
    availability = SiteDateAvailable()
    browser, page = await browse_to_site(park_url, show_ui)
    t0 = datetime.utcnow()
    k = 0
    while window_last_day < season_last_day and k < n:
        logger.info('parse: window last day = {}; season last day = {}'.format(window_last_day,season_last_day))
        k += 1
        html = await page.evaluate('()=>document.body.innerHTML')

        # with open('site.html', 'w') as f:
        #     f.write(html)
        #     sys.exit()

        season_last_day = parse_season_last_day(html)
        this_availability = await parse_availability(page)
        availability = availability.append(this_availability)
        window_last_day = this_availability.window_last_day()
        if window_last_day < season_last_day:
            next_url = get_next_url(html)
            logger.info('go to next url {}'.format(next_url))
            await page.goto(next_url)
    t1 = datetime.utcnow()
    store.put(availability, t0=t0, t1=t1)
    await browser.close()


# TODO add asyncio unit tests


@click.command()
@click.option('--logging-config', '-l', default=config.LOGGING_CONFIG_FILE)
@click.option('--park-url', '-p', type=str)
@click.option('--show-ui', '-u', is_flag=True)
@click.option('--max-pages', '-n', default=9999)
@click.option('--write-to-console', '-c', is_flag=True)
def main(park_url, show_ui, max_pages, logging_config, write_to_console):
    config_logging(logging_config)
    store = ConsoleStore() if write_to_console else DbStore(get_session())
    asyncio.get_event_loop().run_until_complete(
        get_availability(park_url, show_ui, max_pages, store)
    )


if __name__ == "__main__":
    main()
