import numpy.testing as npt
import asyncio
import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Optional

import click
import numpy as np
from bs4 import BeautifulSoup
# noinspection PyUnresolvedReferences
from nose.tools import set_trace

from sitemkt import config
from sitemkt.data import Availability, SiteDateAvailable
from sitemkt.exception import DimensionException, UnexpectedValueException
from sitemkt.model import get_session
from sitemkt.store import Store, DbStore, ConsoleStore
from sitemkt.util import config_logging, browse_to_site

URL = r'https://recreation.gov'
logger = logging.getLogger(__name__)


def load_test_html() -> str:
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'doe-point.html')
    with open(test_file, 'r') as f:
        return f.read()


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
    i = 0
    for day in days:
        if day < prev_day:
            i+=1
        month = months[i]
        prev_day = day
        dates.append(month + timedelta(days=day - 1))
    return dates

def test_decode_dates():
    result = decode_dates(months=[date(2018,8,1), date(2018,9,1)],days=[31,1,2])
    npt.assert_array_equal(result, [date(2018,8,31), date(2018,9,1), date(2018,9,2)])


def parse_site_number(cell) -> Optional[int]:
    label = cell.find(lambda tag: tag.name == 'div' and tag.has_attr("class") and \
                                  'siteListLabel' in tag['class'])
    if not label:
        return None
    a = label.find(lambda tag: tag.name == 'a')
    try:
        return int(a.contents[0])
    except ValueError:
        logger.error('failed to parse as int {}'.format(a.contents[0]))
        return None


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
        if not site_number:
            continue
        sites.append(site_number)
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
    ONLY = 3


def page_first_last(html) -> FirstLast:
    next, prev = get_prev_next(html)
    no_prev = prev.has_attr('class') and 'disabled' in prev['class']
    no_next = next.has_attr('class') and 'disabled' in next['class']
    if no_prev and no_next:
        return FirstLast.ONLY
    if no_prev:
        return FirstLast.FIRST
    elif no_next:
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


async def parse_availability(page, n=20) -> SiteDateAvailable:
    html = await page.evaluate('()=>document.body.innerHTML')
    page_order = page_first_last(html)

    if page_order == FirstLast.ONLY:
        html = await page.evaluate('()=>document.body.innerHTML')
        return parse_subset_availability(html)


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

    return SiteDateAvailable(sites=sites, dates=dates, is_available=is_available) if sites else SiteDateAvailable()


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


def get_next_url(html) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    return urllib.parse.urljoin(URL, soup.select('#nextWeek')[0]['href'])


def build_url(park_id) -> str:
    return r'https://www.recreation.gov/camping/Doe-Point-Campground/r/campsiteCalendar.do?page=calendar&search=site&contractCode=NRSO&parkId=119240'


async def get_availability(show_ui=False, n=9999, store: Store = ConsoleStore()):
    window_last_day = date.today()
    season_last_day = date(2050, 1, 1)
    availability = SiteDateAvailable()
    park_url = store.get_url()
    browser, page = await browse_to_site(park_url, show_ui)

    link = await page.querySelector("#campCalendar")
    assert link
    await page.evaluate('(link) => link.click()', link)
    await page.waitForNavigation()

    # input('hit any key')
    t0 = datetime.utcnow()
    k = 0
    while window_last_day < season_last_day and k < n:
        logger.info('parse: window last day = {}; season last day = {}'.format(window_last_day,season_last_day))
        k += 1
        html = await page.evaluate('()=>document.body.innerHTML')

        season_last_day = parse_season_last_day(html)
        this_availability = await parse_availability(page)
        if this_availability.isempty():
            logger.error("empty availability; exit")
            break

        availability = availability.append(this_availability)
        window_last_day = this_availability.window_last_day()
        if window_last_day < season_last_day:
            next_url = get_next_url(html)
            logger.info('go to next url {}'.format(next_url))
            await page.goto(next_url)
    t1 = datetime.utcnow()

    if not availability.isempty():
        store.put(availability, t0=t0, t1=t1)

    await browser.close()


# TODO add asyncio unit tests


@click.command()
@click.option('--logging-config', '-l', default=config.LOGGING_CONFIG_FILE)
@click.option('--park-id', '-p', type=int)
@click.option('--show-ui', '-u', is_flag=True)
@click.option('--max-pages', '-n', default=9999)
@click.option('--write-to-console', '-c', is_flag=True)
def main(park_id, show_ui, max_pages, logging_config, write_to_console):
    config_logging(logging_config)
    store = ConsoleStore() if write_to_console else DbStore(session=get_session(), park_id=park_id)
    asyncio.get_event_loop().run_until_complete(
        get_availability(show_ui, max_pages, store)
    )


if __name__ == "__main__":
    main()
