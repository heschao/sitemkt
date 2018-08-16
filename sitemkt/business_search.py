"""
Search Colorado Secretary of State Business Entities
"""
import asyncio
import logging
import os
from datetime import datetime, date
from time import sleep
from typing import List

import click
from bs4 import BeautifulSoup

from sitemkt import config
from sitemkt.util import config_logging, browse_to_site

logger = logging.getLogger(__name__)
URL = "https://www.sos.state.co.us/biz/AdvancedSearchCriteria.do"


# URL = "https://google.com"

@click.group()
@click.option('--logging-config-file', '-l', default=config.LOGGING_CONFIG_FILE)
def cli(logging_config_file):
    config_logging(logging_config_file)


@cli.command(name='search')
@click.option('--show-ui', '-u', is_flag=True)
@click.option('--term', '-t', default="electrical contractor")
@click.option('--outfile', '-o', default="business_search_result.txt")
@click.option('--from-date','-f',default='01/01/2001')
@click.option('--to-date','-g',default='01/01/2002')
def search_cli(show_ui, term, from_date, to_date, outfile):
    logger.info('show_ui={}, term={}, from_date={}, to_date={}, outfile={}'.format(
        show_ui, term, from_date, to_date, outfile
    ))
    asyncio.get_event_loop().run_until_complete(
        search(show_ui=show_ui, term=term, from_date=from_date, to_date=to_date, outfile=outfile)
    )


class Entry(object):
    def __init__(self, name, formation_date, event, status, form):
        self.name = name
        self.formation_date = formation_date
        self.event = event
        self.status = status
        self.form = form

    def __repr__(self):
        return 'Entry(name={}, formation_date={}, event={}, status={}, form={})'.format(
            self.name, self.formation_date, self.event, self.status, self.form)


def extract_content(cell):
    return cell.contents[0].strip()


def parse_formation_date(cell) -> date:
    s = cell.contents[0].strip()
    if s:
        return datetime.strptime(s, '%m/%d/%Y').date()
    else:
        return date(1970, 1, 1)


def parse_row(row) -> Entry:
    cells = row.findAll('td')
    return Entry(
        name=extract_content(cells[3]),
        event=extract_content(cells[4]),
        status=extract_content(cells[5]),
        form=extract_content(cells[6]),
        formation_date = parse_formation_date(cells[7]),
    )


def parse_entries(html) -> List[Entry]:
    soup = BeautifulSoup(html, 'html.parser')
    th = soup.select('th.headerNoLink')[0]
    tbody = th.find_parent().find_parent()
    rows = tbody.findAll('tr')
    entries = []
    for row in rows[1:]:
        entries.append(parse_row(row))
    return entries


def load_html():
    dirname = os.path.dirname(os.path.abspath(__file__))
    result_file = os.path.abspath(os.path.join(dirname, '..', 'data', 'business_search_result.html'))
    with open(result_file, 'r') as f:
        return f.read()


def test_parse_entries():
    html = load_html()
    result = parse_entries(html)
    assert len(result) == 20, result
    assert result[0].name == "2 - H ELECTRICAL CONTRACTORS, LLP", result[0]


def write_entries(entries: List[Entry], outfile: str):
    with open(outfile, 'w') as f:
        f.write('\t'.join(('name', 'formation_date', 'event', 'status', 'form')))
        f.write('\n')
        for entry in entries:
            f.write('\t'.join((entry.name, str(entry.formation_date), entry.event, entry.status, entry.form)))
            f.write('\n')


async def search(show_ui: bool, term: str, from_date:str, to_date:str, outfile: str):
    logger.info('search term={}, from_date={}, to_date={}'.format(term, from_date, to_date))
    browser, page = await browse_to_site(URL, show_ui)
    await page.waitForNavigation()

    await page.type('input[name="dateFrom"]', from_date)
    await page.type('input[name="dateTo"]', to_date)
    await page.type('input[name="searchName"]', term)
    checkbox = await page.querySelector('input[name="includeEntity"]')
    assert checkbox
    await page.evaluate('(checkbox) => checkbox.click()', checkbox)
    button = await page.querySelector('input[name="cmd"]')
    assert button
    await page.evaluate('(button) => button.click()', button)
    await page.waitForNavigation()

    input('hit any key')

    filename = 'site.html'
    with open('%s' % filename, 'w') as f:
        html = await page.evaluate('()=>document.body.innerHTML')
        f.write(html)
        logger.info('wrote page to {}'.format(filename))

    entries = await gather_entries(page)

    write_entries(entries, outfile)

    await browser.close()


async def a_next(page):
    for a in await page.querySelectorAll('a'):
        title = await page.evaluate('(a) => a.title', a)
        if title is not None and 'Next' in title:
            return a
    return None


async def gather_entries(page):
    html = await page.evaluate('()=>document.body.innerHTML')
    entries = parse_entries(html)
    logger.info('got {} entries from first page'.format(len(entries)))
    a = await a_next(page)

    while a:
        await page.evaluate('(a) => a.click()', a)
        await page.waitForNavigation()
        html = await page.evaluate('()=>document.body.innerHTML')
        entries += parse_entries(html)
        logger.info('entries now at {}'.format(len(entries)))
        a = await a_next(page)

    return entries


if __name__ == "__main__":
    cli()
