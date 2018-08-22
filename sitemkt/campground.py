import logging
import random
import urllib.parse as urlparse
import re
import click
from time import sleep
import asyncio
from pyppeteer import launch

from sitemkt.model import Campground
from sitemkt.states import states
from sitemkt.util import browse_to_site

logger = logging.getLogger(__name__)

URL = 'http://recreation.gov'

def extract_park_id(href):
    result = urlparse.urlparse(href)
    return int(urlparse.parse_qs(result.query)['parkId'][0])

def test_extract_park_id():
    result = extract_park_id(
        r'/unifSearchInterface.do?interface=camping&contractCode=NRSO&parkId=73562')
    assert result == 73562, result

async def search(term, delay, max_pages, show_ui):
    logger.info('launch browser')
    browser, page = await browse_to_site(show_ui=show_ui, url=URL)

    logger.info('search term = {}'.format(term))
    await page.type('#locationCriteria',term)
    form = await page.querySelector('form')
    await page.evaluate('(form) => form.submit() ', form)
    await page.waitForNavigation()

    radio = await page.querySelector('input[type="radio"][value="camping"]')
    assert radio
    await page.evaluate('(radio) => radio.click()', radio)
    await page.waitForNavigation()

    campgrounds = {}
    nextbtn = True
    pageid = 0
    while nextbtn and pageid<max_pages:
        pageid+=1
        logger.info('page {}'.format(pageid))
        for card in await page.querySelectorAll('div#search_results_list > div.eufacility_view_card > div > div > div > a.facility_link'):
            title = await page.evaluate('(card) => card.getAttribute("title")',card)
            href = await page.evaluate('(card) => card.getAttribute("href")',card)
            try:
                park_id = extract_park_id(href)
                c = Campground(park_id=park_id, name=title, state='', url=href, )
                campgrounds[park_id] = c
                logger.info('got {}'.format(c))
            except Exception as e:
                logger.error('failed to parse park id. title={}, href={}'.format(title,href))

        if pageid >= max_pages:
            logger.info('{} pages reached; exit loop.'.format(pageid))
            break

        sleep(delay)
        nextbtn =  await page.querySelector('a[title="Next"]')
        if not nextbtn:
            logger.info('no "Next" button found; exit loop.')
            break
        await page.evaluate('UnifSearchEngine.nextPage()')
        await page.waitForNavigation()

    await browser.close()
    return campgrounds


def search_state(state_code:str, delay=1, max_pages=9999, show_ui=False):
    state_name = states[state_code]
    term = "{} Campgrounds".format(state_name)
    campgrounds = asyncio.get_event_loop().run_until_complete(
        search(term=term, delay=delay, max_pages=max_pages, show_ui=show_ui)
    )
    for c in campgrounds.values():
        c.state = state_code

    return list(campgrounds.values())
