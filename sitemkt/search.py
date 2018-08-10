import logging
import random
import urllib.parse as urlparse
import re
import click
from time import sleep
import asyncio
from pyppeteer import launch

logger = logging.getLogger(__name__)

def extract_park_id(href):
    result = urlparse.urlparse(href)
    return int(urlparse.parse_qs(result.query)['parkId'][0])

def test_extract_park_id():
    result = extract_park_id(
        r'/unifSearchInterface.do?interface=camping&contractCode=NRSO&parkId=73562')
    assert result == 73562, result

@click.command()
@click.argument('term')
@click.option('--min-delay','-n', default = 1.0)
@click.option('--max-delay','-x', default = 5.0)
@click.option('--max-pages','-p', default = 999999)
@click.option('--show-ui','-u', is_flag=True)
@click.option('--debug','-d', is_flag=True)
def main(term, min_delay, max_delay, max_pages,show_ui, debug):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    asyncio.get_event_loop().run_until_complete(search(term, min_delay, max_delay, max_pages,show_ui))

async def search(term, min_delay, max_delay, max_pages, show_ui):
    logger.info('launch browser')
    browser = await launch(headless=not show_ui)
    page = await browser.newPage()

    logger.info('navigate to url')
    await page.goto('http://recreation.gov')

    logger.info('search term = {}'.format(term))
    await page.type('#locationCriteria',term)
    form = await page.querySelector('form')
    await page.evaluate('(form) => form.submit() ', form)
    await page.waitForNavigation()

    nextbtn = True
    pageid = 0
    while nextbtn and pageid<max_pages:
        pageid+=1
        logger.info('page {}'.format(pageid))
        for card in await page.querySelectorAll('.facility_link'):
            title = await page.evaluate('(card) => card.getAttribute("title")',card)
            href = await page.evaluate('(card) => card.getAttribute("href")',card)
            try:
                park_id = extract_park_id(href)
                print('park_id={}: {}'.format(park_id,title))
            except Exception as e:
                logger.error('failed to parse park id. title={}, href={}'.format(title,href))

        if pageid >= max_pages:
            logger.info('{} pages reached; exit loop.'.format(pageid))
            break

        delay = random.randrange(min_delay, max_delay)
        sleep(delay)
        nextbtn =  await page.querySelector('a[title="Next"]')
        if not nextbtn:
            logger.info('no "Next" button found; exit loop.')
            break
        await page.evaluate('UnifSearchEngine.nextPage()')
        await page.waitForNavigation()

    await browser.close()


if __name__ == "__main__":
    main()