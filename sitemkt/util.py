import logging
from logging.config import dictConfig
import yaml
from pyppeteer import launch

logger = logging.getLogger(__name__)

def config_logging(config_file):
    with open(config_file,'r') as f:
        d = yaml.load(f)
        dictConfig(d)


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