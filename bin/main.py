import asyncio
import logging
import sys

import click
from sqlalchemy import create_engine

from sitemkt import config, campground
from sitemkt.dates_available import get_availability
from sitemkt.model import Base, get_session, Campground, Site, RawAvailable
from sitemkt.availabilitystore import DbCampgroundStore, DbAvailabilityStore
from sitemkt.util import config_logging, GracefulKiller

logger = logging.getLogger(__name__)

@click.group()
def cli():
    config_logging(config.LOGGING_CONFIG_FILE)

@cli.command(name='init-db')
@click.option('--clean', '-c', is_flag=True, help='drop tables')
@click.option('--connection-string', '-s', default=config.CONNECTION_STRING)
def init_db(clean: bool, connection_string):
    engine = create_engine(connection_string)
    if clean:
        click.echo('drop tables...')
        Base.metadata.drop_all(bind=engine)
    click.echo('create tables...')
    Base.metadata.create_all(bind=engine, checkfirst=True)
    click.echo('done')


@cli.command(name='populate-campgrounds')
@click.option('--state','-s', default='CO')
@click.option('--show-ui','-u', is_flag=True)
@click.option('--max-pages','-n', default=99999)
def populate_campgrounds_cli(state, show_ui, max_pages):
    session = get_session()
    store = DbCampgroundStore(session=session)
    campgrounds = campground.search_state(state_code=state, show_ui=show_ui, max_pages=max_pages)
    store.put(campgrounds)

@cli.command(name='status')
def status_cli():
    session = get_session()
    n_campgrounds = session.query(Campground).count()
    n_sites = session.query(Site).count()
    n_records = session.query(RawAvailable).count()
    print('''
{:12,.0f} campgrounds
{:12,.0f} sites
{:12,.0f} records
'''.format(n_campgrounds,n_sites,n_records))

@cli.command(name='check-availability')
@click.option('--max-pages','-n',default=9999)
def check_availability_cli(max_pages):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(check_all_sites(max_pages))


async def check_all_sites(max_pages):
    campground_store = DbCampgroundStore(session=get_session())
    for c in campground_store.get():
        logger.info('campground={}'.format(c))
        store = DbAvailabilityStore(session=get_session(), park_id=c.park_id)
        try:
            await get_availability(show_ui=False, n=max_pages, store=store)
        except KeyboardInterrupt as e:
            logger.info('break detected... exit')
            break
        except Exception as e:
            logger.error('failed with {}; moving on...'.format(e))
            


if __name__ == "__main__":
    cli()

# TODO site number is really a string
