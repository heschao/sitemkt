import click
from sqlalchemy import create_engine

from sitemkt import config, campground
from sitemkt.model import Base, get_session, Campground, Site, RawAvailable
from sitemkt.store import CampgroundStoreDb
from sitemkt.util import config_logging


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
    store = CampgroundStoreDb(session=session)
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

# TODO Package into installable on hda
# TODO Set up cron job to run on hda
# TODO Visualize data from db

if __name__ == "__main__":
    cli()
