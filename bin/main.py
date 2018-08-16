import click
from sqlalchemy import create_engine

from sitemkt import config, campground
from sitemkt.model import Base, get_session
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


# TODO Package into installable on hda
# TODO Set up cron job to run on hda
# TODO Visualize data from db
# TODO create unique site_id that encapsulates park_id and site_number
# TODO build url from park_id
# TODO catalogue campgrounds by name and park_id and other info

if __name__ == "__main__":
    cli()
