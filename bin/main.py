import click
from sqlalchemy import create_engine

from sitemkt import config
from sitemkt.model import Base


@click.group()
def main():
    pass

@main.command(name='init-db')
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


if __name__ == "__main__":
    main()