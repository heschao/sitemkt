import logging
from datetime import timedelta, datetime

import click

from sitemkt import config
from sitemkt.data import SiteDateAvailable
from sitemkt.availabilitystore import CompressedAvailabilityStore
from sitemkt.model import get_session
from sitemkt.util import config_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--logging-config-file', '-l', default=config.LOGGING_CONFIG_FILE)
def main(logging_config_file):
    config_logging(logging_config_file)


def mkdate(s: str):
    return datetime.strptime(s, '%Y-%m-%d').date()


def calc_transitions(a: SiteDateAvailable, b: SiteDateAvailable) -> (int, int):
    n_changed, n_same = 0, 0
    return n_changed, n_same


@click.command(name='transition-prob')
@click.argument('day_0', type=mkdate)
@click.argument('day_1', type=mkdate)
def transition_prob_cli(day_0, day_1):
    store = CompressedAvailabilityStore(get_session())
    n_days = (day_1 - day_0).days
    timestamps = [day_0 + timedelta(days=i) for i in range(n_days)]
    a_prev = store.get(timestamps[0])
    n_changed, n_same = 0, 0
    for t in timestamps[1:]:
        a = store.get(t)
        c, s = calc_transitions(a, a_prev)
        n_changed += c
        n_same += s
        a_prev = a

    print(
        '{} changed; {} same; {:.1f}% transition probility'.format(n_changed, n_same, n_changed / (n_changed + n_same)))


if __name__ == "__main__":
    main()
