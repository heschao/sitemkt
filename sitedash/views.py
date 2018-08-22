from flask import render_template

from sitedash import app
from sitemkt.model import get_session, Campground, Site, RawAvailable


@app.route('/')
def index():
    session = get_session()
    n_campgrounds = session.query(Campground).count()
    n_sites = session.query(Site).count()
    n_records = session.query(RawAvailable).count()
    stats = {
        'n_campgrounds' : n_campgrounds,
        'n_sites' : n_sites,
        'n_records' : n_records
    }
    return render_template('index.html', stats=stats)

