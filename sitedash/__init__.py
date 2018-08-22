import os
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

__version__ = "0.1.0"

dirname = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(dirname,'templates')
static_folder = os.path.join(dirname,'static')
print('template_folder={} static_folder={}'.format(template_folder,static_folder))
app = Flask(__name__,
            template_folder=template_folder,
            static_folder=static_folder,
            static_url_path=''
            )
app.config.from_object('sitedash.config')

db = SQLAlchemy(app)
lm = LoginManager()
lm.init_app(app)
lm.login_view = 'login'
lm.login_message='Please log in to view this page'
from sitedash import views
