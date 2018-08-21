import os

import sitemkt

pkgdir = os.path.dirname(sitemkt.__file__)
LOGGING_CONFIG_FILE = os.path.join( os.path.dirname(pkgdir), 'config', 'logging.yml')
CONNECTION_STRING = os.environ['CONNECTION_STRING']