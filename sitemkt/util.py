from logging.config import dictConfig
import yaml

def config_logging(config_file):
    with open(config_file,'r') as f:
        d = yaml.load(f)
        dictConfig(d)