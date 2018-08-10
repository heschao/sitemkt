import os

from sqlalchemy import create_engine, String, Column, DateTime, Integer, Date, SmallInteger
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from sitemkt import config

Base = declarative_base()


class ConnectionException(Exception):
    pass


def get_session(connection_string=config.CONNECTION_STRING) -> Session:
    if not connection_string:
        raise ConnectionException('empty connection string')
    engine = create_engine(connection_string)
    Session = sessionmaker(bind=engine)
    return Session()


class SiteAvailable(Base):
    __tablename__ = 'site_available'
    site_id = Column(String, primary_key=True)
    t1 = Column(DateTime, primary_key=True)
    t0 = Column(DateTime)
    c0 = Column(Integer)
    c1 = Column(Integer)
    c2 = Column(Integer)
    c3 = Column(Integer)
    c4 = Column(Integer)
    c5 = Column(Integer)
    c6 = Column(Integer)


class RawAvailable(Base):
    __tablename__ = "raw_available"
    site_id = Column(String, primary_key=True)
    t0 = Column(DateTime)
    t1 = Column(DateTime, primary_key=True)
    date = Column(Date, primary_key=True)
    availability = Column(SmallInteger)