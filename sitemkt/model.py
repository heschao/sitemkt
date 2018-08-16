import os
from uuid import uuid4

from sqlalchemy import create_engine, String, Column, DateTime, Integer, Date, SmallInteger, ForeignKey, \
    UniqueConstraint
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from sitemkt import config

Base = declarative_base()

def uuidstr():
    return str(uuid4())

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
    site_id = Column(String, ForeignKey('site.id'), primary_key=True)
    t0 = Column(DateTime)
    t1 = Column(DateTime, primary_key=True)
    date = Column(Date, primary_key=True)
    availability = Column(SmallInteger)

    def __repr__(self):
        return 'RawAvailable(site_id={},t0={},t1={},date={},availability={})'.format(
            self.site_id,self.t0,self.t1,self.date,self.availability
        )

class Campground(Base):
    __tablename__ = 'campground'
    park_id = Column(Integer,primary_key=True)
    name = Column(String)
    state = Column(String)
    url = Column(String)

    def __repr__(self):
        return 'Campground(park_id={}, name={}, state={}, url={})'.format(
            self.park_id, self.name, self.state, self.url
        )

class Site(Base):
    __tablename__ = 'site'
    __table_args__ = (UniqueConstraint('park_id','number'),)
    id = Column(String,default=uuidstr,primary_key=True)
    park_id = Column(Integer, ForeignKey('campground.park_id'))
    number = Column(Integer)