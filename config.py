"""Database connection settings.

Credentials come from the environment. The fallbacks are the values that were
previously hard-coded in five places, so a local checkout keeps working
untouched while a deployment can supply its own.

Engines are cached per database name and never shared between databases.
publication and all_data_aug_2024 hold 11 output names in common with
different numbers behind them, so nothing that talks to one is allowed to
reach the other by accident.
"""

import os
import threading
from urllib.parse import quote_plus

from sqlalchemy import create_engine

HOST = os.environ.get("JP_DB_HOST", "localhost")
PORT = int(os.environ.get("JP_DB_PORT", "3306"))
USER = os.environ.get("JP_DB_USER", "root")
# MYSQL_PWD is the name the migration and validation scripts already read.
PASSWORD = os.environ.get("JP_DB_PASSWORD") or os.environ.get("MYSQL_PWD") or "password"

POOL_SIZE = int(os.environ.get("JP_DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.environ.get("JP_DB_MAX_OVERFLOW", "20"))
POOL_RECYCLE_SECONDS = 3600

_ENGINES = {}
_ENGINE_LOCK = threading.Lock()


def connector_kwargs(database):
    """Arguments for a raw mysql.connector connection."""
    return {
        "host": HOST,
        "port": PORT,
        "user": USER,
        "password": PASSWORD,
        "database": database,
    }


def url(database):
    return "mysql+mysqlconnector://{}:{}@{}:{}/{}".format(
        quote_plus(USER), quote_plus(PASSWORD), HOST, PORT, database)


def engine(database, pool_size=None, max_overflow=None):
    """The pooled engine for one database, created once per process.

    pool_size and max_overflow are honoured only on the first call for a given
    database; later callers share the engine that already exists.
    """
    with _ENGINE_LOCK:
        if database not in _ENGINES:
            _ENGINES[database] = create_engine(
                url(database),
                pool_size=POOL_SIZE if pool_size is None else pool_size,
                max_overflow=MAX_OVERFLOW if max_overflow is None else max_overflow,
                pool_pre_ping=True,
                pool_recycle=POOL_RECYCLE_SECONDS,
            )
        return _ENGINES[database]
