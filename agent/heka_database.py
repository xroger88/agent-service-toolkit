import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ServerSelectionTimeoutError
import logging

logger = logging.getLogger(__name__)

for name, value in os.environ.items():
    print("*** {0}: {1}\n".format(name, value))

collections = {
    "company": {
        "db_name": "heka",
        "collection_name": "company",
    },
    "user": {
        "db_name": "heka",
        "collection_name": "user",
    },
    "organization": {
        "db_name": "heka",
        "collection_name": "organization",
    },
    "reset_token": {
        "db_name": "heka",
        "collection_name": "reset_token",
    },
    "token_blacklist": {
        "db_name": "heka",
        "collection_name": "token_blacklist",
    },
    "verification": {"db_name": "heka", "collection_name": "verification"},
    "exchange_rate": {
        "db_name": "heka",
        "collection_name": "exchange_rate",
    },
    "aws_account": {
        "db_name": "heka",
        "collection_name": "aws_account",
    },
    "ncp_account": {
        "db_name": "heka",
        "collection_name": "ncp_account",
    },
    "nhn_account": {
        "db_name": "heka",
        "collection_name": "nhn_account",
    },
    "aws_invoice": {
        "db_name": "heka",
        "collection_name": "aws_invoice",
    },
    "ncp_invoice": {
        "db_name": "heka",
        "collection_name": "ncp_invoice",
    },
    "nhn_invoice": {
        "db_name": "heka",
        "collection_name": "nhn_invoice",
    },
    "nhn_invoice_usage": {
        "db_name": "heka",
        "collection_name": "nhn_invoice_usage",
    },
    "extra_info": {
        "db_name": "heka",
        "collection_name": "extra_info",
    },
    "activity_log": {
        "db_name": "heka",
        "collection_name": "activity_log",
    },
    "report_setting": {
        "db_name": "heka",
        "collection_name": "report_setting",
    },
    "white_label": {
        "db_name": "heka",
        "collection_name": "white_label",
    },
    "contract": {
        "db_name": "heka",
        "collection_name": "contract",
    },
    "notification": {
        "db_name": "heka",
        "collection_name": "notification",
    },
    "temporary_data": {
        "db_name": "heka",
        "collection_name": "temporary_data",
    },
}


def _get_database_name(db_name) -> str:
    """Get the database name, aware of testing environment"""
    env = os.getenv("TEST")
    if env:
        db_name = f"TEST_{db_name}"
    return db_name


def _check_db(mongo_client: MongoClient) -> None:
    # The ismaster command is cheap and does not require auth.
    try:
        mongo_client.admin.command("ismaster")
    except ServerSelectionTimeoutError as err:
        logger.exception("Mongdb Server is not available", exc_info=err)
        raise


def _get_database_client() -> MongoClient:
    """Create a Database connection pool."""

    mongo_client = MongoClient(
        host=os.getenv("MONGO_HOST", os.getenv("MONGO_HOST", "host.docker.internal")),
        port=int(os.getenv("MONGO_PORT", "27017")),
        username=os.getenv("MONGO_USER", "root"),
        password=os.getenv("MONGO_PASSWORD", "GrmtM0n6o"),
    )

    _check_db(mongo_client=mongo_client)

    return mongo_client


def _get_database(db_name: str) -> Database:
    mongo_client = _get_database_client()
    db_name = _get_database_name(db_name=db_name)
    return mongo_client[db_name]


def get_database_collection(collection_details: dict[str, str]) -> Collection:
    mongo_client = _get_database(collection_details["db_name"])
    return mongo_client[collection_details["collection_name"]]


def database_cleanup() -> None:
    db_name = _get_database_name("heka")
    if "TEST_" in db_name:
        db_client = _get_database_client()
        db_client.drop_database(name_or_database=db_name)
