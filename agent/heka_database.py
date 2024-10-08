import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
import logging
from functools import lru_cache
from typing import List
#import pprint

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = "heka"

@lru_cache
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

@lru_cache
def _get_database_client() -> MongoClient:
    """Create a Database connection pool."""

    mongo_client = MongoClient(
        host=os.getenv("MONGO_HOST", os.getenv("MONGO_HOST", "host.docker.internal")),
        port=int(os.getenv("MONGO_PORT", "27017")),
        username=os.getenv("MONGO_USER", "root"),
        password=os.getenv("MONGO_PASSWORD", "GrmtM0n6o"),
        timeoutMS=5000,
        connectTimeoutMS=10000,
    )

    _check_db(mongo_client=mongo_client)

    return mongo_client

@lru_cache
def _get_database(db_name: str = DEFAULT_DB_NAME) -> Database:
    mongo_client = _get_database_client()
    db_name = _get_database_name(db_name=db_name)
    return mongo_client[db_name]

@lru_cache
def get_database_collection(collection_name, database_name: str = DEFAULT_DB_NAME) -> Collection:
    mongo_client = _get_database(database_name)
    return mongo_client[collection_name]

@lru_cache
def get_schema(database_name: str = DEFAULT_DB_NAME):
    # Connect to the specified database
    db = _get_database(database_name)
    # Retrieve a list of all collection names in the database
    collection_names = db.list_collection_names()
    # Initialize a dictionary to store the schema of each collection
    all_schemas = {}
    # Iterate through each collection in the database
    for collection_name in collection_names:
        # Access the specific collection
        collection = db[collection_name]
        # Retrieve all documents within the collection
        documents = collection.find({}, limit=3)
        # Initialize a dictionary to store the schema of the current collection
        schema = {}
        # Iterate through each document in the collection
        for doc in documents:
            # Iterate through each key in the document
            for key in doc:
                # If the key is not already in the schema dictionary, add it
                if key not in schema:
                    # Store the type of the value associated with the key
                    schema[key] = type(doc[key]).__name__
        # remove unnecessary attributes like '_id'
        for key in ["_id"]:
            if schema.get(key):
                schema.pop(key)
        # Assign the schema of the current collection to the all_schemas dictionary
        all_schemas[collection_name] = schema
    # Return the schema of all collections in the database
    return all_schemas


@lru_cache
def get_unique_names():
    """Find unique names in collections."""

    collections = [{"collection_name": "user",
                    "name_attributes": ["first_name", "last_name"]},
                   {"collection_name": "company",
                    "name_attributes": ["name"]},
                   {"collection_name": "organization",
                    "name_attributes": ["name"]}]

    unique_names = {}
    for coll_info in collections:
        coll_name = coll_info['collection_name']
        db = get_database_collection(coll_name)
        name_attributes = coll_info['name_attributes']
        names = set()
        if name_attributes:
            projection = {}
            for attr in name_attributes:
                projection[attr] = 1
            results = db.find({"delete_at": None},
                              {"_id": 0 } | projection)
            for doc in results:
                names.update(str(value) for key, value in doc.items() if value)
        unique_names[coll_name] = list(names)

    return unique_names


def get_user_info(id: str = None, limit: int = 1, details: bool = True) -> List[dict]:
    "Get user information in heka database"
    try:
        db = get_database_collection("user")
        query = {"deactivate": False,
                 "$or": [{"uid": id}, {"email": id}, {"first_name": id}, {"last_name": id},
                         {"uid": {"$regex": id, "$options": "i"}},
                         {"email":{"$regex": id, "$options": "i"}},
                         {"first_name": {"$regex": id, "$options": "i"}},
                         {"last_name": {"$regex": id, "$options": "i"}},
                         ],
                 } if id else {}
        if details:
            cursor = db.find(query,
                             {"_id": 0, "refresh_tokens": 0, "password": 0},
                             limit=limit).sort({"email": 1})
        else:
            cursor = db.find(query,
                             {"_id": 0,
                              "uid": 1, "created_at": 1, "deactivate": 1,
                              "email": 1, "company_id": 1,
                              "department": 1, "title": 1,
                              "phone_number": 1, "roles": 1, "organizations": 1,
                              "last_login": 1, "first_name": 1, "last_name": 1,
                              }, limit=limit).sort({"email": 1})

        result = [user for user in cursor]
        return result
    except Exception as e:
        return {"error": f"Invalid query: {e}"}

def get_user_info_by_company_name(company_name: str, limit: int = 1, details: bool = True) -> List[dict]:
    """
    Get the users of a company (MSP) in heka database.
    """
    try:
        db = get_database_collection("company")
        company = db.find_one({"name": company_name}, {"_id": 0, "uid": 1})
        if not company:
            return None

        db = get_database_collection("user")
        query = {"company_id": company["uid"]}

        if details:
            cursor = db.find(query,
                             {"_id": 0, "refresh_tokens": 0, "password": 0},
                             limit=limit).sort({"email": 1})
        else:
            cursor = db.find(query,
                             {"_id": 0,
                              "uid": 1, "created_at": 1, "deactivate": 1,
                              "email": 1, "company_id": 1,
                              "department": 1, "title": 1,
                              "phone_number": 1, "roles": 1, "organizations": 1,
                              "last_login": 1, "first_name": 1, "last_name": 1,
                              }, limit=limit).sort({"email": 1})

        result = [user for user in cursor]
        return result
    except Exception as e:
        return {"error": f"Invalid query: {e}"}


def get_company_info(id: str = None, limit: int = 1, details: bool = True) -> List[dict]:
    "Get company information in heka database"
    try:
        db = get_database_collection("company")
        query = {"$or": [{"uid": id}, {"name": id},
                         {"uid": {"$regex": id, "$options": "i"}},
                         {"name": {"$regex": id, "$options": "i"}}]} if id else {}
        if details:
            cursor = db.find(query,
                             {"_id": 0, "deleted_at": 0, "updated_at": 0, "trial_date": 0,
                              }, limit=limit).sort({"name": 1})
        else:
            cursor = db.find(query,
                             {"_id": 0, "uid": 1, "created_at": 1, "deactivate": 1,
                              "name": 1, "email": 1, "representative": 1,
                              "address": 1, "telephone": 1, "website": 1,
                              }, limit=limit).sort({"name": 1})

        result = [company for company in cursor]
        return result
    except Exception as e:
        return {"error": f"Invalid query: {e}"}

def get_organization_info(id: str = None, limit: int = 1, details: bool = True) -> List[dict]:
    "Get organization information in heka database"
    try:
        db = get_database_collection("organization")
        query = {"$or": [{"uid": id}, {"name": id},
                         {"uid": {"$regex": id, "$options": "i"}},
                         {"name": {"$regex": id, "$options": "i"}}]} if id else {}
        if details:
            cursor = db.find(query,
                             {"_id": 0, "deleted_at": 0, "updated_at": 0,
                              }, limit=limit).sort({"name": 1})
        else:
            cursor = db.find(query,
                             {"_id": 0, "uid": 1, "created_at": 1, "status": 1,
                              "name": 1, "representative": 1, "website": 1,
                              "company_id": 1, "telephone": 1, "business_location": 1,
                              "cloud_accounts": 1, "last_invoiced": 1,
                              }, limit=limit).sort({"name": 1})

        result = [org for org in cursor]
        return result
    except Exception as e:
        return {"error": f"Invalid query: {e}"}

def get_organization_info_by_company_name(company_name: str, limit: int = 1, details: bool = True) -> List[dict]:
    """
    Get the organizations of a company (MSP) in heka database.
    """
    try:
        db = get_database_collection("company")
        company = db.find_one({"name": company_name}, {"_id": 0, "uid": 1})
        if not company:
            return None

        db = get_database_collection("organization")
        query = {"company_id": company["uid"]}
        if details:
            cursor = db.find(query,
                             {"_id": 0, "deleted_at": 0, "updated_at": 0,
                              }, limit=limit).sort({"name": 1})
        else:
            cursor = db.find(query,
                             {"_id": 0, "uid": 1, "created_at": 1, "status": 1,
                              "name": 1, "representative": 1, "website": 1,
                              "company_id": 1, "telephone": 1, "business_location": 1,
                              "cloud_accounts": 1, "last_invoiced": 1,
                              }, limit=limit).sort({"name": 1})

        result = [org for org in cursor]
        return result
    except Exception as e:
        return {"error": f"Invalid query: {e}"}


# def insert(insert_data, collection_name, database_name: str = DEFAULT_DB_NAME):
#     # Access the database using the global CLIENT instance and DATABASE_NAME
#     db = _get_database(database_name)
#     # Access the specified collection within the database
#     collection = db[collection_name]
#     # Perform the insertion of the provided data into the collection
#     result = collection.insert_one(insert_data)
#     # Return a success message including the ID of the inserted document
#     return f"Insertion successful. Document ID: {result.inserted_id}"

exclude_projection = {"_id": 0, "refresh_tokens": 0, "password": 0, }

def find(collection_name:str, query:dict = {}, projection:dict = {},  limit:int = 30):
    "Perform the find operation for mongo database collection"
    try:
        db = _get_database(DEFAULT_DB_NAME)
        # Access the specified collection within the database
        collection = db[collection_name]
        # Perform the query to find documents matching the criteria
        if not projection:
            new_projection = exclude_projection
        else:
            new_projection = projection | {"_id": 0}

            print(f"*** projection={projection}, new_projection={new_projection}")
        documents = collection.find(query,
                                    projection=new_projection,
                                    limit=limit)
        # Convert the cursor to a list and return the documents found
        return list(documents)
    except OperationFailure as e:
        print(f"*** pymongo operation failure: {e}")
        return str(e)

def aggregate(collection_name:str, pipeline:List[dict] = []):
    "Perform the aggreate operation using pipeline for mongo database collection"
    try:
        db = _get_database(DEFAULT_DB_NAME)
        # Access the specified collection within the database
        collection = db[collection_name]
        result = []

        if not pipeline:
            # do not list up all docs fro empty pipeline
            pipeline.append({"$limit": 1})

        pipeline.append({"$project": {"_id": 0}})

        cursor = collection.aggregate(pipeline)
        for doc in cursor:
            result.append(doc)
        return result
    except OperationFailure as e:
        print(f"*** pymongo operation failure: {e}")
        return str(e)

def count_documents(collection_name:str, filter:dict = {}):
    "Get the total number of documents in mongo database collection"
    try:
        db = _get_database(DEFAULT_DB_NAME)
        # Access the specified collection within the database
        collection = db[collection_name]
        return collection.count_documents(filter)
    except OperationFailure as e:
        print(f"*** pymongo operation failure: {e}")
        return str(e)


# def database_cleanup() -> None:
#     db_name = _get_database_name(DEFAULT_DB_NAME)
#     if "TEST_" in db_name:
#         db_client = _get_database_client()
#         db_client.drop_database(name_or_database=db_name)
