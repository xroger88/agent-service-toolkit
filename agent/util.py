import json
import datetime
from bson.objectid import ObjectId

class MongoDbEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
          return obj.astimezone().strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        if isinstance(obj, ObjectId):
          return str(obj)
        return json.JSONEncoder.default(self, obj)

def to_json(doc):
  return json.dumps(doc, cls=MongoDbEncoder)

def from_json(json_str):
  return json.loads(json_str)

def to_json_serializable_doc(doc):
  return from_json(to_json(doc))
