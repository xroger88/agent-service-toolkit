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

def decode_mongo_type(dict_obj):
  if dict_obj.get("$date", None) and len(dict_obj) == 1:
    try:
      return datetime.datetime.fromisoformat(dict_obj["$date"])
    except ValueError as e:
      print(f"*** value error in isoformat for {dict_obj}")
      return dict_obj

  return dict_obj

def convert_date_item_to_datetime_obj(doc):
  json_str = to_json(doc)
  doc = json.loads(json_str, object_hook=decode_mongo_type)
  return doc
