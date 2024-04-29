import json
import os
from pprint import pprint
import jsonschema as jsc
import tensorflow as tf
from datetime import datetime
import numpy as np

def get_manual_input(*inputs):
    "For future development. Currently all manual input are made in final JSON"
    manual_inputs = {"CreatedForProject": inputs, 
                     "CreatedAt": inputs}
    return manual_inputs

def get_config_value_type(value):
    """
    Check the type of the given value and return its type as a string.
    The returned type is one of the following: ["boolean", "number", "string", "tuple", "list", "None"].
    """
    type_map = {
        bool: "boolean",
        int: "number",
        float: "number",
        np.float32: "number",
        str: "string",
        tuple: "tuple",
        list: "list",
        type(None): "None",
    }
    
    value_type = type(value)
    return type_map.get(value_type, str(value_type))

def to_pascal_case(text):
    # Replace underscores and dashes with spaces to uniformly split the text
    text = text.replace('_', ' ').replace('-', ' ')
    # Split the text into words
    words = text.split()
    # Process all words: capitalize the first letter and join them
    return ''.join([word.capitalize() for word in words])

def to_str(config_value):
    """
    Input: object
    Output: string
    """
    if not isinstance(config_value, (int, bool, str)):
        config_value = str(config_value)
    return config_value

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def dict_to_hashable(d):
    """
    Recursively converts a dictionary into a hashable type (frozenset) by converting
    dictionaries to frozensets of tuples and lists to tuples.
    """
    if isinstance(d, dict):
        return frozenset((key, dict_to_hashable(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(dict_to_hashable(item) for item in d)
    else:
        return d

def is_all_dicts_identical(dict_list):
    """
    Checks if all dictionaries in the list are identical (including nested structures).
    Returns True and one of the dictionaries if they are all identical, 
    otherwise returns False and the list of all dictionaries.
    For checking Initializer dict.
    """
    unique_hashes = {dict_to_hashable(d): d for d in dict_list}
    
    if len(unique_hashes) == 1:
        return True, list(unique_hashes.values())[0]  # All dictionaries are identical
    else:
        return False, dict_list  # Dictionaries are not identical, return all


def to_json(mllo_dict):
    """
    Input: MLLO dict
    Output: MLLO JSON
    """
    model_name = mllo_dict['model_name']
    str_time = datetime.now()
    d = datetime.strftime(str_time, "%Y-%m-%dT%H_%M_%S%f")
    filename = f"{model_name}_{d}.json"
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(mllo_dict, file)
    return filename

def json_validation(schema_path, json_path):
    
    f = open(schema_path)
    jsonsch = json.load(f)
    f.close()

    f = open(json_path)
    print(f"load {json_path}")
    loaded_json = json.load(f)
    f.close()

    jsc.validate(loaded_json, jsonsch)
    print('pass')

    return
