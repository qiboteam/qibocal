import json


def deserialize(raw: dict):
    """Deserialization of nested dict."""
    for key, value in raw.items():
        if isinstance(value, dict):
            raw[key] = deserialize(value)

    return {load(key): value for key, value in raw.items()}


def serialize(raw: dict):
    """JSON-friendly serialization for nested dict."""
    for key, value in raw.items():
        if isinstance(value, dict):
            raw[key] = serialize(value)

    return {
        key if isinstance(key, str) else json.dumps(key): value
        for key, value in raw.items()
    }


# TODO: distinguish between tuples and strings
def load(key):
    """Evaluate key converting string of lists to tuples."""
    raw_load = json.loads(key)
    if isinstance(raw_load, list):
        return tuple(raw_load)
    return raw_load
