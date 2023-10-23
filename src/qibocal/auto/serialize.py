import json


def deserialize(raw: dict):
    """Deserialization of nested dict."""
    return {
        # TODO: don't apply load(key) if the key is an actual string
        load(key): value if not isinstance(value, dict) else deserialize(value)
        for key, value in raw.items()
    }


def serialize(raw: dict):
    """JSON-friendly serialization for nested dict."""
    return {
        json.dumps(key): (value if not isinstance(value, dict) else serialize(value))
        for key, value in raw.items()
    }


# TODO: distinguish between tuples and strings
def load(key):
    """Evaluate key converting string of lists to tuples."""
    raw_load = json.loads(key)
    if isinstance(raw_load, list):
        return tuple(raw_load)
    return raw_load
