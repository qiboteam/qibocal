import json
from typing import Union


class Key:
    """Custom key loading and dumping.

    Class to handle keys for np.savez. Wrapper to json.dumps and json.loads
    with list to tuple conversion.

    """

    def load(self, key) -> str:
        """Convert key from str to original format plus
        explict conversion from list to tuple (not recursive yet)."""
        raw_load = json.loads(key)
        if isinstance(raw_load, list):
            return tuple(raw_load)
        return raw_load

    def dump(self, key: str) -> Union[str, int, tuple]:
        """Convert key to string using json.dumps"""
        return json.dumps(key)
