"""Common function for all submodules"""

import hashlib
from collections.abc import Sequence
import struct


def hash_function(data: bytes) -> int:
    """The hash function used to identify similar datasets, methods, configurations, .."""
    return int.from_bytes(hashlib.sha1(data).digest(), "big")


def hash_object_list(objects: Sequence) -> int:
    """hash objects in a list
    Will raise a TypeError if an input value has an unsupported type
    """
    type_handling = {
        int: lambda x: x,
        bytes: hash_function,
        str: lambda x: hash_function(x.encode()),
        list: hash_object_list,
        bool: lambda x: hash_function(bytes(x)),
        float: lambda x: hash_function(struct.pack("d", x)),
    }

    hashes = 0
    for value in objects:
        success = False
        for t, handler in type_handling.items():
            if isinstance(value, t):
                hashes ^= handler(value)
                success = True
                break
        if not success:
            raise TypeError(f"Hashing is not supported for {type(value)}!")
    return hashes
